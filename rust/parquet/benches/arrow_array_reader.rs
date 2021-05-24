use std::{collections::VecDeque, sync::Arc};
use criterion::{black_box, criterion_group, criterion_main, Criterion};
use parquet::{arrow::array_reader::ArrayReader, basic::Encoding, column::page::PageIterator, data_type::{Int32Type, ByteArrayType}, schema::types::{ColumnDescPtr, SchemaDescPtr}};

fn build_test_schema() -> SchemaDescPtr {
    use parquet::schema::{types::SchemaDescriptor, parser::parse_message_type};
    let message_type = "
        message test_schema {
            REQUIRED INT32 mandatory_int32_leaf;
            REPEATED Group test_mid_int32 {
                OPTIONAL INT32 optional_int32_leaf;
            }
            REQUIRED BYTE_ARRAY mandatory_string_leaf (UTF8);
            REPEATED Group test_mid_string {
                OPTIONAL BYTE_ARRAY optional_string_leaf (UTF8);
            }
        }
        ";
    parse_message_type(message_type)
        .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
        .unwrap()
}

// test data params
const NUM_ROW_GROUPS: usize = 1;
const PAGES_PER_GROUP: usize = 2;
const VALUES_PER_PAGE: usize = 10_000;
const BATCH_SIZE: usize = 8192;

use rand::{Rng, SeedableRng, rngs::StdRng};

pub fn seedable_rng() -> StdRng {
    StdRng::seed_from_u64(42)
}

fn build_plain_encoded_int32_page_iterator(schema: SchemaDescPtr, column_desc: ColumnDescPtr, null_density: f32) -> impl PageIterator + Clone {
    use parquet::util::test_common::page_util::{InMemoryPageIterator, DataPageBuilderImpl, DataPageBuilder};
    let max_def_level = column_desc.max_def_level();
    let max_rep_level = column_desc.max_rep_level();
    let rep_levels = vec![max_rep_level; VALUES_PER_PAGE];
    let mut rng = seedable_rng();
    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    let mut int32_value = 0;
    for _i in 0..NUM_ROW_GROUPS {
        let mut column_chunk_pages = Vec::new();
        for _j in 0..PAGES_PER_GROUP {
            // generate page
            let mut values = Vec::with_capacity(VALUES_PER_PAGE);
            let mut def_levels = Vec::with_capacity(VALUES_PER_PAGE);
            for _k in 0..VALUES_PER_PAGE {
                let def_level = if rng.gen::<f32>() < null_density {
                    max_def_level - 1
                } else {
                    max_def_level
                };
                if def_level == max_def_level {
                    int32_value += 1;
                    values.push(int32_value);
                }
                def_levels.push(def_level);
            }
            let mut page_builder = DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);
            page_builder.add_rep_levels(max_rep_level, &rep_levels);
            page_builder.add_def_levels(max_def_level, &def_levels);
            page_builder.add_values::<Int32Type>(Encoding::PLAIN, &values);
            column_chunk_pages.push(page_builder.consume());
        }
        pages.push(column_chunk_pages);
    }

    InMemoryPageIterator::new(schema, column_desc, pages)
}

fn build_dictionary_encoded_int32_page_iterator(schema: SchemaDescPtr, column_desc: ColumnDescPtr, null_density: f32) -> impl PageIterator + Clone {
    use parquet::util::test_common::page_util::{InMemoryPageIterator, DataPageBuilderImpl, DataPageBuilder};
    use parquet::encoding::{Encoder, DictEncoder};
    let max_def_level = column_desc.max_def_level();
    let max_rep_level = column_desc.max_rep_level();
    let rep_levels = vec![max_rep_level; VALUES_PER_PAGE];
    // generate 1% unique values
    const NUM_UNIQUE_VALUES: usize = VALUES_PER_PAGE / 100;
    let unique_values = 
        (0..NUM_UNIQUE_VALUES)
        .map(|x| (x + 1) as i32)
        .collect::<Vec<_>>();
    let mut rng = seedable_rng();
    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    for _i in 0..NUM_ROW_GROUPS {
        let mut column_chunk_pages = VecDeque::new();
        let mem_tracker = Arc::new(parquet::memory::MemTracker::new());
        let mut dict_encoder = DictEncoder::<Int32Type>::new(column_desc.clone(), mem_tracker);
        // add data pages
        for _j in 0..PAGES_PER_GROUP {
            // generate page
            let mut values = Vec::with_capacity(VALUES_PER_PAGE);
            let mut def_levels = Vec::with_capacity(VALUES_PER_PAGE);
            for _k in 0..VALUES_PER_PAGE {
                let def_level = if rng.gen::<f32>() < null_density {
                    max_def_level - 1
                } else {
                    max_def_level
                };
                if def_level == max_def_level {
                    // select random value from list of unique values
                    let int32_value = unique_values[rng.gen_range(0..NUM_UNIQUE_VALUES)];
                    values.push(int32_value);
                }
                def_levels.push(def_level);
            }
            let mut page_builder = DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);
            page_builder.add_rep_levels(max_rep_level, &rep_levels);
            page_builder.add_def_levels(max_def_level, &def_levels);
            let _ = dict_encoder.put(&values);
            let indices = dict_encoder
                .write_indices()
                .expect("write_indices() should be OK");
            page_builder.add_indices(indices);
            column_chunk_pages.push_back(page_builder.consume());
        }
        // add dictionary page
        let dict = dict_encoder
            .write_dict()
            .expect("write_dict() should be OK");
        let dict_page = parquet::column::page::Page::DictionaryPage {
            buf: dict,
            num_values: dict_encoder.num_entries() as u32,
            encoding: Encoding::RLE_DICTIONARY,
            is_sorted: false,
        };
        column_chunk_pages.push_front(dict_page);
        pages.push(column_chunk_pages.into());
    }

    InMemoryPageIterator::new(schema, column_desc, pages)
}

fn build_plain_encoded_string_page_iterator(schema: SchemaDescPtr, column_desc: ColumnDescPtr, null_density: f32) -> impl PageIterator + Clone {
    use parquet::util::test_common::page_util::{InMemoryPageIterator, DataPageBuilderImpl, DataPageBuilder};
    let max_def_level = column_desc.max_def_level();
    let max_rep_level = column_desc.max_rep_level();
    let rep_levels = vec![max_rep_level; VALUES_PER_PAGE];
    let mut rng = seedable_rng();
    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    for i in 0..NUM_ROW_GROUPS {
        let mut column_chunk_pages = Vec::new();
        for j in 0..PAGES_PER_GROUP {
            // generate page
            let mut values = Vec::with_capacity(VALUES_PER_PAGE);
            let mut def_levels = Vec::with_capacity(VALUES_PER_PAGE);
            for k in 0..VALUES_PER_PAGE {
                let def_level = if rng.gen::<f32>() < null_density {
                    max_def_level - 1
                } else {
                    max_def_level
                };
                if def_level == max_def_level {
                    let string_value = format!("Test value {}, row group: {}, page: {}", k, i, j);
                    values.push(parquet::data_type::ByteArray::from(string_value.as_str()));
                }
                def_levels.push(def_level);
            }
            let mut page_builder = DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);
            page_builder.add_rep_levels(max_rep_level, &rep_levels);
            page_builder.add_def_levels(max_def_level, &def_levels);
            page_builder.add_values::<ByteArrayType>(Encoding::PLAIN, &values);
            column_chunk_pages.push(page_builder.consume());
        }
        pages.push(column_chunk_pages);
    }

    InMemoryPageIterator::new(schema, column_desc, pages)
}

fn build_dictionary_encoded_string_page_iterator(schema: SchemaDescPtr, column_desc: ColumnDescPtr, null_density: f32) -> impl PageIterator + Clone {
    use parquet::util::test_common::page_util::{InMemoryPageIterator, DataPageBuilderImpl, DataPageBuilder};
    use parquet::encoding::{Encoder, DictEncoder};
    let max_def_level = column_desc.max_def_level();
    let max_rep_level = column_desc.max_rep_level();
    let rep_levels = vec![max_rep_level; VALUES_PER_PAGE];
    // generate 1% unique values
    const NUM_UNIQUE_VALUES: usize = VALUES_PER_PAGE / 100;
    let unique_values = 
        (0..NUM_UNIQUE_VALUES)
        .map(|x| format!("Dictionary value {}", x))
        .collect::<Vec<_>>();
    let mut rng = seedable_rng();
    let mut pages: Vec<Vec<parquet::column::page::Page>> = Vec::new();
    for _i in 0..NUM_ROW_GROUPS {
        let mut column_chunk_pages = VecDeque::new();
        let mem_tracker = Arc::new(parquet::memory::MemTracker::new());
        let mut dict_encoder = DictEncoder::<ByteArrayType>::new(column_desc.clone(), mem_tracker);
        // add data pages
        for _j in 0..PAGES_PER_GROUP {
            // generate page
            let mut values = Vec::with_capacity(VALUES_PER_PAGE);
            let mut def_levels = Vec::with_capacity(VALUES_PER_PAGE);
            for _k in 0..VALUES_PER_PAGE {
                let def_level = if rng.gen::<f32>() < null_density {
                    max_def_level - 1
                } else {
                    max_def_level
                };
                if def_level == max_def_level {
                    // select random value from list of unique values
                    let string_value = unique_values[rng.gen_range(0..NUM_UNIQUE_VALUES)].as_str();
                    values.push(parquet::data_type::ByteArray::from(string_value));
                }
                def_levels.push(def_level);
            }
            let mut page_builder = DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);
            page_builder.add_rep_levels(max_rep_level, &rep_levels);
            page_builder.add_def_levels(max_def_level, &def_levels);
            let _ = dict_encoder.put(&values);
            let indices = dict_encoder
                .write_indices()
                .expect("write_indices() should be OK");
            page_builder.add_indices(indices);
            column_chunk_pages.push_back(page_builder.consume());
        }
        // add dictionary page
        let dict = dict_encoder
            .write_dict()
            .expect("write_dict() should be OK");
        let dict_page = parquet::column::page::Page::DictionaryPage {
            buf: dict,
            num_values: dict_encoder.num_entries() as u32,
            encoding: Encoding::RLE_DICTIONARY,
            is_sorted: false,
        };
        column_chunk_pages.push_front(dict_page);
        pages.push(column_chunk_pages.into());
    }

    InMemoryPageIterator::new(schema, column_desc, pages)
}

fn bench_array_reader(mut array_reader: impl ArrayReader) -> usize {
    // test procedure: read data in batches of 8192 until no more data
    let mut total_count = 0;
    loop {
        let array = array_reader.next_batch(BATCH_SIZE);
        let array_len = array.unwrap().len();
        total_count += array_len;
        if array_len < BATCH_SIZE {
            break;
        }
    }
    total_count
}

fn create_int32_arrow_array_reader(page_iterator: impl PageIterator + 'static, column_desc: ColumnDescPtr) -> impl ArrayReader {
    use parquet::arrow::arrow_array_reader::{PrimitiveArrayConverter, ArrowArrayReader};
    let converter = PrimitiveArrayConverter::<arrow::datatypes::Int32Type>::new();
    ArrowArrayReader::try_new(
        page_iterator, column_desc, converter, None
    ).unwrap()
}

fn create_int32_primitive_array_reader(page_iterator: impl PageIterator + 'static, column_desc: ColumnDescPtr) -> impl ArrayReader {
    use parquet::arrow::array_reader::PrimitiveArrayReader;
    PrimitiveArrayReader::<Int32Type>::new(
        Box::new(page_iterator), column_desc, None,
    ).unwrap()
}

fn create_string_arrow_array_reader(page_iterator: impl PageIterator + 'static, column_desc: ColumnDescPtr) -> impl ArrayReader {
    use parquet::arrow::arrow_array_reader::{StringArrayConverter, ArrowArrayReader};
    let converter = StringArrayConverter::new();
    ArrowArrayReader::try_new(
        page_iterator, column_desc, converter, None
    ).unwrap()
}

fn create_string_complex_array_reader(page_iterator: impl PageIterator + 'static, column_desc: ColumnDescPtr) -> impl ArrayReader {
    use parquet::arrow::array_reader::ComplexObjectArrayReader;
    use parquet::arrow::converter::{Utf8Converter, Utf8ArrayConverter};
    let converter = Utf8Converter::new(Utf8ArrayConverter {});
    ComplexObjectArrayReader::<parquet::data_type::ByteArrayType, Utf8Converter>::new(
        Box::new(page_iterator), column_desc, converter, None
    ).unwrap()
}


fn add_benches(c: &mut Criterion) {
    let mut group = c.benchmark_group("arrow_array_reader");

    let mut count: usize = 0;

    let schema = build_test_schema();
    let mandatory_int32_column_desc = schema.column(0);
    let optional_int32_column_desc = schema.column(1);
    let mandatory_string_column_desc = schema.column(2);
    // println!("mandatory_string_column_desc: {:?}", mandatory_string_column_desc);
    let optional_string_column_desc = schema.column(3);
    // println!("optional_string_column_desc: {:?}", optional_string_column_desc);

    // primitive / int32 benchmarks
    // =============================
    let plain_int32_no_null_data = build_plain_encoded_int32_page_iterator(schema.clone(), mandatory_int32_column_desc.clone(), 0.0);
    // group.bench_function("clone benchmark data", |b| b.iter(|| {
    //     let data = plain_string_no_null_data.clone();
    //     count = data.flatten().count();
    // }));
    // println!("read {} pages", count);

    // int32, plain encoded, no NULLs
    group.bench_function("read Int32Array, plain encoded, mandatory, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(plain_int32_no_null_data.clone(), mandatory_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, plain encoded, mandatory, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(plain_int32_no_null_data.clone(), mandatory_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    let plain_int32_no_null_data = build_plain_encoded_int32_page_iterator(schema.clone(), optional_int32_column_desc.clone(), 0.0);
    group.bench_function("read Int32Array, plain encoded, optional, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(plain_int32_no_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, plain encoded, optional, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(plain_int32_no_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // int32, plain encoded, half NULLs
    let plain_int32_half_null_data = build_plain_encoded_int32_page_iterator(schema.clone(), optional_int32_column_desc.clone(), 0.5);
    group.bench_function("read Int32Array, plain encoded, optional, half NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(plain_int32_half_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, plain encoded, optional, half NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(plain_int32_half_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // int32, dictionary encoded, no NULLs
    let dictionary_int32_no_null_data = build_dictionary_encoded_int32_page_iterator(schema.clone(), mandatory_int32_column_desc.clone(), 0.0);
    group.bench_function("read Int32Array, dictionary encoded, mandatory, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(dictionary_int32_no_null_data.clone(), mandatory_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, dictionary encoded, mandatory, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(dictionary_int32_no_null_data.clone(), mandatory_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    let dictionary_int32_no_null_data = build_dictionary_encoded_int32_page_iterator(schema.clone(), optional_int32_column_desc.clone(), 0.0);
    group.bench_function("read Int32Array, dictionary encoded, optional, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(dictionary_int32_no_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, dictionary encoded, optional, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(dictionary_int32_no_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // int32, dictionary encoded, half NULLs
    let dictionary_int32_half_null_data = build_dictionary_encoded_int32_page_iterator(schema.clone(), optional_int32_column_desc.clone(), 0.5);
    group.bench_function("read Int32Array, dictionary encoded, optional, half NULLs - old", |b| b.iter(|| {
        let array_reader = create_int32_primitive_array_reader(dictionary_int32_half_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read Int32Array, dictionary encoded, optional, half NULLs - new", |b| b.iter(|| {
        let array_reader = create_int32_arrow_array_reader(dictionary_int32_half_null_data.clone(), optional_int32_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);


    // string benchmarks
    //==============================
    let plain_string_no_null_data = build_plain_encoded_string_page_iterator(schema.clone(), mandatory_string_column_desc.clone(), 0.0);
    // group.bench_function("clone benchmark data", |b| b.iter(|| {
    //     let data = plain_string_no_null_data.clone();
    //     count = data.flatten().count();
    // }));
    // println!("read {} pages", count);

    // string, plain encoded, no NULLs
    group.bench_function("read StringArray, plain encoded, mandatory, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(plain_string_no_null_data.clone(), mandatory_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, plain encoded, mandatory, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(plain_string_no_null_data.clone(), mandatory_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    let plain_string_no_null_data = build_plain_encoded_string_page_iterator(schema.clone(), optional_string_column_desc.clone(), 0.0);
    group.bench_function("read StringArray, plain encoded, optional, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(plain_string_no_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, plain encoded, optional, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(plain_string_no_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // string, plain encoded, half NULLs
    let plain_string_half_null_data = build_plain_encoded_string_page_iterator(schema.clone(), optional_string_column_desc.clone(), 0.5);
    group.bench_function("read StringArray, plain encoded, optional, half NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(plain_string_half_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, plain encoded, optional, half NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(plain_string_half_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // string, dictionary encoded, no NULLs
    let dictionary_string_no_null_data = build_dictionary_encoded_string_page_iterator(schema.clone(), mandatory_string_column_desc.clone(), 0.0);
    group.bench_function("read StringArray, dictionary encoded, mandatory, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(dictionary_string_no_null_data.clone(), mandatory_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, dictionary encoded, mandatory, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(dictionary_string_no_null_data.clone(), mandatory_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    let dictionary_string_no_null_data = build_dictionary_encoded_string_page_iterator(schema.clone(), optional_string_column_desc.clone(), 0.0);
    group.bench_function("read StringArray, dictionary encoded, optional, no NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(dictionary_string_no_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, dictionary encoded, optional, no NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(dictionary_string_no_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    // string, dictionary encoded, half NULLs
    let dictionary_string_half_null_data = build_dictionary_encoded_string_page_iterator(schema.clone(), optional_string_column_desc.clone(), 0.5);
    group.bench_function("read StringArray, dictionary encoded, optional, half NULLs - old", |b| b.iter(|| {
        let array_reader = create_string_complex_array_reader(dictionary_string_half_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.bench_function("read StringArray, dictionary encoded, optional, half NULLs - new", |b| b.iter(|| {
        let array_reader = create_string_arrow_array_reader(dictionary_string_half_null_data.clone(), optional_string_column_desc.clone());
        count = bench_array_reader(array_reader);
    }));
    println!("read {} values", count);

    group.finish();
}

criterion_group!(benches, add_benches);
criterion_main!(benches);
