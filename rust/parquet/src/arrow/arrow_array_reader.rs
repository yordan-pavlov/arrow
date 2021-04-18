use std::{any::Any, collections::VecDeque};
use std::{rc::Rc, cell::RefCell};
use arrow::{array::ArrayRef, datatypes::{DataType as ArrowType}, buffer::{MutableBuffer, Buffer}};
use crate::{column::page::{Page, PageIterator}, memory::ByteBufferPtr, schema::types::{ColumnDescPtr, ColumnDescriptor}};
use crate::arrow::schema::parquet_to_arrow_field;
use crate::errors::{ParquetError, Result};
use super::array_reader::ArrayReader;

struct UnzipIter<Source, Target, State>
{
    shared_state: Rc<RefCell<State>>,
    select_item_buffer: fn(&mut State) -> &mut VecDeque<Target>,
    consume_source_item: fn(source_item: Source, state: &mut State) -> Target,
}

impl<Source, Target, State> UnzipIter<Source, Target, State>
{
    fn new(
        shared_state: Rc<RefCell<State>>, 
        item_buffer_selector: fn(&mut State) -> &mut VecDeque<Target>, 
        source_item_consumer: fn(source_item: Source, state: &mut State) -> Target
    ) -> Self {
        Self {
            shared_state,
            select_item_buffer: item_buffer_selector,
            consume_source_item: source_item_consumer,
        }
    }
}

trait UnzipIterState<T> {
    type SourceIter: Iterator<Item = T>;
    fn source_iter(&mut self) -> &mut Self::SourceIter;
}

impl<Source, Target, State: UnzipIterState<Source>> Iterator for UnzipIter<Source, Target, State> {
    type Item = Target;

    fn next(&mut self) -> Option<Self::Item> {
        let mut inner = self.shared_state.borrow_mut();
        // try to get one from the stored data
        (self.select_item_buffer)(&mut *inner).pop_front().or_else(|| 
            // nothing stored, we need a new element.
            inner.source_iter().next().map(|s| {
                (self.consume_source_item)(s, &mut inner)
            }))
    }
}

struct PageBufferUnzipIterState<V, L, It> {
    iter: It,
    value_iter_buffer: VecDeque<V>,
    def_level_iter_buffer: VecDeque<L>,
    rep_level_iter_buffer: VecDeque<L>,
}

impl<V, L, It: Iterator<Item = (V, L, L)>> UnzipIterState<(V, L, L)> for PageBufferUnzipIterState<V, L, It> {
    type SourceIter = It;

    #[inline]
    fn source_iter(&mut self) -> &mut Self::SourceIter {
        &mut self.iter
    }
}

fn unzip_iter<V, L, It: Iterator<Item = (V, L, L)>>(it: It) -> (
    UnzipIter<(V, L, L), V, PageBufferUnzipIterState<V, L, It>>, 
    UnzipIter<(V, L, L), L, PageBufferUnzipIterState<V, L, It>>,
    UnzipIter<(V, L, L), L, PageBufferUnzipIterState<V, L, It>>,
) {
    let shared_data = Rc::new(RefCell::new(PageBufferUnzipIterState { 
        iter: it,
        value_iter_buffer: VecDeque::new(),
        def_level_iter_buffer: VecDeque::new(),
        rep_level_iter_buffer: VecDeque::new(),
    }));

    let value_iter = UnzipIter::new(
        shared_data.clone(),
        |state| &mut state.value_iter_buffer,
        |(v, d, r), state| { 
            state.def_level_iter_buffer.push_back(d); 
            state.rep_level_iter_buffer.push_back(r);
            v
        }, 
    );

    let def_level_iter = UnzipIter::new(
        shared_data.clone(),
        |state| &mut state.def_level_iter_buffer,
        |(v, d, r), state| {
            state.value_iter_buffer.push_back(v);
            state.rep_level_iter_buffer.push_back(r);
            d
        }, 
    );

    let rep_level_iter = UnzipIter::new(
        shared_data,
        |state| &mut state.rep_level_iter_buffer,
        |(v, d, r), state| {
            state.value_iter_buffer.push_back(v);
            state.def_level_iter_buffer.push_back(d);
            r
        }, 
    );

    (value_iter, def_level_iter, rep_level_iter)
}

pub trait Splittable {
    type BufferType: Splittable<BufferType = Self::BufferType>;
    type OutputType;

    fn len(&self) -> usize;
    fn split(self, len: usize) -> (Self::BufferType, Self::BufferType);
}

pub struct SplittableBatchingIteratorFactory<'iter, T: Splittable> {
    source_iter: Box<dyn Iterator<Item = T> + 'iter>,
    buffer_item: Option<T::BufferType>,
}

impl<'iter, T: Splittable + 'iter> SplittableBatchingIteratorFactory<'iter, T>
{
    pub fn new(source_iter: impl IntoIterator<Item = T> + 'iter) -> Self {
        Self {
            source_iter: Box::new(source_iter.into_iter()),
            buffer_item: None,
        }
    }

    // fn get_batch_iter<'a>(&'a mut self, batch_size: usize) -> impl Iterator<Item = Box<dyn AsRef<[u8]> + 'iter>> +'a {
    pub fn get_batch_iter<'a>(&'a mut self, batch_size: usize) -> SplittableBatchingIterator<'iter, 'a, T> {
        SplittableBatchingIterator::new(self, batch_size)
    }
}

pub struct SplittableBatchingIterator<'iter, 'a, T: Splittable> {
    source_state: &'a mut SplittableBatchingIteratorFactory<'iter, T>,
    batch_size: usize,
    iter_pos: usize,
}

impl<'iter, 'a, T: Splittable> SplittableBatchingIterator<'iter, 'a, T> {
    fn new(source_state: &'a mut SplittableBatchingIteratorFactory<'iter, T>, batch_size: usize) -> Self {
        Self {
            source_state,
            batch_size,
            iter_pos: 0,
        }
    }
}

pub trait Convert<T>: Sized {
    /// Performs the conversion.
    fn convert(self) -> T;
}

impl<T> Convert<T> for T {
    #[inline]
    fn convert(self) -> T {
        self
    }
}

impl<'iter, 'a, I: Splittable + 'iter> Iterator for SplittableBatchingIterator<'iter, 'a, I>
where
    I: Convert<I::OutputType>, 
    I::BufferType: Convert<I::OutputType>,
{
    type Item = I::OutputType;

    fn next(&mut self) -> Option<Self::Item> {
        let items_left = self.batch_size - self.iter_pos;
        if items_left <= 0 {
            return None;
        }
        
        if self.source_state.buffer_item.is_some()  {
            // if buffered item is some
            let buffer_item = std::mem::replace(&mut self.source_state.buffer_item, None).unwrap();
            let buffered_len = buffer_item.len();
            let byte_buffer = if buffered_len <= items_left {
                // consume buffered item fully
                self.iter_pos += buffered_len;
                buffer_item
            }
            else {
                // consume buffered item partially
                self.iter_pos += items_left;
                let (result, buffer_item) = buffer_item.split(items_left);
                self.source_state.buffer_item = Some(buffer_item);
                result
            };
            return Some(byte_buffer.convert());
        }
        
        if let Some(byte_slice) = self.source_state.source_iter.next() {
            let slice_len = byte_slice.len();
            // else if there are more items in source iterator
            let result_slice = if slice_len <= items_left {
                // consume source iterator item fully
                self.iter_pos += slice_len;
                byte_slice.convert()
            }
            else {
                // consume source iterator item partially
                let (result_slice, buffered_item) = byte_slice.split(items_left);
                self.source_state.buffer_item = Some(buffered_item);
                self.iter_pos += items_left;
                result_slice.convert()
            };
            return Some(result_slice);
        }

        // source iterator exhausted
        return None;
    }
}

impl Splittable for Result<(usize, ByteBufferPtr)> {
    type BufferType = Result<(usize, ByteBufferPtr)>;
    type OutputType = Result<(usize, ByteBufferPtr)>;

    #[inline]
    fn len(&self) -> usize {
        match self {
            Ok(x) => x.0,
            _ => 0
        }
    }

    #[inline]
    fn split(self, len: usize) -> (Self::BufferType, Self::BufferType) {
        let (value_count, mut byte_buffer) = if let Ok(item) = self {
            item
        }
        else {
            // this shouldn't happen as len() returns 0 for error
            // so it should always be fully consumed and never split
            return (self.clone(), self);
        };
        let value_size = byte_buffer.len() / value_count;
        (Ok((len, byte_buffer.split_to( len * value_size))), Ok((value_count - len, byte_buffer)))
    }
}

struct ArrowArrayReader<'a> {
    column_desc: ColumnDescPtr,
    data_type: ArrowType,
    def_level_iter: Box<dyn Iterator<Item = i16> + 'a>,
    rep_level_iter: Box<dyn Iterator<Item = i16> + 'a>,
    value_iter_factory: SplittableBatchingIteratorFactory<'a, Result<(usize, ByteBufferPtr)>>,
    last_def_levels: Option<Vec<i16>>,
}

impl<'a> ArrowArrayReader<'a> {
    fn try_new<P: PageIterator + 'a>(column_chunk_iterator: P, column_desc: ColumnDescPtr, arrow_type: Option<ArrowType>) -> Result<Self> {
        let data_type = match arrow_type {
            Some(t) => t,
            None => parquet_to_arrow_field(column_desc.as_ref())?
                .data_type()
                .clone(),
        };
        
        // TODO: attach column chunk context
        let page_iter = column_chunk_iterator
            // build iterator of pages across column chunks
            .flat_map(|x| -> Box<dyn Iterator<Item = Result<Page>>> {
                match x {
                    Ok(page_reader) => Box::new(page_reader.into_iter()),
                    // errors from reading column chunks / row groups are propagated to page level
                    Err(e) => Box::new(std::iter::once(Err(e)))
                }
            });
        // capture a clone of column_desc in closure so that it can outlive current function
        let map_page_fn = (|column_desc: ColumnDescPtr| {
            // move |x: Result<Page>|  match x {
            //     Ok(p) => Self::map_page(p, column_desc.as_ref()),
            //     Err(e) => Err(e),
            // }
            move |x: Result<Page>| x.and_then(|p| Self::map_page(p, column_desc.as_ref()))
        })(column_desc.clone());
        // map page iterator into tuple of buffer iterators for (values, def levels, rep levels)
        // errors from lower levels are surfaced through the value decoder iterator
        let decoder_iter = page_iter
            .map(map_page_fn)
            .map(|x| match x {
                Ok(iter_tuple) => iter_tuple,
                // errors from reading pages are propagated to decoder iterator level
                Err(e) => Self::map_page_error(e)
            });
        // split tuple iterator into separate iterators for (values, def levels, rep levels)
        let (value_iter, def_level_iter, rep_level_iter) = unzip_iter(decoder_iter);
        let value_iter = value_iter.flat_map(|x| x.into_iter());
        let def_level_iter = def_level_iter.flat_map(|x| x.into_iter());
        let rep_level_iter = rep_level_iter.flat_map(|x| x.into_iter());
        
        Ok(Self {
            column_desc,
            data_type,
            def_level_iter: Box::new(def_level_iter),
            rep_level_iter: Box::new(rep_level_iter),
            value_iter_factory: SplittableBatchingIteratorFactory::new(value_iter),
            last_def_levels: None,
        })
    }

    #[inline]
    fn def_levels_available(column_desc: &ColumnDescriptor) -> bool {
        column_desc.max_def_level() > 0
    }

    #[inline]
    fn rep_levels_available(column_desc: &ColumnDescriptor) -> bool {
        column_desc.max_rep_level() > 0
    }

    fn map_page_error(err: ParquetError) -> (Box<dyn Iterator<Item = Result<(usize, ByteBufferPtr)>>>, Box<dyn Iterator<Item = i16>>, Box<dyn Iterator<Item = i16>>)
    {
        (Box::new(std::iter::once(Err(err))), Box::new(std::iter::empty::<i16>()), Box::new(std::iter::empty::<i16>()))
    }

    // Split Result<Page> into Result<(Iterator<Values>, Iterator<DefLevels>, Iterator<RepLevels>)>
    // this method could fail, e.g. if the page encoding is not supported
    fn map_page(page: Page, column_desc: &ColumnDescriptor) -> Result<(Box<dyn Iterator<Item = Result<(usize, ByteBufferPtr)>>>, Box<dyn Iterator<Item = i16>>, Box<dyn Iterator<Item = i16>>)> 
    {
        // process page (V1, V2, Dictionary)
        match page {
            Page::DataPageV2 {
                buf,
                num_values,
                encoding,
                num_nulls: _,
                num_rows: _,
                def_levels_byte_len,
                rep_levels_byte_len,
                is_compressed: _,
                statistics: _,
            } => {
                // create rep level decoder iterator
                let rep_level_iter: Box<dyn Iterator<Item = i16>> = if Self::rep_levels_available(&column_desc) {
                    // TODO: create actual rep level decoder iterator
                    Box::new(Vec::<i16>::new().into_iter())
                }
                else {
                    // is an empty iterator a good choice for rep levels?
                    Box::new(std::iter::empty::<i16>())
                };
                // create def level decoder iterator
                let def_level_iter: Box<dyn Iterator<Item = i16>> = if Self::def_levels_available(&column_desc) {
                    // TODO: create actual def level decoder iterator
                    Box::new(Vec::<i16>::new().into_iter())
                }
                else {
                    // create def level iterators where all values are included
                    // so that if there are no def levels, then attempting to read def levels
                    // will not buffer all value decoders
                    Box::new(std::iter::repeat(column_desc.max_def_level()).take(num_values as usize))
                };

                // create value decoder iterator
                let value_iter = vec![Ok((0, ByteBufferPtr::new(Vec::<u8>::new())))];

                Ok((
                    Box::new(value_iter.into_iter()),
                    def_level_iter,
                    rep_level_iter
                ))
            },
            p @ _ => Err(ParquetError::General(
                format!("type of page not supported: {}", p.page_type()))
            )
        }
    }

    fn convert_value_buffers(value_buffers: Vec<(usize, ByteBufferPtr)>) -> Result<arrow::array::ArrayData> {
        use arrow::datatypes::ArrowNativeType;
        let data_len = value_buffers.len();
        let offset_size = std::mem::size_of::<i32>();
        let mut offsets = MutableBuffer::new((data_len + 1) * offset_size);
        let values_byte_len = value_buffers.iter().map(|(_len, bytes)| bytes.len()).sum();
        let mut values = MutableBuffer::new(values_byte_len);

        let mut length_so_far = i32::default();
        offsets.push(length_so_far);

        for (value_len, value_bytes) in value_buffers {
            debug_assert_eq!(
                value_len, 1,
                "offset length value buffers can only contain bytes for a single value"
            );
            length_so_far += <i32 as ArrowNativeType>::from_usize(value_bytes.len()).unwrap();
            offsets.push(length_so_far);
            values.extend_from_slice(value_bytes.data());
        }
        let array_data = arrow::array::ArrayData::builder(ArrowType::Utf8)
            .len(data_len)
            .add_buffer(offsets.into())
            .add_buffer(values.into())
            .build();
        Ok(array_data)
    }
}

impl ArrayReader for ArrowArrayReader<'static> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef> {
        // check if def levels are available
        let (values_to_read, null_bitmap_array) = if !Self::def_levels_available(&self.column_desc) {
            // if no def levels - just read (up to) batch_size values
            (batch_size, None)
        }
        else {
            // if def levels are available - they determine how many values will be read
            let def_levels = self.def_level_iter
                .by_ref()
                .take(batch_size)
                .collect::<Vec<_>>();
            let def_level_count = def_levels.len();
            let max_def_level = self.column_desc.max_def_level();
            let null_bitmap_iter = def_levels.iter().map(|d| d == &max_def_level);
            // use of from_trusted_len_iter_bool is safe because take() sets upper size hint
            let null_bit_buffer: Buffer = unsafe { MutableBuffer::from_trusted_len_iter_bool(null_bitmap_iter).into() };
            self.last_def_levels = Some(def_levels);
            // it should be faster to use count_set_bits instead of incrementing during iteration
            let values_to_read = null_bit_buffer.count_set_bits_offset(0, def_level_count);
            let null_bitmap_array = {
                let builder = arrow::array::ArrayData::builder(ArrowType::Boolean)
                    .len(def_level_count)
                    .add_buffer(null_bit_buffer);
                let data = builder.build();
                arrow::array::BooleanArray::from(data)
            };
            (values_to_read, Some(null_bitmap_array))
        };

        // read a batch of values
        let value_iter = self.value_iter_factory.get_batch_iter(values_to_read);
        // collect and unwrap values into Vec, return first error if any
        // this will separate reading and decoding values from creating an arrow array
        // extra memory is allocated for the Vec, but the values still point to the page buffer
        let values: Vec<(usize, ByteBufferPtr)> = value_iter.collect::<Result<_>>()?;
        // converter only creates a no-null / all value array data
        let mut value_array_data = Self::convert_value_buffers(values)?;

        if let Some(null_bitmap_array) = null_bitmap_array {
            // Only if def levels are available - insert null values efficiently using MutableArrayData.
            // This will require value bytes to be copied again, but converter requirements are reduced.
            // With a small number of NULLs, this will only be a few copies of large byte sequences.
            let actual_batch_size = null_bitmap_array.len();
            // TODO: optimize MutableArrayData::extend_offsets for sequential slices
            // use_nulls is false, because null_bitmap_array is already calculated and re-used
            let mut mutable = arrow::array::MutableArrayData::new(vec![&value_array_data], false, actual_batch_size);
            // SlicesIterator slices only the true values, NULLs are inserted to fill any gaps
            arrow::compute::SlicesIterator::new(&null_bitmap_array).for_each(|(start, end)| {
                // the gap needs to be filled with NULLs
                if start > mutable.len() {
                    let nulls_to_add = start - mutable.len();
                    mutable.extend_nulls(nulls_to_add);
                }
                // fill values, adjust start and end with NULL count so far
                let nulls_added = mutable.null_count();
                mutable.extend(0, start - nulls_added, end - nulls_added);
            });
            // any remaining part is NULLs
            if mutable.len() < actual_batch_size {
                let nulls_to_add = actual_batch_size - mutable.len();
                mutable.extend_nulls(nulls_to_add);
            }
            
            value_array_data = mutable
                .into_builder()
                .null_bit_buffer(null_bitmap_array.values().clone())
                .build();
        }

        Ok(arrow::array::make_array(value_array_data))
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.last_def_levels.as_deref()
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        if Self::rep_levels_available(&self.column_desc) {
            // TODO: return actual rep levels
        }
        None
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::basic::{Encoding};
    use crate::column::page::{Page};
    use crate::data_type::{ByteArray};
    use crate::data_type::{ByteArrayType};
    use crate::schema::parser::parse_message_type;
    use crate::schema::types::{SchemaDescriptor};
    use crate::util::test_common::page_util::{
        DataPageBuilder, DataPageBuilderImpl, InMemoryPageIterator,
    };
    use arrow::array::{StringArray};
    use rand::{thread_rng, Rng};
    use std::sync::Arc;

    #[test]
    fn test_arrow_array_reader() {
        // Construct column schema
        let message_type = "
        message test_schema {
            REPEATED Group test_mid {
                OPTIONAL BYTE_ARRAY leaf (UTF8);
            }
        }
        ";
        let num_pages = 2;
        let values_per_page = 100;
        let str_base = "Hello World";

        let schema = parse_message_type(message_type)
            .map(|t| Arc::new(SchemaDescriptor::new(Arc::new(t))))
            .unwrap();

        let max_def_level = schema.column(0).max_def_level();
        let max_rep_level = schema.column(0).max_rep_level();

        assert_eq!(max_def_level, 2);
        assert_eq!(max_rep_level, 1);

        let mut rng = thread_rng();
        let column_desc = schema.column(0);
        let mut pages: Vec<Vec<Page>> = Vec::new();

        let mut rep_levels = Vec::with_capacity(num_pages * values_per_page);
        let mut def_levels = Vec::with_capacity(num_pages * values_per_page);
        let mut all_values = Vec::with_capacity(num_pages * values_per_page);

        for i in 0..num_pages {
            let mut values = Vec::with_capacity(values_per_page);

            for _ in 0..values_per_page {
                let def_level = rng.gen_range(0..max_def_level + 1);
                let rep_level = rng.gen_range(0..max_rep_level + 1);
                if def_level == max_def_level {
                    let len = rng.gen_range(1..str_base.len());
                    let slice = &str_base[..len];
                    values.push(ByteArray::from(slice));
                    all_values.push(Some(slice.to_string()));
                } else {
                    all_values.push(None)
                }
                rep_levels.push(rep_level);
                def_levels.push(def_level)
            }

            let range = i * values_per_page..(i + 1) * values_per_page;
            let mut pb =
                DataPageBuilderImpl::new(column_desc.clone(), values.len() as u32, true);

            pb.add_rep_levels(max_rep_level, &rep_levels.as_slice()[range.clone()]);
            pb.add_def_levels(max_def_level, &def_levels.as_slice()[range]);
            pb.add_values::<ByteArrayType>(Encoding::PLAIN, values.as_slice());

            let data_page = pb.consume();
            pages.push(vec![data_page]);
        }

        let page_iterator = InMemoryPageIterator::new(schema, column_desc.clone(), pages);

        // let converter = Utf8Converter::new(Utf8ArrayConverter {});
        // let mut array_reader =
        //     ComplexObjectArrayReader::<ByteArrayType, Utf8Converter>::new(
        //         Box::new(page_iterator),
        //         column_desc,
        //         converter,
        //         None,
        //     )
        //     .unwrap();

        let mut array_reader = ArrowArrayReader::try_new(page_iterator, column_desc, None).unwrap();

        let mut accu_len: usize = 0;

        let array = array_reader.next_batch(values_per_page / 2).unwrap();
        assert_eq!(array.len(), values_per_page / 2);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
        accu_len += array.len();

        // Read next values_per_page values, the first values_per_page/2 ones are from the first column chunk,
        // and the last values_per_page/2 ones are from the second column chunk
        let array = array_reader.next_batch(values_per_page).unwrap();
        assert_eq!(array.len(), values_per_page);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
        let strings = array.as_any().downcast_ref::<StringArray>().unwrap();
        for i in 0..array.len() {
            if array.is_valid(i) {
                assert_eq!(
                    all_values[i + accu_len].as_ref().unwrap().as_str(),
                    strings.value(i)
                )
            } else {
                assert_eq!(all_values[i + accu_len], None)
            }
        }
        accu_len += array.len();

        // Try to read values_per_page values, however there are only values_per_page/2 values
        let array = array_reader.next_batch(values_per_page).unwrap();
        assert_eq!(array.len(), values_per_page / 2);
        assert_eq!(
            Some(&def_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_def_levels()
        );
        assert_eq!(
            Some(&rep_levels[accu_len..(accu_len + array.len())]),
            array_reader.get_rep_levels()
        );
    }
}