use std::{any::Any, collections::VecDeque};
use std::{rc::Rc, cell::RefCell};
use arrow::{array::{ArrayRef, Int16Array}, buffer::MutableBuffer, datatypes::{DataType as ArrowType}};
use crate::{column::page::{Page, PageIterator}, memory::{ByteBufferPtr, BufferPtr}, schema::types::{ColumnDescPtr, ColumnDescriptor}};
use crate::arrow::schema::parquet_to_arrow_field;
use crate::errors::{ParquetError, Result};
use crate::basic::Encoding;
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

    fn size_hint(&self) -> (usize, Option<usize>) {
        (0, Some(self.batch_size))
    }
}

impl<T: Clone> Splittable for Result<BufferPtr<T>> {
    type BufferType = Result<BufferPtr<T>>;
    type OutputType = Result<BufferPtr<T>>;

    #[inline]
    fn len(&self) -> usize {
        match self {
            Ok(x) => x.len(),
            _ => 0
        }
    }

    #[inline]
    fn split(self, len: usize) -> (Self::BufferType, Self::BufferType) {
        let mut buffer = if let Ok(item) = self {
            item
        }
        else {
            // this shouldn't happen as len() returns 0 for error
            // so it should always be fully consumed and never split
            return (self.clone(), self);
        };
        (Ok(buffer.split_to( len)), Ok(buffer))
    }
}

use crate::encodings::decoding::ValueByteChunk;

impl Splittable for Result<ValueByteChunk> {
    type BufferType = Result<ValueByteChunk>;
    type OutputType = Result<ValueByteChunk>;

    #[inline]
    fn len(&self) -> usize {
        match self {
            Ok(x) => x.value_count,
            _ => 0
        }
    }

    #[inline]
    fn split(self, len: usize) -> (Self::BufferType, Self::BufferType) {
        let mut value_byte_chunk = if let Ok(item) = self {
            item
        }
        else {
            // this shouldn't happen as len() returns 0 for error
            // so it should always be fully consumed and never split
            return (self.clone(), self);
        };
        let value_bit_len = value_byte_chunk.value_bit_len;
        let bit_len = len * value_bit_len;
        // TODO: use bit_offset when splitting bit / bool values
        assert!(bit_len % 8 == 0, "value byte buffer can only be split into whole bytes");
        let byte_len = bit_len / 8;

        let split_value_chunk = ValueByteChunk::new(
            value_byte_chunk.data.split_to(byte_len),
                len,
                value_bit_len,
        );
        value_byte_chunk.value_count -= len;
        (Ok(split_value_chunk), Ok(value_byte_chunk))
    }
}

type LevelBufferPtr = BufferPtr<i16>;

pub trait ArrayConverter {
    fn convert_value_chunks(&self, value_byte_chunks: impl IntoIterator<Item = Result<ValueByteChunk>>) -> Result<arrow::array::ArrayData>;
}

pub struct ArrowArrayReader<'a, C: ArrayConverter + 'a> {
    column_desc: ColumnDescPtr,
    data_type: ArrowType,
    def_level_iter_factory: SplittableBatchingIteratorFactory<'a, Result<LevelBufferPtr>>,
    rep_level_iter_factory: SplittableBatchingIteratorFactory<'a, Result<LevelBufferPtr>>,
    value_iter_factory: SplittableBatchingIteratorFactory<'a, Result<ValueByteChunk>>,
    last_def_levels: Option<Int16Array>,
    last_rep_levels: Option<Int16Array>,
    array_converter: C,
}

pub(crate) struct ColumnChunkContext {
    dictionary_values: Option<Vec<ValueByteChunk>>,
}

impl ColumnChunkContext {
    fn new() -> Self {
        Self {
            dictionary_values: None,
        }
    }
}

impl<'a, C: ArrayConverter + 'a> ArrowArrayReader<'a, C> {
    pub fn try_new<P: PageIterator + 'a>(column_chunk_iterator: P, column_desc: ColumnDescPtr, array_converter: C, arrow_type: Option<ArrowType>) -> Result<Self> {
        let data_type = match arrow_type {
            Some(t) => t,
            None => parquet_to_arrow_field(column_desc.as_ref())?
                .data_type()
                .clone(),
        };
        // println!("ArrowArrayReader::try_new, data_type: {}", data_type);
        let page_iter = column_chunk_iterator
            // build iterator of pages across column chunks
            .flat_map(|x| -> Box<dyn Iterator<Item = Result<(Page, Rc<RefCell<ColumnChunkContext>>)>>> {
                // attach column chunk context
                let context = Rc::new(RefCell::new(ColumnChunkContext::new()));
                match x {
                    Ok(page_reader) => Box::new(page_reader.map(move |pr| pr.and_then(|p| Ok((p, context.clone()))))),
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
            move |x: Result<(Page, Rc<RefCell<ColumnChunkContext>>)>| x.and_then(
                |(page, context)| Self::map_page(page, context, column_desc.as_ref())
            )
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
            def_level_iter_factory: SplittableBatchingIteratorFactory::new(def_level_iter),
            rep_level_iter_factory: SplittableBatchingIteratorFactory::new(rep_level_iter),
            value_iter_factory: SplittableBatchingIteratorFactory::new(value_iter),
            last_def_levels: None,
            last_rep_levels: None,
            array_converter,
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

    fn map_page_error(err: ParquetError) -> (Box<dyn Iterator<Item = Result<ValueByteChunk>>>, Box<dyn Iterator<Item = Result<LevelBufferPtr>>>, Box<dyn Iterator<Item = Result<LevelBufferPtr>>>)
    {
        (Box::new(std::iter::once(Err(err.clone()))), Box::new(std::iter::once(Err(err.clone()))), Box::new(std::iter::once(Err(err))))
    }

    // Split Result<Page> into Result<(Iterator<Values>, Iterator<DefLevels>, Iterator<RepLevels>)>
    // this method could fail, e.g. if the page encoding is not supported
    fn map_page(page: Page, column_chunk_context: Rc<RefCell<ColumnChunkContext>>, column_desc: &ColumnDescriptor) -> Result<(Box<dyn Iterator<Item = Result<ValueByteChunk>>>, Box<dyn Iterator<Item = Result<LevelBufferPtr>>>, Box<dyn Iterator<Item = Result<LevelBufferPtr>>>)> 
    {
        use crate::encodings::levels::LevelDecoder;
        match page {
            Page::DictionaryPage {
                buf,
                num_values,
                encoding,
                ..
            } => {
                let mut column_chunk_context = column_chunk_context.borrow_mut();
                if column_chunk_context.dictionary_values.is_some() {
                    return Err(general_err!("Column chunk cannot have more than one dictionary"));
                }
                // create plain decoder for dictionary values
                let value_iter = Self::decode_dictionary_page(buf, num_values as usize, encoding, column_desc)?;
                // decode and cache dictionary values
                let dictionary_values: Vec<ValueByteChunk> = value_iter.collect::<Result<_>>()?;
                column_chunk_context.dictionary_values = Some(dictionary_values);

                // a dictionary page doesn't return any values
                Ok((
                    Box::new(std::iter::empty()),
                    Box::new(std::iter::empty()),
                    Box::new(std::iter::empty()),
                ))
            }
            Page::DataPage {
                buf,
                num_values,
                encoding,
                def_level_encoding,
                rep_level_encoding,
                statistics: _,
            } => {
                let mut buffer_ptr = buf;
                // create rep level decoder iterator
                let rep_level_iter: Box<dyn Iterator<Item = Result<LevelBufferPtr>>> = if Self::rep_levels_available(&column_desc) {
                    let mut rep_decoder =
                        LevelDecoder::v1(rep_level_encoding, column_desc.max_rep_level());
                    let rep_level_byte_len = rep_decoder.set_data(
                        num_values as usize,
                        buffer_ptr.all(),
                    );
                    // advance buffer pointer
                    buffer_ptr = buffer_ptr.start_from(rep_level_byte_len);
                    Box::new(rep_decoder)
                }
                else {
                    Box::new(std::iter::once(Err(ParquetError::General(format!("rep levels are not available")))))
                };
                // create def level decoder iterator
                let def_level_iter: Box<dyn Iterator<Item = Result<LevelBufferPtr>>> = if Self::def_levels_available(&column_desc) {
                    let mut def_decoder = LevelDecoder::v1(
                        def_level_encoding,
                        column_desc.max_def_level(),
                    );
                    let def_levels_byte_len = def_decoder.set_data(
                        num_values as usize,
                        buffer_ptr.all(),
                    );
                    // advance buffer pointer
                    buffer_ptr = buffer_ptr.start_from(def_levels_byte_len);
                    Box::new(def_decoder)
                }
                else {
                    Box::new(std::iter::once(Err(ParquetError::General(format!("def levels are not available")))))
                };
                // create value decoder iterator
                let value_iter = Self::get_values_decoder_iter(
                    buffer_ptr, num_values as usize, encoding, column_desc, column_chunk_context
                )?;
                Ok((
                    value_iter,
                    def_level_iter,
                    rep_level_iter
                ))
            }
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
                let mut offset = 0;
                // create rep level decoder iterator
                let rep_level_iter: Box<dyn Iterator<Item = Result<LevelBufferPtr>>> = if Self::rep_levels_available(&column_desc) {
                    let rep_levels_byte_len = rep_levels_byte_len as usize;
                    let mut rep_decoder =
                        LevelDecoder::v2(column_desc.max_rep_level());
                    rep_decoder.set_data_range(
                        num_values as usize,
                        &buf,
                        offset,
                        rep_levels_byte_len,
                    );
                    offset += rep_levels_byte_len;
                    Box::new(rep_decoder)
                }
                else {
                    Box::new(std::iter::once(Err(ParquetError::General(format!("rep levels are not available")))))
                };
                // create def level decoder iterator
                let def_level_iter: Box<dyn Iterator<Item = Result<LevelBufferPtr>>> = if Self::def_levels_available(&column_desc) {
                    let def_levels_byte_len = def_levels_byte_len as usize;
                    let mut def_decoder =
                        LevelDecoder::v2(column_desc.max_def_level());
                    def_decoder.set_data_range(
                        num_values as usize,
                        &buf,
                        offset,
                        def_levels_byte_len,
                    );
                    offset += def_levels_byte_len;
                    Box::new(def_decoder)
                }
                else {
                    Box::new(std::iter::once(Err(ParquetError::General(format!("def levels are not available")))))
                };

                // create value decoder iterator
                let values_buffer = buf.start_from(offset);
                let value_iter = Self::get_values_decoder_iter(
                    values_buffer, num_values as usize, encoding, column_desc, column_chunk_context
                )?;
                Ok((
                    value_iter,
                    def_level_iter,
                    rep_level_iter
                ))
            }
        }
    }

    fn decode_dictionary_page(values_buffer: ByteBufferPtr, num_values: usize, mut encoding: Encoding, column_desc: &ColumnDescriptor) -> Result<Box<dyn Iterator<Item = Result<ValueByteChunk>>>> {
        if encoding == Encoding::PLAIN || encoding == Encoding::PLAIN_DICTIONARY {
            encoding = Encoding::RLE_DICTIONARY
        }

        if encoding == Encoding::RLE_DICTIONARY {
            Ok(Self::get_plain_decoder_iterator(values_buffer, num_values, column_desc))
        } else {
            Err(nyi_err!(
                "Invalid/Unsupported encoding type for dictionary: {}",
                encoding
            ))
        }
    }

    fn get_values_decoder_iter(values_buffer: ByteBufferPtr, num_values: usize, mut encoding: Encoding, column_desc: &ColumnDescriptor, column_chunk_context: Rc<RefCell<ColumnChunkContext>>) -> Result<Box<dyn Iterator<Item = Result<ValueByteChunk>>>> {
        if encoding == Encoding::PLAIN_DICTIONARY {
            encoding = Encoding::RLE_DICTIONARY;
        }

        match encoding {
            Encoding::PLAIN => Ok(Self::get_plain_decoder_iterator(values_buffer, num_values, column_desc)),
            Encoding::RLE_DICTIONARY => {
                // TODO: add support for fixed-length types
                if column_chunk_context.borrow().dictionary_values.is_some() {
                    Ok(Box::new(DictionaryDecoder::new(
                        column_chunk_context, values_buffer, num_values
                    )))
                }
                else {
                    Err(general_err!(
                        "Dictionary values have not been initialized."
                    ))
                }
            }
            // Encoding::RLE => Box::new(RleValueDecoder::new()),
            // Encoding::DELTA_BINARY_PACKED => Box::new(DeltaBitPackDecoder::new()),
            // Encoding::DELTA_LENGTH_BYTE_ARRAY => Box::new(DeltaLengthByteArrayDecoder::new()),
            // Encoding::DELTA_BYTE_ARRAY => Box::new(DeltaByteArrayDecoder::new()),
            e => return Err(nyi_err!("Encoding {} is not supported", e)),
        }
    }

    fn get_plain_decoder_iterator(values_buffer: ByteBufferPtr, num_values: usize, column_desc: &ColumnDescriptor) -> Box<dyn Iterator<Item = Result<ValueByteChunk>>> {
        use crate::encodings::decoding::{FixedLenPlainDecoder, VariableLenPlainDecoder};
        use crate::basic::Type as PhysicalType;

        // parquet only supports a limited number of physical types
        // later converters cast to a more specific arrow / logical type if necessary
        let value_bit_len: usize = match column_desc.physical_type() {
            PhysicalType::BOOLEAN => 1,
            PhysicalType::INT32 | PhysicalType::FLOAT => 32,
            PhysicalType::INT64 | PhysicalType::DOUBLE => 64,
            PhysicalType::INT96 => 96,
            PhysicalType::BYTE_ARRAY => { 
                return Box::new(VariableLenPlainDecoder::new(values_buffer, num_values));
            },
            PhysicalType::FIXED_LEN_BYTE_ARRAY => column_desc.type_length() as usize * 8,
        };

        Box::new(FixedLenPlainDecoder::new(values_buffer, num_values, value_bit_len))
    }

    fn build_level_array(level_buffers: Vec<LevelBufferPtr>) -> Int16Array {
        let value_count = level_buffers.iter().map(|levels| levels.len()).sum();
        let values_byte_len = value_count * std::mem::size_of::<i16>();
        let mut value_buffer = MutableBuffer::new(values_byte_len);
        for level_buffer in level_buffers {
            value_buffer.extend_from_slice(level_buffer.data());
        }
        let array_data = arrow::array::ArrayData::builder(ArrowType::Int16)
            .len(value_count)
            .add_buffer(value_buffer.into())
            .build();
        Int16Array::from(array_data)
    }
}

impl<C: ArrayConverter> ArrayReader for ArrowArrayReader<'static, C> {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn get_data_type(&self) -> &ArrowType {
        &self.data_type
    }

    fn next_batch(&mut self, batch_size: usize) -> Result<ArrayRef> {
        if Self::rep_levels_available(&self.column_desc) {
            // read rep levels if available
            let rep_level_iter = self.rep_level_iter_factory.get_batch_iter(batch_size);
            let rep_level_buffers: Vec<LevelBufferPtr> = rep_level_iter.collect::<Result<_>>()?;
            let rep_level_array = Self::build_level_array(rep_level_buffers);
            self.last_rep_levels = Some(rep_level_array);
        }
        
        // check if def levels are available
        let (values_to_read, null_bitmap_array) = if !Self::def_levels_available(&self.column_desc) {
            // if no def levels - just read (up to) batch_size values
            (batch_size, None)
        }
        else {
            // if def levels are available - they determine how many values will be read
            let def_level_iter = self.def_level_iter_factory.get_batch_iter(batch_size);
            // decode def levels, return first error if any
            let def_level_buffers: Vec<LevelBufferPtr> = def_level_iter.collect::<Result<_>>()?;
            let def_level_array = Self::build_level_array(def_level_buffers);
            let def_level_count = def_level_array.len();
            // use eq_scalar to efficiently build null bitmap array from def levels
            let null_bitmap_array = arrow::compute::eq_scalar(&def_level_array, self.column_desc.max_def_level())?;
            self.last_def_levels = Some(def_level_array);
            // efficiently calculate values to read
            let values_to_read = null_bitmap_array.values().count_set_bits_offset(0, def_level_count);
            (values_to_read, Some(null_bitmap_array))
        };

        // read a batch of values
        let value_iter = self.value_iter_factory.get_batch_iter(values_to_read);
        
        // NOTE: collecting value chunks here is actually slower
        // TODO: re-evaluate when iterators are migrated to async streams
        // collect and unwrap values into Vec, return first error if any
        // this will separate reading and decoding values from creating an arrow array
        // extra memory is allocated for the Vec, but the values still point to the page buffer
        // let values: Vec<ValueByteChunk> = value_iter.collect::<Result<_>>()?;

        // converter only creates a no-null / all value array data
        let mut value_array_data = self.array_converter.convert_value_chunks(value_iter)?;

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
        // TODO: cast array to self.data_type if necessary
        Ok(arrow::array::make_array(value_array_data))
    }

    fn get_def_levels(&self) -> Option<&[i16]> {
        self.last_def_levels.as_ref().map(|x| x.values())
    }

    fn get_rep_levels(&self) -> Option<&[i16]> {
        self.last_rep_levels.as_ref().map(|x| x.values())
    }
}

use crate::encodings::rle::RleDecoder;

pub(crate) struct DictionaryDecoder {
    context_ref: Rc<RefCell<ColumnChunkContext>>,
    key_data_bufer: ByteBufferPtr,
    num_values: usize,
    rle_decoder: RleDecoder,
    // value_buffer: VecDeque<ValueByteChunk>,
    // keys_buffer: Vec<i32>,
}

impl DictionaryDecoder {
    pub(crate) fn new(column_chunk_context: Rc<RefCell<ColumnChunkContext>>, key_data_bufer: ByteBufferPtr, num_values: usize) -> Self {
        // First byte in `data` is bit width
        let bit_width = key_data_bufer.data()[0];
        let mut rle_decoder = RleDecoder::new(bit_width);
        rle_decoder.set_data(key_data_bufer.start_from(1));
        
        Self {
            context_ref: column_chunk_context,
            key_data_bufer,
            num_values,
            rle_decoder,
            // value_buffer: VecDeque::with_capacity(128),
            // keys_buffer: vec![0; 128],
        }
    }
}

impl Iterator for DictionaryDecoder {
    type Item = Result<ValueByteChunk>;

    fn next(&mut self) -> Option<Self::Item> {
        // this simpler, non-buffering implementation is actually a bit faster
        // TODO: re-evaluate when iterators are replaced with async streams
        if self.num_values > 0 {
            let value_index = match self.rle_decoder.get::<i32>() {
                Ok(maybe_key) => match maybe_key {
                    Some(key) => key,
                    None => return None,
                }
                Err(e) => return Some(Err(e)),
            };
            let context = self.context_ref.borrow();
            let value_chunk = &context.dictionary_values.as_ref().unwrap()[value_index as usize];
            self.num_values -= 1;
            return Some(Ok(value_chunk.clone()));
        }
        return None;

        // match self.value_buffer.pop_front() {
        //     Some(value) => Some(Ok(value)),
        //     None => {
        //         if self.num_values <= 0 {
        //             return None;
        //         }
        //         let values_to_read = std::cmp::min(self.num_values, self.keys_buffer.len());
        //         let keys_read = match self.rle_decoder.get_batch(&mut self.keys_buffer[..values_to_read]) {
        //             Ok(keys_read) => keys_read,
        //             Err(e) => return Some(Err(e)),
        //         };
        //         if keys_read == 0 {
        //             self.num_values = 0;
        //             return None;
        //         }
        //         let context = self.context_ref.borrow();
        //         let values = context.dictionary_values.as_ref().unwrap();
        //         let first_value = values[self.keys_buffer[0] as usize].clone();
        //         let values_iter = 
        //             self.keys_buffer[1..keys_read].iter()
        //             .map(|key| values[*key as usize].clone());
        //         self.value_buffer.extend(values_iter);
                
        //         self.num_values -= keys_read;
        //         Some(Ok(first_value))
        //     }
        // }
    }
}

pub struct StringArrayConverter {}

impl StringArrayConverter {
    pub fn new() -> Self {
        Self {}
    }
}

impl ArrayConverter for StringArrayConverter {
    fn convert_value_chunks(&self, value_byte_chunks: impl IntoIterator<Item = Result<ValueByteChunk>>) -> Result<arrow::array::ArrayData> {
        use arrow::datatypes::ArrowNativeType;
        let value_chunks_iter = value_byte_chunks.into_iter();
        let value_capacity = value_chunks_iter.size_hint().1
            .ok_or_else(|| ParquetError::ArrowError("StringArrayConverter expects input iterator to declare an upper size hint.".to_string()))?;
        let offset_size = std::mem::size_of::<i32>();
        let mut offsets_buffer = MutableBuffer::new((value_capacity + 1) * offset_size);
        // NOTE: calculating exact value byte capacity is actually slower with current implementation
        // TODO: re-evaluate when iterators are migrated to async streams
        // calculate values_byte_capacity
        // let mut values_byte_capacity = 0;
        // let mut value_chunks = Vec::<ValueByteChunk>::with_capacity(value_capacity);
        // for value_chunk in value_iter {
        //     let value_chunk = value_chunk?;
        //     values_byte_capacity += value_chunk.data.len();
        //     value_chunks.push(value_chunk);
        // }

        // allocate initial capacity of 1 byte for each item
        let values_byte_capacity = value_capacity;
        let mut values_buffer = MutableBuffer::new(values_byte_capacity);

        let mut length_so_far = i32::default();
        offsets_buffer.push(length_so_far);

        for value_chunk in value_chunks_iter {
            let value_chunk = value_chunk?;
            debug_assert_eq!(
                value_chunk.value_count, 1,
                "offset length value buffers can only contain bytes for a single value"
            );
            let value_bytes = value_chunk.data;
            length_so_far += <i32 as ArrowNativeType>::from_usize(value_bytes.len()).unwrap();
            offsets_buffer.push(length_so_far);
            values_buffer.extend_from_slice(value_bytes.data());
        }
        // calculate actual data_len, which may be different from the iterator's upper bound
        let data_len = (offsets_buffer.len() / offset_size) - 1;
        let array_data = arrow::array::ArrayData::builder(ArrowType::Utf8)
            .len(data_len)
            .add_buffer(offsets_buffer.into())
            .add_buffer(values_buffer.into())
            .build();
        Ok(array_data)
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
        let column_desc = schema.column(0);
        let max_def_level = column_desc.max_def_level();
        let max_rep_level = column_desc.max_rep_level();

        assert_eq!(max_def_level, 2);
        assert_eq!(max_rep_level, 1);

        let mut rng = thread_rng();
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
        let converter = StringArrayConverter::new();
        let mut array_reader = ArrowArrayReader::try_new(
            page_iterator, column_desc, converter, None
        ).unwrap();

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