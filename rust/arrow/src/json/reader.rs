// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

//! JSON Reader
//!
//! This JSON reader allows JSON line-delimited files to be read into the Arrow memory
//! model. Records are loaded in batches and are then converted from row-based data to
//! columnar data.
//!
//! Example:
//!
//! ```
//! use arrow::datatypes::{DataType, Field, Schema};
//! use arrow::json;
//! use std::fs::File;
//! use std::io::BufReader;
//! use std::sync::Arc;
//!
//! let schema = Schema::new(vec![
//!     Field::new("a", DataType::Float64, false),
//!     Field::new("b", DataType::Float64, false),
//!     Field::new("c", DataType::Float64, false),
//! ]);
//!
//! let file = File::open("test/data/basic.json").unwrap();
//!
//! let mut json = json::Reader::new(BufReader::new(file), Arc::new(schema), 1024, None);
//! let batch = json.next().unwrap().unwrap();
//! ```

use indexmap::map::IndexMap as HashMap;
use indexmap::set::IndexSet as HashSet;
use std::io::{BufRead, BufReader, Read, Seek, SeekFrom};
use std::sync::Arc;

use serde_json::Value;

use crate::array::*;
use crate::buffer::MutableBuffer;
use crate::datatypes::*;
use crate::error::{ArrowError, Result};
use crate::record_batch::RecordBatch;
use crate::util::bit_util;

/// Coerce data type during inference
///
/// * `Int64` and `Float64` should be `Float64`
/// * Lists and scalars are coerced to a list of a compatible scalar
/// * All other types are coerced to `Utf8`
fn coerce_data_type(dt: Vec<&DataType>) -> Result<DataType> {
    match dt.len() {
        1 => Ok(dt[0].clone()),
        2 => {
            // there can be a case where a list and scalar both exist
            if dt.contains(&&DataType::List(Box::new(NullableDataType::new(
                DataType::Float64,
                true,
            )))) || dt.contains(&&DataType::List(Box::new(NullableDataType::new(
                DataType::Int64,
                true,
            )))) || dt.contains(&&DataType::List(Box::new(NullableDataType::new(
                DataType::Boolean,
                true,
            )))) || dt.contains(&&DataType::List(Box::new(NullableDataType::new(
                DataType::Utf8,
                true,
            )))) {
                // we have a list and scalars, so we should get the values and coerce them
                let mut dt = dt;
                // sorting guarantees that the list will be the second value
                dt.sort();
                match (dt[0], dt[1]) {
                    (t1, DataType::List(e)) if e.data_type() == &DataType::Float64 => {
                        if t1 == &DataType::Float64 {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                DataType::Float64,
                                true,
                            ))))
                        } else {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                coerce_data_type(vec![t1, &DataType::Float64])?,
                                true,
                            ))))
                        }
                    }
                    (t1, DataType::List(e)) if e.data_type() == &DataType::Int64 => {
                        if t1 == &DataType::Int64 {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                DataType::Int64,
                                true,
                            ))))
                        } else {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                coerce_data_type(vec![t1, &DataType::Int64])?,
                                true,
                            ))))
                        }
                    }
                    (t1, DataType::List(e)) if e.data_type() == &DataType::Boolean => {
                        if t1 == &DataType::Boolean {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                DataType::Boolean,
                                true,
                            ))))
                        } else {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                coerce_data_type(vec![t1, &DataType::Boolean])?,
                                true,
                            ))))
                        }
                    }
                    (t1, DataType::List(e)) if e.data_type() == &DataType::Utf8 => {
                        if t1 == &DataType::Utf8 {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                DataType::Utf8,
                                true,
                            ))))
                        } else {
                            Ok(DataType::List(Box::new(NullableDataType::new(
                                coerce_data_type(vec![t1, &DataType::Utf8])?,
                                true,
                            ))))
                        }
                    }
                    (t1, t2) => Err(ArrowError::JsonError(format!(
                        "Cannot coerce data types for {:?} and {:?}",
                        t1, t2
                    ))),
                }
            } else if dt.contains(&&DataType::Float64) && dt.contains(&&DataType::Int64) {
                Ok(DataType::Float64)
            } else {
                Ok(DataType::Utf8)
            }
        }
        _ => {
            // TODO(nevi_me) It's possible to have [float, int, list(float)], which should
            // return list(float). Will hash this out later
            Ok(DataType::List(Box::new(NullableDataType::new(
                DataType::Utf8,
                true,
            ))))
        }
    }
}

/// Generate schema from JSON field names and inferred data types
fn generate_schema(spec: HashMap<String, HashSet<DataType>>) -> Result<SchemaRef> {
    let fields: Result<Vec<Field>> = spec
        .iter()
        .map(|(k, hs)| {
            let v: Vec<&DataType> = hs.iter().collect();
            coerce_data_type(v).map(|t| Field::new(k, t, true))
        })
        .collect();
    match fields {
        Ok(fields) => {
            let schema = Schema::new(fields);
            Ok(Arc::new(schema))
        }
        Err(e) => Err(e),
    }
}

/// Infer the fields of a JSON file by reading the first n records of the file, with
/// `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its field types.
///
/// Contrary to [`infer_json_schema`], this function will seek back to the start of the `reader`.
/// That way, the `reader` can be used immediately afterwards to create a [`Reader`].
///
/// # Examples
/// ```
/// use std::fs::File;
/// use std::io::BufReader;
/// use arrow::json::reader::infer_json_schema_from_seekable;
///
/// let file = File::open("test/data/mixed_arrays.json").unwrap();
/// // file's cursor's offset at 0
/// let mut reader = BufReader::new(file);
/// let inferred_schema = infer_json_schema_from_seekable(&mut reader, None).unwrap();
/// // file's cursor's offset automatically set at 0
/// ```
pub fn infer_json_schema_from_seekable<R: Read + Seek>(
    reader: &mut BufReader<R>,
    max_read_records: Option<usize>,
) -> Result<SchemaRef> {
    let schema = infer_json_schema(reader, max_read_records);
    // return the reader seek back to the start
    reader.seek(SeekFrom::Start(0))?;

    schema
}

/// Infer the fields of a JSON file by reading the first n records of the buffer, with
/// `max_read_records` controlling the maximum number of records to read.
///
/// If `max_read_records` is not set, the whole file is read to infer its field types.
///
/// This function will not seek back to the start of the `reader`. The user has to manage the
/// original file's cursor. This function is useful when the `reader`'s cursor is not available
/// (does not implement [`Seek`]), such is the case for compressed streams decoders.
///
/// # Examples
/// ```
/// use std::fs::File;
/// use std::io::{BufReader, SeekFrom, Seek};
/// use flate2::read::GzDecoder;
/// use arrow::json::reader::infer_json_schema;
///
/// let mut file = File::open("test/data/mixed_arrays.json.gz").unwrap();
///
/// // file's cursor's offset at 0
/// let mut reader = BufReader::new(GzDecoder::new(&file));
/// let inferred_schema = infer_json_schema(&mut reader, None).unwrap();
/// // cursor's offset at end of file
///
/// // seek back to start so that the original file is usable again
/// file.seek(SeekFrom::Start(0)).unwrap();
/// ```
pub fn infer_json_schema<R: Read>(
    reader: &mut BufReader<R>,
    max_read_records: Option<usize>,
) -> Result<SchemaRef> {
    let mut values: HashMap<String, HashSet<DataType>> = HashMap::new();

    let mut line = String::new();
    for _ in 0..max_read_records.unwrap_or(std::usize::MAX) {
        reader.read_line(&mut line)?;
        if line.is_empty() {
            break;
        }
        let record: Value = serde_json::from_str(&line.trim()).expect("Not valid JSON");

        line = String::new();

        match record {
            Value::Object(map) => {
                let res = map.iter().try_for_each(|(k, v)| {
                    match v {
                        Value::Array(a) => {
                            // collect the data types in array
                            let types: Result<Vec<Option<&DataType>>> = a
                                .iter()
                                .map(|a| match a {
                                    Value::Null => Ok(None),
                                    Value::Number(n) => {
                                        if n.is_i64() {
                                            Ok(Some(&DataType::Int64))
                                        } else {
                                            Ok(Some(&DataType::Float64))
                                        }
                                    }
                                    Value::Bool(_) => Ok(Some(&DataType::Boolean)),
                                    Value::String(_) => Ok(Some(&DataType::Utf8)),
                                    Value::Array(_) | Value::Object(_) => {
                                        Err(ArrowError::JsonError(
                                            "Nested lists and structs not supported"
                                                .to_string(),
                                        ))
                                    }
                                })
                                .collect();
                            match types {
                                Ok(types) => {
                                    // unwrap the Option and discard None values (from
                                    // JSON nulls)
                                    let mut types: Vec<&DataType> =
                                        types.into_iter().filter_map(|t| t).collect();
                                    types.dedup();
                                    // if a record contains only nulls, it is not
                                    // added to values
                                    if !types.is_empty() {
                                        let dt = coerce_data_type(types)?;

                                        if values.contains_key(k) {
                                            let x = values.get_mut(k).unwrap();
                                            x.insert(DataType::List(Box::new(
                                                NullableDataType::new(dt, true),
                                            )));
                                        } else {
                                            // create hashset and add value type
                                            let mut hs = HashSet::new();
                                            hs.insert(DataType::List(Box::new(
                                                NullableDataType::new(dt, true),
                                            )));
                                            values.insert(k.to_string(), hs);
                                        }
                                    }
                                    Ok(())
                                }
                                Err(e) => Err(e),
                            }
                        }
                        Value::Bool(_) => {
                            if values.contains_key(k) {
                                let x = values.get_mut(k).unwrap();
                                x.insert(DataType::Boolean);
                            } else {
                                // create hashset and add value type
                                let mut hs = HashSet::new();
                                hs.insert(DataType::Boolean);
                                values.insert(k.to_string(), hs);
                            }
                            Ok(())
                        }
                        Value::Null => {
                            // do nothing, we treat json as nullable by default when
                            // inferring
                            Ok(())
                        }
                        Value::Number(n) => {
                            if n.is_f64() {
                                if values.contains_key(k) {
                                    let x = values.get_mut(k).unwrap();
                                    x.insert(DataType::Float64);
                                } else {
                                    // create hashset and add value type
                                    let mut hs = HashSet::new();
                                    hs.insert(DataType::Float64);
                                    values.insert(k.to_string(), hs);
                                }
                            } else {
                                // default to i64
                                if values.contains_key(k) {
                                    let x = values.get_mut(k).unwrap();
                                    x.insert(DataType::Int64);
                                } else {
                                    // create hashset and add value type
                                    let mut hs = HashSet::new();
                                    hs.insert(DataType::Int64);
                                    values.insert(k.to_string(), hs);
                                }
                            }
                            Ok(())
                        }
                        Value::String(_) => {
                            if values.contains_key(k) {
                                let x = values.get_mut(k).unwrap();
                                x.insert(DataType::Utf8);
                            } else {
                                // create hashset and add value type
                                let mut hs = HashSet::new();
                                hs.insert(DataType::Utf8);
                                values.insert(k.to_string(), hs);
                            }
                            Ok(())
                        }
                        Value::Object(_) => Err(ArrowError::JsonError(
                            "Inferring schema from nested JSON structs currently not supported"
                                .to_string(),
                        )),
                    }
                });
                match res {
                    Ok(()) => {}
                    Err(e) => return Err(e),
                }
            }
            t => {
                return Err(ArrowError::JsonError(format!(
                    "Expected JSON record to be an object, found {:?}",
                    t
                )));
            }
        };
    }

    generate_schema(values)
}

/// JSON file reader
#[derive(Debug)]
pub struct Reader<R: Read> {
    /// Explicit schema for the JSON file
    schema: SchemaRef,
    /// Optional projection for which columns to load (case-sensitive names)
    projection: Option<Vec<String>>,
    /// File reader
    reader: BufReader<R>,
    /// Batch size (number of records to load each time)
    batch_size: usize,
}

impl<R: Read> Reader<R> {
    /// Create a new JSON Reader from any value that implements the `Read` trait.
    ///
    /// If reading a `File`, you can customise the Reader, such as to enable schema
    /// inference, use `ReaderBuilder`.
    pub fn new(
        reader: R,
        schema: SchemaRef,
        batch_size: usize,
        projection: Option<Vec<String>>,
    ) -> Self {
        Self::from_buf_reader(BufReader::new(reader), schema, batch_size, projection)
    }

    /// Returns the schema of the reader, useful for getting the schema without reading
    /// record batches
    pub fn schema(&self) -> SchemaRef {
        match &self.projection {
            Some(projection) => {
                let fields = self.schema.fields();
                let projected_fields: Vec<Field> = fields
                    .iter()
                    .filter_map(|field| {
                        if projection.contains(field.name()) {
                            Some(field.clone())
                        } else {
                            None
                        }
                    })
                    .collect();

                Arc::new(Schema::new(projected_fields))
            }
            None => self.schema.clone(),
        }
    }

    /// Create a new JSON Reader from a `BufReader<R: Read>`
    ///
    /// To customize the schema, such as to enable schema inference, use `ReaderBuilder`
    pub fn from_buf_reader(
        reader: BufReader<R>,
        schema: SchemaRef,
        batch_size: usize,
        projection: Option<Vec<String>>,
    ) -> Self {
        Self {
            schema,
            projection,
            reader,
            batch_size,
        }
    }

    /// Read the next batch of records
    #[allow(clippy::should_implement_trait)]
    pub fn next(&mut self) -> Result<Option<RecordBatch>> {
        let mut rows: Vec<Value> = Vec::with_capacity(self.batch_size);
        let mut line = String::new();
        for _ in 0..self.batch_size {
            let bytes_read = self.reader.read_line(&mut line)?;
            if bytes_read > 0 {
                rows.push(serde_json::from_str(&line).expect("Not valid JSON"));
                line = String::new();
            } else {
                break;
            }
        }

        if rows.is_empty() {
            // reached end of file
            return Ok(None);
        }

        let rows = &rows[..];
        let projection = self.projection.clone().unwrap_or_else(Vec::new);
        let arrays = self.build_struct_array(rows, self.schema.fields(), &projection);

        let projected_fields: Vec<Field> = if projection.is_empty() {
            self.schema.fields().to_vec()
        } else {
            projection
                .iter()
                .map(|name| self.schema.column_with_name(name))
                .filter_map(|c| c)
                .map(|(_, field)| field.clone())
                .collect()
        };

        let projected_schema = Arc::new(Schema::new(projected_fields));

        arrays.and_then(|arr| RecordBatch::try_new(projected_schema, arr).map(Some))
    }

    fn build_wrapped_list_array(
        &self,
        rows: &[Value],
        col_name: &str,
        key_type: &DataType,
    ) -> Result<ArrayRef> {
        match *key_type {
            DataType::Int8 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::Int8),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<Int8Type>(&dtype, col_name, rows)
            }
            DataType::Int16 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::Int16),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<Int16Type>(&dtype, col_name, rows)
            }
            DataType::Int32 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::Int32),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<Int32Type>(&dtype, col_name, rows)
            }
            DataType::Int64 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::Int64),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<Int64Type>(&dtype, col_name, rows)
            }
            DataType::UInt8 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::UInt8),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<UInt8Type>(&dtype, col_name, rows)
            }
            DataType::UInt16 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::UInt16),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<UInt16Type>(&dtype, col_name, rows)
            }
            DataType::UInt32 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::UInt32),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<UInt32Type>(&dtype, col_name, rows)
            }
            DataType::UInt64 => {
                let dtype = DataType::Dictionary(
                    Box::new(DataType::UInt64),
                    Box::new(DataType::Utf8),
                );
                self.list_array_string_array_builder::<UInt64Type>(&dtype, col_name, rows)
            }
            ref e => Err(ArrowError::JsonError(format!(
                "Data type is currently not supported for dictionaries in list : {:?}",
                e
            ))),
        }
    }

    #[inline(always)]
    fn list_array_string_array_builder<DICT_TY>(
        &self,
        data_type: &DataType,
        col_name: &str,
        rows: &[Value],
    ) -> Result<ArrayRef>
    where
        DICT_TY: ArrowPrimitiveType + ArrowDictionaryKeyType,
    {
        let mut builder: Box<dyn ArrayBuilder> = match data_type {
            DataType::Utf8 => {
                let values_builder = StringBuilder::new(rows.len() * 5);
                Box::new(ListBuilder::new(values_builder))
            }
            DataType::Dictionary(_, _) => {
                let values_builder =
                    self.build_string_dictionary_builder::<DICT_TY>(rows.len() * 5)?;
                Box::new(ListBuilder::new(values_builder))
            }
            e => {
                return Err(ArrowError::JsonError(format!(
                    "Nested list data builder type is not supported: {:?}",
                    e
                )))
            }
        };

        for row in rows {
            if let Some(value) = row.get(col_name) {
                // value can be an array or a scalar
                let vals: Vec<Option<String>> = if let Value::String(v) = value {
                    vec![Some(v.to_string())]
                } else if let Value::Array(n) = value {
                    n.iter()
                        .map(|v: &Value| {
                            if v.is_string() {
                                Some(v.as_str().unwrap().to_string())
                            } else if v.is_array() || v.is_object() || v.is_null() {
                                // implicitly drop nested values
                                // TODO support deep-nesting
                                None
                            } else {
                                Some(v.to_string())
                            }
                        })
                        .collect()
                } else if let Value::Null = value {
                    vec![None]
                } else if !value.is_object() {
                    vec![Some(value.to_string())]
                } else {
                    return Err(ArrowError::JsonError(
                        "Only scalars are currently supported in JSON arrays".to_string(),
                    ));
                };

                // TODO: ARROW-10335: APIs of dictionary arrays and others are different. Unify
                // them.
                match data_type {
                    DataType::Utf8 => {
                        let builder = builder
                            .as_any_mut()
                            .downcast_mut::<ListBuilder<StringBuilder>>()
                            .ok_or_else(||ArrowError::JsonError(
                                "Cast failed for ListBuilder<StringBuilder> during nested data parsing".to_string(),
                            ))?;
                        for val in vals {
                            if let Some(v) = val {
                                builder.values().append_value(&v)?
                            } else {
                                builder.values().append_null()?
                            };
                        }

                        // Append to the list
                        builder.append(true)?;
                    }
                    DataType::Dictionary(_, _) => {
                        let builder = builder.as_any_mut().downcast_mut::<ListBuilder<StringDictionaryBuilder<DICT_TY>>>().ok_or_else(||ArrowError::JsonError(
                            "Cast failed for ListBuilder<StringDictionaryBuilder> during nested data parsing".to_string(),
                        ))?;
                        for val in vals {
                            if let Some(v) = val {
                                let _ = builder.values().append(&v)?;
                            } else {
                                builder.values().append_null()?
                            };
                        }

                        // Append to the list
                        builder.append(true)?;
                    }
                    e => {
                        return Err(ArrowError::JsonError(format!(
                            "Nested list data builder type is not supported: {:?}",
                            e
                        )))
                    }
                }
            }
        }

        Ok(builder.finish() as ArrayRef)
    }

    #[inline(always)]
    fn build_string_dictionary_builder<T>(
        &self,
        row_len: usize,
    ) -> Result<StringDictionaryBuilder<T>>
    where
        T: ArrowPrimitiveType + ArrowDictionaryKeyType,
    {
        let key_builder = PrimitiveBuilder::<T>::new(row_len);
        let values_builder = StringBuilder::new(row_len * 5);
        Ok(StringDictionaryBuilder::new(key_builder, values_builder))
    }

    #[inline(always)]
    fn build_string_dictionary_array(
        &self,
        rows: &[Value],
        col_name: &str,
        key_type: &DataType,
        value_type: &DataType,
    ) -> Result<ArrayRef> {
        if let DataType::Utf8 = *value_type {
            match *key_type {
                DataType::Int8 => self.build_dictionary_array::<Int8Type>(rows, col_name),
                DataType::Int16 => {
                    self.build_dictionary_array::<Int16Type>(rows, col_name)
                }
                DataType::Int32 => {
                    self.build_dictionary_array::<Int32Type>(rows, col_name)
                }
                DataType::Int64 => {
                    self.build_dictionary_array::<Int64Type>(rows, col_name)
                }
                DataType::UInt8 => {
                    self.build_dictionary_array::<UInt8Type>(rows, col_name)
                }
                DataType::UInt16 => {
                    self.build_dictionary_array::<UInt16Type>(rows, col_name)
                }
                DataType::UInt32 => {
                    self.build_dictionary_array::<UInt32Type>(rows, col_name)
                }
                DataType::UInt64 => {
                    self.build_dictionary_array::<UInt64Type>(rows, col_name)
                }
                _ => Err(ArrowError::JsonError(
                    "unsupported dictionary key type".to_string(),
                )),
            }
        } else {
            Err(ArrowError::JsonError(
                "dictionary types other than UTF-8 not yet supported".to_string(),
            ))
        }
    }

    fn build_boolean_array(&self, rows: &[Value], col_name: &str) -> Result<ArrayRef> {
        let mut builder = BooleanBuilder::new(rows.len());
        for row in rows {
            if let Some(value) = row.get(&col_name) {
                if let Some(boolean) = value.as_bool() {
                    builder.append_value(boolean)?
                } else {
                    builder.append_null()?;
                }
            } else {
                builder.append_null()?;
            }
        }
        Ok(Arc::new(builder.finish()))
    }

    fn build_boolean_list_array(
        &self,
        rows: &[Value],
        col_name: &str,
    ) -> Result<ArrayRef> {
        let values_builder = BooleanBuilder::new(rows.len() * 5);
        let mut builder = ListBuilder::new(values_builder);
        for row in rows {
            if let Some(value) = row.get(col_name) {
                // value can be an array or a scalar
                let vals: Vec<Option<bool>> = if let Value::Bool(v) = value {
                    vec![Some(*v)]
                } else if let Value::Array(n) = value {
                    n.iter().map(|v: &Value| v.as_bool()).collect()
                } else if let Value::Null = value {
                    vec![None]
                } else {
                    return Err(ArrowError::JsonError(
                        "2Only scalars are currently supported in JSON arrays"
                            .to_string(),
                    ));
                };
                for val in vals {
                    match val {
                        Some(v) => builder.values().append_value(v)?,
                        None => builder.values().append_null()?,
                    };
                }
            }
            builder.append(true)?
        }
        Ok(Arc::new(builder.finish()))
    }

    fn build_primitive_array<T: ArrowPrimitiveType>(
        &self,
        rows: &[Value],
        col_name: &str,
    ) -> Result<ArrayRef>
    where
        T: ArrowNumericType,
        T::Native: num::NumCast,
    {
        Ok(Arc::new(
            rows.iter()
                .map(|row| {
                    row.get(&col_name)
                        .and_then(|value| value.as_f64())
                        .and_then(num::cast::cast)
                })
                .collect::<PrimitiveArray<T>>(),
        ))
    }

    fn build_list_array<T: ArrowPrimitiveType>(
        &self,
        rows: &[Value],
        col_name: &str,
    ) -> Result<ArrayRef>
    where
        T::Native: num::NumCast,
    {
        let values_builder: PrimitiveBuilder<T> = PrimitiveBuilder::new(rows.len());
        let mut builder = ListBuilder::new(values_builder);
        for row in rows {
            if let Some(value) = row.get(&col_name) {
                // value can be an array or a scalar
                let vals: Vec<Option<f64>> = if let Value::Number(value) = value {
                    vec![value.as_f64()]
                } else if let Value::Array(n) = value {
                    n.iter().map(|v: &Value| v.as_f64()).collect()
                } else if let Value::Null = value {
                    vec![None]
                } else {
                    return Err(ArrowError::JsonError(
                        "3Only scalars are currently supported in JSON arrays"
                            .to_string(),
                    ));
                };
                for val in vals {
                    match val {
                        Some(v) => match num::cast::cast(v) {
                            Some(v) => builder.values().append_value(v)?,
                            None => builder.values().append_null()?,
                        },
                        None => builder.values().append_null()?,
                    };
                }
            }
            builder.append(true)?
        }
        Ok(Arc::new(builder.finish()))
    }

    fn build_struct_array(
        &self,
        rows: &[Value],
        struct_fields: &[Field],
        projection: &[String],
    ) -> Result<Vec<ArrayRef>> {
        let arrays: Result<Vec<ArrayRef>> =
            struct_fields
                .iter()
                .filter(|field| {
                    if projection.is_empty() {
                        return true;
                    }
                    projection.contains(field.name())
                })
                .map(|field| {
                    match field.data_type() {
                        DataType::Null => unimplemented!(),
                        DataType::Boolean => self.build_boolean_array(rows, field.name()),
                        DataType::Float64 => {
                            self.build_primitive_array::<Float64Type>(rows, field.name())
                        }
                        DataType::Float32 => {
                            self.build_primitive_array::<Float32Type>(rows, field.name())
                        }
                        DataType::Int64 => {
                            self.build_primitive_array::<Int64Type>(rows, field.name())
                        }
                        DataType::Int32 => {
                            self.build_primitive_array::<Int32Type>(rows, field.name())
                        }
                        DataType::Int16 => {
                            self.build_primitive_array::<Int16Type>(rows, field.name())
                        }
                        DataType::Int8 => {
                            self.build_primitive_array::<Int8Type>(rows, field.name())
                        }
                        DataType::UInt64 => {
                            self.build_primitive_array::<UInt64Type>(rows, field.name())
                        }
                        DataType::UInt32 => {
                            self.build_primitive_array::<UInt32Type>(rows, field.name())
                        }
                        DataType::UInt16 => {
                            self.build_primitive_array::<UInt16Type>(rows, field.name())
                        }
                        DataType::UInt8 => {
                            self.build_primitive_array::<UInt8Type>(rows, field.name())
                        }
                        DataType::Timestamp(unit, _) => match unit {
                            TimeUnit::Second => self
                                .build_primitive_array::<TimestampSecondType>(
                                    rows,
                                    field.name(),
                                ),
                            TimeUnit::Microsecond => self
                                .build_primitive_array::<TimestampMicrosecondType>(
                                    rows,
                                    field.name(),
                                ),
                            TimeUnit::Millisecond => self
                                .build_primitive_array::<TimestampMillisecondType>(
                                    rows,
                                    field.name(),
                                ),
                            TimeUnit::Nanosecond => self
                                .build_primitive_array::<TimestampNanosecondType>(
                                    rows,
                                    field.name(),
                                ),
                        },
                        DataType::Date64(_) => {
                            self.build_primitive_array::<Date64Type>(rows, field.name())
                        }
                        DataType::Date32(_) => {
                            self.build_primitive_array::<Date32Type>(rows, field.name())
                        }
                        DataType::Time64(unit) => match unit {
                            TimeUnit::Microsecond => self
                                .build_primitive_array::<Time64MicrosecondType>(
                                    rows,
                                    field.name(),
                                ),
                            TimeUnit::Nanosecond => self
                                .build_primitive_array::<Time64NanosecondType>(
                                    rows,
                                    field.name(),
                                ),
                            _ => unimplemented!(),
                        },
                        DataType::Time32(unit) => match unit {
                            TimeUnit::Second => self
                                .build_primitive_array::<Time32SecondType>(
                                    rows,
                                    field.name(),
                                ),
                            TimeUnit::Millisecond => self
                                .build_primitive_array::<Time32MillisecondType>(
                                    rows,
                                    field.name(),
                                ),
                            _ => unimplemented!(),
                        },
                        DataType::Utf8 => {
                            let mut builder = StringBuilder::new(rows.len());
                            for row in rows {
                                if let Some(value) = row.get(field.name()) {
                                    if let Some(str_v) = value.as_str() {
                                        builder.append_value(str_v)?
                                    } else {
                                        builder.append(false)?
                                    }
                                } else {
                                    builder.append(false)?
                                }
                            }
                            Ok(Arc::new(builder.finish()) as ArrayRef)
                        }
                        DataType::List(ref t) => {
                            match t.data_type() {
                                DataType::Int8 => {
                                    self.build_list_array::<Int8Type>(rows, field.name())
                                }
                                DataType::Int16 => {
                                    self.build_list_array::<Int16Type>(rows, field.name())
                                }
                                DataType::Int32 => {
                                    self.build_list_array::<Int32Type>(rows, field.name())
                                }
                                DataType::Int64 => {
                                    self.build_list_array::<Int64Type>(rows, field.name())
                                }
                                DataType::UInt8 => {
                                    self.build_list_array::<UInt8Type>(rows, field.name())
                                }
                                DataType::UInt16 => self
                                    .build_list_array::<UInt16Type>(rows, field.name()),
                                DataType::UInt32 => self
                                    .build_list_array::<UInt32Type>(rows, field.name()),
                                DataType::UInt64 => self
                                    .build_list_array::<UInt64Type>(rows, field.name()),
                                DataType::Float32 => self
                                    .build_list_array::<Float32Type>(rows, field.name()),
                                DataType::Float64 => self
                                    .build_list_array::<Float64Type>(rows, field.name()),
                                DataType::Null => unimplemented!(),
                                DataType::Boolean => {
                                    self.build_boolean_list_array(rows, field.name())
                                }
                                ref dtype @ DataType::Utf8 => {
                                    // UInt64Type passed down below is a fake type for dictionary builder.
                                    // It is there to make compiler happy.
                                    self.list_array_string_array_builder::<UInt64Type>(
                                        &dtype,
                                        field.name(),
                                        rows,
                                    )
                                }
                                DataType::Dictionary(ref key_ty, _) => self
                                    .build_wrapped_list_array(rows, field.name(), key_ty),
                                ref e => Err(ArrowError::JsonError(format!(
                            "Data type is currently not supported in a list : {:?}",
                            e
                        ))),
                            }
                        }
                        DataType::Dictionary(ref key_ty, ref val_ty) => self
                            .build_string_dictionary_array(
                                rows,
                                field.name(),
                                key_ty,
                                val_ty,
                            ),
                        DataType::Struct(fields) => {
                            // TODO: add a check limiting recursion
                            let len = rows.len();
                            let num_bytes = bit_util::ceil(len, 8);
                            let mut null_buffer = MutableBuffer::new(num_bytes)
                                .with_bitset(num_bytes, false);
                            let struct_rows = rows
                                .iter()
                                .enumerate()
                                .map(|(i, row)| {
                                    (
                                        i,
                                        row.as_object()
                                            .map(|v| v.get(field.name()))
                                            .flatten(),
                                    )
                                })
                                .map(|(i, v)| match v {
                                    // we want the field as an object, if it's not, we treat as null
                                    Some(Value::Object(value)) => {
                                        bit_util::set_bit(null_buffer.data_mut(), i);
                                        Value::Object(value.clone())
                                    }
                                    _ => Value::Object(Default::default()),
                                })
                                .collect::<Vec<Value>>();
                            let arrays =
                                self.build_struct_array(&struct_rows, fields, &[])?;
                            // construct a struct array's data in order to set null buffer
                            let data_type = DataType::Struct(fields.clone());
                            let data = ArrayDataBuilder::new(data_type)
                                .len(len)
                                .null_bit_buffer(null_buffer.freeze())
                                .child_data(
                                    arrays.into_iter().map(|a| a.data()).collect(),
                                )
                                .build();
                            Ok(make_array(data))
                        }
                        _ => Err(ArrowError::JsonError(format!(
                            "{:?} type is not supported",
                            field.data_type()
                        ))),
                    }
                })
                .collect();
        arrays
    }

    #[inline(always)]
    fn build_dictionary_array<T>(
        &self,
        rows: &[Value],
        col_name: &str,
    ) -> Result<ArrayRef>
    where
        T::Native: num::NumCast,
        T: ArrowPrimitiveType + ArrowDictionaryKeyType,
    {
        let mut builder: StringDictionaryBuilder<T> =
            self.build_string_dictionary_builder(rows.len())?;
        for row in rows {
            if let Some(value) = row.get(&col_name) {
                if let Some(str_v) = value.as_str() {
                    builder.append(str_v).map(drop)?
                } else {
                    builder.append_null()?
                }
            } else {
                builder.append_null()?
            }
        }
        Ok(Arc::new(builder.finish()) as ArrayRef)
    }
}

/// JSON file reader builder
#[derive(Debug)]
pub struct ReaderBuilder {
    /// Optional schema for the JSON file
    ///
    /// If the schema is not supplied, the reader will try to infer the schema
    /// based on the JSON structure.
    schema: Option<SchemaRef>,
    /// Optional maximum number of records to read during schema inference
    ///
    /// If a number is not provided, all the records are read.
    max_records: Option<usize>,
    /// Batch size (number of records to load each time)
    ///
    /// The default batch size when using the `ReaderBuilder` is 1024 records
    batch_size: usize,
    /// Optional projection for which columns to load (zero-based column indices)
    projection: Option<Vec<String>>,
}

impl Default for ReaderBuilder {
    fn default() -> Self {
        Self {
            schema: None,
            max_records: None,
            batch_size: 1024,
            projection: None,
        }
    }
}

impl ReaderBuilder {
    /// Create a new builder for configuring JSON parsing options.
    ///
    /// To convert a builder into a reader, call `Reader::from_builder`
    ///
    /// # Example
    ///
    /// ```
    /// extern crate arrow;
    ///
    /// use arrow::json;
    /// use std::fs::File;
    ///
    /// fn example() -> json::Reader<File> {
    ///     let file = File::open("test/data/basic.json").unwrap();
    ///
    ///     // create a builder, inferring the schema with the first 100 records
    ///     let builder = json::ReaderBuilder::new().infer_schema(Some(100));
    ///
    ///     let reader = builder.build::<File>(file).unwrap();
    ///
    ///     reader
    /// }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the JSON file's schema
    pub fn with_schema(mut self, schema: SchemaRef) -> Self {
        self.schema = Some(schema);
        self
    }

    /// Set the JSON reader to infer the schema of the file
    pub fn infer_schema(mut self, max_records: Option<usize>) -> Self {
        // remove any schema that is set
        self.schema = None;
        self.max_records = max_records;
        self
    }

    /// Set the batch size (number of records to load at one time)
    pub fn with_batch_size(mut self, batch_size: usize) -> Self {
        self.batch_size = batch_size;
        self
    }

    /// Set the reader's column projection
    pub fn with_projection(mut self, projection: Vec<String>) -> Self {
        self.projection = Some(projection);
        self
    }

    /// Create a new `Reader` from the `ReaderBuilder`
    pub fn build<R: Read + Seek>(self, source: R) -> Result<Reader<R>> {
        let mut buf_reader = BufReader::new(source);

        // check if schema should be inferred
        let schema = match self.schema {
            Some(schema) => schema,
            None => infer_json_schema_from_seekable(&mut buf_reader, self.max_records)?,
        };

        Ok(Reader::from_buf_reader(
            buf_reader,
            schema,
            self.batch_size,
            self.projection,
        ))
    }
}

#[cfg(test)]
mod tests {
    use crate::{
        buffer::Buffer,
        datatypes::DataType::{Dictionary, List},
    };

    use super::*;
    use flate2::read::GzDecoder;
    use std::fs::File;

    #[test]
    fn test_json_basic() {
        let builder = ReaderBuilder::new().infer_schema(None).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(4, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(0, a.0);
        assert_eq!(&DataType::Int64, a.1.data_type());
        let b = schema.column_with_name("b").unwrap();
        assert_eq!(1, b.0);
        assert_eq!(&DataType::Float64, b.1.data_type());
        let c = schema.column_with_name("c").unwrap();
        assert_eq!(2, c.0);
        assert_eq!(&DataType::Boolean, c.1.data_type());
        let d = schema.column_with_name("d").unwrap();
        assert_eq!(3, d.0);
        assert_eq!(&DataType::Utf8, d.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(1, aa.value(0));
        assert_eq!(-10, aa.value(1));
        let bb = batch
            .column(b.0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert!(2.0 - bb.value(0) < f64::EPSILON);
        assert!(-3.5 - bb.value(1) < f64::EPSILON);
        let cc = batch
            .column(c.0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(false, cc.value(0));
        assert_eq!(true, cc.value(10));
        let dd = batch
            .column(d.0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!("4", dd.value(0));
        assert_eq!("text", dd.value(8));
    }

    #[test]
    fn test_json_basic_with_nulls() {
        let builder = ReaderBuilder::new().infer_schema(None).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(4, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(&DataType::Int64, a.1.data_type());
        let b = schema.column_with_name("b").unwrap();
        assert_eq!(&DataType::Float64, b.1.data_type());
        let c = schema.column_with_name("c").unwrap();
        assert_eq!(&DataType::Boolean, c.1.data_type());
        let d = schema.column_with_name("d").unwrap();
        assert_eq!(&DataType::Utf8, d.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(true, aa.is_valid(0));
        assert_eq!(false, aa.is_valid(1));
        assert_eq!(false, aa.is_valid(11));
        let bb = batch
            .column(b.0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .unwrap();
        assert_eq!(true, bb.is_valid(0));
        assert_eq!(false, bb.is_valid(2));
        assert_eq!(false, bb.is_valid(11));
        let cc = batch
            .column(c.0)
            .as_any()
            .downcast_ref::<BooleanArray>()
            .unwrap();
        assert_eq!(true, cc.is_valid(0));
        assert_eq!(false, cc.is_valid(4));
        assert_eq!(false, cc.is_valid(11));
        let dd = batch
            .column(d.0)
            .as_any()
            .downcast_ref::<StringArray>()
            .unwrap();
        assert_eq!(false, dd.is_valid(0));
        assert_eq!(true, dd.is_valid(1));
        assert_eq!(false, dd.is_valid(4));
        assert_eq!(false, dd.is_valid(11));
    }

    #[test]
    fn test_json_basic_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float32, false),
            Field::new("c", DataType::Boolean, false),
            Field::new("d", DataType::Utf8, false),
        ]);

        let mut reader: Reader<File> = Reader::new(
            File::open("test/data/basic.json").unwrap(),
            Arc::new(schema.clone()),
            1024,
            None,
        );
        let reader_schema = reader.schema();
        assert_eq!(reader_schema, Arc::new(schema));
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(4, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = batch.schema();

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(&DataType::Int32, a.1.data_type());
        let b = schema.column_with_name("b").unwrap();
        assert_eq!(&DataType::Float32, b.1.data_type());
        let c = schema.column_with_name("c").unwrap();
        assert_eq!(&DataType::Boolean, c.1.data_type());
        let d = schema.column_with_name("d").unwrap();
        assert_eq!(&DataType::Utf8, d.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Int32Array>()
            .unwrap();
        assert_eq!(1, aa.value(0));
        // test that a 64bit value is returned as null due to overflowing
        assert_eq!(false, aa.is_valid(11));
        let bb = batch
            .column(b.0)
            .as_any()
            .downcast_ref::<Float32Array>()
            .unwrap();
        assert!(2.0 - bb.value(0) < f32::EPSILON);
        assert!(-3.5 - bb.value(1) < f32::EPSILON);
    }

    #[test]
    fn test_json_basic_schema_projection() {
        // We test implicit and explicit projection:
        // Implicit: omitting fields from a schema
        // Explicit: supplying a vec of fields to take
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Float32, false),
            Field::new("c", DataType::Boolean, false),
        ]);

        let mut reader: Reader<File> = Reader::new(
            File::open("test/data/basic.json").unwrap(),
            Arc::new(schema),
            1024,
            Some(vec!["a".to_string(), "c".to_string()]),
        );
        let reader_schema = reader.schema();
        let expected_schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("c", DataType::Boolean, false),
        ]));
        assert_eq!(reader_schema, expected_schema);

        let batch = reader.next().unwrap().unwrap();

        assert_eq!(2, batch.num_columns());
        assert_eq!(2, batch.schema().fields().len());
        assert_eq!(12, batch.num_rows());

        let schema = batch.schema();
        assert_eq!(reader_schema, schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(0, a.0);
        assert_eq!(&DataType::Int32, a.1.data_type());
        let c = schema.column_with_name("c").unwrap();
        assert_eq!(1, c.0);
        assert_eq!(&DataType::Boolean, c.1.data_type());
    }

    #[test]
    fn test_json_arrays() {
        let builder = ReaderBuilder::new().infer_schema(None).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/arrays.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(4, batch.num_columns());
        assert_eq!(3, batch.num_rows());

        let schema = batch.schema();

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(&DataType::Int64, a.1.data_type());
        let b = schema.column_with_name("b").unwrap();
        assert_eq!(
            &DataType::List(Box::new(NullableDataType::new(DataType::Float64, true))),
            b.1.data_type()
        );
        let c = schema.column_with_name("c").unwrap();
        assert_eq!(
            &DataType::List(Box::new(NullableDataType::new(DataType::Boolean, true))),
            c.1.data_type()
        );
        let d = schema.column_with_name("d").unwrap();
        assert_eq!(&DataType::Utf8, d.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .unwrap();
        assert_eq!(1, aa.value(0));
        assert_eq!(-10, aa.value(1));
        let bb = batch
            .column(b.0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let bb = bb.values();
        let bb = bb.as_any().downcast_ref::<Float64Array>().unwrap();
        assert_eq!(9, bb.len());
        assert!(2.0 - bb.value(0) < f64::EPSILON);
        assert!(-6.1 - bb.value(5) < f64::EPSILON);
        assert_eq!(false, bb.is_valid(7));

        let cc = batch
            .column(c.0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let cc = cc.values();
        let cc = cc.as_any().downcast_ref::<BooleanArray>().unwrap();
        assert_eq!(6, cc.len());
        assert_eq!(false, cc.value(0));
        assert_eq!(false, cc.value(4));
        assert_eq!(false, cc.is_valid(5));
    }

    #[test]
    #[should_panic(expected = "Not valid JSON")]
    fn test_invalid_file() {
        let builder = ReaderBuilder::new().infer_schema(None).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/uk_cities_with_headers.csv").unwrap())
            .unwrap();
        let _batch = reader.next().unwrap().unwrap();
    }

    #[test]
    fn test_coersion_scalar_and_list() {
        use crate::datatypes::DataType::*;

        assert_eq!(
            List(Box::new(NullableDataType::new(Float64, true))),
            coerce_data_type(vec![
                &Float64,
                &List(Box::new(NullableDataType::new(Float64, true)))
            ])
            .unwrap()
        );
        assert_eq!(
            List(Box::new(NullableDataType::new(Float64, true))),
            coerce_data_type(vec![
                &Float64,
                &List(Box::new(NullableDataType::new(Int64, true)))
            ])
            .unwrap()
        );
        assert_eq!(
            List(Box::new(NullableDataType::new(Int64, true))),
            coerce_data_type(vec![
                &Int64,
                &List(Box::new(NullableDataType::new(Int64, true)))
            ])
            .unwrap()
        );
        // boolean and number are incompatible, return utf8
        assert_eq!(
            List(Box::new(NullableDataType::new(Utf8, true))),
            coerce_data_type(vec![
                &Boolean,
                &List(Box::new(NullableDataType::new(Float64, true)))
            ])
            .unwrap()
        );
    }

    #[test]
    fn test_mixed_json_arrays() {
        let builder = ReaderBuilder::new().infer_schema(None).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/mixed_arrays.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        let mut file = File::open("test/data/mixed_arrays.json.gz").unwrap();
        let mut reader = BufReader::new(GzDecoder::new(&file));
        let schema = infer_json_schema(&mut reader, None).unwrap();
        file.seek(SeekFrom::Start(0)).unwrap();

        let reader = BufReader::new(GzDecoder::new(&file));
        let mut reader = Reader::from_buf_reader(reader, schema, 64, None);
        let batch_gz = reader.next().unwrap().unwrap();

        for batch in vec![batch, batch_gz] {
            assert_eq!(4, batch.num_columns());
            assert_eq!(4, batch.num_rows());

            let schema = batch.schema();

            let a = schema.column_with_name("a").unwrap();
            assert_eq!(&DataType::Int64, a.1.data_type());
            let b = schema.column_with_name("b").unwrap();
            assert_eq!(
                &DataType::List(Box::new(NullableDataType::new(DataType::Float64, true))),
                b.1.data_type()
            );
            let c = schema.column_with_name("c").unwrap();
            assert_eq!(
                &DataType::List(Box::new(NullableDataType::new(DataType::Boolean, true))),
                c.1.data_type()
            );
            let d = schema.column_with_name("d").unwrap();
            assert_eq!(
                &DataType::List(Box::new(NullableDataType::new(DataType::Utf8, true))),
                d.1.data_type()
            );

            let bb = batch
                .column(b.0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let bb = bb.values();
            let bb = bb.as_any().downcast_ref::<Float64Array>().unwrap();
            assert_eq!(10, bb.len());
            assert!(4.0 - bb.value(9) < f64::EPSILON);

            let cc = batch
                .column(c.0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let cc = cc.values();
            let cc = cc.as_any().downcast_ref::<BooleanArray>().unwrap();
            assert_eq!(6, cc.len());
            assert_eq!(false, cc.value(0));
            assert_eq!(false, cc.value(3));
            assert_eq!(false, cc.is_valid(2));
            assert_eq!(false, cc.is_valid(4));

            let dd = batch
                .column(d.0)
                .as_any()
                .downcast_ref::<ListArray>()
                .unwrap();
            let dd = dd.values();
            let dd = dd.as_any().downcast_ref::<StringArray>().unwrap();
            assert_eq!(7, dd.len());
            assert_eq!(false, dd.is_valid(1));
            assert_eq!("text", dd.value(2));
            assert_eq!("1", dd.value(3));
            assert_eq!("false", dd.value(4));
            assert_eq!("array", dd.value(5));
            assert_eq!("2.4", dd.value(6));
        }
    }

    #[test]
    fn test_nested_struct_json_arrays() {
        let c_field = Field::new(
            "c",
            DataType::Struct(vec![Field::new("d", DataType::Utf8, true)]),
            true,
        );
        let a_field = Field::new(
            "a",
            DataType::Struct(vec![
                Field::new("b", DataType::Boolean, true),
                c_field.clone(),
            ]),
            true,
        );
        let schema = Arc::new(Schema::new(vec![a_field.clone()]));
        let builder = ReaderBuilder::new().with_schema(schema).with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/nested_structs.json").unwrap())
            .unwrap();

        // build expected output
        let d = StringArray::from(vec![Some("text"), None, Some("text"), None]);
        let c = ArrayDataBuilder::new(c_field.data_type().clone())
            .null_count(2)
            .len(4)
            .add_child_data(d.data())
            .null_bit_buffer(Buffer::from(vec![0b00000101]))
            .build();
        let b = BooleanArray::from(vec![Some(true), Some(false), Some(true), None]);
        let a = ArrayDataBuilder::new(a_field.data_type().clone())
            .null_count(1)
            .len(4)
            .add_child_data(b.data())
            .add_child_data(c)
            .null_bit_buffer(Buffer::from(vec![0b00000111]))
            .build();
        let expected = make_array(a);

        // compare `a` with result from json reader
        let batch = reader.next().unwrap().unwrap();
        let read = batch.column(0);
        assert!(
            expected.data_ref() == read.data_ref(),
            format!("{:?} != {:?}", expected.data(), read.data())
        );
    }

    #[test]
    fn test_dictionary_from_json_basic_with_nulls() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::Int16), Box::new(DataType::Utf8)),
            d.1.data_type()
        );

        let dd = batch
            .column(d.0)
            .as_any()
            .downcast_ref::<DictionaryArray<Int16Type>>()
            .unwrap();
        assert_eq!(false, dd.is_valid(0));
        assert_eq!(true, dd.is_valid(1));
        assert_eq!(true, dd.is_valid(2));
        assert_eq!(false, dd.is_valid(11));

        assert_eq!(
            dd.keys(),
            &Int16Array::from(vec![
                None,
                Some(0),
                Some(1),
                Some(0),
                None,
                None,
                Some(0),
                None,
                Some(1),
                Some(0),
                Some(0),
                None
            ])
        );
    }

    #[test]
    fn test_dictionary_from_json_int8() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::Int8), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_dictionary_from_json_int32() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::Int32), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_dictionary_from_json_int64() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::Int64), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::Int64), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_list_of_string_dictionary_from_json() {
        let schema = Schema::new(vec![Field::new(
            "events",
            List(Box::new(NullableDataType::new(
                Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
                true,
            ))),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/list_string_dict_nested.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(3, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let events = schema.column_with_name("events").unwrap();
        assert_eq!(
            &List(Box::new(NullableDataType::new(
                Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
                true
            ))),
            events.1.data_type()
        );

        let evs_list = batch
            .column(events.0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let evs_list = evs_list.values();
        let evs_list = evs_list
            .as_any()
            .downcast_ref::<DictionaryArray<UInt64Type>>()
            .unwrap();
        assert_eq!(6, evs_list.len());
        assert_eq!(true, evs_list.is_valid(1));
        assert_eq!(DataType::Utf8, evs_list.value_type());

        // dict from the events list
        let dict_el = evs_list.values();
        let dict_el = dict_el.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(3, dict_el.len());
        assert_eq!("Elect Leader", dict_el.value(0));
        assert_eq!("Do Ballot", dict_el.value(1));
        assert_eq!("Send Data", dict_el.value(2));
    }

    #[test]
    fn test_list_of_string_dictionary_from_json_with_nulls() {
        let schema = Schema::new(vec![Field::new(
            "events",
            List(Box::new(NullableDataType::new(
                Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
                true,
            ))),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(
                File::open("test/data/list_string_dict_nested_nulls.json").unwrap(),
            )
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(3, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let events = schema.column_with_name("events").unwrap();
        assert_eq!(
            &List(Box::new(NullableDataType::new(
                Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
                true
            ))),
            events.1.data_type()
        );

        let evs_list = batch
            .column(events.0)
            .as_any()
            .downcast_ref::<ListArray>()
            .unwrap();
        let evs_list = evs_list.values();
        let evs_list = evs_list
            .as_any()
            .downcast_ref::<DictionaryArray<UInt64Type>>()
            .unwrap();
        assert_eq!(8, evs_list.len());
        assert_eq!(true, evs_list.is_valid(1));
        assert_eq!(DataType::Utf8, evs_list.value_type());

        // dict from the events list
        let dict_el = evs_list.values();
        let dict_el = dict_el.as_any().downcast_ref::<StringArray>().unwrap();
        assert_eq!(2, evs_list.null_count());
        assert_eq!(3, dict_el.len());
        assert_eq!("Elect Leader", dict_el.value(0));
        assert_eq!("Do Ballot", dict_el.value(1));
        assert_eq!("Send Data", dict_el.value(2));
    }

    #[test]
    fn test_dictionary_from_json_uint8() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::UInt8), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_dictionary_from_json_uint32() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::UInt32), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_dictionary_from_json_uint64() {
        let schema = Schema::new(vec![Field::new(
            "d",
            Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
            true,
        )]);
        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let d = schema.column_with_name("d").unwrap();
        assert_eq!(
            &Dictionary(Box::new(DataType::UInt64), Box::new(DataType::Utf8)),
            d.1.data_type()
        );
    }

    #[test]
    fn test_with_multiple_batches() {
        let builder = ReaderBuilder::new()
            .infer_schema(Some(4))
            .with_batch_size(5);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();

        let mut num_records = Vec::new();
        while let Some(rb) = reader.next().unwrap() {
            num_records.push(rb.num_rows());
        }

        assert_eq!(vec![5, 5, 2], num_records);
    }

    #[test]
    fn test_json_infer_schema() {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int64, true),
            Field::new(
                "b",
                DataType::List(Box::new(NullableDataType::new(DataType::Float64, true))),
                true,
            ),
            Field::new(
                "c",
                DataType::List(Box::new(NullableDataType::new(DataType::Boolean, true))),
                true,
            ),
            Field::new(
                "d",
                DataType::List(Box::new(NullableDataType::new(DataType::Utf8, true))),
                true,
            ),
        ]);

        let mut reader =
            BufReader::new(File::open("test/data/mixed_arrays.json").unwrap());
        let inferred_schema = infer_json_schema_from_seekable(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, Arc::new(schema.clone()));

        let file = File::open("test/data/mixed_arrays.json.gz").unwrap();
        let mut reader = BufReader::new(GzDecoder::new(&file));
        let inferred_schema = infer_json_schema(&mut reader, None).unwrap();

        assert_eq!(inferred_schema, Arc::new(schema));
    }

    #[test]
    fn test_timestamp_from_json_seconds() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Second, None),
            true,
        )]);

        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(
            &DataType::Timestamp(TimeUnit::Second, None),
            a.1.data_type()
        );

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<TimestampSecondArray>()
            .unwrap();
        assert_eq!(true, aa.is_valid(0));
        assert_eq!(false, aa.is_valid(1));
        assert_eq!(false, aa.is_valid(2));
        assert_eq!(1, aa.value(0));
        assert_eq!(1, aa.value(3));
        assert_eq!(5, aa.value(7));
    }

    #[test]
    fn test_timestamp_from_json_milliseconds() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Timestamp(TimeUnit::Millisecond, None),
            true,
        )]);

        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(
            &DataType::Timestamp(TimeUnit::Millisecond, None),
            a.1.data_type()
        );

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<TimestampMillisecondArray>()
            .unwrap();
        assert_eq!(true, aa.is_valid(0));
        assert_eq!(false, aa.is_valid(1));
        assert_eq!(false, aa.is_valid(2));
        assert_eq!(1, aa.value(0));
        assert_eq!(1, aa.value(3));
        assert_eq!(5, aa.value(7));
    }

    #[test]
    fn test_date_from_json_milliseconds() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Date64(DateUnit::Millisecond),
            true,
        )]);

        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(&DataType::Date64(DateUnit::Millisecond), a.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Date64Array>()
            .unwrap();
        assert_eq!(true, aa.is_valid(0));
        assert_eq!(false, aa.is_valid(1));
        assert_eq!(false, aa.is_valid(2));
        assert_eq!(1, aa.value(0));
        assert_eq!(1, aa.value(3));
        assert_eq!(5, aa.value(7));
    }

    #[test]
    fn test_time_from_json_nanoseconds() {
        let schema = Schema::new(vec![Field::new(
            "a",
            DataType::Time64(TimeUnit::Nanosecond),
            true,
        )]);

        let builder = ReaderBuilder::new()
            .with_schema(Arc::new(schema))
            .with_batch_size(64);
        let mut reader: Reader<File> = builder
            .build::<File>(File::open("test/data/basic_nulls.json").unwrap())
            .unwrap();
        let batch = reader.next().unwrap().unwrap();

        assert_eq!(1, batch.num_columns());
        assert_eq!(12, batch.num_rows());

        let schema = reader.schema();
        let batch_schema = batch.schema();
        assert_eq!(schema, batch_schema);

        let a = schema.column_with_name("a").unwrap();
        assert_eq!(&DataType::Time64(TimeUnit::Nanosecond), a.1.data_type());

        let aa = batch
            .column(a.0)
            .as_any()
            .downcast_ref::<Time64NanosecondArray>()
            .unwrap();
        assert_eq!(true, aa.is_valid(0));
        assert_eq!(false, aa.is_valid(1));
        assert_eq!(false, aa.is_valid(2));
        assert_eq!(1, aa.value(0));
        assert_eq!(1, aa.value(3));
        assert_eq!(5, aa.value(7));
    }
}
