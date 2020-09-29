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

//! Common unit test utility methods

use crate::datasource::{MemTable, TableProvider};
use crate::error::Result;
use crate::execution::context::ExecutionContext;
use crate::logical_plan::{LogicalPlan, LogicalPlanBuilder};
use crate::physical_plan::ExecutionPlan;
use arrow::array;
use arrow::array::PrimitiveArrayOps;
use arrow::datatypes::{DataType, Field, Schema, SchemaRef};
use arrow::record_batch::RecordBatch;
use std::env;
use std::fs::File;
use std::io::prelude::*;
use std::io::{BufReader, BufWriter};
use std::sync::Arc;
use tempfile::TempDir;

pub fn create_table_dual() -> Box<dyn TableProvider + Send + Sync> {
    let dual_schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Int32, false),
        Field::new("name", DataType::Utf8, false),
    ]));
    let batch = RecordBatch::try_new(
        dual_schema.clone(),
        vec![
            Arc::new(array::Int32Array::from(vec![1])),
            Arc::new(array::StringArray::from(vec!["a"])),
        ],
    )
    .unwrap();
    let provider = MemTable::new(dual_schema.clone(), vec![vec![batch.clone()]]).unwrap();
    Box::new(provider)
}

/// Get the value of the ARROW_TEST_DATA environment variable
pub fn arrow_testdata_path() -> String {
    env::var("ARROW_TEST_DATA").expect("ARROW_TEST_DATA not defined")
}

/// Execute a physical plan and collect the results
pub async fn execute(plan: Arc<dyn ExecutionPlan>) -> Result<Vec<RecordBatch>> {
    let ctx = ExecutionContext::new();
    ctx.collect(plan).await
}

/// Generated partitioned copy of a CSV file
pub fn create_partitioned_csv(filename: &str, partitions: usize) -> Result<String> {
    let testdata = arrow_testdata_path();
    let path = format!("{}/csv/{}", testdata, filename);

    let tmp_dir = TempDir::new()?;

    let mut writers = vec![];
    for i in 0..partitions {
        let filename = format!("partition-{}.csv", i);
        let filename = tmp_dir.path().join(&filename);

        let writer = BufWriter::new(File::create(&filename).unwrap());
        writers.push(writer);
    }

    let f = File::open(&path)?;
    let f = BufReader::new(f);
    let mut i = 0;
    for line in f.lines() {
        let line = line.unwrap();

        if i == 0 {
            // write header to all partitions
            for w in writers.iter_mut() {
                w.write_all(line.as_bytes()).unwrap();
                w.write_all(b"\n").unwrap();
            }
        } else {
            // write data line to single partition
            let partition = i % partitions;
            writers[partition].write_all(line.as_bytes()).unwrap();
            writers[partition].write_all(b"\n").unwrap();
        }

        i += 1;
    }
    for w in writers.iter_mut() {
        w.flush().unwrap();
    }

    Ok(tmp_dir.into_path().to_str().unwrap().to_string())
}

/// Get the schema for the aggregate_test_* csv files
pub fn aggr_test_schema() -> SchemaRef {
    Arc::new(Schema::new(vec![
        Field::new("c1", DataType::Utf8, false),
        Field::new("c2", DataType::UInt32, false),
        Field::new("c3", DataType::Int8, false),
        Field::new("c4", DataType::Int16, false),
        Field::new("c5", DataType::Int32, false),
        Field::new("c6", DataType::Int64, false),
        Field::new("c7", DataType::UInt8, false),
        Field::new("c8", DataType::UInt16, false),
        Field::new("c9", DataType::UInt32, false),
        Field::new("c10", DataType::UInt64, false),
        Field::new("c11", DataType::Float32, false),
        Field::new("c12", DataType::Float64, false),
        Field::new("c13", DataType::Utf8, false),
    ]))
}

/// Format a batch as csv
pub fn format_batch(batch: &RecordBatch) -> Vec<String> {
    let mut rows = vec![];
    for row_index in 0..batch.num_rows() {
        let mut s = String::new();
        for column_index in 0..batch.num_columns() {
            if column_index > 0 {
                s.push(',');
            }
            let array = batch.column(column_index);
            match array.data_type() {
                DataType::Int8 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Int8Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::Int16 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Int16Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::Int32 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Int32Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::Int64 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Int64Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::UInt8 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::UInt8Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::UInt16 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::UInt16Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::UInt32 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::UInt32Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::UInt64 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::UInt64Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::Float32 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Float32Array>()
                        .unwrap()
                        .value(row_index)
                )),
                DataType::Float64 => s.push_str(&format!(
                    "{:?}",
                    array
                        .as_any()
                        .downcast_ref::<array::Float64Array>()
                        .unwrap()
                        .value(row_index)
                )),
                _ => s.push('?'),
            }
        }
        rows.push(s);
    }
    rows
}

/// all tests share a common table
pub fn test_table_scan() -> Result<LogicalPlan> {
    let schema = Schema::new(vec![
        Field::new("a", DataType::UInt32, false),
        Field::new("b", DataType::UInt32, false),
        Field::new("c", DataType::UInt32, false),
    ]);
    LogicalPlanBuilder::scan("default", "test", &schema, None)?.build()
}

pub fn assert_fields_eq(plan: &LogicalPlan, expected: Vec<&str>) {
    let actual: Vec<String> = plan
        .schema()
        .fields()
        .iter()
        .map(|f| f.name().clone())
        .collect();
    assert_eq!(actual, expected);
}

pub mod variable;
