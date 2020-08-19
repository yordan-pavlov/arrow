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

//! Defines the SORT plan

use std::sync::{Arc, Mutex};

use arrow::array::ArrayRef;
pub use arrow::compute::SortOptions;
use arrow::compute::{concat, lexsort_to_indices, take, SortColumn, TakeOptions};
use arrow::datatypes::SchemaRef;
use arrow::record_batch::{RecordBatch, RecordBatchReader};

use crate::error::Result;
use crate::execution::physical_plan::common::RecordBatchIterator;
use crate::execution::physical_plan::expressions::PhysicalSortExpr;
use crate::execution::physical_plan::merge::MergeExec;
use crate::execution::physical_plan::{common, ExecutionPlan, Partition};

/// Sort execution plan
#[derive(Debug)]
pub struct SortExec {
    /// Input schema
    input: Arc<dyn ExecutionPlan>,
    expr: Vec<PhysicalSortExpr>,
    /// Number of threads to execute input partitions on before combining into a single partition
    concurrency: usize,
}

impl SortExec {
    /// Create a new sort execution plan
    pub fn try_new(
        expr: Vec<PhysicalSortExpr>,
        input: Arc<dyn ExecutionPlan>,
        concurrency: usize,
    ) -> Result<Self> {
        Ok(Self {
            expr,
            input,
            concurrency,
        })
    }
}

impl ExecutionPlan for SortExec {
    fn schema(&self) -> SchemaRef {
        self.input.schema().clone()
    }

    fn partitions(&self) -> Result<Vec<Arc<dyn Partition>>> {
        Ok(vec![
            (Arc::new(SortPartition {
                input: self.input.partitions()?,
                expr: self.expr.clone(),
                schema: self.schema(),
                concurrency: self.concurrency,
            })),
        ])
    }
}

/// Represents a single partition of a Sort execution plan
#[derive(Debug)]
struct SortPartition {
    schema: SchemaRef,
    expr: Vec<PhysicalSortExpr>,
    input: Vec<Arc<dyn Partition>>,
    /// Number of threads to execute input partitions on before combining into a single partition
    concurrency: usize,
}

impl Partition for SortPartition {
    /// Execute the sort
    fn execute(&self) -> Result<Arc<Mutex<dyn RecordBatchReader + Send + Sync>>> {
        // sort needs to operate on a single partition currently
        let merge =
            MergeExec::new(self.schema.clone(), self.input.clone(), self.concurrency);
        let merge_partitions = merge.partitions()?;
        // MergeExec must always produce a single partition
        assert_eq!(1, merge_partitions.len());
        let it = merge_partitions[0].execute()?;
        let batches = common::collect(it)?;

        // combine all record batches into one for each column
        let combined_batch = RecordBatch::try_new(
            self.schema.clone(),
            self.schema
                .fields()
                .iter()
                .enumerate()
                .map(|(i, _)| -> Result<ArrayRef> {
                    Ok(concat(
                        &batches
                            .iter()
                            .map(|batch| batch.columns()[i].clone())
                            .collect::<Vec<ArrayRef>>(),
                    )?)
                })
                .collect::<Result<Vec<ArrayRef>>>()?,
        )?;

        // sort combined record batch
        let indices = lexsort_to_indices(
            &self
                .expr
                .iter()
                .map(|e| e.evaluate_to_sort_column(&combined_batch))
                .collect::<Result<Vec<SortColumn>>>()?,
        )?;

        // reorder all rows based on sorted indices
        let sorted_batch = RecordBatch::try_new(
            self.schema.clone(),
            combined_batch
                .columns()
                .iter()
                .map(|column| -> Result<ArrayRef> {
                    Ok(take(
                        column,
                        &indices,
                        // disable bound check overhead since indices are already generated from
                        // the same record batch
                        Some(TakeOptions {
                            check_bounds: false,
                        }),
                    )?)
                })
                .collect::<Result<Vec<ArrayRef>>>()?,
        )?;

        Ok(Arc::new(Mutex::new(RecordBatchIterator::new(
            self.schema.clone(),
            vec![Arc::new(sorted_batch)],
        ))))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::physical_plan::csv::{CsvExec, CsvReadOptions};
    use crate::execution::physical_plan::expressions::col;
    use crate::test;
    use arrow::array::*;
    use arrow::datatypes::*;

    #[test]
    fn test_sort() -> Result<()> {
        let schema = test::aggr_test_schema();
        let partitions = 4;
        let path = test::create_partitioned_csv("aggregate_test_100.csv", partitions)?;
        let csv =
            CsvExec::try_new(&path, CsvReadOptions::new().schema(&schema), None, 1024)?;

        let sort_exec = SortExec::try_new(
            vec![
                // c1 string column
                PhysicalSortExpr {
                    expr: col("c1"),
                    options: SortOptions::default(),
                },
                // c2 uin32 column
                PhysicalSortExpr {
                    expr: col("c2"),
                    options: SortOptions::default(),
                },
                // c7 uin8 column
                PhysicalSortExpr {
                    expr: col("c7"),
                    options: SortOptions::default(),
                },
            ],
            Arc::new(csv),
            2,
        )?;

        let result: Vec<RecordBatch> = test::execute(&sort_exec)?;
        assert_eq!(result.len(), 1);

        let columns = result[0].columns();

        let c1 = as_string_array(&columns[0]);
        assert_eq!(c1.value(0), "a");
        assert_eq!(c1.value(c1.len() - 1), "e");

        let c2 = as_primitive_array::<UInt32Type>(&columns[1]);
        assert_eq!(c2.value(0), 1);
        assert_eq!(c2.value(c2.len() - 1), 5,);

        let c7 = as_primitive_array::<UInt8Type>(&columns[6]);
        assert_eq!(c7.value(0), 15);
        assert_eq!(c7.value(c7.len() - 1), 254,);

        Ok(())
    }
}
