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

//! Defines the EXPLAIN operator

use crate::error::Result;
use crate::{
    execution::physical_plan::{common::RecordBatchIterator, ExecutionPlan, Partition},
    logicalplan::StringifiedPlan,
};
use arrow::{
    array::StringArray,
    datatypes::SchemaRef,
    record_batch::{RecordBatch, RecordBatchReader},
};

use std::sync::{Arc, Mutex};

/// Explain execution plan operator. This operator contains the string
/// values of the various plans it has when it is created, and passes
/// them to its output.
#[derive(Debug)]
pub struct ExplainExec {
    /// The schema that this exec plan node outputs
    schema: SchemaRef,

    /// The strings to be printed
    stringified_plans: Vec<StringifiedPlan>,
}

impl ExplainExec {
    /// Create a new MergeExec
    pub fn new(schema: SchemaRef, stringified_plans: Vec<StringifiedPlan>) -> Self {
        ExplainExec {
            schema,
            stringified_plans,
        }
    }
}

impl ExecutionPlan for ExplainExec {
    fn schema(&self) -> SchemaRef {
        self.schema.clone()
    }

    fn partitions(&self) -> Result<Vec<Arc<dyn Partition>>> {
        Ok(vec![Arc::new(ExplainPartition {
            schema: self.schema.clone(),
            stringified_plans: self.stringified_plans.clone(),
        })])
    }
}

#[derive(Debug)]
struct ExplainPartition {
    /// Input schema
    schema: SchemaRef,
    /// The various plans that were created.
    stringified_plans: Vec<StringifiedPlan>,
}

impl Partition for ExplainPartition {
    fn execute(&self) -> Result<Arc<Mutex<dyn RecordBatchReader + Send + Sync>>> {
        let mut type_builder = StringArray::builder(self.stringified_plans.len());
        let mut plan_builder = StringArray::builder(self.stringified_plans.len());

        for p in &self.stringified_plans {
            type_builder.append_value(&String::from(&p.plan_type))?;
            plan_builder.append_value(&p.plan)?;
        }

        let record_batch = RecordBatch::try_new(
            self.schema.clone(),
            vec![
                Arc::new(type_builder.finish()),
                Arc::new(plan_builder.finish()),
            ],
        )?;

        Ok(Arc::new(Mutex::new(RecordBatchIterator::new(
            self.schema.clone(),
            vec![Arc::new(record_batch)],
        ))))
    }
}
