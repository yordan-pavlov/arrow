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

//! Implementation of DataFrame API

use std::sync::{Arc, Mutex};

use crate::arrow::record_batch::RecordBatch;
use crate::dataframe::*;
use crate::error::Result;
use crate::execution::context::{ExecutionContext, ExecutionContextState};
use crate::logicalplan::{col, Expr, LogicalPlan, LogicalPlanBuilder};
use arrow::datatypes::Schema;

/// Implementation of DataFrame API
pub struct DataFrameImpl {
    ctx_state: Arc<Mutex<ExecutionContextState>>,
    plan: LogicalPlan,
}

impl DataFrameImpl {
    /// Create a new Table based on an existing logical plan
    pub fn new(ctx_state: Arc<Mutex<ExecutionContextState>>, plan: &LogicalPlan) -> Self {
        Self {
            ctx_state,
            plan: plan.clone(),
        }
    }
}

impl DataFrame for DataFrameImpl {
    /// Apply a projection based on a list of column names
    fn select_columns(&self, columns: Vec<&str>) -> Result<Arc<dyn DataFrame>> {
        let exprs = columns
            .iter()
            .map(|name| {
                self.plan
                    .schema()
                    // take the index to ensure that the column exists in the schema
                    .index_of(name.to_owned())
                    .and_then(|_| Ok(col(name)))
                    .map_err(|e| e.into())
            })
            .collect::<Result<Vec<_>>>()?;
        self.select(exprs)
    }

    /// Create a projection based on arbitrary expressions
    fn select(&self, expr_list: Vec<Expr>) -> Result<Arc<dyn DataFrame>> {
        let plan = LogicalPlanBuilder::from(&self.plan)
            .project(expr_list)?
            .build()?;
        Ok(Arc::new(DataFrameImpl::new(self.ctx_state.clone(), &plan)))
    }

    /// Create a filter based on a predicate expression
    fn filter(&self, predicate: Expr) -> Result<Arc<dyn DataFrame>> {
        let plan = LogicalPlanBuilder::from(&self.plan)
            .filter(predicate)?
            .build()?;
        Ok(Arc::new(DataFrameImpl::new(self.ctx_state.clone(), &plan)))
    }

    /// Perform an aggregate query
    fn aggregate(
        &self,
        group_expr: Vec<Expr>,
        aggr_expr: Vec<Expr>,
    ) -> Result<Arc<dyn DataFrame>> {
        let plan = LogicalPlanBuilder::from(&self.plan)
            .aggregate(group_expr, aggr_expr)?
            .build()?;
        Ok(Arc::new(DataFrameImpl::new(self.ctx_state.clone(), &plan)))
    }

    /// Limit the number of rows
    fn limit(&self, n: usize) -> Result<Arc<dyn DataFrame>> {
        let plan = LogicalPlanBuilder::from(&self.plan).limit(n)?.build()?;
        Ok(Arc::new(DataFrameImpl::new(self.ctx_state.clone(), &plan)))
    }

    /// Sort by specified sorting expressions
    fn sort(&self, expr: Vec<Expr>) -> Result<Arc<dyn DataFrame>> {
        let plan = LogicalPlanBuilder::from(&self.plan).sort(expr)?.build()?;
        Ok(Arc::new(DataFrameImpl::new(self.ctx_state.clone(), &plan)))
    }

    /// Convert to logical plan
    fn to_logical_plan(&self) -> LogicalPlan {
        self.plan.clone()
    }

    fn collect(&self) -> Result<Vec<RecordBatch>> {
        let mut ctx = ExecutionContext::from(self.ctx_state.clone());
        ctx.collect_plan(&self.plan.clone())
    }

    /// Returns the schema from the logical plan
    fn schema(&self) -> &Schema {
        self.plan.schema().as_ref()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::datasource::csv::CsvReadOptions;
    use crate::execution::context::ExecutionContext;
    use crate::logicalplan::*;
    use crate::test;

    #[test]
    fn select_columns() -> Result<()> {
        // build plan using Table API
        let t = test_table()?;
        let t2 = t.select_columns(vec!["c1", "c2", "c11"])?;
        let plan = t2.to_logical_plan();

        // build query using SQL
        let sql_plan = create_plan("SELECT c1, c2, c11 FROM aggregate_test_100")?;

        // the two plans should be identical
        assert_same_plan(&plan, &sql_plan);

        Ok(())
    }

    #[test]
    fn select_expr() -> Result<()> {
        // build plan using Table API
        let t = test_table()?;
        let t2 = t.select(vec![col("c1"), col("c2"), col("c11")])?;
        let plan = t2.to_logical_plan();

        // build query using SQL
        let sql_plan = create_plan("SELECT c1, c2, c11 FROM aggregate_test_100")?;

        // the two plans should be identical
        assert_same_plan(&plan, &sql_plan);

        Ok(())
    }

    #[test]
    fn aggregate() -> Result<()> {
        // build plan using DataFrame API
        let df = test_table()?;
        let group_expr = vec![col("c1")];
        let aggr_expr = vec![
            min(col("c12")),
            max(col("c12")),
            avg(col("c12")),
            sum(col("c12")),
            count(col("c12")),
        ];

        let df = df.aggregate(group_expr.clone(), aggr_expr.clone())?;

        let plan = df.to_logical_plan();

        // build same plan using SQL API
        let sql = "SELECT c1, MIN(c12), MAX(c12), AVG(c12), SUM(c12), COUNT(c12) \
                   FROM aggregate_test_100 \
                   GROUP BY c1";
        let sql_plan = create_plan(sql)?;

        // the two plans should be identical
        assert_same_plan(&plan, &sql_plan);

        Ok(())
    }

    #[test]
    fn limit() -> Result<()> {
        // build query using Table API
        let t = test_table()?;
        let t2 = t.select_columns(vec!["c1", "c2", "c11"])?.limit(10)?;
        let plan = t2.to_logical_plan();

        // build query using SQL
        let sql_plan =
            create_plan("SELECT c1, c2, c11 FROM aggregate_test_100 LIMIT 10")?;

        // the two plans should be identical
        assert_same_plan(&plan, &sql_plan);

        Ok(())
    }

    /// Compare the formatted string representation of two plans for equality
    fn assert_same_plan(plan1: &LogicalPlan, plan2: &LogicalPlan) {
        assert_eq!(format!("{:?}", plan1), format!("{:?}", plan2));
    }

    /// Create a logical plan from a SQL query
    fn create_plan(sql: &str) -> Result<LogicalPlan> {
        let mut ctx = ExecutionContext::new();
        register_aggregate_csv(&mut ctx)?;
        ctx.create_logical_plan(sql)
    }

    fn test_table() -> Result<Arc<dyn DataFrame + 'static>> {
        let mut ctx = ExecutionContext::new();
        register_aggregate_csv(&mut ctx)?;
        ctx.table("aggregate_test_100")
    }

    fn register_aggregate_csv(ctx: &mut ExecutionContext) -> Result<()> {
        let schema = test::aggr_test_schema();
        let testdata = test::arrow_testdata_path();
        ctx.register_csv(
            "aggregate_test_100",
            &format!("{}/csv/aggregate_test_100.csv", testdata),
            CsvReadOptions::new().schema(&schema),
        )?;
        Ok(())
    }
}
