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

//! Collection of utility functions that are leveraged by the query optimizer rules

use std::{collections::HashSet, sync::Arc};

use arrow::datatypes::{Schema, SchemaRef};

use super::optimizer::OptimizerRule;
use crate::error::{ExecutionError, Result};
use crate::logical_plan::{Expr, LogicalPlan, PlanType, StringifiedPlan};

/// Recursively walk a list of expression trees, collecting the unique set of column
/// names referenced in the expression
pub fn exprlist_to_column_names(
    expr: &[Expr],
    accum: &mut HashSet<String>,
) -> Result<()> {
    for e in expr {
        expr_to_column_names(e, accum)?;
    }
    Ok(())
}

/// Recursively walk an expression tree, collecting the unique set of column names
/// referenced in the expression
pub fn expr_to_column_names(expr: &Expr, accum: &mut HashSet<String>) -> Result<()> {
    match expr {
        Expr::Alias(expr, _) => expr_to_column_names(expr, accum),
        Expr::Column(name) => {
            accum.insert(name.clone());
            Ok(())
        }
        Expr::ScalarVariable(var_names) => {
            accum.insert(var_names.join("."));
            Ok(())
        }
        Expr::Literal(_) => {
            // not needed
            Ok(())
        }
        Expr::Not(e) => expr_to_column_names(e, accum),
        Expr::IsNull(e) => expr_to_column_names(e, accum),
        Expr::IsNotNull(e) => expr_to_column_names(e, accum),
        Expr::BinaryExpr { left, right, .. } => {
            expr_to_column_names(left, accum)?;
            expr_to_column_names(right, accum)?;
            Ok(())
        }
        Expr::Cast { expr, .. } => expr_to_column_names(expr, accum),
        Expr::Sort { expr, .. } => expr_to_column_names(expr, accum),
        Expr::AggregateFunction { args, .. } => exprlist_to_column_names(args, accum),
        Expr::AggregateUDF { args, .. } => exprlist_to_column_names(args, accum),
        Expr::ScalarFunction { args, .. } => exprlist_to_column_names(args, accum),
        Expr::ScalarUDF { args, .. } => exprlist_to_column_names(args, accum),
        Expr::Wildcard => Err(ExecutionError::General(
            "Wildcard expressions are not valid in a logical query plan".to_owned(),
        )),
        Expr::Nested(e) => expr_to_column_names(e, accum),
    }
}

/// Create a `LogicalPlan::Explain` node by running `optimizer` on the
/// input plan and capturing the resulting plan string
pub fn optimize_explain(
    optimizer: &mut impl OptimizerRule,
    verbose: bool,
    plan: &LogicalPlan,
    stringified_plans: &Vec<StringifiedPlan>,
    schema: &Schema,
) -> Result<LogicalPlan> {
    // These are the fields of LogicalPlan::Explain It might be nice
    // to transform that enum Variant into its own struct and avoid
    // passing the fields individually
    let plan = Arc::new(optimizer.optimize(plan)?);
    let mut stringified_plans = stringified_plans.clone();
    let optimizer_name = optimizer.name().into();
    stringified_plans.push(StringifiedPlan::new(
        PlanType::OptimizedLogicalPlan { optimizer_name },
        format!("{:#?}", plan),
    ));
    let schema = SchemaRef::new(schema.clone());

    Ok(LogicalPlan::Explain {
        verbose,
        plan,
        stringified_plans,
        schema,
    })
}

/// returns all expressions (non-recursively) in the current logical plan node.
pub fn expressions(plan: &LogicalPlan) -> Vec<Expr> {
    match plan {
        LogicalPlan::Projection { expr, .. } => expr.clone(),
        LogicalPlan::Filter { predicate, .. } => vec![predicate.clone()],
        LogicalPlan::Aggregate {
            group_expr,
            aggr_expr,
            ..
        } => {
            let mut result = group_expr.clone();
            result.extend(aggr_expr.clone());
            result
        }
        LogicalPlan::Sort { expr, .. } => expr.clone(),
        LogicalPlan::Extension { node } => node.expressions(),
        // plans without expressions
        LogicalPlan::TableScan { .. }
        | LogicalPlan::InMemoryScan { .. }
        | LogicalPlan::ParquetScan { .. }
        | LogicalPlan::CsvScan { .. }
        | LogicalPlan::EmptyRelation { .. }
        | LogicalPlan::Limit { .. }
        | LogicalPlan::CreateExternalTable { .. }
        | LogicalPlan::Explain { .. } => vec![],
    }
}

/// returns all inputs in the logical plan
pub fn inputs(plan: &LogicalPlan) -> Vec<&LogicalPlan> {
    match plan {
        LogicalPlan::Projection { input, .. } => vec![input],
        LogicalPlan::Filter { input, .. } => vec![input],
        LogicalPlan::Aggregate { input, .. } => vec![input],
        LogicalPlan::Sort { input, .. } => vec![input],
        LogicalPlan::Limit { input, .. } => vec![input],
        LogicalPlan::Extension { node } => node.inputs(),
        // plans without inputs
        LogicalPlan::TableScan { .. }
        | LogicalPlan::InMemoryScan { .. }
        | LogicalPlan::ParquetScan { .. }
        | LogicalPlan::CsvScan { .. }
        | LogicalPlan::EmptyRelation { .. }
        | LogicalPlan::CreateExternalTable { .. }
        | LogicalPlan::Explain { .. } => vec![],
    }
}

/// Returns a new logical plan based on the original one with inputs and expressions replaced
pub fn from_plan(
    plan: &LogicalPlan,
    expr: &Vec<Expr>,
    inputs: &Vec<LogicalPlan>,
) -> Result<LogicalPlan> {
    match plan {
        LogicalPlan::Projection { schema, .. } => Ok(LogicalPlan::Projection {
            expr: expr.clone(),
            input: Arc::new(inputs[0].clone()),
            schema: schema.clone(),
        }),
        LogicalPlan::Filter { .. } => Ok(LogicalPlan::Filter {
            predicate: expr[0].clone(),
            input: Arc::new(inputs[0].clone()),
        }),
        LogicalPlan::Aggregate {
            group_expr, schema, ..
        } => Ok(LogicalPlan::Aggregate {
            group_expr: expr[0..group_expr.len()].to_vec(),
            aggr_expr: expr[group_expr.len()..].to_vec(),
            input: Arc::new(inputs[0].clone()),
            schema: schema.clone(),
        }),
        LogicalPlan::Sort { .. } => Ok(LogicalPlan::Sort {
            expr: expr.clone(),
            input: Arc::new(inputs[0].clone()),
        }),
        LogicalPlan::Limit { n, .. } => Ok(LogicalPlan::Limit {
            n: *n,
            input: Arc::new(inputs[0].clone()),
        }),
        LogicalPlan::Extension { node } => Ok(LogicalPlan::Extension {
            node: node.from_template(expr, inputs),
        }),
        LogicalPlan::EmptyRelation { .. }
        | LogicalPlan::TableScan { .. }
        | LogicalPlan::InMemoryScan { .. }
        | LogicalPlan::ParquetScan { .. }
        | LogicalPlan::CsvScan { .. }
        | LogicalPlan::CreateExternalTable { .. }
        | LogicalPlan::Explain { .. } => Ok(plan.clone()),
    }
}

/// Returns all direct children `Expression`s of `expr`.
/// E.g. if the expression is "(a + 1) + 1", it returns ["a + 1", "1"] (as Expr objects)
pub fn expr_sub_expressions(expr: &Expr) -> Result<Vec<&Expr>> {
    match expr {
        Expr::BinaryExpr { left, right, .. } => Ok(vec![left, right]),
        Expr::IsNull(e) => Ok(vec![e]),
        Expr::IsNotNull(e) => Ok(vec![e]),
        Expr::ScalarFunction { args, .. } => Ok(args.iter().collect()),
        Expr::ScalarUDF { args, .. } => Ok(args.iter().collect()),
        Expr::AggregateFunction { args, .. } => Ok(args.iter().collect()),
        Expr::AggregateUDF { args, .. } => Ok(args.iter().collect()),
        Expr::Cast { expr, .. } => Ok(vec![expr]),
        Expr::Column(_) => Ok(vec![]),
        Expr::Alias(expr, ..) => Ok(vec![expr]),
        Expr::Literal(_) => Ok(vec![]),
        Expr::ScalarVariable(_) => Ok(vec![]),
        Expr::Not(expr) => Ok(vec![expr]),
        Expr::Sort { expr, .. } => Ok(vec![expr]),
        Expr::Wildcard { .. } => Err(ExecutionError::General(
            "Wildcard expressions are not valid in a logical query plan".to_owned(),
        )),
        Expr::Nested(expr) => Ok(vec![expr]),
    }
}

/// returns a new expression where the expressions in expr are replaced by the ones in `expr`.
/// This is used in conjunction with ``expr_expressions`` to re-write expressions.
pub fn rewrite_expression(expr: &Expr, expressions: &Vec<Expr>) -> Result<Expr> {
    match expr {
        Expr::BinaryExpr { op, .. } => Ok(Expr::BinaryExpr {
            left: Box::new(expressions[0].clone()),
            op: op.clone(),
            right: Box::new(expressions[1].clone()),
        }),
        Expr::IsNull(_) => Ok(Expr::IsNull(Box::new(expressions[0].clone()))),
        Expr::IsNotNull(_) => Ok(Expr::IsNotNull(Box::new(expressions[0].clone()))),
        Expr::ScalarFunction { fun, .. } => Ok(Expr::ScalarFunction {
            fun: fun.clone(),
            args: expressions.clone(),
        }),
        Expr::ScalarUDF { fun, .. } => Ok(Expr::ScalarUDF {
            fun: fun.clone(),
            args: expressions.clone(),
        }),
        Expr::AggregateFunction { fun, .. } => Ok(Expr::AggregateFunction {
            fun: fun.clone(),
            args: expressions.clone(),
        }),
        Expr::AggregateUDF { fun, .. } => Ok(Expr::AggregateUDF {
            fun: fun.clone(),
            args: expressions.clone(),
        }),
        Expr::Cast { data_type, .. } => Ok(Expr::Cast {
            expr: Box::new(expressions[0].clone()),
            data_type: data_type.clone(),
        }),
        Expr::Alias(_, alias) => {
            Ok(Expr::Alias(Box::new(expressions[0].clone()), alias.clone()))
        }
        Expr::Not(_) => Ok(Expr::Not(Box::new(expressions[0].clone()))),
        Expr::Column(_) => Ok(expr.clone()),
        Expr::Literal(_) => Ok(expr.clone()),
        Expr::ScalarVariable(_) => Ok(expr.clone()),
        Expr::Sort {
            asc, nulls_first, ..
        } => Ok(Expr::Sort {
            expr: Box::new(expressions[0].clone()),
            asc: asc.clone(),
            nulls_first: nulls_first.clone(),
        }),
        Expr::Wildcard { .. } => Err(ExecutionError::General(
            "Wildcard expressions are not valid in a logical query plan".to_owned(),
        )),
        Expr::Nested(_) => Ok(Expr::Nested(Box::new(expressions[0].clone()))),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::logical_plan::{col, LogicalPlanBuilder};
    use arrow::datatypes::DataType;
    use std::collections::HashSet;

    #[test]
    fn test_collect_expr() -> Result<()> {
        let mut accum: HashSet<String> = HashSet::new();
        expr_to_column_names(
            &Expr::Cast {
                expr: Box::new(col("a")),
                data_type: DataType::Float64,
            },
            &mut accum,
        )?;
        expr_to_column_names(
            &Expr::Cast {
                expr: Box::new(col("a")),
                data_type: DataType::Float64,
            },
            &mut accum,
        )?;
        assert_eq!(1, accum.len());
        assert!(accum.contains("a"));
        Ok(())
    }

    struct TestOptimizer {}

    impl OptimizerRule for TestOptimizer {
        fn optimize(&mut self, plan: &LogicalPlan) -> Result<LogicalPlan> {
            Ok(plan.clone())
        }

        fn name(&self) -> &str {
            return "test_optimizer";
        }
    }

    #[test]
    fn test_optimize_explain() -> Result<()> {
        let mut optimizer = TestOptimizer {};

        let empty_plan = LogicalPlanBuilder::empty().build()?;
        let schema = LogicalPlan::explain_schema();

        let optimized_explain = optimize_explain(
            &mut optimizer,
            true,
            &empty_plan,
            &vec![StringifiedPlan::new(PlanType::LogicalPlan, "...")],
            &*schema,
        )?;

        match &optimized_explain {
            LogicalPlan::Explain {
                verbose,
                stringified_plans,
                ..
            } => {
                assert_eq!(*verbose, true);

                let expected_stringified_plans = vec![
                    StringifiedPlan::new(PlanType::LogicalPlan, "..."),
                    StringifiedPlan::new(
                        PlanType::OptimizedLogicalPlan {
                            optimizer_name: "test_optimizer".into(),
                        },
                        "EmptyRelation",
                    ),
                ];
                assert_eq!(*stringified_plans, expected_stringified_plans);
            }
            _ => assert!(
                false,
                "Expected explain plan but got {:?}",
                optimized_explain
            ),
        }

        Ok(())
    }
}
