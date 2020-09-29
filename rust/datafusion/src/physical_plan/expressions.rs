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

//! Defines physical expressions that can evaluated at runtime during query execution

use std::fmt;
use std::rc::Rc;
use std::sync::Arc;
use std::{cell::RefCell, convert::TryFrom};

use crate::error::{ExecutionError, Result};
use crate::logical_plan::Operator;
use crate::physical_plan::{Accumulator, AggregateExpr, PhysicalExpr};
use crate::scalar::ScalarValue;
use arrow::array::{
    Float32Builder, Float64Builder, Int16Builder, Int32Builder, Int64Builder,
    Int8Builder, LargeStringArray, StringBuilder, UInt16Builder, UInt32Builder,
    UInt64Builder, UInt8Builder,
};
use arrow::compute;
use arrow::compute::kernels;
use arrow::compute::kernels::arithmetic::{add, divide, multiply, subtract};
use arrow::compute::kernels::boolean::{and, or};
use arrow::compute::kernels::comparison::{eq, gt, gt_eq, lt, lt_eq, neq};
use arrow::compute::kernels::comparison::{
    eq_utf8, gt_eq_utf8, gt_utf8, like_utf8, lt_eq_utf8, lt_utf8, neq_utf8, nlike_utf8,
};
use arrow::compute::kernels::sort::{SortColumn, SortOptions};
use arrow::datatypes::{DataType, Schema, TimeUnit};
use arrow::record_batch::RecordBatch;
use arrow::{
    array::{
        ArrayRef, BooleanArray, Float32Array, Float64Array, Int16Array, Int32Array,
        Int64Array, Int8Array, StringArray, TimestampNanosecondArray, UInt16Array,
        UInt32Array, UInt64Array, UInt8Array,
    },
    datatypes::Field,
};

/// returns the name of the state
pub fn format_state_name(name: &str, state_name: &str) -> String {
    format!("{}[{}]", name, state_name)
}

/// Represents the column at a given index in a RecordBatch
#[derive(Debug)]
pub struct Column {
    name: String,
}

impl Column {
    /// Create a new column expression
    pub fn new(name: &str) -> Self {
        Self {
            name: name.to_owned(),
        }
    }
}

impl fmt::Display for Column {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.name)
    }
}

impl PhysicalExpr for Column {
    /// Get the data type of this expression, given the schema of the input
    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        Ok(input_schema
            .field_with_name(&self.name)?
            .data_type()
            .clone())
    }

    /// Decide whehter this expression is nullable, given the schema of the input
    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(input_schema.field_with_name(&self.name)?.is_nullable())
    }

    /// Evaluate the expression
    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        Ok(batch.column(batch.schema().index_of(&self.name)?).clone())
    }
}

/// Create a column expression
pub fn col(name: &str) -> Arc<dyn PhysicalExpr> {
    Arc::new(Column::new(name))
}

/// SUM aggregate expression
#[derive(Debug)]
pub struct Sum {
    name: String,
    data_type: DataType,
    expr: Arc<dyn PhysicalExpr>,
    nullable: bool,
}

/// function return type of a sum
pub fn sum_return_type(arg_type: &DataType) -> Result<DataType> {
    match arg_type {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => {
            Ok(DataType::Int64)
        }
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => {
            Ok(DataType::UInt64)
        }
        DataType::Float32 => Ok(DataType::Float32),
        DataType::Float64 => Ok(DataType::Float64),
        other => Err(ExecutionError::General(format!(
            "SUM does not support type \"{:?}\"",
            other
        ))),
    }
}

impl Sum {
    /// Create a new SUM aggregate function
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: String, data_type: DataType) -> Self {
        Self {
            name,
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Sum {
    fn field(&self) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.data_type.clone(),
            self.nullable,
        ))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "sum"),
            self.data_type.clone(),
            self.nullable,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Rc<RefCell<dyn Accumulator>>> {
        Ok(Rc::new(RefCell::new(SumAccumulator::try_new(
            &self.data_type,
        )?)))
    }
}

#[derive(Debug)]
struct SumAccumulator {
    sum: ScalarValue,
}

impl SumAccumulator {
    /// new sum accumulator
    pub fn try_new(data_type: &DataType) -> Result<Self> {
        Ok(Self {
            sum: ScalarValue::try_from(data_type)?,
        })
    }
}

// returns the new value after sum with the new values, taking nullability into account
macro_rules! typed_sum_delta_batch {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let delta = compute::sum(array);
        ScalarValue::$SCALAR(delta)
    }};
}

// sums the array and returns a ScalarValue of its corresponding type.
fn sum_batch(values: &ArrayRef) -> Result<ScalarValue> {
    Ok(match values.data_type() {
        DataType::Float64 => typed_sum_delta_batch!(values, Float64Array, Float64),
        DataType::Float32 => typed_sum_delta_batch!(values, Float32Array, Float32),
        DataType::Int64 => typed_sum_delta_batch!(values, Int64Array, Int64),
        DataType::Int32 => typed_sum_delta_batch!(values, Int32Array, Int32),
        DataType::Int16 => typed_sum_delta_batch!(values, Int16Array, Int16),
        DataType::Int8 => typed_sum_delta_batch!(values, Int8Array, Int8),
        DataType::UInt64 => typed_sum_delta_batch!(values, UInt64Array, UInt64),
        DataType::UInt32 => typed_sum_delta_batch!(values, UInt32Array, UInt32),
        DataType::UInt16 => typed_sum_delta_batch!(values, UInt16Array, UInt16),
        DataType::UInt8 => typed_sum_delta_batch!(values, UInt8Array, UInt8),
        e => {
            return Err(ExecutionError::InternalError(format!(
                "Sum is not expected to receive the type {:?}",
                e
            )))
        }
    })
}

// returns the sum of two scalar values, including coercion into $TYPE.
macro_rules! typed_sum {
    ($OLD_VALUE:expr, $DELTA:expr, $SCALAR:ident, $TYPE:ident) => {{
        ScalarValue::$SCALAR(match ($OLD_VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone() as $TYPE),
            (Some(a), Some(b)) => Some(a + (*b as $TYPE)),
        })
    }};
}

fn sum(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    Ok(match (lhs, rhs) {
        // float64 coerces everything to f64
        (ScalarValue::Float64(lhs), ScalarValue::Float64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Float32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int16(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::Int8(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt64(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt32(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt16(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        (ScalarValue::Float64(lhs), ScalarValue::UInt8(rhs)) => {
            typed_sum!(lhs, rhs, Float64, f64)
        }
        // float32 has no cast
        (ScalarValue::Float32(lhs), ScalarValue::Float32(rhs)) => {
            typed_sum!(lhs, rhs, Float32, f32)
        }
        // u64 coerces u* to u64
        (ScalarValue::UInt64(lhs), ScalarValue::UInt64(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt32(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt16(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        (ScalarValue::UInt64(lhs), ScalarValue::UInt8(rhs)) => {
            typed_sum!(lhs, rhs, UInt64, u64)
        }
        // i64 coerces i* to u64
        (ScalarValue::Int64(lhs), ScalarValue::Int64(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int32(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int16(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        (ScalarValue::Int64(lhs), ScalarValue::Int8(rhs)) => {
            typed_sum!(lhs, rhs, Int64, i64)
        }
        e => {
            return Err(ExecutionError::InternalError(format!(
                "Sum is not expected to receive a scalar {:?}",
                e
            )))
        }
    })
}

impl Accumulator for SumAccumulator {
    fn update_batch(&mut self, values: &Vec<ArrayRef>) -> Result<()> {
        let values = &values[0];
        self.sum = sum(&self.sum, &sum_batch(values)?)?;
        Ok(())
    }

    fn update(&mut self, values: &Vec<ScalarValue>) -> Result<()> {
        // sum(v1, v2, v3) = v1 + v2 + v3
        self.sum = sum(&self.sum, &values[0])?;
        Ok(())
    }

    fn merge(&mut self, states: &Vec<ScalarValue>) -> Result<()> {
        // sum(sum1, sum2) = sum1 + sum2
        self.update(states)
    }

    fn merge_batch(&mut self, states: &Vec<ArrayRef>) -> Result<()> {
        // sum(sum1, sum2, sum3, ...) = sum1 + sum2 + sum3 + ...
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.sum.clone()])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.sum.clone())
    }
}

/// AVG aggregate expression
#[derive(Debug)]
pub struct Avg {
    name: String,
    data_type: DataType,
    nullable: bool,
    expr: Arc<dyn PhysicalExpr>,
}

/// function return type of an average
pub fn avg_return_type(arg_type: &DataType) -> Result<DataType> {
    match arg_type {
        DataType::Int8
        | DataType::Int16
        | DataType::Int32
        | DataType::Int64
        | DataType::UInt8
        | DataType::UInt16
        | DataType::UInt32
        | DataType::UInt64
        | DataType::Float32
        | DataType::Float64 => Ok(DataType::Float64),
        other => Err(ExecutionError::General(format!(
            "AVG does not support {:?}",
            other
        ))),
    }
}

impl Avg {
    /// Create a new AVG aggregate function
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: String, data_type: DataType) -> Self {
        Self {
            name,
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Avg {
    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, DataType::Float64, true))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![
            Field::new(
                &format_state_name(&self.name, "count"),
                DataType::UInt64,
                true,
            ),
            Field::new(
                &format_state_name(&self.name, "sum"),
                DataType::Float64,
                true,
            ),
        ])
    }

    fn create_accumulator(&self) -> Result<Rc<RefCell<dyn Accumulator>>> {
        Ok(Rc::new(RefCell::new(AvgAccumulator::try_new(
            // avg is f64
            &DataType::Float64,
        )?)))
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }
}

/// An accumulator to compute the average
#[derive(Debug)]
pub(crate) struct AvgAccumulator {
    // sum is used for null
    sum: ScalarValue,
    count: u64,
}

impl AvgAccumulator {
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            sum: ScalarValue::try_from(datatype)?,
            count: 0,
        })
    }
}

impl Accumulator for AvgAccumulator {
    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![ScalarValue::from(self.count), self.sum.clone()])
    }

    fn update(&mut self, values: &Vec<ScalarValue>) -> Result<()> {
        let values = &values[0];

        self.count += (!values.is_null()) as u64;
        self.sum = sum(&self.sum, values)?;

        Ok(())
    }

    fn update_batch(&mut self, values: &Vec<ArrayRef>) -> Result<()> {
        let values = &values[0];

        self.count += (values.len() - values.data().null_count()) as u64;
        self.sum = sum(&self.sum, &sum_batch(values)?)?;
        Ok(())
    }

    fn merge(&mut self, states: &Vec<ScalarValue>) -> Result<()> {
        let count = &states[0];
        // counts are summed
        if let ScalarValue::UInt64(Some(c)) = count {
            self.count += c
        } else {
            unreachable!()
        };

        // sums are summed
        self.sum = sum(&self.sum, &states[1])?;
        Ok(())
    }

    fn merge_batch(&mut self, states: &Vec<ArrayRef>) -> Result<()> {
        let counts = states[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        // counts are summed
        self.count += compute::sum(counts).unwrap_or(0);

        // sums are summed
        self.sum = sum(&self.sum, &sum_batch(&states[1])?)?;
        Ok(())
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        match self.sum {
            ScalarValue::Float64(e) => Ok(ScalarValue::Float64(match e {
                Some(f) => Some(f / self.count as f64),
                None => None,
            })),
            _ => Err(ExecutionError::InternalError(
                "Sum should be f64 on average".to_string(),
            )),
        }
    }
}

/// MAX aggregate expression
#[derive(Debug)]
pub struct Max {
    name: String,
    data_type: DataType,
    nullable: bool,
    expr: Arc<dyn PhysicalExpr>,
}

impl Max {
    /// Create a new MAX aggregate function
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: String, data_type: DataType) -> Self {
        Self {
            name,
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Max {
    fn field(&self) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.data_type.clone(),
            self.nullable,
        ))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "max"),
            self.data_type.clone(),
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Rc<RefCell<dyn Accumulator>>> {
        Ok(Rc::new(RefCell::new(MaxAccumulator::try_new(
            &self.data_type,
        )?)))
    }
}

// Statically-typed version of min/max(array) -> ScalarValue for string types.
macro_rules! typed_min_max_batch_string {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let value = compute::$OP(array);
        let value = value.and_then(|e| Some(e.to_string()));
        ScalarValue::$SCALAR(value)
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue for non-string types.
macro_rules! typed_min_max_batch {
    ($VALUES:expr, $ARRAYTYPE:ident, $SCALAR:ident, $OP:ident) => {{
        let array = $VALUES.as_any().downcast_ref::<$ARRAYTYPE>().unwrap();
        let value = compute::$OP(array);
        ScalarValue::$SCALAR(value)
    }};
}

// Statically-typed version of min/max(array) -> ScalarValue  for non-string types.
// this is a macro to support both operations (min and max).
macro_rules! min_max_batch {
    ($VALUES:expr, $OP:ident) => {{
        match $VALUES.data_type() {
            // all types that have a natural order
            DataType::Float64 => {
                typed_min_max_batch!($VALUES, Float64Array, Float64, $OP)
            }
            DataType::Float32 => {
                typed_min_max_batch!($VALUES, Float32Array, Float32, $OP)
            }
            DataType::Int64 => typed_min_max_batch!($VALUES, Int64Array, Int64, $OP),
            DataType::Int32 => typed_min_max_batch!($VALUES, Int32Array, Int32, $OP),
            DataType::Int16 => typed_min_max_batch!($VALUES, Int16Array, Int16, $OP),
            DataType::Int8 => typed_min_max_batch!($VALUES, Int8Array, Int8, $OP),
            DataType::UInt64 => typed_min_max_batch!($VALUES, UInt64Array, UInt64, $OP),
            DataType::UInt32 => typed_min_max_batch!($VALUES, UInt32Array, UInt32, $OP),
            DataType::UInt16 => typed_min_max_batch!($VALUES, UInt16Array, UInt16, $OP),
            DataType::UInt8 => typed_min_max_batch!($VALUES, UInt8Array, UInt8, $OP),
            other => {
                return Err(ExecutionError::NotImplemented(format!(
                    "Min/Max accumulator not implemented for type {:?}",
                    other
                )))
            }
        }
    }};
}

/// dynamically-typed min(array) -> ScalarValue
fn min_batch(values: &ArrayRef) -> Result<ScalarValue> {
    Ok(match values.data_type() {
        DataType::Utf8 => {
            typed_min_max_batch_string!(values, StringArray, Utf8, min_string)
        }
        DataType::LargeUtf8 => {
            typed_min_max_batch_string!(values, LargeStringArray, LargeUtf8, min_string)
        }
        _ => min_max_batch!(values, min),
    })
}

/// dynamically-typed max(array) -> ScalarValue
fn max_batch(values: &ArrayRef) -> Result<ScalarValue> {
    Ok(match values.data_type() {
        DataType::Utf8 => {
            typed_min_max_batch_string!(values, StringArray, Utf8, max_string)
        }
        DataType::LargeUtf8 => {
            typed_min_max_batch_string!(values, LargeStringArray, LargeUtf8, max_string)
        }
        _ => min_max_batch!(values, max),
    })
}

// min/max of two non-string scalar values.
macro_rules! typed_min_max {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some((*a).$OP(*b)),
        })
    }};
}

// min/max of two scalar string values.
macro_rules! typed_min_max_string {
    ($VALUE:expr, $DELTA:expr, $SCALAR:ident, $OP:ident) => {{
        ScalarValue::$SCALAR(match ($VALUE, $DELTA) {
            (None, None) => None,
            (Some(a), None) => Some(a.clone()),
            (None, Some(b)) => Some(b.clone()),
            (Some(a), Some(b)) => Some((a).$OP(b).clone()),
        })
    }};
}

// min/max of two scalar values of the same type
macro_rules! min_max {
    ($VALUE:expr, $DELTA:expr, $OP:ident) => {{
        Ok(match ($VALUE, $DELTA) {
            (ScalarValue::Float64(lhs), ScalarValue::Float64(rhs)) => {
                typed_min_max!(lhs, rhs, Float64, $OP)
            }
            (ScalarValue::Float32(lhs), ScalarValue::Float32(rhs)) => {
                typed_min_max!(lhs, rhs, Float32, $OP)
            }
            (ScalarValue::UInt64(lhs), ScalarValue::UInt64(rhs)) => {
                typed_min_max!(lhs, rhs, UInt64, $OP)
            }
            (ScalarValue::UInt32(lhs), ScalarValue::UInt32(rhs)) => {
                typed_min_max!(lhs, rhs, UInt32, $OP)
            }
            (ScalarValue::UInt16(lhs), ScalarValue::UInt16(rhs)) => {
                typed_min_max!(lhs, rhs, UInt16, $OP)
            }
            (ScalarValue::UInt8(lhs), ScalarValue::UInt8(rhs)) => {
                typed_min_max!(lhs, rhs, UInt8, $OP)
            }
            (ScalarValue::Int64(lhs), ScalarValue::Int64(rhs)) => {
                typed_min_max!(lhs, rhs, Int64, $OP)
            }
            (ScalarValue::Int32(lhs), ScalarValue::Int32(rhs)) => {
                typed_min_max!(lhs, rhs, Int32, $OP)
            }
            (ScalarValue::Int16(lhs), ScalarValue::Int16(rhs)) => {
                typed_min_max!(lhs, rhs, Int16, $OP)
            }
            (ScalarValue::Int8(lhs), ScalarValue::Int8(rhs)) => {
                typed_min_max!(lhs, rhs, Int8, $OP)
            }
            (ScalarValue::Utf8(lhs), ScalarValue::Utf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, Utf8, $OP)
            }
            (ScalarValue::LargeUtf8(lhs), ScalarValue::LargeUtf8(rhs)) => {
                typed_min_max_string!(lhs, rhs, LargeUtf8, $OP)
            }
            e => {
                return Err(ExecutionError::InternalError(format!(
                    "MIN/MAX is not expected to receive a scalar {:?}",
                    e
                )))
            }
        })
    }};
}

/// the minimum of two scalar values
fn min(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    min_max!(lhs, rhs, min)
}

/// the maximum of two scalar values
fn max(lhs: &ScalarValue, rhs: &ScalarValue) -> Result<ScalarValue> {
    min_max!(lhs, rhs, max)
}

#[derive(Debug)]
struct MaxAccumulator {
    max: ScalarValue,
}

impl MaxAccumulator {
    /// new max accumulator
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            max: ScalarValue::try_from(datatype)?,
        })
    }
}

impl Accumulator for MaxAccumulator {
    fn update_batch(&mut self, values: &Vec<ArrayRef>) -> Result<()> {
        let values = &values[0];
        let delta = &max_batch(values)?;
        self.max = max(&self.max, delta)?;
        Ok(())
    }

    fn update(&mut self, values: &Vec<ScalarValue>) -> Result<()> {
        let value = &values[0];
        self.max = max(&self.max, value)?;
        Ok(())
    }

    fn merge(&mut self, states: &Vec<ScalarValue>) -> Result<()> {
        self.update(states)
    }

    fn merge_batch(&mut self, states: &Vec<ArrayRef>) -> Result<()> {
        self.update_batch(states)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.max.clone()])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.max.clone())
    }
}

/// MIN aggregate expression
#[derive(Debug)]
pub struct Min {
    name: String,
    data_type: DataType,
    nullable: bool,
    expr: Arc<dyn PhysicalExpr>,
}

impl Min {
    /// Create a new MIN aggregate function
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: String, data_type: DataType) -> Self {
        Self {
            name,
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Min {
    fn field(&self) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.data_type.clone(),
            self.nullable,
        ))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "min"),
            self.data_type.clone(),
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Rc<RefCell<dyn Accumulator>>> {
        Ok(Rc::new(RefCell::new(MinAccumulator::try_new(
            &self.data_type,
        )?)))
    }
}

#[derive(Debug)]
struct MinAccumulator {
    min: ScalarValue,
}

impl MinAccumulator {
    /// new min accumulator
    pub fn try_new(datatype: &DataType) -> Result<Self> {
        Ok(Self {
            min: ScalarValue::try_from(datatype)?,
        })
    }
}

impl Accumulator for MinAccumulator {
    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.min.clone()])
    }

    fn update_batch(&mut self, values: &Vec<ArrayRef>) -> Result<()> {
        let values = &values[0];
        let delta = &min_batch(values)?;
        self.min = min(&self.min, delta)?;
        Ok(())
    }

    fn update(&mut self, values: &Vec<ScalarValue>) -> Result<()> {
        let value = &values[0];
        self.min = min(&self.min, value)?;
        Ok(())
    }

    fn merge(&mut self, states: &Vec<ScalarValue>) -> Result<()> {
        self.update(states)
    }

    fn merge_batch(&mut self, states: &Vec<ArrayRef>) -> Result<()> {
        self.update_batch(states)
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.min.clone())
    }
}

/// COUNT aggregate expression
/// Returns the amount of non-null values of the given expression.
#[derive(Debug)]
pub struct Count {
    name: String,
    data_type: DataType,
    nullable: bool,
    expr: Arc<dyn PhysicalExpr>,
}

impl Count {
    /// Create a new COUNT aggregate function.
    pub fn new(expr: Arc<dyn PhysicalExpr>, name: String, data_type: DataType) -> Self {
        Self {
            name,
            expr,
            data_type,
            nullable: true,
        }
    }
}

impl AggregateExpr for Count {
    fn field(&self) -> Result<Field> {
        Ok(Field::new(
            &self.name,
            self.data_type.clone(),
            self.nullable,
        ))
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format_state_name(&self.name, "count"),
            self.data_type.clone(),
            true,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        vec![self.expr.clone()]
    }

    fn create_accumulator(&self) -> Result<Rc<RefCell<dyn Accumulator>>> {
        Ok(Rc::new(RefCell::new(CountAccumulator::new())))
    }
}

#[derive(Debug)]
struct CountAccumulator {
    count: ScalarValue,
}

impl CountAccumulator {
    /// new count accumulator
    pub fn new() -> Self {
        Self {
            count: ScalarValue::from(0u64),
        }
    }

    fn update_from_option(&mut self, delta: &Option<u64>) -> Result<()> {
        self.count = ScalarValue::UInt64(match (&self.count, delta) {
            (ScalarValue::UInt64(None), None) => None,
            (ScalarValue::UInt64(None), Some(rhs)) => Some(rhs.clone()),
            (ScalarValue::UInt64(Some(lhs)), None) => Some(lhs.clone()),
            (ScalarValue::UInt64(Some(lhs)), Some(rhs)) => Some(lhs + rhs),
            _ => {
                return Err(ExecutionError::InternalError(
                    "Code should not be reached reach".to_string(),
                ))
            }
        });
        Ok(())
    }
}

impl Accumulator for CountAccumulator {
    fn update_batch(&mut self, values: &Vec<ArrayRef>) -> Result<()> {
        let array = &values[0];
        let delta = if array.len() == array.data().null_count() {
            None
        } else {
            Some((array.len() - array.data().null_count()) as u64)
        };
        self.update_from_option(&delta)
    }

    fn update(&mut self, values: &Vec<ScalarValue>) -> Result<()> {
        let value = &values[0];
        self.count = match (&self.count, value.is_null()) {
            (ScalarValue::UInt64(None), false) => ScalarValue::from(1u64),
            (ScalarValue::UInt64(Some(count)), false) => ScalarValue::from(count + 1),
            // value is null => no change in count
            (e, true) => e.clone(),
            (_, false) => {
                return Err(ExecutionError::InternalError(
                    "Count is always of type u64".to_string(),
                ))
            }
        };
        Ok(())
    }

    fn merge(&mut self, states: &Vec<ScalarValue>) -> Result<()> {
        let count = &states[0];
        if let ScalarValue::UInt64(delta) = count {
            self.update_from_option(delta)
        } else {
            unreachable!()
        }
    }

    fn merge_batch(&mut self, states: &Vec<ArrayRef>) -> Result<()> {
        let counts = states[0].as_any().downcast_ref::<UInt64Array>().unwrap();
        let delta = &compute::sum(counts);
        self.update_from_option(delta)
    }

    fn state(&self) -> Result<Vec<ScalarValue>> {
        Ok(vec![self.count.clone()])
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(self.count.clone())
    }
}

/// Invoke a compute kernel on a pair of binary data arrays
macro_rules! compute_utf8_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new(paste::expr! {[<$OP _utf8>]}(&ll, &rr)?))
    }};
}

/// Invoke a compute kernel on a pair of arrays
macro_rules! compute_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident, $DT:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<$DT>()
            .expect("compute_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}

macro_rules! binary_string_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            other => Err(ExecutionError::General(format!(
                "Unsupported data type {:?}",
                other
            ))),
        }
    }};
}

/// Invoke a compute kernel on a pair of arrays
/// The binary_primitive_array_op macro only evaluates for primitive types
/// like integers and floats.
macro_rules! binary_primitive_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Int8 => compute_op!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op!($LEFT, $RIGHT, $OP, Float64Array),
            other => Err(ExecutionError::General(format!(
                "Unsupported data type {:?}",
                other
            ))),
        }
    }};
}

/// The binary_array_op macro includes types that extend beyond the primitive,
/// such as Utf8 strings.
macro_rules! binary_array_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        match $LEFT.data_type() {
            DataType::Int8 => compute_op!($LEFT, $RIGHT, $OP, Int8Array),
            DataType::Int16 => compute_op!($LEFT, $RIGHT, $OP, Int16Array),
            DataType::Int32 => compute_op!($LEFT, $RIGHT, $OP, Int32Array),
            DataType::Int64 => compute_op!($LEFT, $RIGHT, $OP, Int64Array),
            DataType::UInt8 => compute_op!($LEFT, $RIGHT, $OP, UInt8Array),
            DataType::UInt16 => compute_op!($LEFT, $RIGHT, $OP, UInt16Array),
            DataType::UInt32 => compute_op!($LEFT, $RIGHT, $OP, UInt32Array),
            DataType::UInt64 => compute_op!($LEFT, $RIGHT, $OP, UInt64Array),
            DataType::Float32 => compute_op!($LEFT, $RIGHT, $OP, Float32Array),
            DataType::Float64 => compute_op!($LEFT, $RIGHT, $OP, Float64Array),
            DataType::Utf8 => compute_utf8_op!($LEFT, $RIGHT, $OP, StringArray),
            DataType::Timestamp(TimeUnit::Nanosecond, None) => {
                compute_op!($LEFT, $RIGHT, $OP, TimestampNanosecondArray)
            }
            other => Err(ExecutionError::General(format!(
                "Unsupported data type {:?}",
                other
            ))),
        }
    }};
}

/// Invoke a boolean kernel on a pair of arrays
macro_rules! boolean_op {
    ($LEFT:expr, $RIGHT:expr, $OP:ident) => {{
        let ll = $LEFT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        let rr = $RIGHT
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        Ok(Arc::new($OP(&ll, &rr)?))
    }};
}
/// Binary expression
#[derive(Debug)]
pub struct BinaryExpr {
    left: Arc<dyn PhysicalExpr>,
    op: Operator,
    right: Arc<dyn PhysicalExpr>,
}

impl BinaryExpr {
    /// Create new binary expression
    pub fn new(
        left: Arc<dyn PhysicalExpr>,
        op: Operator,
        right: Arc<dyn PhysicalExpr>,
    ) -> Self {
        Self { left, op, right }
    }
}

impl fmt::Display for BinaryExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} {} {}", self.left, self.op, self.right)
    }
}

// the type that both lhs and rhs can be casted to for the purpose of a string computation
fn string_coercion(lhs_type: &DataType, rhs_type: &DataType) -> Option<DataType> {
    use arrow::datatypes::DataType::*;
    match (lhs_type, rhs_type) {
        (Utf8, Utf8) => Some(Utf8),
        (LargeUtf8, Utf8) => Some(LargeUtf8),
        (Utf8, LargeUtf8) => Some(LargeUtf8),
        (LargeUtf8, LargeUtf8) => Some(LargeUtf8),
        _ => None,
    }
}

/// coercion rule for numerical types
pub fn numerical_coercion(lhs_type: &DataType, rhs_type: &DataType) -> Option<DataType> {
    use arrow::datatypes::DataType::*;

    // error on any non-numeric type
    if !is_numeric(lhs_type) || !is_numeric(rhs_type) {
        return None;
    };

    // same type => all good
    if lhs_type == rhs_type {
        return Some(lhs_type.clone());
    }

    // these are ordered from most informative to least informative so
    // that the coercion removes the least amount of information
    match (lhs_type, rhs_type) {
        (Float64, _) => Some(Float64),
        (_, Float64) => Some(Float64),

        (_, Float32) => Some(Float32),
        (Float32, _) => Some(Float32),

        (Int64, _) => Some(Int64),
        (_, Int64) => Some(Int64),

        (Int32, _) => Some(Int32),
        (_, Int32) => Some(Int32),

        (Int16, _) => Some(Int16),
        (_, Int16) => Some(Int16),

        (Int8, _) => Some(Int8),
        (_, Int8) => Some(Int8),

        (UInt64, _) => Some(UInt64),
        (_, UInt64) => Some(UInt64),

        (UInt32, _) => Some(UInt32),
        (_, UInt32) => Some(UInt32),

        (UInt16, _) => Some(UInt16),
        (_, UInt16) => Some(UInt16),

        (UInt8, _) => Some(UInt8),
        (_, UInt8) => Some(UInt8),

        _ => None,
    }
}

// coercion rules for equality operations. This is a superset of all numerical coercion rules.
fn eq_coercion(lhs_type: &DataType, rhs_type: &DataType) -> Option<DataType> {
    if lhs_type == rhs_type {
        // same type => equality is possible
        return Some(lhs_type.clone());
    }
    numerical_coercion(lhs_type, rhs_type)
}

// coercion rules that assume an ordered set, such as "less than".
// These are the union of all numerical coercion rules and all string coercion rules
fn order_coercion(lhs_type: &DataType, rhs_type: &DataType) -> Option<DataType> {
    if lhs_type == rhs_type {
        // same type => all good
        return Some(lhs_type.clone());
    }

    match numerical_coercion(lhs_type, rhs_type) {
        None => {
            // strings are naturally ordered, and thus ordering can be applied to them.
            string_coercion(lhs_type, rhs_type)
        }
        t => t,
    }
}

/// coercion rules for all binary operators
fn common_binary_type(
    lhs_type: &DataType,
    op: &Operator,
    rhs_type: &DataType,
) -> Result<DataType> {
    // This result MUST be compatible with `binary_coerce`
    let result = match op {
        Operator::And | Operator::Or => match (lhs_type, rhs_type) {
            // logical binary boolean operators can only be evaluated in bools
            (DataType::Boolean, DataType::Boolean) => Some(DataType::Boolean),
            _ => None,
        },
        // logical equality operators have their own rules, and always return a boolean
        Operator::Eq | Operator::NotEq => eq_coercion(lhs_type, rhs_type),
        // "like" operators operate on strings and always return a boolean
        Operator::Like | Operator::NotLike => string_coercion(lhs_type, rhs_type),
        // order-comparison operators have their own rules
        Operator::Lt | Operator::Gt | Operator::GtEq | Operator::LtEq => {
            order_coercion(lhs_type, rhs_type)
        }
        // for math expressions, the final value of the coercion is also the return type
        // because coercion favours higher information types
        Operator::Plus | Operator::Minus | Operator::Divide | Operator::Multiply => {
            numerical_coercion(lhs_type, rhs_type)
        }
        Operator::Modulus => {
            return Err(ExecutionError::NotImplemented(
                "Modulus operator is still not supported".to_string(),
            ))
        }
    };

    // re-write the error message of failed coercions to include the operator's information
    match result {
        None => Err(ExecutionError::General(
            format!(
                "'{:?} {} {:?}' can't be evaluated because there isn't a common type to coerce the types to",
                lhs_type, op, rhs_type
            )
            .to_string(),
        )),
        Some(t) => Ok(t)
    }
}

/// Returns the return type of a binary operator or an error when the binary operator cannot
/// perform the computation between the argument's types, even after type coercion.
///
/// This function makes some assumptions about the underlying available computations.
pub fn binary_operator_data_type(
    lhs_type: &DataType,
    op: &Operator,
    rhs_type: &DataType,
) -> Result<DataType> {
    // validate that it is possible to perform the operation on incoming types.
    // (or the return datatype cannot be infered)
    let common_type = common_binary_type(lhs_type, op, rhs_type)?;

    match op {
        // operators that return a boolean
        Operator::Eq
        | Operator::NotEq
        | Operator::And
        | Operator::Or
        | Operator::Like
        | Operator::NotLike
        | Operator::Lt
        | Operator::Gt
        | Operator::GtEq
        | Operator::LtEq => Ok(DataType::Boolean),
        // math operations return the same value as the common coerced type
        Operator::Plus | Operator::Minus | Operator::Divide | Operator::Multiply => {
            Ok(common_type)
        }
        Operator::Modulus => Err(ExecutionError::NotImplemented(
            "Modulus operator is still not supported".to_string(),
        )),
    }
}

/// return two physical expressions that are optionally coerced to a
/// common type that the binary operator supports.
fn binary_cast(
    lhs: Arc<dyn PhysicalExpr>,
    op: &Operator,
    rhs: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<(Arc<dyn PhysicalExpr>, Arc<dyn PhysicalExpr>)> {
    let lhs_type = &lhs.data_type(input_schema)?;
    let rhs_type = &rhs.data_type(input_schema)?;

    let cast_type = common_binary_type(lhs_type, op, rhs_type)?;

    Ok((
        cast(lhs, input_schema, cast_type.clone())?,
        cast(rhs, input_schema, cast_type)?,
    ))
}

impl PhysicalExpr for BinaryExpr {
    fn data_type(&self, input_schema: &Schema) -> Result<DataType> {
        binary_operator_data_type(
            &self.left.data_type(input_schema)?,
            &self.op,
            &self.right.data_type(input_schema)?,
        )
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        Ok(self.left.nullable(input_schema)? || self.right.nullable(input_schema)?)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let left = self.left.evaluate(batch)?;
        let right = self.right.evaluate(batch)?;
        if left.data_type() != right.data_type() {
            return Err(ExecutionError::General(format!(
                "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                self.op,
                left.data_type(),
                right.data_type()
            )));
        }
        match &self.op {
            Operator::Like => binary_string_array_op!(left, right, like),
            Operator::NotLike => binary_string_array_op!(left, right, nlike),
            Operator::Lt => binary_array_op!(left, right, lt),
            Operator::LtEq => binary_array_op!(left, right, lt_eq),
            Operator::Gt => binary_array_op!(left, right, gt),
            Operator::GtEq => binary_array_op!(left, right, gt_eq),
            Operator::Eq => binary_array_op!(left, right, eq),
            Operator::NotEq => binary_array_op!(left, right, neq),
            Operator::Plus => binary_primitive_array_op!(left, right, add),
            Operator::Minus => binary_primitive_array_op!(left, right, subtract),
            Operator::Multiply => binary_primitive_array_op!(left, right, multiply),
            Operator::Divide => binary_primitive_array_op!(left, right, divide),
            Operator::And => {
                if left.data_type() == &DataType::Boolean {
                    boolean_op!(left, right, and)
                } else {
                    return Err(ExecutionError::General(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op,
                        left.data_type(),
                        right.data_type()
                    )));
                }
            }
            Operator::Or => {
                if left.data_type() == &DataType::Boolean {
                    boolean_op!(left, right, or)
                } else {
                    return Err(ExecutionError::General(format!(
                        "Cannot evaluate binary expression {:?} with types {:?} and {:?}",
                        self.op,
                        left.data_type(),
                        right.data_type()
                    )));
                }
            }
            Operator::Modulus => Err(ExecutionError::NotImplemented(
                "Modulus operator is still not supported".to_string(),
            )),
        }
    }
}

/// Create a binary expression whose arguments are correctly coerced.
/// This function errors if it is not possible to coerce the arguments
/// to computational types supported by the operator.
pub fn binary(
    lhs: Arc<dyn PhysicalExpr>,
    op: Operator,
    rhs: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let (l, r) = binary_cast(lhs, &op, rhs, input_schema)?;
    Ok(Arc::new(BinaryExpr::new(l, op, r)))
}

/// Not expression
#[derive(Debug)]
pub struct NotExpr {
    arg: Arc<dyn PhysicalExpr>,
}

impl NotExpr {
    /// Create new not expression
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self { arg }
    }
}

impl fmt::Display for NotExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "NOT {}", self.arg)
    }
}
impl PhysicalExpr for NotExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        return Ok(DataType::Boolean);
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.arg.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let arg = self.arg.evaluate(batch)?;
        let arg = arg
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("boolean_op failed to downcast array");
        return Ok(Arc::new(arrow::compute::kernels::boolean::not(arg)?));
    }
}

/// Creates a unary expression NOT
///
/// # Errors
///
/// This function errors when the argument's type is not boolean
pub fn not(
    arg: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
) -> Result<Arc<dyn PhysicalExpr>> {
    let data_type = arg.data_type(input_schema)?;
    if data_type != DataType::Boolean {
        Err(ExecutionError::General(
            format!(
                "NOT '{:?}' can't be evaluated because the expression's type is {:?}, not boolean",
                arg, data_type,
            )
            .to_string(),
        ))
    } else {
        Ok(Arc::new(NotExpr::new(arg)))
    }
}

/// IS NULL expression
#[derive(Debug)]
pub struct IsNullExpr {
    arg: Arc<dyn PhysicalExpr>,
}

impl IsNullExpr {
    /// Create new not expression
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self { arg }
    }
}

impl fmt::Display for IsNullExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} IS NULL", self.arg)
    }
}
impl PhysicalExpr for IsNullExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        return Ok(DataType::Boolean);
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let arg = self.arg.evaluate(batch)?;
        return Ok(Arc::new(arrow::compute::is_null(&arg)?));
    }
}

/// Create an IS NULL expression
pub fn is_null(arg: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(IsNullExpr::new(arg)))
}

/// IS NULL expression
#[derive(Debug)]
pub struct IsNotNullExpr {
    arg: Arc<dyn PhysicalExpr>,
}

impl IsNotNullExpr {
    /// Create new not expression
    pub fn new(arg: Arc<dyn PhysicalExpr>) -> Self {
        Self { arg }
    }
}

impl fmt::Display for IsNotNullExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{} IS NOT NULL", self.arg)
    }
}
impl PhysicalExpr for IsNotNullExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        return Ok(DataType::Boolean);
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(false)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let arg = self.arg.evaluate(batch)?;
        return Ok(Arc::new(arrow::compute::is_not_null(&arg)?));
    }
}

/// Create an IS NOT NULL expression
pub fn is_not_null(arg: Arc<dyn PhysicalExpr>) -> Result<Arc<dyn PhysicalExpr>> {
    Ok(Arc::new(IsNotNullExpr::new(arg)))
}

/// CAST expression casts an expression to a specific data type
#[derive(Debug)]
pub struct CastExpr {
    /// The expression to cast
    expr: Arc<dyn PhysicalExpr>,
    /// The data type to cast to
    cast_type: DataType,
}

/// Determine if a DataType is numeric or not
pub fn is_numeric(dt: &DataType) -> bool {
    match dt {
        DataType::Int8 | DataType::Int16 | DataType::Int32 | DataType::Int64 => true,
        DataType::UInt8 | DataType::UInt16 | DataType::UInt32 | DataType::UInt64 => true,
        DataType::Float16 | DataType::Float32 | DataType::Float64 => true,
        _ => false,
    }
}

impl fmt::Display for CastExpr {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "CAST({} AS {:?})", self.expr, self.cast_type)
    }
}

impl PhysicalExpr for CastExpr {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.cast_type.clone())
    }

    fn nullable(&self, input_schema: &Schema) -> Result<bool> {
        self.expr.nullable(input_schema)
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        let value = self.expr.evaluate(batch)?;
        Ok(kernels::cast::cast(&value, &self.cast_type)?)
    }
}

/// Returns a cast operation, if casting needed.
pub fn cast(
    expr: Arc<dyn PhysicalExpr>,
    input_schema: &Schema,
    cast_type: DataType,
) -> Result<Arc<dyn PhysicalExpr>> {
    let expr_type = expr.data_type(input_schema)?;
    if expr_type == cast_type {
        return Ok(expr.clone());
    }
    if is_numeric(&expr_type) && (is_numeric(&cast_type) || cast_type == DataType::Utf8) {
        Ok(Arc::new(CastExpr { expr, cast_type }))
    } else if expr_type == DataType::Binary && cast_type == DataType::Utf8 {
        Ok(Arc::new(CastExpr { expr, cast_type }))
    } else if is_numeric(&expr_type)
        && cast_type == DataType::Timestamp(TimeUnit::Nanosecond, None)
    {
        Ok(Arc::new(CastExpr { expr, cast_type }))
    } else {
        Err(ExecutionError::General(format!(
            "Invalid CAST from {:?} to {:?}",
            expr_type, cast_type
        )))
    }
}

/// Represents a non-null literal value
#[derive(Debug)]
pub struct Literal {
    value: ScalarValue,
}

impl Literal {
    /// Create a literal value expression
    pub fn new(value: ScalarValue) -> Self {
        Self { value }
    }
}

/// Build array containing the same literal value repeated. This is necessary because the Arrow
/// memory model does not have the concept of a scalar value currently.
macro_rules! build_literal_array {
    ($BATCH:ident, $BUILDER:ident, $VALUE:expr) => {{
        let mut builder = $BUILDER::new($BATCH.num_rows());
        if $VALUE.is_none() {
            for _ in 0..$BATCH.num_rows() {
                builder.append_null()?;
            }
        } else {
            for _ in 0..$BATCH.num_rows() {
                builder.append_value($VALUE.unwrap())?;
            }
        }
        Ok(Arc::new(builder.finish()))
    }};
}

impl fmt::Display for Literal {
    fn fmt(&self, f: &mut fmt::Formatter) -> fmt::Result {
        write!(f, "{}", self.value)
    }
}

impl PhysicalExpr for Literal {
    fn data_type(&self, _input_schema: &Schema) -> Result<DataType> {
        Ok(self.value.get_datatype())
    }

    fn nullable(&self, _input_schema: &Schema) -> Result<bool> {
        Ok(self.value.is_null())
    }

    fn evaluate(&self, batch: &RecordBatch) -> Result<ArrayRef> {
        match &self.value {
            ScalarValue::Int8(value) => build_literal_array!(batch, Int8Builder, *value),
            ScalarValue::Int16(value) => {
                build_literal_array!(batch, Int16Builder, *value)
            }
            ScalarValue::Int32(value) => {
                build_literal_array!(batch, Int32Builder, *value)
            }
            ScalarValue::Int64(value) => {
                build_literal_array!(batch, Int64Builder, *value)
            }
            ScalarValue::UInt8(value) => {
                build_literal_array!(batch, UInt8Builder, *value)
            }
            ScalarValue::UInt16(value) => {
                build_literal_array!(batch, UInt16Builder, *value)
            }
            ScalarValue::UInt32(value) => {
                build_literal_array!(batch, UInt32Builder, *value)
            }
            ScalarValue::UInt64(value) => {
                build_literal_array!(batch, UInt64Builder, *value)
            }
            ScalarValue::Float32(value) => {
                build_literal_array!(batch, Float32Builder, *value)
            }
            ScalarValue::Float64(value) => {
                build_literal_array!(batch, Float64Builder, *value)
            }
            ScalarValue::Utf8(value) => build_literal_array!(
                batch,
                StringBuilder,
                value.as_ref().and_then(|e| Some(&*e))
            ),
            other => Err(ExecutionError::General(format!(
                "Unsupported literal type {:?}",
                other
            ))),
        }
    }
}

/// Create a literal expression
pub fn lit(value: ScalarValue) -> Arc<dyn PhysicalExpr> {
    Arc::new(Literal::new(value))
}

/// Represents Sort operation for a column in a RecordBatch
#[derive(Clone, Debug)]
pub struct PhysicalSortExpr {
    /// Physical expression representing the column to sort
    pub expr: Arc<dyn PhysicalExpr>,
    /// Option to specify how the given column should be sorted
    pub options: SortOptions,
}

impl PhysicalSortExpr {
    /// evaluate the sort expression into SortColumn that can be passed into arrow sort kernel
    pub fn evaluate_to_sort_column(&self, batch: &RecordBatch) -> Result<SortColumn> {
        Ok(SortColumn {
            values: self.expr.evaluate(batch)?,
            options: Some(self.options),
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::error::Result;
    use arrow::array::{
        LargeStringArray, PrimitiveArray, PrimitiveArrayOps, StringArray,
        Time64NanosecondArray,
    };
    use arrow::datatypes::*;

    // Create a binary expression without coercion. Used here when we do not want to coerce the expressions
    // to valid types. Usage can result in an execution (after plan) error.
    fn binary_simple(
        l: Arc<dyn PhysicalExpr>,
        op: Operator,
        r: Arc<dyn PhysicalExpr>,
    ) -> Arc<dyn PhysicalExpr> {
        Arc::new(BinaryExpr::new(l, op, r))
    }

    #[test]
    fn binary_comparison() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )?;

        // expression: "a < b"
        let lt = binary_simple(col("a"), Operator::Lt, col("b"));
        let result = lt.evaluate(&batch)?;
        assert_eq!(result.len(), 5);

        let expected = vec![false, false, true, true, true];
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");
        for i in 0..5 {
            assert_eq!(result.value(i), expected[i]);
        }

        Ok(())
    }

    #[test]
    fn binary_nested() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![2, 4, 6, 8, 10]);
        let b = Int32Array::from(vec![2, 5, 4, 8, 8]);
        let batch = RecordBatch::try_new(
            Arc::new(schema.clone()),
            vec![Arc::new(a), Arc::new(b)],
        )?;

        // expression: "a < b OR a == b"
        let expr = binary_simple(
            binary_simple(col("a"), Operator::Lt, col("b")),
            Operator::Or,
            binary_simple(col("a"), Operator::Eq, col("b")),
        );
        assert_eq!("a < b OR a = b", format!("{}", expr));

        let result = expr.evaluate(&batch)?;
        assert_eq!(result.len(), 5);

        let expected = vec![true, true, false, true, false];
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");
        for i in 0..5 {
            assert_eq!(result.value(i), expected[i]);
        }

        Ok(())
    }

    #[test]
    fn literal_i32() -> Result<()> {
        // create an arbitrary record bacth
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        // create and evaluate a literal expression
        let literal_expr = lit(ScalarValue::from(42i32));
        assert_eq!("42", format!("{}", literal_expr));

        let literal_array = literal_expr.evaluate(&batch)?;
        let literal_array = literal_array.as_any().downcast_ref::<Int32Array>().unwrap();

        // note that the contents of the literal array are unrelated to the batch contents except for the length of the array
        assert_eq!(literal_array.len(), 5); // 5 rows in the batch
        for i in 0..literal_array.len() {
            assert_eq!(literal_array.value(i), 42);
        }

        Ok(())
    }

    // runs an end-to-end test of physical type coercion:
    // 1. construct a record batch with two columns of type A and B
    // 2. construct a physical expression of A OP B
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type C
    macro_rules! test_coercion {
        ($A_ARRAY:ident, $A_TYPE:expr, $A_VEC:expr, $B_ARRAY:ident, $B_TYPE:expr, $B_VEC:expr, $OP:expr, $TYPEARRAY:ident, $TYPE:expr, $VEC:expr) => {{
            let schema = Schema::new(vec![
                Field::new("a", $A_TYPE, false),
                Field::new("b", $B_TYPE, false),
            ]);
            let a = $A_ARRAY::from($A_VEC);
            let b = $B_ARRAY::from($B_VEC);
            let batch = RecordBatch::try_new(
                Arc::new(schema.clone()),
                vec![Arc::new(a), Arc::new(b)],
            )?;

            // verify that we can construct the expression
            let expression = binary(col("a"), $OP, col("b"), &schema)?;

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $TYPE);

            // compute
            let result = expression.evaluate(&batch)?;

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $TYPE);

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$TYPEARRAY>()
                .expect("failed to downcast");
            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                assert_eq!(result.value(i), *x);
            }
        }};
    }

    #[test]
    fn test_type_coersion() -> Result<()> {
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32, 2i32],
            UInt32Array,
            DataType::UInt32,
            vec![1u32, 2u32],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            vec![2i32, 4i32]
        );
        test_coercion!(
            Int32Array,
            DataType::Int32,
            vec![1i32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Int32Array,
            DataType::Int32,
            vec![2i32]
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![1f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Plus,
            Float32Array,
            DataType::Float32,
            vec![2f32]
        );
        test_coercion!(
            Float32Array,
            DataType::Float32,
            vec![2f32],
            UInt16Array,
            DataType::UInt16,
            vec![1u16],
            Operator::Multiply,
            Float32Array,
            DataType::Float32,
            vec![2f32]
        );
        test_coercion!(
            StringArray,
            DataType::Utf8,
            vec!["hello world", "world"],
            StringArray,
            DataType::Utf8,
            vec!["%hello%", "%hello%"],
            Operator::Like,
            BooleanArray,
            DataType::Boolean,
            vec![true, false]
        );
        Ok(())
    }

    #[test]
    fn test_coersion_error() -> Result<()> {
        let expr =
            common_binary_type(&DataType::Float32, &Operator::Plus, &DataType::Utf8);

        if let Err(ExecutionError::General(e)) = expr {
            assert_eq!(e, "'Float32 + Utf8' can't be evaluated because there isn't a common type to coerce the types to");
            Ok(())
        } else {
            Err(ExecutionError::General(
                "Coercion should have returned an ExecutionError::General".to_string(),
            ))
        }
    }

    // runs an end-to-end test of physical type cast
    // 1. construct a record batch with a column "a" of type A
    // 2. construct a physical expression of CAST(a AS B)
    // 3. evaluate the expression
    // 4. verify that the resulting expression is of type B
    // 5. verify that the resulting values are downcastable and correct
    macro_rules! generic_test_cast {
        ($A_ARRAY:ident, $A_TYPE:expr, $A_VEC:expr, $TYPEARRAY:ident, $TYPE:expr, $VEC:expr) => {{
            let schema = Schema::new(vec![Field::new("a", $A_TYPE, false)]);
            let a = $A_ARRAY::from($A_VEC);
            let batch =
                RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

            // verify that we can construct the expression
            let expression = cast(col("a"), &schema, $TYPE)?;

            // verify that its display is correct
            assert_eq!(format!("CAST(a AS {:?})", $TYPE), format!("{}", expression));

            // verify that the expression's type is correct
            assert_eq!(expression.data_type(&schema)?, $TYPE);

            // compute
            let result = expression.evaluate(&batch)?;

            // verify that the array's data_type is correct
            assert_eq!(*result.data_type(), $TYPE);

            // verify that the len is correct
            assert_eq!(result.len(), $A_VEC.len());

            // verify that the data itself is downcastable
            let result = result
                .as_any()
                .downcast_ref::<$TYPEARRAY>()
                .expect("failed to downcast");

            // verify that the result itself is correct
            for (i, x) in $VEC.iter().enumerate() {
                assert_eq!(result.value(i), *x);
            }
        }};
    }

    #[test]
    fn test_cast_i32_u32() -> Result<()> {
        generic_test_cast!(
            Int32Array,
            DataType::Int32,
            vec![1, 2, 3, 4, 5],
            UInt32Array,
            DataType::UInt32,
            vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]
        );
        Ok(())
    }

    #[test]
    fn test_cast_i32_utf8() -> Result<()> {
        generic_test_cast!(
            Int32Array,
            DataType::Int32,
            vec![1, 2, 3, 4, 5],
            StringArray,
            DataType::Utf8,
            vec!["1", "2", "3", "4", "5"]
        );
        Ok(())
    }

    #[test]
    fn test_cast_i64_t64() -> Result<()> {
        let original = vec![1, 2, 3, 4, 5];
        let expected: Vec<i64> = original
            .iter()
            .map(|i| Time64NanosecondArray::from(vec![*i]).value(0))
            .collect();
        generic_test_cast!(
            Int64Array,
            DataType::Int64,
            original.clone(),
            TimestampNanosecondArray,
            DataType::Timestamp(TimeUnit::Nanosecond, None),
            expected
        );
        Ok(())
    }

    #[test]
    fn invalid_cast() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);
        let result = cast(col("a"), &schema, DataType::Int32);
        result.expect_err("Invalid CAST from Utf8 to Int32");
        Ok(())
    }

    #[test]
    fn sum_i32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::Int64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(15i64);

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn avg_i32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_i32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5i32);

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn max_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);

        let a = StringArray::from(vec!["d", "a", "c", "b"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Utf8));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Utf8(Some("d".to_string()));

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_large_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::LargeUtf8, false)]);

        let a = LargeStringArray::from(vec!["d", "a", "c", "b"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::LargeUtf8));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::LargeUtf8(Some("d".to_string()));

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_i32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(1i32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);

        let a = StringArray::from(vec!["d", "a", "c", "b"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Utf8));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Utf8(Some("a".to_string()));

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn min_large_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::LargeUtf8, false)]);

        let a = LargeStringArray::from(vec!["d", "a", "c", "b"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::LargeUtf8));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::LargeUtf8(Some("a".to_string()));

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn sum_i32_with_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::Int64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(13i64);

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn avg_i32_with_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3.25f64);

        assert_eq!(expected, actual);

        Ok(())
    }

    #[test]
    fn max_i32_with_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5i32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_i32_with_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![Some(1), None, Some(3), Some(4), Some(5)]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(1i32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn sum_i32_all_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![None, None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::Int64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Int64(None);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_i32_all_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![None, None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Int32(None);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_i32_all_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![None, None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Int32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Int32(None);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn avg_i32_all_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);

        let a = Int32Array::from(vec![None, None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::Float64(None);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn sum_u32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

        let a = UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(15u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn avg_u32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

        let a = UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3.0f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_u32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

        let a = UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::UInt32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5u32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_u32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::UInt32, false)]);

        let a = UInt32Array::from(vec![1_u32, 2_u32, 3_u32, 4_u32, 5_u32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::UInt32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(1u32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn sum_f32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);

        let a = Float32Array::from(vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::Float32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(15_f32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn avg_f32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);

        let a = Float32Array::from(vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_f32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);

        let a = Float32Array::from(vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Float32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5_f32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_f32() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float32, false)]);

        let a = Float32Array::from(vec![1_f32, 2_f32, 3_f32, 4_f32, 5_f32]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Float32));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(1_f32);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn sum_f64() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float64, false)]);

        let a = Float64Array::from(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Sum::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(15_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn avg_f64() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float64, false)]);

        let a = Float64Array::from(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Avg::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn max_f64() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float64, false)]);

        let a = Float64Array::from(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Max::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn min_f64() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Float64, false)]);

        let a = Float64Array::from(vec![1_f64, 2_f64, 3_f64, 4_f64, 5_f64]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Min::new(col("a"), "bla".to_string(), DataType::Float64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(1_f64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_elements() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_with_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Int32, false)]);
        let a = Int32Array::from(vec![Some(1), Some(2), None, None, Some(3), None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(3u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_all_nulls() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, false)]);
        let a = BooleanArray::from(vec![None, None, None, None, None, None, None, None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(0u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_empty() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, false)]);
        let a = BooleanArray::from(Vec::<bool>::new());
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(0u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, false)]);
        let a = StringArray::from(vec!["a", "bb", "ccc", "dddd", "ad"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    #[test]
    fn count_large_utf8() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::LargeUtf8, false)]);
        let a = LargeStringArray::from(vec!["a", "bb", "ccc", "dddd", "ad"]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        let agg = Arc::new(Count::new(col("a"), "bla".to_string(), DataType::UInt64));
        let actual = aggregate(&batch, agg)?;
        let expected = ScalarValue::from(5u64);

        assert_eq!(expected, actual);
        Ok(())
    }

    fn aggregate(
        batch: &RecordBatch,
        agg: Arc<dyn AggregateExpr>,
    ) -> Result<ScalarValue> {
        let accum = agg.create_accumulator()?;
        let expr = agg.expressions();
        let values = expr
            .iter()
            .map(|e| e.evaluate(batch))
            .collect::<Result<Vec<_>>>()?;
        let mut accum = accum.borrow_mut();
        accum.update_batch(&values)?;
        accum.evaluate()
    }

    #[test]
    fn plus_op() -> Result<()> {
        let schema = Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]);
        let a = Int32Array::from(vec![1, 2, 3, 4, 5]);
        let b = Int32Array::from(vec![1, 2, 4, 8, 16]);

        apply_arithmetic::<Int32Type>(
            Arc::new(schema),
            vec![Arc::new(a), Arc::new(b)],
            Operator::Plus,
            Int32Array::from(vec![2, 4, 7, 12, 21]),
        )?;

        Ok(())
    }

    #[test]
    fn minus_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![1, 2, 4, 8, 16]));
        let b = Arc::new(Int32Array::from(vec![1, 2, 3, 4, 5]));

        apply_arithmetic::<Int32Type>(
            schema.clone(),
            vec![a.clone(), b.clone()],
            Operator::Minus,
            Int32Array::from(vec![0, 0, 1, 4, 11]),
        )?;

        // should handle have negative values in result (for signed)
        apply_arithmetic::<Int32Type>(
            schema.clone(),
            vec![b.clone(), a.clone()],
            Operator::Minus,
            Int32Array::from(vec![0, 0, -1, -4, -11]),
        )?;

        Ok(())
    }

    #[test]
    fn multiply_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![4, 8, 16, 32, 64]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Multiply,
            Int32Array::from(vec![8, 32, 128, 512, 2048]),
        )?;

        Ok(())
    }

    #[test]
    fn divide_op() -> Result<()> {
        let schema = Arc::new(Schema::new(vec![
            Field::new("a", DataType::Int32, false),
            Field::new("b", DataType::Int32, false),
        ]));
        let a = Arc::new(Int32Array::from(vec![8, 32, 128, 512, 2048]));
        let b = Arc::new(Int32Array::from(vec![2, 4, 8, 16, 32]));

        apply_arithmetic::<Int32Type>(
            schema,
            vec![a, b],
            Operator::Divide,
            Int32Array::from(vec![4, 8, 16, 32, 64]),
        )?;

        Ok(())
    }

    fn apply_arithmetic<T: ArrowNumericType>(
        schema: SchemaRef,
        data: Vec<ArrayRef>,
        op: Operator,
        expected: PrimitiveArray<T>,
    ) -> Result<()> {
        let arithmetic_op = binary_simple(col("a"), op, col("b"));
        let batch = RecordBatch::try_new(schema, data)?;
        let result = arithmetic_op.evaluate(&batch)?;

        assert_array_eq::<T>(expected, result);

        Ok(())
    }

    fn assert_array_eq<T: ArrowNumericType>(
        expected: PrimitiveArray<T>,
        actual: ArrayRef,
    ) {
        let actual = actual
            .as_any()
            .downcast_ref::<PrimitiveArray<T>>()
            .expect("Actual array should unwrap to type of expected array");

        for i in 0..expected.len() {
            assert_eq!(expected.value(i), actual.value(i));
        }
    }

    #[test]
    fn neg_op() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Boolean, true)]);

        let expr = not(col("a"), &schema)?;
        assert_eq!(expr.data_type(&schema)?, DataType::Boolean);
        assert_eq!(expr.nullable(&schema)?, true);

        let input = BooleanArray::from(vec![Some(true), None, Some(false)]);
        let expected = &BooleanArray::from(vec![Some(false), None, Some(true)]);

        let batch =
            RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(input)])?;

        let result = expr.evaluate(&batch)?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");
        assert_eq!(result, expected);

        Ok(())
    }

    /// verify that expression errors when the input expression is not a boolean.
    #[test]
    fn neg_op_not_null() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]);

        let expr = not(col("a"), &schema);
        assert!(expr.is_err());

        Ok(())
    }

    #[test]
    fn is_null_op() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]);
        let a = StringArray::from(vec![Some("foo"), None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        // expression: "a is null"
        let expr = is_null(col("a")).unwrap();
        let result = expr.evaluate(&batch)?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");

        let expected = &BooleanArray::from(vec![false, true]);

        assert_eq!(expected, result);

        Ok(())
    }

    #[test]
    fn is_not_null_op() -> Result<()> {
        let schema = Schema::new(vec![Field::new("a", DataType::Utf8, true)]);
        let a = StringArray::from(vec![Some("foo"), None]);
        let batch = RecordBatch::try_new(Arc::new(schema.clone()), vec![Arc::new(a)])?;

        // expression: "a is not null"
        let expr = is_not_null(col("a")).unwrap();
        let result = expr.evaluate(&batch)?;
        let result = result
            .as_any()
            .downcast_ref::<BooleanArray>()
            .expect("failed to downcast to BooleanArray");

        let expected = &BooleanArray::from(vec![true, false]);

        assert_eq!(expected, result);

        Ok(())
    }
}
