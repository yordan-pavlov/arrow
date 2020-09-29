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

#[macro_use]
extern crate criterion;
use criterion::Criterion;

use rand::distributions::{Alphanumeric, Distribution, Standard};
use rand::prelude::random;
use rand::Rng;

use std::sync::Arc;

extern crate arrow;

use arrow::array::*;
use arrow::compute::{cast, take};
use arrow::datatypes::*;

// cast array from specified primitive array type to desired data type
fn create_numeric<T>(size: usize) -> ArrayRef
where
    T: ArrowNumericType,
    Standard: Distribution<T::Native>,
    PrimitiveArray<T>: std::convert::From<Vec<T::Native>>,
{
    Arc::new(PrimitiveArray::<T>::from(vec![random::<T::Native>(); size])) as ArrayRef
}

fn create_strings(size: usize) -> ArrayRef {
    let v = (0..size)
        .map(|_| {
            rand::thread_rng()
                .sample_iter(&Alphanumeric)
                .take(5)
                .collect::<String>()
        })
        .collect::<Vec<_>>();

    Arc::new(StringArray::from(
        v.iter().map(|x| &**x).collect::<Vec<&str>>(),
    ))
}

fn create_random_index(size: usize) -> UInt32Array {
    let mut rng = rand::thread_rng();
    let ints = Int32Array::from(vec![rng.gen_range(-24i32, size as i32); size]);
    // cast to u32, conveniently marking negative values as nulls
    UInt32Array::from(
        cast(&(Arc::new(ints) as ArrayRef), &DataType::UInt32)
            .unwrap()
            .data(),
    )
}

fn bench_take(values: &ArrayRef, indices: &UInt32Array) {
    criterion::black_box(take(&values, &indices, None).unwrap());
}

fn add_benchmark(c: &mut Criterion) {
    let values = create_numeric::<Int32Type>(512);
    let indices = create_random_index(512);
    c.bench_function("take i32 512", |b| b.iter(|| bench_take(&values, &indices)));
    let values = create_numeric::<Int32Type>(1024);
    let indices = create_random_index(1024);
    c.bench_function("take i32 1024", |b| {
        b.iter(|| bench_take(&values, &indices))
    });

    let values = Arc::new(BooleanArray::from(vec![random::<bool>(); 512])) as ArrayRef;
    let indices = create_random_index(512);
    c.bench_function("take bool 512", |b| {
        b.iter(|| bench_take(&values, &indices))
    });

    let values = Arc::new(BooleanArray::from(vec![random::<bool>(); 1024])) as ArrayRef;
    let indices = create_random_index(1024);
    c.bench_function("take bool 1024", |b| {
        b.iter(|| bench_take(&values, &indices))
    });

    let values = create_strings(512);
    let indices = create_random_index(512);
    c.bench_function("take str 512", |b| b.iter(|| bench_take(&values, &indices)));

    let values = create_strings(1024);
    let indices = create_random_index(1024);
    c.bench_function("take str 1024", |b| {
        b.iter(|| bench_take(&values, &indices))
    });
}

criterion_group!(benches, add_benchmark);
criterion_main!(benches);
