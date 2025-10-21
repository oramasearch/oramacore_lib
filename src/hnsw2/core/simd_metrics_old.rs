// Old unoptimized SIMD implementation for benchmarking comparison
use super::calc::same_dimension;
use pulp::{Arch, Simd, WithSimd};

pub trait SIMDOptmizedOld<T = Self> {
    fn real_dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
}

impl SIMDOptmizedOld for f64 {
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        let a = Self::dot_product(a, b)?;
        Ok(-a)
    }

    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProductOld(a, b)))
    }

    fn manhattan_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(ManhattanDistanceOld(a, b)))
    }

    fn euclidean_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(EuclideanDistanceOld(a, b)))
    }
}

impl SIMDOptmizedOld for f32 {
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        let a = Self::dot_product(a, b)?;
        Ok(-a)
    }

    #[inline(always)]
    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProductOld(a, b)))
    }

    #[inline(always)]
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(ManhattanDistanceOld(a, b)))
    }

    #[inline(always)]
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(EuclideanDistanceOld(a, b)))
    }
}

struct EuclideanDistanceOld<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for EuclideanDistanceOld<'a, f64> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f64s(self.0);
        let (b_head, b_tail) = S::as_simd_f64s(self.1);

        let mut sum = simd.splat_f64s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let diff = simd.sub_f64s(a, b);
            sum = simd.mul_add_f64s(diff, diff, sum);
        }

        let mut scalar_sum = simd.reduce_sum_f64s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += (a - b).powi(2);
        }
        scalar_sum
    }
}

impl<'a> WithSimd for EuclideanDistanceOld<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        let mut sum = simd.splat_f32s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let diff = simd.sub_f32s(a, b);
            sum = simd.mul_add_f32s(diff, diff, sum);
        }

        let mut scalar_sum = simd.reduce_sum_f32s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += (a - b).powi(2);
        }
        scalar_sum
    }
}

struct ManhattanDistanceOld<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for ManhattanDistanceOld<'a, f64> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f64s(self.0);
        let (b_head, b_tail) = S::as_simd_f64s(self.1);

        let mut sum = simd.splat_f64s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let diff = simd.sub_f64s(a, b);
            let abs_diff = simd.abs_f64s(diff);
            sum = simd.add_f64s(sum, abs_diff);
        }

        let mut scalar_sum = simd.reduce_sum_f64s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += (a - b).abs();
        }
        scalar_sum
    }
}

impl<'a> WithSimd for ManhattanDistanceOld<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        let mut sum = simd.splat_f32s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let diff = simd.sub_f32s(a, b);
            let abs_diff = simd.abs_f32s(diff);
            sum = simd.add_f32s(sum, abs_diff);
        }

        let mut scalar_sum = simd.reduce_sum_f32s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += (a - b).abs();
        }
        scalar_sum
    }
}

struct DotProductOld<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for DotProductOld<'a, f64> {
    type Output = f64;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f64s(self.0);
        let (b_head, b_tail) = S::as_simd_f64s(self.1);

        let mut sum = simd.splat_f64s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let product = simd.mul_f64s(a, b);
            sum = simd.add_f64s(sum, product);
        }

        let mut scalar_sum = simd.reduce_sum_f64s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += a * b;
        }
        -scalar_sum
    }
}

impl<'a> WithSimd for DotProductOld<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        let mut sum = simd.splat_f32s(0.0);
        for (&a, &b) in a_head.iter().zip(b_head) {
            let product = simd.mul_f32s(a, b);
            sum = simd.add_f32s(sum, product);
        }

        let mut scalar_sum = simd.reduce_sum_f32s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += a * b;
        }
        -scalar_sum
    }
}
