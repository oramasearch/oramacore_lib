use super::calc::same_dimension;
use pulp::{Arch, Simd, WithSimd};

pub trait SIMDOptmized<T = Self> {
    fn real_dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
}

impl SIMDOptmized for f64 {
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        let a = Self::dot_product(a, b)?;
        Ok(-a)
    }

    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProduct(a, b)))
    }

    fn manhattan_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(ManhattanDistance(a, b)))
    }

    fn euclidean_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(EuclideanDistance(a, b)))
    }
}

impl SIMDOptmized for f32 {
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        let a = Self::dot_product(a, b)?;
        Ok(-a)
    }

    #[inline(always)]
    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProduct(a, b)))
    }

    #[inline(always)]
    fn manhattan_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(ManhattanDistance(a, b)))
    }

    #[inline(always)]
    fn euclidean_distance(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(EuclideanDistance(a, b)))
    }
}

struct EuclideanDistance<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for EuclideanDistance<'a, f64> {
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

impl<'a> WithSimd for EuclideanDistance<'a, f32> {
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

struct ManhattanDistance<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for ManhattanDistance<'a, f64> {
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

impl<'a> WithSimd for ManhattanDistance<'a, f32> {
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

struct DotProduct<'a, T>(&'a [T], &'a [T]);
impl<'a> WithSimd for DotProduct<'a, f64> {
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

impl<'a> WithSimd for DotProduct<'a, f32> {
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

#[cfg(test)]
mod simd_metrics_tests {
    use super::*;

    #[test]
    fn test_euclidean_distance_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let dist = f32::euclidean_distance(&a, &b).unwrap();
        assert_eq!(dist, 40.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f32>();

        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_euclidean_distance_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f64, 4.0, 3.0, 2.0, 1.0];
        let dist = f64::euclidean_distance(&a, &b).unwrap();
        assert_eq!(dist, 40.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>();

        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_manhattan_distance_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dist = f32::manhattan_distance(&a, &b).unwrap();
        assert_eq!(dist, 9.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>();
        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_manhattan_distance_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dist = f64::manhattan_distance(&a, &b).unwrap();
        assert_eq!(dist, 9.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f64>();
        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dot = f32::dot_product(&a, &b).unwrap();
        // assert_eq!(dot, 32.0);

        let previous_output = -(a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>());
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dot = f64::dot_product(&a, &b).unwrap();
        // assert_eq!(dot, 32.0);

        let previous_output = -(a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>());
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_real_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dot = f32::real_dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>();
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_real_dot_product_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dot = f64::real_dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);

        let previous_output = a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>();
        assert_eq!(dot, previous_output);
    }
}
