use super::calc::same_dimension;
use pulp::{Arch, Simd, WithSimd};

pub trait SIMDOptmized<T = Self> {
    fn real_dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn dot_product(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn manhattan_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
    fn euclidean_distance(a: &[T], b: &[T]) -> Result<T, &'static str>;
}

impl SIMDOptmized for f64 {
    #[inline(always)]
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProduct(a, b)))
    }

    #[inline(always)]
    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(-arch.dispatch(DotProduct(a, b)))
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

impl SIMDOptmized for f32 {
    #[inline(always)]
    fn real_dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(arch.dispatch(DotProduct(a, b)))
    }

    #[inline(always)]
    fn dot_product(a: &[Self], b: &[Self]) -> Result<Self, &'static str> {
        same_dimension(a, b)?;
        let arch = Arch::new();
        Ok(-arch.dispatch(DotProduct(a, b)))
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

        let mut sum1 = simd.splat_f64s(0.0);
        let mut sum2 = simd.splat_f64s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            let diff1 = simd.sub_f64s(a_head[i], b_head[i]);
            sum1 = simd.mul_add_f64s(diff1, diff1, sum1);
            let diff2 = simd.sub_f64s(a_head[i + 1], b_head[i + 1]);
            sum2 = simd.mul_add_f64s(diff2, diff2, sum2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            let diff = simd.sub_f64s(a_head[i], b_head[i]);
            sum1 = simd.mul_add_f64s(diff, diff, sum1);
        }

        let sum = simd.add_f64s(sum1, sum2);
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

        let mut sum1 = simd.splat_f32s(0.0);
        let mut sum2 = simd.splat_f32s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            let diff1 = simd.sub_f32s(a_head[i], b_head[i]);
            sum1 = simd.mul_add_f32s(diff1, diff1, sum1);
            let diff2 = simd.sub_f32s(a_head[i + 1], b_head[i + 1]);
            sum2 = simd.mul_add_f32s(diff2, diff2, sum2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            let diff = simd.sub_f32s(a_head[i], b_head[i]);
            sum1 = simd.mul_add_f32s(diff, diff, sum1);
        }

        let sum = simd.add_f32s(sum1, sum2);
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

        let mut sum1 = simd.splat_f64s(0.0);
        let mut sum2 = simd.splat_f64s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            let diff1 = simd.sub_f64s(a_head[i], b_head[i]);
            let abs_diff1 = simd.abs_f64s(diff1);
            sum1 = simd.add_f64s(sum1, abs_diff1);
            let diff2 = simd.sub_f64s(a_head[i + 1], b_head[i + 1]);
            let abs_diff2 = simd.abs_f64s(diff2);
            sum2 = simd.add_f64s(sum2, abs_diff2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            let diff = simd.sub_f64s(a_head[i], b_head[i]);
            let abs_diff = simd.abs_f64s(diff);
            sum1 = simd.add_f64s(sum1, abs_diff);
        }

        let sum = simd.add_f64s(sum1, sum2);
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

        let mut sum1 = simd.splat_f32s(0.0);
        let mut sum2 = simd.splat_f32s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            let diff1 = simd.sub_f32s(a_head[i], b_head[i]);
            let abs_diff1 = simd.abs_f32s(diff1);
            sum1 = simd.add_f32s(sum1, abs_diff1);
            let diff2 = simd.sub_f32s(a_head[i + 1], b_head[i + 1]);
            let abs_diff2 = simd.abs_f32s(diff2);
            sum2 = simd.add_f32s(sum2, abs_diff2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            let diff = simd.sub_f32s(a_head[i], b_head[i]);
            let abs_diff = simd.abs_f32s(diff);
            sum1 = simd.add_f32s(sum1, abs_diff);
        }

        let sum = simd.add_f32s(sum1, sum2);
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

        let mut sum1 = simd.splat_f64s(0.0);
        let mut sum2 = simd.splat_f64s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            sum1 = simd.mul_add_f64s(a_head[i], b_head[i], sum1);
            sum2 = simd.mul_add_f64s(a_head[i + 1], b_head[i + 1], sum2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            sum1 = simd.mul_add_f64s(a_head[i], b_head[i], sum1);
        }

        let sum = simd.add_f64s(sum1, sum2);
        let mut scalar_sum = simd.reduce_sum_f64s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += a * b;
        }
        scalar_sum
    }
}

impl<'a> WithSimd for DotProduct<'a, f32> {
    type Output = f32;

    #[inline(always)]
    fn with_simd<S: Simd>(self, simd: S) -> Self::Output {
        let (a_head, a_tail) = S::as_simd_f32s(self.0);
        let (b_head, b_tail) = S::as_simd_f32s(self.1);

        let mut sum1 = simd.splat_f32s(0.0);
        let mut sum2 = simd.splat_f32s(0.0);

        let mut i = 0;
        while i + 1 < a_head.len() {
            sum1 = simd.mul_add_f32s(a_head[i], b_head[i], sum1);
            sum2 = simd.mul_add_f32s(a_head[i + 1], b_head[i + 1], sum2);
            i += 2;
        }

        // Handle remainder if odd number of SIMD vectors
        if i < a_head.len() {
            sum1 = simd.mul_add_f32s(a_head[i], b_head[i], sum1);
        }

        let sum = simd.add_f32s(sum1, sum2);
        let mut scalar_sum = simd.reduce_sum_f32s(sum);
        for (&a, &b) in a_tail.iter().zip(b_tail) {
            scalar_sum += a * b;
        }
        scalar_sum
    }
}

#[cfg(test)]
mod simd_metrics_tests {
    use super::*;

    // Scalar implementations for comparison
    fn scalar_euclidean_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f32>()
    }

    fn scalar_manhattan_distance_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f32>()
    }

    fn scalar_real_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>()
    }

    fn scalar_dot_product_f32(a: &[f32], b: &[f32]) -> f32 {
        -(a.iter().zip(b).map(|(p, q)| p * q).sum::<f32>())
    }

    fn scalar_euclidean_distance_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(p, q)| (p - q).powi(2)).sum::<f64>()
    }

    fn scalar_manhattan_distance_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(p, q)| (p - q).abs()).sum::<f64>()
    }

    fn scalar_real_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
        a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>()
    }

    fn scalar_dot_product_f64(a: &[f64], b: &[f64]) -> f64 {
        -(a.iter().zip(b).map(|(p, q)| p * q).sum::<f64>())
    }

    #[test]
    fn test_euclidean_distance_f32() {
        let a = vec![1.0f32, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f32, 4.0, 3.0, 2.0, 1.0];
        let dist = f32::euclidean_distance(&a, &b).unwrap();
        assert_eq!(dist, 40.0);

        let previous_output = scalar_euclidean_distance_f32(&a, &b);

        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_euclidean_distance_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0, 5.0];
        let b = vec![5.0f64, 4.0, 3.0, 2.0, 1.0];
        let dist = f64::euclidean_distance(&a, &b).unwrap();
        assert_eq!(dist, 40.0);

        let previous_output = scalar_euclidean_distance_f64(&a, &b);

        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_manhattan_distance_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dist = f32::manhattan_distance(&a, &b).unwrap();
        assert_eq!(dist, 9.0);

        let previous_output = scalar_manhattan_distance_f32(&a, &b);
        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_manhattan_distance_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dist = f64::manhattan_distance(&a, &b).unwrap();
        assert_eq!(dist, 9.0);

        let previous_output = scalar_manhattan_distance_f64(&a, &b);
        assert_eq!(dist, previous_output);
    }

    #[test]
    fn test_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dot = f32::dot_product(&a, &b).unwrap();
        // assert_eq!(dot, 32.0);

        let previous_output = scalar_dot_product_f32(&a, &b);
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_dot_product_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dot = f64::dot_product(&a, &b).unwrap();
        // assert_eq!(dot, 32.0);

        let previous_output = scalar_dot_product_f64(&a, &b);
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_real_dot_product_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0, 6.0];
        let dot = f32::real_dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);

        let previous_output = scalar_real_dot_product_f32(&a, &b);
        assert_eq!(dot, previous_output);
    }

    #[test]
    fn test_real_dot_product_f64() {
        let a = vec![1.0f64, 2.0, 3.0];
        let b = vec![4.0f64, 5.0, 6.0];
        let dot = f64::real_dot_product(&a, &b).unwrap();
        assert_eq!(dot, 32.0);

        let previous_output = scalar_real_dot_product_f64(&a, &b);
        assert_eq!(dot, previous_output);
    }

    // Error Handling Tests
    #[test]
    fn test_mismatched_dimensions_f32() {
        let a = vec![1.0f32, 2.0, 3.0];
        let b = vec![4.0f32, 5.0];

        assert!(f32::euclidean_distance(&a, &b).is_err());
        assert!(f32::manhattan_distance(&a, &b).is_err());
        assert!(f32::dot_product(&a, &b).is_err());
        assert!(f32::real_dot_product(&a, &b).is_err());
    }

    #[test]
    fn test_mismatched_dimensions_f64() {
        let a = vec![1.0f64, 2.0, 3.0, 4.0];
        let b = vec![5.0f64, 6.0];

        assert!(f64::euclidean_distance(&a, &b).is_err());
        assert!(f64::manhattan_distance(&a, &b).is_err());
        assert!(f64::dot_product(&a, &b).is_err());
        assert!(f64::real_dot_product(&a, &b).is_err());
    }

    #[test]
    fn test_empty_vectors_f32() {
        let a: Vec<f32> = vec![];
        let b: Vec<f32> = vec![];

        // Empty vectors should succeed (sum of nothing is 0)
        assert_eq!(f32::euclidean_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::manhattan_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::dot_product(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::real_dot_product(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_empty_vectors_f64() {
        let a: Vec<f64> = vec![];
        let b: Vec<f64> = vec![];

        // Empty vectors should succeed (sum of nothing is 0)
        assert_eq!(f64::euclidean_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::manhattan_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::dot_product(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::real_dot_product(&a, &b).unwrap(), 0.0);
    }

    // Edge Case Tests
    #[test]
    fn test_single_element_f32() {
        let a = vec![3.0f32];
        let b = vec![4.0f32];

        assert_eq!(
            f32::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f32(&a, &b)
        );
        assert_eq!(
            f32::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f32(&a, &b)
        );
    }

    #[test]
    fn test_single_element_f64() {
        let a = vec![3.0f64];
        let b = vec![4.0f64];

        assert_eq!(
            f64::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f64(&a, &b)
        );
        assert_eq!(
            f64::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f64(&a, &b)
        );
    }

    #[test]
    fn test_two_elements_f32() {
        let a = vec![1.0f32, 2.0];
        let b = vec![3.0f32, 4.0];

        assert_eq!(
            f32::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f32(&a, &b)
        );
        assert_eq!(
            f32::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f32(&a, &b)
        );
    }

    #[test]
    fn test_two_elements_f64() {
        let a = vec![1.0f64, 2.0];
        let b = vec![3.0f64, 4.0];

        assert_eq!(
            f64::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f64(&a, &b)
        );
        assert_eq!(
            f64::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f64(&a, &b)
        );
    }

    #[test]
    fn test_zero_vectors_f32() {
        let a = vec![0.0f32, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f32, 0.0, 0.0, 0.0, 0.0];

        assert_eq!(f32::euclidean_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::manhattan_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::dot_product(&a, &b).unwrap(), 0.0);
        assert_eq!(f32::real_dot_product(&a, &b).unwrap(), 0.0);
    }

    #[test]
    fn test_zero_vectors_f64() {
        let a = vec![0.0f64, 0.0, 0.0, 0.0, 0.0];
        let b = vec![0.0f64, 0.0, 0.0, 0.0, 0.0];

        assert_eq!(f64::euclidean_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::manhattan_distance(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::dot_product(&a, &b).unwrap(), 0.0);
        assert_eq!(f64::real_dot_product(&a, &b).unwrap(), 0.0);
    }

    // SIMD Path Coverage Tests - Various Vector Sizes
    fn test_vector_size_f32(size: usize) {
        let a: Vec<f32> = (0..size).map(|i| i as f32 * 0.5).collect();
        let b: Vec<f32> = (0..size).map(|i| (size - i) as f32 * 0.3).collect();

        let euclidean = f32::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f32(&a, &b);
        // Use relative tolerance for larger values
        let tolerance = if expected_euclidean.abs() > 1.0 {
            expected_euclidean.abs() * 1e-4
        } else {
            1e-4
        };
        assert!(
            (euclidean - expected_euclidean).abs() < tolerance,
            "Size {}: euclidean={}, expected={}",
            size,
            euclidean,
            expected_euclidean
        );

        let manhattan = f32::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f32(&a, &b);
        let tolerance = if expected_manhattan.abs() > 1.0 {
            expected_manhattan.abs() * 1e-4
        } else {
            1e-4
        };
        assert!(
            (manhattan - expected_manhattan).abs() < tolerance,
            "Size {}: manhattan={}, expected={}",
            size,
            manhattan,
            expected_manhattan
        );

        let dot = f32::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f32(&a, &b);
        let tolerance = if expected_dot.abs() > 1.0 {
            expected_dot.abs() * 1e-4
        } else {
            1e-4
        };
        assert!(
            (dot - expected_dot).abs() < tolerance,
            "Size {}: dot={}, expected={}",
            size,
            dot,
            expected_dot
        );

        let real_dot = f32::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f32(&a, &b);
        let tolerance = if expected_real_dot.abs() > 1.0 {
            expected_real_dot.abs() * 1e-4
        } else {
            1e-4
        };
        assert!(
            (real_dot - expected_real_dot).abs() < tolerance,
            "Size {}: real_dot={}, expected={}",
            size,
            real_dot,
            expected_real_dot
        );
    }

    fn test_vector_size_f64(size: usize) {
        let a: Vec<f64> = (0..size).map(|i| i as f64 * 0.5).collect();
        let b: Vec<f64> = (0..size).map(|i| (size - i) as f64 * 0.3).collect();

        let euclidean = f64::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f64(&a, &b);
        assert!(
            (euclidean - expected_euclidean).abs() < 1e-10,
            "Size {}: euclidean={}, expected={}",
            size,
            euclidean,
            expected_euclidean
        );

        let manhattan = f64::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f64(&a, &b);
        assert!(
            (manhattan - expected_manhattan).abs() < 1e-10,
            "Size {}: manhattan={}, expected={}",
            size,
            manhattan,
            expected_manhattan
        );

        let dot = f64::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f64(&a, &b);
        assert!(
            (dot - expected_dot).abs() < 1e-10,
            "Size {}: dot={}, expected={}",
            size,
            dot,
            expected_dot
        );

        let real_dot = f64::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f64(&a, &b);
        assert!(
            (real_dot - expected_real_dot).abs() < 1e-10,
            "Size {}: real_dot={}, expected={}",
            size,
            real_dot,
            expected_real_dot
        );
    }

    #[test]
    fn test_size_4_f32() {
        test_vector_size_f32(4);
    }

    #[test]
    fn test_size_4_f64() {
        test_vector_size_f64(4);
    }

    #[test]
    fn test_size_7_f32() {
        test_vector_size_f32(7);
    }

    #[test]
    fn test_size_7_f64() {
        test_vector_size_f64(7);
    }

    #[test]
    fn test_size_8_f32() {
        test_vector_size_f32(8);
    }

    #[test]
    fn test_size_8_f64() {
        test_vector_size_f64(8);
    }

    #[test]
    fn test_size_15_f32() {
        test_vector_size_f32(15);
    }

    #[test]
    fn test_size_15_f64() {
        test_vector_size_f64(15);
    }

    #[test]
    fn test_size_16_f32() {
        test_vector_size_f32(16);
    }

    #[test]
    fn test_size_16_f64() {
        test_vector_size_f64(16);
    }

    #[test]
    fn test_size_31_f32() {
        test_vector_size_f32(31);
    }

    #[test]
    fn test_size_31_f64() {
        test_vector_size_f64(31);
    }

    #[test]
    fn test_size_32_f32() {
        test_vector_size_f32(32);
    }

    #[test]
    fn test_size_32_f64() {
        test_vector_size_f64(32);
    }

    #[test]
    fn test_size_100_f32() {
        test_vector_size_f32(100);
    }

    #[test]
    fn test_size_100_f64() {
        test_vector_size_f64(100);
    }

    #[test]
    fn test_size_1000_f32() {
        test_vector_size_f32(1000);
    }

    #[test]
    fn test_size_1000_f64() {
        test_vector_size_f64(1000);
    }

    // Numerical Edge Case Tests
    #[test]
    fn test_all_negative_f32() {
        let a = vec![-1.0f32, -2.0, -3.0, -4.0, -5.0];
        let b = vec![-5.0f32, -4.0, -3.0, -2.0, -1.0];

        assert_eq!(
            f32::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f32(&a, &b)
        );
        assert_eq!(
            f32::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f32(&a, &b)
        );
        assert_eq!(
            f32::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f32(&a, &b)
        );
    }

    #[test]
    fn test_all_negative_f64() {
        let a = vec![-1.0f64, -2.0, -3.0, -4.0, -5.0];
        let b = vec![-5.0f64, -4.0, -3.0, -2.0, -1.0];

        assert_eq!(
            f64::euclidean_distance(&a, &b).unwrap(),
            scalar_euclidean_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::manhattan_distance(&a, &b).unwrap(),
            scalar_manhattan_distance_f64(&a, &b)
        );
        assert_eq!(
            f64::dot_product(&a, &b).unwrap(),
            scalar_dot_product_f64(&a, &b)
        );
        assert_eq!(
            f64::real_dot_product(&a, &b).unwrap(),
            scalar_real_dot_product_f64(&a, &b)
        );
    }

    #[test]
    fn test_mixed_signs_f32() {
        let a = vec![1.0f32, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let b = vec![-1.0f32, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

        let euclidean = f32::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f32(&a, &b);
        assert!((euclidean - expected_euclidean).abs() < 1e-4);

        let manhattan = f32::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f32(&a, &b);
        assert!((manhattan - expected_manhattan).abs() < 1e-4);

        let dot = f32::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f32(&a, &b);
        assert!((dot - expected_dot).abs() < 1e-4);

        let real_dot = f32::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f32(&a, &b);
        assert!((real_dot - expected_real_dot).abs() < 1e-4);
    }

    #[test]
    fn test_mixed_signs_f64() {
        let a = vec![1.0f64, -2.0, 3.0, -4.0, 5.0, -6.0, 7.0, -8.0];
        let b = vec![-1.0f64, 2.0, -3.0, 4.0, -5.0, 6.0, -7.0, 8.0];

        let euclidean = f64::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f64(&a, &b);
        assert!((euclidean - expected_euclidean).abs() < 1e-10);

        let manhattan = f64::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f64(&a, &b);
        assert!((manhattan - expected_manhattan).abs() < 1e-10);

        let dot = f64::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f64(&a, &b);
        assert!((dot - expected_dot).abs() < 1e-10);

        let real_dot = f64::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f64(&a, &b);
        assert!((real_dot - expected_real_dot).abs() < 1e-10);
    }

    #[test]
    fn test_large_values_f32() {
        let a = vec![1e6f32, 2e6, 3e6, 4e6, 5e6];
        let b = vec![5e6f32, 4e6, 3e6, 2e6, 1e6];

        let euclidean = f32::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f32(&a, &b);
        assert!((euclidean - expected_euclidean).abs() / expected_euclidean < 1e-4);

        let manhattan = f32::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f32(&a, &b);
        assert!((manhattan - expected_manhattan).abs() / expected_manhattan < 1e-4);

        let dot = f32::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f32(&a, &b);
        assert!((dot - expected_dot).abs() / expected_dot.abs() < 1e-4);

        let real_dot = f32::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f32(&a, &b);
        assert!((real_dot - expected_real_dot).abs() / expected_real_dot < 1e-4);
    }

    #[test]
    fn test_large_values_f64() {
        let a = vec![1e10f64, 2e10, 3e10, 4e10, 5e10];
        let b = vec![5e10f64, 4e10, 3e10, 2e10, 1e10];

        let euclidean = f64::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f64(&a, &b);
        assert!((euclidean - expected_euclidean).abs() / expected_euclidean < 1e-10);

        let manhattan = f64::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f64(&a, &b);
        assert!((manhattan - expected_manhattan).abs() / expected_manhattan < 1e-10);

        let dot = f64::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f64(&a, &b);
        assert!((dot - expected_dot).abs() / expected_dot.abs() < 1e-10);

        let real_dot = f64::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f64(&a, &b);
        assert!((real_dot - expected_real_dot).abs() / expected_real_dot < 1e-10);
    }

    #[test]
    fn test_small_values_f32() {
        let a = vec![1e-6f32, 2e-6, 3e-6, 4e-6, 5e-6];
        let b = vec![5e-6f32, 4e-6, 3e-6, 2e-6, 1e-6];

        let euclidean = f32::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f32(&a, &b);
        assert!((euclidean - expected_euclidean).abs() < 1e-10);

        let manhattan = f32::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f32(&a, &b);
        assert!((manhattan - expected_manhattan).abs() < 1e-10);

        let dot = f32::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f32(&a, &b);
        assert!((dot - expected_dot).abs() < 1e-15);

        let real_dot = f32::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f32(&a, &b);
        assert!((real_dot - expected_real_dot).abs() < 1e-15);
    }

    #[test]
    fn test_small_values_f64() {
        let a = vec![1e-10f64, 2e-10, 3e-10, 4e-10, 5e-10];
        let b = vec![5e-10f64, 4e-10, 3e-10, 2e-10, 1e-10];

        let euclidean = f64::euclidean_distance(&a, &b).unwrap();
        let expected_euclidean = scalar_euclidean_distance_f64(&a, &b);
        assert!((euclidean - expected_euclidean).abs() < 1e-20);

        let manhattan = f64::manhattan_distance(&a, &b).unwrap();
        let expected_manhattan = scalar_manhattan_distance_f64(&a, &b);
        assert!((manhattan - expected_manhattan).abs() < 1e-20);

        let dot = f64::dot_product(&a, &b).unwrap();
        let expected_dot = scalar_dot_product_f64(&a, &b);
        assert!((dot - expected_dot).abs() < 1e-30);

        let real_dot = f64::real_dot_product(&a, &b).unwrap();
        let expected_real_dot = scalar_real_dot_product_f64(&a, &b);
        assert!((real_dot - expected_real_dot).abs() < 1e-30);
    }
}
