use super::{calc::dot, node::FloatElement};
use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, Copy, Eq, PartialEq, Serialize, Deserialize, Default)]
pub enum Metric {
    #[default]
    Unknown,
    Manhattan,
    DotProduct,
    Euclidean,
    CosineSimilarity,
    Angular,
}

// TODO: make these func private
pub fn metric<T>(vec1: &[T], vec2: &[T], mt: Metric) -> Result<T, &'static str>
where
    T: FloatElement,
{
    match mt {
        Metric::Euclidean => euclidean_distance(vec1, vec2),
        Metric::Manhattan => manhattan_distance(vec1, vec2),
        Metric::DotProduct => dot_product(vec1, vec2),
        Metric::CosineSimilarity => cosine_similarity(vec1, vec2),
        Metric::Angular => angular_distance(vec1, vec2),
        Metric::Unknown => Result::Err("unknown method"),
    }
}

#[allow(dead_code)]
pub fn range_metric<T>(
    vec1: &[T],
    vec2: &[T],
    mt: Metric,
    begin: usize,
    end: usize,
) -> Result<T, &'static str>
where
    T: FloatElement,
{
    metric(&vec1[begin..end], &vec2[begin..end], mt)
}

fn dot_product<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    assert_eq!(vec1.len(), vec2.len());
    // smaller means closer.
    dot(vec1, vec2).map(|x| -x)
}

fn manhattan_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    T::manhattan_distance(vec1, vec2)
}

fn euclidean_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    T::euclidean_distance(vec1, vec2)
}

fn cosine_similarity<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    assert_eq!(vec1.len(), vec2.len());
    // smaller means closer.
    Ok(-dot(vec1, vec2).unwrap()
        / (dot(vec1, vec1).unwrap().sqrt() * dot(vec2, vec2).unwrap().sqrt()))
}

// (a/|a| - b/|b|)^2
// = a^2 / a^2 + b^2 / b^2 - 2ab/|a||b|
// = 2 - 2cos
fn angular_distance<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    assert_eq!(vec1.len(), vec2.len());
    let rhd = dot(vec1, vec1).unwrap();
    let lhd = dot(vec2, vec2).unwrap();
    let rldot = dot(vec1, vec2).unwrap();
    let rlmul = rhd * lhd;
    let two = T::float_two();
    if rlmul > T::float_zero() {
        Ok(two - two * rldot / rlmul.sqrt())
    } else {
        Ok(two)
    }
}

pub fn real_cosine_similarity<T>(vec1: &[T], vec2: &[T]) -> Result<T, &'static str>
where
    T: FloatElement,
{
    let a = T::real_dot_product(vec1, vec2).unwrap();
    let b = T::real_dot_product(vec1, vec1).unwrap();
    let c = T::real_dot_product(vec2, vec2).unwrap();

    Ok(a / (b.sqrt() * c.sqrt()))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cosine() {
        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![0.0, 1.0, 0.0];
        let d = real_cosine_similarity(&v1, &v2).unwrap();
        assert_eq!(d, 0.0);

        let v1 = vec![1.0, 0.0, 0.0];
        let v2 = vec![1.0, 0.0, 0.0];
        let d = real_cosine_similarity(&v1, &v2).unwrap();
        assert_eq!(d, 1.0);
    }
}
