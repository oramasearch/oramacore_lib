use std::{cmp::Reverse, fmt::Debug};

use crate::{data_structures::capped_heap::CappedHeap, hnsw2::core::simd_metrics::SIMDOptmized};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

#[derive(Debug, Serialize, Deserialize)]
pub struct VectorBruteForce<DocumentId: Debug + Clone + Serialize> {
    dim: usize,
    data: Vec<(DocumentId, Box<[f32]>, f32)>, // (id, vector, magnitude)
}

fn real_cosine_similarity(
    (vec1, magnitude_vec1): (&[f32], f32),
    (vec2, magnitude_vec2): (&[f32], f32),
) -> Result<f32, &'static str> {
    let a = f32::real_dot_product(vec1, vec2).unwrap();

    Ok(a / (magnitude_vec1.sqrt() * magnitude_vec2.sqrt()))
}

impl<DocumentId: Debug + Clone + Copy + Serialize + Ord + Send + Sync>
    VectorBruteForce<DocumentId>
{
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            data: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn get_data(&self) -> impl Iterator<Item = (DocumentId, &[f32])> + '_ {
        self.data
            .iter()
            .map(|(id, vec_box, _)| (*id, vec_box.as_ref()))
    }

    pub fn into_data(self) -> impl Iterator<Item = (DocumentId, Vec<f32>)> {
        self.data
            .into_iter()
            .map(|(id, vec_box, _)| (id, vec_box.into_vec()))
    }

    pub fn set_capacity(&mut self, capacity: usize) {
        if self.data.len() >= capacity {
            return;
        }
        self.data.reserve_exact(capacity - self.data.len());
    }

    pub fn add_owned(&mut self, point: Vec<f32>, id: DocumentId) {
        let magnitude = f32::real_dot_product(&point, &point).unwrap();
        self.data.push((id, point.into_boxed_slice(), magnitude));
    }

    pub fn search(&self, target: &[f32], limit: usize) -> Vec<(DocumentId, f32)> {
        let target_magnitude = f32::real_dot_product(target, target).unwrap();

        let half_data_len = self.data.len() / 2;
        let (first_half, second_half) = self.data.split_at(half_data_len);

        let (capped_head_one, capped_head_two) = rayon::join(
            || {
                let mut capped_head_one = CappedHeap::new(limit);

                for (id, vec, magnitude) in first_half {
                    // The cosine similarity isnt a distance in the math sense
                    // https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
                    // Anyway, it is good for ranking purposes
                    // 1 means the vectors are equal
                    // 0 means the vectors are orthogonal
                    let score =
                        real_cosine_similarity((vec, *magnitude), (target, target_magnitude))
                            .expect("real_cosine_similarity should not return an error");

                    capped_head_one.insert(*id, OrderedFloat(score));
                }
                capped_head_one
            },
            || {
                let mut capped_head_two = CappedHeap::new(limit);

                for (id, vec, magnitude) in second_half {
                    // The cosine similarity isnt a distance in the math sense
                    // https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
                    // Anyway, it is good for ranking purposes
                    // 1 means the vectors are equal
                    // 0 means the vectors are orthogonal
                    let score =
                        real_cosine_similarity((vec, *magnitude), (target, target_magnitude))
                            .expect("real_cosine_similarity should not return an error");

                    capped_head_two.insert(*id, OrderedFloat(score));
                }
                capped_head_two
            },
        );

        let mut output: Vec<_> = capped_head_one
            .into_top()
            .map(|(id, OrderedFloat(score))| (id, score))
            .chain(
                capped_head_two
                    .into_top()
                    .map(|(id, OrderedFloat(score))| (id, score)),
            )
            .collect();

        output.sort_by_key(|(_, score)| Reverse(OrderedFloat(*score)));
        output
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::distr::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_basic3_index() {
        let dim = 3;

        let mut index = VectorBruteForce::new(dim);

        let points = [
            vec![255.0, 0.0, 0.0],
            vec![0.0, 255.0, 0.0],
            vec![0.0, 0.0, 255.0],
        ];

        for (id, point) in points.iter().enumerate() {
            let id = id;
            index.add_owned(point.clone(), id);
        }

        let target = vec![255.0, 0.0, 0.0];
        let v = index.search(&target, 10);

        let res: HashMap<_, _> = v.into_iter().collect();

        assert_eq!(res, HashMap::from([(0, 1.0), (1, 0.0), (2, 0.0),]))
    }

    #[test]
    fn test_basic3_serialize_deserialize() {
        let n = 10_000;
        let dimension = 64;

        let normal = Uniform::new(0.0, 10.0).unwrap();
        let samples = (0..n)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect::<Vec<f32>>()
            })
            .collect::<Vec<Vec<f32>>>();
        let mut index = VectorBruteForce::new(dimension);
        for (i, sample) in samples.into_iter().enumerate() {
            index.add_owned(sample.clone(), i);
        }

        let decoded = bincode::serialize(&index).unwrap();
        let new_index: VectorBruteForce<usize> = bincode::deserialize(&decoded).unwrap();

        let target = (0..dimension)
            .map(|_| normal.sample(&mut rand::rng()))
            .collect::<Vec<f32>>();

        let v1 = index.search(&target, 10);
        let v2 = new_index.search(&target, 10);

        assert_eq!(v1, v2);
    }
}
