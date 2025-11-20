use std::fmt::Debug;

use crate::{data_structures::capped_heap::CappedHeap, hnsw2::core::simd_metrics::SIMDOptmized};
use ordered_float::OrderedFloat;
use serde::{Deserialize, Serialize};

pub trait DocId: Debug + Clone + Serialize + Ord {
    fn from_f32(v: [f32; 2]) -> Self;
    fn to_f32(self) -> [f32; 2];
}

impl DocId for u64 {
    fn from_f32(v: [f32; 2]) -> Self {
        unsafe { std::mem::transmute(v) }
    }

    fn to_f32(self) -> [f32; 2] {
        unsafe { std::mem::transmute(self) }
    }
}

const BLOCK_SIZE: usize = 1024;

#[derive(Debug, Serialize, Deserialize)]
struct Block<DocumentId: DocId> {
    data: Box<[f32]>,
    offset: usize,
    length: usize,
    phantom: std::marker::PhantomData<DocumentId>,
}

impl<DocumentId: DocId> Block<DocumentId> {
    fn into_data(self) -> impl Iterator<Item = (DocumentId, Vec<f32>)> {
        vec![].into_iter() // Placeholder implementation
    }

    fn add_item(&mut self, point: &[f32], id: DocumentId, magnitude: f32) {
        let start = self.length * self.offset;

        let to_f32 = id.to_f32();
        self.data[start] = to_f32[0];
        self.data[start + 1] = to_f32[1];
        self.data[start + 2] = magnitude;

        self.data[start + 3..start + 3 + point.len()].copy_from_slice(&point);

        self.length += 1;
    }

    fn search_in_block(
        &self,
        target: &[f32],
        target_magnitude: f32,
        capped_head: &mut CappedHeap<DocumentId, OrderedFloat<f32>>,
    ) {
        for i in 0..self.length {
            let start = i * self.offset;

            let id_f32 = [self.data[start], self.data[start + 1]];
            let id = DocumentId::from_f32(id_f32);
            let magnitude = self.data[start + 2];
            let vec = &self.data[start + 3..start + 3 + target.len()];

            let score = real_cosine_similarity((vec, magnitude), (target, target_magnitude))
                .expect("real_cosine_similarity should not return an error");

            capped_head.insert(id, OrderedFloat(score));
        }
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct BasicIndex2<DocumentId: DocId> {
    dim: usize,
    blocks: Vec<Block<DocumentId>>, // (id, vector, magnitude)
}

fn real_cosine_similarity(
    (vec1, magnitude_vec1): (&[f32], f32),
    (vec2, magnitude_vec2): (&[f32], f32),
) -> Result<f32, &'static str> {
    let a = f32::real_dot_product(vec1, vec2).unwrap();

    Ok(a / (magnitude_vec1.sqrt() * magnitude_vec2.sqrt()))
}

impl<DocumentId: DocId> BasicIndex2<DocumentId> {
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            blocks: Vec::new(),
        }
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn len(&self) -> usize {
        self.blocks.iter().map(|b| b.length).sum()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn into_data(self) -> impl Iterator<Item = (DocumentId, Vec<f32>)> {
        self.blocks.into_iter().flat_map(|block| block.into_data())
    }

    pub fn add_owned(&mut self, point: &[f32], id: DocumentId) {
        let magnitude = f32::real_dot_product(&point, &point).unwrap();

        let last_block = self.blocks.last_mut();
        let last_block = if let Some(block) = last_block {
            if block.length < BLOCK_SIZE - 1 {
                block
            } else {
                self.blocks.push(Block {
                    data: vec![0.0; BLOCK_SIZE * ((3 + self.dim) * 4)].into_boxed_slice(),
                    offset: 2 + self.dim,
                    length: 0,
                    phantom: std::marker::PhantomData,
                });
                self.blocks.last_mut().unwrap()
            }
        } else {
            self.blocks.push(Block {
                data: vec![0.0; BLOCK_SIZE * ((3 + self.dim) * 4)].into_boxed_slice(),
                offset: 2 + self.dim,
                length: 0,
                phantom: std::marker::PhantomData,
            });
            self.blocks.last_mut().unwrap()
        };

        last_block.add_item(point, id, magnitude);
    }

    pub fn search(&self, target: &[f32], limit: usize) -> Vec<(DocumentId, f32)> {
        let mut capped_head = CappedHeap::new(limit);

        let target_magnitude = f32::real_dot_product(target, target).unwrap();

        for block in &self.blocks {
            block.search_in_block(target, target_magnitude, &mut capped_head);
        }

        capped_head
            .into_top()
            .map(|(id, OrderedFloat(score))| (id, score))
            .collect()
    }
}

#[cfg(test)]
mod tests {
    use std::collections::HashMap;

    use rand::distr::{Distribution, Uniform};

    use super::*;

    #[test]
    fn test_basic2_index() {
        let dim = 3;

        let mut index = BasicIndex2::new(dim);

        let points = [
            vec![255.0, 0.0, 0.0],
            vec![0.0, 255.0, 0.0],
            vec![0.0, 0.0, 255.0],
        ];

        for (id, point) in points.iter().enumerate() {
            let id = id;
            index.add_owned(point, id as u64);
        }

        let target = vec![255.0, 0.0, 0.0];
        let v = index.search(&target, 10);

        let res: HashMap<_, _> = v.into_iter().collect();

        assert_eq!(res, HashMap::from([(0, 1.0), (1, 0.0), (2, 0.0),]))
    }

    #[test]
    fn test_basic2_serialize_deserialize() {
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
        let mut index = BasicIndex2::new(dimension);
        for (i, sample) in samples.into_iter().enumerate() {
            index.add_owned(&sample, i as u64);
        }

        let decoded = bincode::serialize(&index).unwrap();
        let new_index: BasicIndex2<u64> = bincode::deserialize(&decoded).unwrap();

        let target = (0..dimension)
            .map(|_| normal.sample(&mut rand::rng()))
            .collect::<Vec<f32>>();

        let v1 = index.search(&target, 10);
        let v2 = new_index.search(&target, 10);

        assert_eq!(v1, v2);
    }
}
