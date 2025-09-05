use std::{fmt::Debug, hash::Hash};

use crate::hnsw2::{
    HNSWIndex,
    core::{
        ann_index::ANNIndex,
        metrics::{Metric, real_cosine_similarity},
        node::{IdxType, Node},
    },
    hnsw_params::HNSWParams,
};
use anyhow::Result;
use serde::{Deserialize, Serialize, Serializer, de::DeserializeOwned};

#[derive(Debug, PartialEq, Eq, Hash, Clone)]
struct DocumentIdWrapper<DocumentId>(DocumentId);
impl<DocumentId: Serialize + Hash + Send + Sync + Ord + Debug + Clone> IdxType
    for DocumentIdWrapper<DocumentId>
{
}
impl<DocumentId: Serialize + Hash + Send + Sync> Default for DocumentIdWrapper<DocumentId> {
    fn default() -> Self {
        panic!("DocumentIdWrapper::default() should not be called");
    }
}

impl<DocumentId: Serialize> Serialize for DocumentIdWrapper<DocumentId> {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: Serializer,
    {
        self.0.serialize(serializer)
    }
}

impl<'de, DocumentId: Deserialize<'de>> Deserialize<'de> for DocumentIdWrapper<DocumentId> {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let inner = DocumentId::deserialize(deserializer)?;
        Ok(DocumentIdWrapper(inner))
    }
}

impl<DocumentId: PartialOrd> PartialOrd for DocumentIdWrapper<DocumentId> {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        self.0.partial_cmp(&other.0)
    }
}
impl<DocumentId: Ord + Eq> Ord for DocumentIdWrapper<DocumentId> {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.0.cmp(&other.0)
    }
}

#[derive(Debug, Serialize, Deserialize)]
pub struct HNSW2Index<DocumentId: Hash + Send + Sync + Ord + Debug + Clone + Serialize> {
    inner: HNSWIndex<f32, DocumentIdWrapper<DocumentId>>,
    dim: usize,
}

impl<
    DocumentId: Serialize + DeserializeOwned + Sync + Send + Copy + Eq + Ord + Hash + Debug + Clone,
> HNSW2Index<DocumentId>
{
    pub fn new(dim: usize) -> Self {
        let params = &HNSWParams::<f32>::default();
        Self {
            inner: HNSWIndex::new(dim, params),
            dim,
        }
    }

    pub fn len(&self) -> usize {
        self.inner.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    pub fn dim(&self) -> usize {
        self.dim
    }

    pub fn into_data(self) -> impl Iterator<Item = (DocumentId, Vec<f32>)> {
        self.inner
            .into_data()
            .map(|(v, DocumentIdWrapper(id))| (id, v))
    }

    pub fn add(&mut self, point: &[f32], id: DocumentId) -> Result<()> {
        self.inner
            .add(point, DocumentIdWrapper(id))
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn build(&mut self) -> Result<()> {
        self.inner
            .build(Metric::Euclidean)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn search(&self, target: Vec<f32>, limit: usize) -> Vec<(DocumentId, f32)> {
        assert_eq!(target.len(), self.dim);

        let v = self.inner.node_search_k(&Node::new(&target), limit);

        let mut result = Vec::new();
        for (node, _) in v {
            let n = node.vectors();

            // The cosine similarity isnt a distance in the math sense
            // https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
            // Anyway, it is good for ranking purposes
            // 1 means the vectors are equal
            // 0 means the vectors are orthogonal
            let score = real_cosine_similarity(n, &target)
                .expect("real_cosine_similarity should not return an error");

            let id = match node.idx() {
                Some(DocumentIdWrapper(id)) => id,
                None => continue,
            };
            result.push((*id, score));
        }

        result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::distr::{Distribution, Uniform};
    use std::collections::HashMap;

    #[test]
    fn test_hnsw2() {
        let dim = 3;
        let points = [
            vec![255.0, 0.0, 0.0],
            vec![0.0, 255.0, 0.0],
            vec![0.0, 0.0, 255.0],
        ];

        let mut index = HNSW2Index::new(dim);
        for (id, point) in points.iter().enumerate() {
            let id = id;
            index.add(point, id).unwrap();
        }
        index.build().unwrap();

        let target = vec![255.0, 0.0, 0.0];
        let v = index.search(target, 10);

        let res: HashMap<_, _> = v.into_iter().collect();

        assert_eq!(res, HashMap::from([(0, 1.0), (1, 0.0), (2, 0.0),]))
    }

    #[test]
    fn test_hnsw2_serialize_deserialize() {
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
        let mut index = HNSW2Index::new(dimension);
        for (i, sample) in samples.into_iter().enumerate() {
            index.add(&sample, i).unwrap();
        }
        index.build().unwrap();

        let decoded = bincode::serialize(&index).unwrap();
        let new_index: HNSW2Index<usize> = bincode::deserialize(&decoded).unwrap();

        let target = (0..dimension)
            .map(|_| normal.sample(&mut rand::rng()))
            .collect::<Vec<f32>>();

        let v1 = index.search(target.clone(), 10);
        let v2 = new_index.search(target.clone(), 10);

        assert_eq!(v1, v2);
    }
}
