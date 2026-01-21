use std::{collections::HashSet, fmt::Debug, hash::Hash};

use crate::hnsw2::{
    HNSWIndex,
    core::{
        ann_index::ANNIndex,
        metrics::{Metric, real_cosine_similarity},
        node::{IdxType, Node},
    },
    hnsw_params::HNSWParams,
    rebuild::{GraphHealthMetrics, RebuildConfig, RebuildResult},
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

    /// Create a new HNSW2Index with deletion support enabled
    pub fn new_with_deletion(dim: usize) -> Self {
        let params = HNSWParams::<f32>::default().has_deletion(true);
        Self {
            inner: HNSWIndex::new(dim, &params),
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

    /// Returns the number of items inserted since the last rebuild
    pub fn insertions_since_rebuild(&self) -> usize {
        self.inner.insertions_since_rebuild()
    }

    /// Returns the number of items deleted since the last rebuild
    pub fn deletions_since_rebuild(&self) -> usize {
        self.inner.deletions_since_rebuild()
    }

    pub fn get_data(&self) -> impl Iterator<Item = (DocumentId, &[f32])> + '_ {
        self.inner.data().map(|(v, DocumentIdWrapper(id))| (id, v))
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

    pub fn batch_add<'a, I: Iterator<Item = (impl AsRef<[f32]> + 'a, DocumentId)>>(
        &mut self,
        items: I,
    ) -> Result<()> {
        self.inner
            .batch_add(items.map(|(v, i)| (v, DocumentIdWrapper(i))))
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn add_owned(&mut self, point: Vec<f32>, id: DocumentId) -> Result<()> {
        self.inner
            .add_owned(point, DocumentIdWrapper(id))
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Delete a document from the index by its DocumentId
    pub fn delete(&mut self, id: DocumentId) -> Result<()> {
        self.inner
            .delete_by_idx(&DocumentIdWrapper(id))
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Delete multiple documents from the index by their DocumentIds in a single pass.
    /// Returns the count of successfully deleted documents.
    pub fn delete_batch(&mut self, ids: &HashSet<DocumentId>) -> usize {
        self.inner
            .delete_batch_where(|DocumentIdWrapper(doc_id)| ids.contains(doc_id))
    }

    pub fn build(&mut self) -> Result<()> {
        self.inner
            .build(Metric::Euclidean)
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn search(&self, target: &[f32], limit: usize) -> Vec<(DocumentId, f32)> {
        assert_eq!(target.len(), self.dim);

        let v = self.inner.node_search_k(&Node::new(target), limit);

        let mut result = Vec::new();
        for (node, _) in v {
            let n = node.vectors();

            // The cosine similarity isnt a distance in the math sense
            // https://en.wikipedia.org/wiki/Distance#Mathematical_formalization
            // Anyway, it is good for ranking purposes
            // 1 means the vectors are equal
            // 0 means the vectors are orthogonal
            let score = real_cosine_similarity(n, target)
                .expect("real_cosine_similarity should not return an error");

            let id = match node.idx() {
                Some(DocumentIdWrapper(id)) => id,
                None => continue,
            };
            result.push((*id, score));
        }

        result
    }

    /// Analyze the health of the HNSW graph and return metrics
    pub fn analyze_health(&self) -> GraphHealthMetrics {
        self.inner.analyze_health()
    }

    /// Analyze the health of the HNSW graph with custom configuration
    pub fn analyze_health_with_config(&self, config: &RebuildConfig) -> GraphHealthMetrics {
        self.inner.analyze_health_with_config(config)
    }

    /// Rebuild the index using automatic strategy selection
    ///
    /// Uses heuristics to decide between:
    /// - NoAction: When deletion ratio < 5%
    /// - PartialRepair: When deletion ratio is 5-40%, repairs broken connections
    /// - FullRebuild: When deletion ratio >= 40%, rebuilds from scratch
    pub fn rebuild(&mut self) -> Result<RebuildResult> {
        self.inner.rebuild_index().map_err(|e| anyhow::anyhow!(e))
    }

    /// Rebuild the index with custom configuration
    pub fn rebuild_with_config(&mut self, config: &RebuildConfig) -> Result<RebuildResult> {
        self.inner
            .rebuild_with_config(config)
            .map_err(|e| anyhow::anyhow!(e))
    }

    /// Force a full rebuild of the index, regardless of deletion ratio
    ///
    /// Returns the number of deleted nodes that were compacted
    pub fn force_full_rebuild(&mut self) -> Result<usize> {
        self.inner
            .force_full_rebuild()
            .map_err(|e| anyhow::anyhow!(e))
    }

    pub fn deserialize_bincode_compat(data: &[u8]) -> Result<Self> {
        let inner: HNSWIndex<f32, DocumentIdWrapper<DocumentId>> =
            HNSWIndex::deserialize_bincode_compat(data)?;
        let dim = inner.dimension();
        Ok(Self { inner, dim })
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
        let v = index.search(&target, 10);

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

        let v1 = index.search(&target, 10);
        let v2 = new_index.search(&target, 10);

        assert_eq!(v1, v2);
    }

    // Helper function to create a test index with deletion support
    fn create_test_hnsw2_index(n: usize, dim: usize) -> HNSW2Index<usize> {
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let mut index = HNSW2Index::new_with_deletion(dim);
        for i in 0..n {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();
        index
    }

    // ==================== Category 1: Add Operations ====================

    #[test]
    fn test_add_single_point() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        assert_eq!(index.len(), 0);
        assert!(index.is_empty());

        let point: Vec<f32> = vec![1.0; dim];
        index.add(&point, 0).unwrap();

        assert_eq!(index.len(), 1);
        assert!(!index.is_empty());
    }

    #[test]
    fn test_add_multiple_points_before_build() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..100 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }

        assert_eq!(index.len(), 100);

        // Build and verify search works
        index.build().unwrap();
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_add_after_build() {
        let dim = 8;
        let normal = Uniform::new(0.0, 10.0).unwrap();

        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add initial points and build
        for i in 0..50 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        assert_eq!(index.len(), 50);

        // Add more points after build (incremental insertion)
        for i in 50..100 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        assert_eq!(index.len(), 100);

        // Verify all points are searchable
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert_eq!(results.len(), 10);
    }

    // ==================== Category 2: Delete Operations ====================

    #[test]
    fn test_delete_single_node() {
        let mut index = create_test_hnsw2_index(10, 8);

        assert_eq!(index.len(), 10);

        // Delete a node
        index.delete(5).unwrap();

        // Length doesn't change (node is just marked as deleted)
        assert_eq!(index.len(), 10);

        // Verify health shows 1 deleted node
        let health = index.analyze_health();
        assert_eq!(health.deleted_nodes, 1);
    }

    #[test]
    fn test_delete_nonexistent_node_error() {
        let mut index = create_test_hnsw2_index(10, 8);

        // Try to delete a node that doesn't exist
        let result = index.delete(999);
        assert!(result.is_err());
    }

    #[test]
    fn test_delete_already_deleted_error() {
        let mut index = create_test_hnsw2_index(10, 8);

        // Delete a node
        index.delete(5).unwrap();

        // Try to delete the same node again
        let result = index.delete(5);
        assert!(result.is_err());
    }

    #[test]
    fn test_deleted_nodes_excluded_from_search() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add known points
        let points = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for (i, point) in points.iter().enumerate() {
            index.add(point, i).unwrap();
        }
        index.build().unwrap();

        // Search for a target close to point 0
        let target = vec![0.9, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results_before = index.search(&target, 5);

        // Verify point 0 is in results
        assert!(results_before.iter().any(|(id, _)| *id == 0));

        // Delete point 0
        index.delete(0).unwrap();

        // Search again
        let results_after = index.search(&target, 5);

        // Verify point 0 is NOT in results anymore
        assert!(!results_after.iter().any(|(id, _)| *id == 0));
    }

    #[test]
    fn test_delete_batch() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add known points
        let points = vec![
            vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.5, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
            vec![0.0, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
        ];

        for (i, point) in points.iter().enumerate() {
            index.add(point, i).unwrap();
        }
        index.build().unwrap();

        // Verify all points are initially searchable
        let target = vec![0.5, 0.5, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results_before = index.search(&target, 6);
        assert_eq!(results_before.len(), 6);

        // Create a HashSet with IDs to delete (0, 2, 4)
        let ids_to_delete: HashSet<usize> = [0, 2, 4].into_iter().collect();

        // Batch delete
        let deleted_count = index.delete_batch(&ids_to_delete);
        assert_eq!(deleted_count, 3);

        // Verify deleted documents are no longer searchable
        let results_after = index.search(&target, 6);

        // Should only find 3 documents (1, 3, 5)
        assert_eq!(results_after.len(), 3);

        // Verify deleted IDs are not in results
        for (id, _) in &results_after {
            assert!(
                !ids_to_delete.contains(id),
                "Deleted document {} should not be in search results",
                id
            );
        }

        // Verify non-deleted documents are still searchable
        let remaining_ids: HashSet<usize> = results_after.iter().map(|(id, _)| *id).collect();
        assert!(remaining_ids.contains(&1));
        assert!(remaining_ids.contains(&3));
        assert!(remaining_ids.contains(&5));

        // Verify health metrics reflect deletions
        let health = index.analyze_health();
        assert_eq!(health.deleted_nodes, 3);
    }

    // ==================== Category 3: Serialization/Deserialization ====================

    #[test]
    fn test_serialize_empty_index() {
        let dim = 8;
        let index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Serialize
        let serialized = bincode::serialize(&index).unwrap();

        // Deserialize
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.len(), 0);
        assert_eq!(deserialized.dim(), dim);
    }

    #[test]
    fn test_serialize_preserves_search_results() {
        let dim = 16;
        let index = create_test_hnsw2_index(100, dim);

        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();

        // Get search results before serialization
        let results_before = index.search(&target, 10);

        // Serialize and deserialize
        let serialized = bincode::serialize(&index).unwrap();
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        // Get search results after deserialization
        let results_after = deserialized.search(&target, 10);

        // Results should be identical
        assert_eq!(results_before, results_after);
    }

    #[test]
    fn test_serialize_with_deleted_nodes() {
        let dim = 16;
        let mut index = create_test_hnsw2_index(100, dim);

        // Delete some nodes
        for i in 0..10 {
            index.delete(i).unwrap();
        }

        let health_before = index.analyze_health();
        assert_eq!(health_before.deleted_nodes, 10);

        // Serialize and deserialize
        let serialized = bincode::serialize(&index).unwrap();
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        // Verify deleted state is preserved
        let health_after = deserialized.analyze_health();
        assert_eq!(health_after.deleted_nodes, 10);

        // Verify deleted nodes are excluded from search
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = deserialized.search(&target, 20);

        for (id, _) in &results {
            assert!(
                *id >= 10,
                "Deleted node {} should not appear in search results",
                id
            );
        }
    }

    #[test]
    fn test_serialize_after_rebuild() {
        let dim = 16;
        let mut index = create_test_hnsw2_index(100, dim);

        // Delete 50% to trigger full rebuild
        for i in 0..50 {
            index.delete(i).unwrap();
        }

        // Rebuild
        let rebuild_result = index.rebuild().unwrap();
        assert_eq!(rebuild_result.nodes_compacted, 50);

        // Serialize and deserialize
        let serialized = bincode::serialize(&index).unwrap();
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        // Verify state
        let health = deserialized.analyze_health();
        assert_eq!(health.deleted_nodes, 0);
        assert_eq!(health.total_nodes, 50);

        // Verify search works
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = deserialized.search(&target, 10);
        assert!(!results.is_empty());
    }

    // ==================== Category 4: Full Workflow Integration ====================

    #[test]
    fn test_full_workflow_add_build_search_delete_search() {
        let dim = 16;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Step 1: Add points
        for i in 0..50 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        assert_eq!(index.len(), 50);

        // Step 2: Build
        index.build().unwrap();

        // Step 3: Search
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results1 = index.search(&target, 10);
        assert_eq!(results1.len(), 10);

        // Step 4: Delete some nodes
        for i in 0..5 {
            index.delete(i).unwrap();
        }

        // Step 5: Search again - deleted nodes should be excluded
        let results2 = index.search(&target, 10);
        for (id, _) in &results2 {
            assert!(*id >= 5, "Deleted node should not appear in results");
        }
    }

    #[test]
    fn test_full_workflow_with_serialization() {
        let dim = 16;
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Stage 1: Create and build
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);
        for i in 0..100 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // Serialize at stage 1
        let ser1 = bincode::serialize(&index).unwrap();

        // Stage 2: Delete 50% of nodes (triggers full rebuild)
        for i in 0..50 {
            index.delete(i).unwrap();
        }

        // Serialize at stage 2
        let ser2 = bincode::serialize(&index).unwrap();

        // Stage 3: Force full rebuild to compact deleted nodes
        index.force_full_rebuild().unwrap();

        // Serialize at stage 3
        let ser3 = bincode::serialize(&index).unwrap();

        // Deserialize all stages and verify
        let index1: HNSW2Index<usize> = bincode::deserialize(&ser1).unwrap();
        let index2: HNSW2Index<usize> = bincode::deserialize(&ser2).unwrap();
        let index3: HNSW2Index<usize> = bincode::deserialize(&ser3).unwrap();

        assert_eq!(index1.analyze_health().deleted_nodes, 0);
        assert_eq!(index2.analyze_health().deleted_nodes, 50);
        assert_eq!(index3.analyze_health().deleted_nodes, 0);
    }

    #[test]
    fn test_multiple_delete_rebuild_cycles() {
        let dim = 16;
        let mut index = create_test_hnsw2_index(100, dim);
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Cycle 1: Delete 20%, rebuild
        for i in 0..20 {
            index.delete(i).unwrap();
        }
        let health1 = index.analyze_health();
        assert_eq!(health1.deleted_nodes, 20);

        // Rebuild (may or may not compact depending on strategy)
        index.rebuild().unwrap();

        // Cycle 2: Delete another batch
        // After rebuild, IDs are remapped, so we need to work with current valid IDs
        let data: Vec<usize> = index.get_data().map(|(id, _)| id).take(10).collect();
        for id in data {
            let _ = index.delete(id); // May fail if already deleted
        }

        // Verify search still works
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert!(!results.is_empty());

        // Cycle 3: Force full rebuild
        index.force_full_rebuild().unwrap();

        // Final verification
        let health_final = index.analyze_health();
        assert_eq!(health_final.deleted_nodes, 0);

        let results_final = index.search(&target, 10);
        assert!(!results_final.is_empty());
    }

    // ==================== Category 5: Edge Cases ====================

    #[test]
    fn test_search_on_empty_index() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);
        index.build().unwrap();

        let target: Vec<f32> = vec![1.0; dim];
        let results = index.search(&target, 10);

        assert!(results.is_empty());
    }

    #[test]
    fn test_search_returns_fewer_than_limit() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add only 3 points
        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..3 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // Request 10 but only 3 exist
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);

        assert_eq!(results.len(), 3);
    }

    #[test]
    fn test_delete_all_nodes() {
        let dim = 8;
        let mut index = create_test_hnsw2_index(10, dim);

        // Delete all nodes
        for i in 0..10 {
            index.delete(i).unwrap();
        }

        let health = index.analyze_health();
        assert_eq!(health.deleted_nodes, 10);
        assert!((health.deletion_ratio - 1.0).abs() < 0.001);

        // Search should return empty results
        let target: Vec<f32> = vec![1.0; dim];
        let results = index.search(&target, 10);
        assert!(results.is_empty());

        // Force rebuild should compact all deleted nodes
        let compacted = index.force_full_rebuild().unwrap();
        assert_eq!(compacted, 10);

        let health_after = index.analyze_health();
        assert_eq!(health_after.total_nodes, 0);
        assert_eq!(health_after.deleted_nodes, 0);
    }

    #[test]
    fn test_large_index_serialization() {
        let n = 10000;
        let dim = 128;
        let index = create_test_hnsw2_index(n, dim);

        assert_eq!(index.len(), n);

        // Serialize
        let serialized = bincode::serialize(&index).unwrap();

        // Deserialize
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        assert_eq!(deserialized.len(), n);
        assert_eq!(deserialized.dim(), dim);

        // Verify search works
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();

        let results_original = index.search(&target, 10);
        let results_deserialized = deserialized.search(&target, 10);

        assert_eq!(results_original, results_deserialized);
    }

    // ==================== Change Tracking Tests ====================

    #[test]
    fn test_change_tracking_counters_increment() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Initially both counters should be 0
        assert_eq!(index.insertions_since_rebuild(), 0);
        assert_eq!(index.deletions_since_rebuild(), 0);

        // Add items
        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..10 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // After build, insertions counter should reflect added items
        assert_eq!(index.insertions_since_rebuild(), 10);
        assert_eq!(index.deletions_since_rebuild(), 0);

        // Delete some items
        for i in 0..3 {
            index.delete(i).unwrap();
        }

        // Deletions counter should increment
        assert_eq!(index.insertions_since_rebuild(), 10);
        assert_eq!(index.deletions_since_rebuild(), 3);
    }

    #[test]
    fn test_change_tracking_persists_after_serialization() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add and delete items
        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..10 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        for i in 0..3 {
            index.delete(i).unwrap();
        }

        // Verify counters before serialization
        assert_eq!(index.insertions_since_rebuild(), 10);
        assert_eq!(index.deletions_since_rebuild(), 3);

        // Serialize and deserialize
        let serialized = bincode::serialize(&index).unwrap();
        let deserialized: HNSW2Index<usize> = bincode::deserialize(&serialized).unwrap();

        // Verify counters persist after deserialization
        assert_eq!(deserialized.insertions_since_rebuild(), 10);
        assert_eq!(deserialized.deletions_since_rebuild(), 3);
    }

    #[test]
    fn test_change_tracking_reset_after_full_rebuild() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add items
        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..20 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // Delete enough to trigger full rebuild (50%)
        for i in 0..10 {
            index.delete(i).unwrap();
        }

        // Verify counters before rebuild
        assert_eq!(index.insertions_since_rebuild(), 20);
        assert_eq!(index.deletions_since_rebuild(), 10);

        // Force full rebuild
        index.force_full_rebuild().unwrap();

        // After full rebuild, both counters should reset to 0
        assert_eq!(index.insertions_since_rebuild(), 0);
        assert_eq!(index.deletions_since_rebuild(), 0);
    }

    #[test]
    fn test_change_tracking_in_health_metrics() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);

        // Add items
        let normal = Uniform::new(0.0, 10.0).unwrap();
        for i in 0..10 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // Delete some items
        for i in 0..3 {
            index.delete(i).unwrap();
        }

        // Verify health metrics include the change tracking counters
        let health = index.analyze_health();
        assert_eq!(health.insertions_since_rebuild, 10);
        assert_eq!(health.deletions_since_rebuild, 3);
    }

    #[test]
    fn test_change_tracking_incremental_additions() {
        let dim = 8;
        let mut index = HNSW2Index::<usize>::new_with_deletion(dim);
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // First batch
        for i in 0..5 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();
        assert_eq!(index.insertions_since_rebuild(), 5);

        // Second batch (incremental)
        for i in 5..10 {
            let point: Vec<f32> = (0..dim).map(|_| normal.sample(&mut rand::rng())).collect();
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        // Counter should reflect all insertions
        assert_eq!(index.insertions_since_rebuild(), 10);
    }
}
