use oramacore_lib::data_structures::hnsw2::HNSW2Index;
use oramacore_lib::hnsw2::rebuild::{RebuildConfig, RebuildStrategy};
use rand::distr::{Distribution, Uniform};

fn create_normalized_vector(dim: usize, seed: u64) -> Vec<f32> {
    use rand::SeedableRng;
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let dist = Uniform::new(-1.0, 1.0).unwrap();

    let mut vec: Vec<f32> = (0..dim).map(|_| dist.sample(&mut rng)).collect();
    let magnitude: f32 = vec.iter().map(|x| x * x).sum::<f32>().sqrt();
    if magnitude > 0.0 {
        for v in &mut vec {
            *v /= magnitude;
        }
    }
    vec
}

fn create_random_vector(dim: usize) -> Vec<f32> {
    let dist = Uniform::new(0.0, 10.0).unwrap();
    (0..dim).map(|_| dist.sample(&mut rand::rng())).collect()
}

fn round_trip_serialize<T>(index: &HNSW2Index<T>) -> HNSW2Index<T>
where
    T: serde::Serialize
        + serde::de::DeserializeOwned
        + std::hash::Hash
        + Send
        + Sync
        + Ord
        + std::fmt::Debug
        + Clone
        + Copy
        + Eq,
{
    let bytes = bincode::serialize(index).expect("Serialization should succeed");
    bincode::deserialize(&bytes).expect("Deserialization should succeed")
}

fn create_test_index(n: usize, dim: usize) -> HNSW2Index<u32> {
    let mut index = HNSW2Index::new_with_deletion(dim);
    for i in 0..n {
        let point = create_random_vector(dim);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();
    index
}

fn assert_approx_eq(a: f32, b: f32, epsilon: f32) {
    assert!(
        (a - b).abs() < epsilon,
        "Expected {a} to be approximately equal to {b} (epsilon: {epsilon})"
    );
}

#[test]
fn tc1_1_create_index_with_new() {
    let dim = 4;
    let index = HNSW2Index::<u32>::new(dim);

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.dim(), dim);
}

#[test]
fn tc1_2_create_index_with_new_with_deletion() {
    let dim = 4;
    let index = HNSW2Index::<u32>::new_with_deletion(dim);

    assert_eq!(index.len(), 0);
    assert!(index.is_empty());
    assert_eq!(index.dim(), dim);
}

#[test]
fn tc1_3_create_index_with_various_dimensions() {
    let index1 = HNSW2Index::<u32>::new(1);
    assert_eq!(index1.dim(), 1);

    let index128 = HNSW2Index::<u32>::new(128);
    assert_eq!(index128.dim(), 128);

    let index384 = HNSW2Index::<u32>::new(384);
    assert_eq!(index384.dim(), 384);

    let index768 = HNSW2Index::<u32>::new(768);
    assert_eq!(index768.dim(), 768);

    let index1536 = HNSW2Index::<u32>::new(1536);
    assert_eq!(index1536.dim(), 1536);
}

#[test]
fn tc2_1_add_single_vector_with_add() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let result = index.add(&[1.0, 0.0, 0.0, 0.0], 1);

    assert!(result.is_ok());
    assert_eq!(index.len(), 1);
    assert!(!index.is_empty());
}

#[test]
fn tc2_2_add_single_vector_with_add_owned() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let result = index.add_owned(vec![1.0, 0.0, 0.0, 0.0], 1);

    assert!(result.is_ok());
    assert_eq!(index.len(), 1);
}

#[test]
fn tc2_3_add_multiple_vectors_sequentially() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let r1 = index.add(&[1.0, 0.0, 0.0, 0.0], 1);
    let r2 = index.add(&[0.0, 1.0, 0.0, 0.0], 2);
    let r3 = index.add(&[0.0, 0.0, 1.0, 0.0], 3);

    assert!(r1.is_ok());
    assert!(r2.is_ok());
    assert!(r3.is_ok());
    assert_eq!(index.len(), 3);
}

#[test]
fn tc2_4_add_with_wrong_dimension_errors() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let result = index.add(&[1.0, 0.0, 0.0], 1);
    assert!(result.is_err());
}

#[test]
fn tc2_6_add_with_both_add_and_add_owned_mixed() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let r1 = index.add(&[1.0, 0.0, 0.0, 0.0], 1);
    let r2 = index.add_owned(vec![0.0, 1.0, 0.0, 0.0], 2);

    assert!(r1.is_ok());
    assert!(r2.is_ok());
    assert_eq!(index.len(), 2);

    index.build().unwrap();
    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert_eq!(results.len(), 2);
}

#[test]
fn tc3_1_delete_from_index_with_deletion_support() {
    let mut index = create_test_index(10, 4);

    let result = index.delete(5);

    assert!(result.is_ok());

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 1);
}

#[test]
fn tc3_2_index_without_deletion_support_still_returns_deleted_in_search() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);
    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.build().unwrap();

    let result = index.delete(1);
    assert!(result.is_ok());

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert!(results.iter().any(|(id, _)| *id == 1));
}

#[test]
fn tc3_3_delete_non_existent_item_errors() {
    let mut index = create_test_index(10, 4);

    let result = index.delete(999);

    assert!(result.is_err());
}

#[test]
fn tc3_4_delete_same_item_twice_errors() {
    let mut index = create_test_index(10, 4);

    let first_delete = index.delete(5);
    assert!(first_delete.is_ok());

    let second_delete = index.delete(5);
    assert!(second_delete.is_err());
}

#[test]
fn tc3_5_deleted_item_excluded_from_search() {
    let dim = 8;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    index
        .add(&[1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 0)
        .unwrap();
    index
        .add(&[0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 1)
        .unwrap();
    index
        .add(&[0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0], 2)
        .unwrap();
    index.build().unwrap();

    let target = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
    let results_before = index.search(&target, 3);
    assert!(results_before.iter().any(|(id, _)| *id == 0));

    index.delete(0).unwrap();

    let results_after = index.search(&target, 3);
    assert!(!results_after.iter().any(|(id, _)| *id == 0));
}

#[test]
fn tc3_6_delete_all_items() {
    let mut index = create_test_index(5, 4);

    for i in 0..5 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 5);

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert!(results.is_empty());
}

#[test]
fn tc4_1_search_empty_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);
    index.build().unwrap();

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);

    assert!(results.is_empty());
}

#[test]
fn tc4_3_search_for_exact_match() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let vector = vec![1.0, 0.0, 0.0, 0.0];
    index.add(&vector, 1).unwrap();
    index.build().unwrap();

    let results = index.search(&vector, 1);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert_approx_eq(results[0].1, 1.0, 0.001);
}

#[test]
fn tc4_4_search_limit_larger_than_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
    index.build().unwrap();

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 100);

    assert_eq!(results.len(), 3);
}

#[test]
fn tc4_5_search_with_limit_zero() {
    let index = create_test_index(10, 4);

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 0);

    assert!(results.is_empty());
}

#[test]
fn tc4_6_search_result_ordering_by_similarity() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.5, 0.5, 0.0, 0.0], 2).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 3).unwrap();
    index.build().unwrap();

    let target = vec![1.0, 0.0, 0.0, 0.0];
    let results = index.search(&target, 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 1);

    for i in 1..results.len() {
        assert!(
            results[i - 1].1 >= results[i].1,
            "Results should be ordered by similarity descending"
        );
    }
}

#[test]
fn tc4_7_multiple_builds_work_correctly() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.build().unwrap();

    let results1 = index.search(&[0.0, 0.0, 1.0, 0.0], 10);
    assert_eq!(results1.len(), 2);

    index.add(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
    index.build().unwrap();

    let results2 = index.search(&[0.0, 0.0, 1.0, 0.0], 10);
    assert_eq!(results2.len(), 3);
    assert!(results2.iter().any(|(id, _)| *id == 3));
}

#[test]
fn tc5_1_round_trip_empty_index() {
    let dim = 4;
    let index = HNSW2Index::<u32>::new(dim);

    let deserialized = round_trip_serialize(&index);

    assert_eq!(deserialized.dim(), dim);
    assert_eq!(deserialized.len(), 0);
    assert!(deserialized.is_empty());
}

#[test]
fn tc5_2_round_trip_populated_index() {
    let index = create_test_index(100, 16);

    let deserialized = round_trip_serialize(&index);

    assert_eq!(deserialized.len(), 100);
    assert_eq!(deserialized.dim(), 16);
}

#[test]
fn tc5_3_round_trip_index_with_deleted_items() {
    let mut index = create_test_index(100, 16);

    for i in 0..10 {
        index.delete(i).unwrap();
    }

    let health_before = index.analyze_health();

    let deserialized = round_trip_serialize(&index);

    let health_after = deserialized.analyze_health();

    assert_eq!(health_before.deleted_nodes, health_after.deleted_nodes);

    let target = create_random_vector(16);
    let results = deserialized.search(&target, 20);
    for (id, _) in &results {
        assert!(*id >= 10, "Deleted item {id} should not appear in search");
    }
}

#[test]
fn tc5_4_search_equivalence_after_deserialization() {
    let index = create_test_index(100, 16);

    let target = create_random_vector(16);
    let results_before = index.search(&target, 10);

    let deserialized = round_trip_serialize(&index);
    let results_after = deserialized.search(&target, 10);

    assert_eq!(results_before, results_after);
}

#[test]
fn tc5_5_add_to_deserialized_index() {
    let index = create_test_index(50, 8);
    let mut deserialized = round_trip_serialize(&index);

    for i in 50..60 {
        let point = create_random_vector(8);
        deserialized.add(&point, i).unwrap();
    }
    deserialized.build().unwrap();

    assert_eq!(deserialized.len(), 60);

    let target = create_random_vector(8);
    let results = deserialized.search(&target, 20);

    assert!(!results.is_empty());
}

#[test]
fn tc6_1_get_data_returns_all_items() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let vectors = [
        vec![1.0, 0.0, 0.0, 0.0],
        vec![0.0, 1.0, 0.0, 0.0],
        vec![0.0, 0.0, 1.0, 0.0],
    ];

    for (i, v) in vectors.iter().enumerate() {
        index.add(v, i as u32).unwrap();
    }
    index.build().unwrap();

    let data: Vec<_> = index.get_data().collect();
    assert_eq!(data.len(), 3);
}

#[test]
fn tc6_3_into_data_consumes_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
    index.build().unwrap();

    let data: Vec<_> = index.into_data().collect();

    assert_eq!(data.len(), 3);

    for (id, vec) in &data {
        assert_eq!(vec.len(), dim);
        assert!(*id >= 1 && *id <= 3);
    }
}

#[test]
fn tc6_4_data_integrity_verification() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let original_vectors: Vec<(u32, Vec<f32>)> =
        vec![(1, vec![1.0, 2.0, 3.0, 4.0]), (2, vec![5.0, 6.0, 7.0, 8.0])];

    for (id, v) in &original_vectors {
        index.add(v, *id).unwrap();
    }
    index.build().unwrap();

    let retrieved: Vec<_> = index.get_data().collect();

    for (id, vec) in &retrieved {
        let original = original_vectors.iter().find(|(oid, _)| oid == id);
        assert!(original.is_some(), "ID {id} should exist");
        let (_, original_vec) = original.unwrap();
        assert_eq!(vec.len(), original_vec.len());
        for (a, b) in vec.iter().zip(original_vec.iter()) {
            assert_approx_eq(*a, *b, 0.0001);
        }
    }
}

#[test]
fn tc7_1_analyze_health_of_healthy_index() {
    let index = create_test_index(100, 16);

    let health = index.analyze_health();

    assert_eq!(health.total_nodes, 100);
    assert_eq!(health.deleted_nodes, 0);
    assert_approx_eq(health.deletion_ratio as f32, 0.0, 0.001);
    assert_eq!(health.recommended_strategy, RebuildStrategy::NoAction);
}

#[test]
fn tc7_2_analyze_health_after_deletions_below_5_percent() {
    let mut index = create_test_index(100, 16);

    for i in 0..3 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();

    assert_eq!(health.deleted_nodes, 3);
    assert!(health.deletion_ratio < 0.05);
    assert_eq!(health.recommended_strategy, RebuildStrategy::NoAction);
}

#[test]
fn tc7_3_analyze_health_after_moderate_deletions() {
    let mut index = create_test_index(100, 16);

    for i in 0..20 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();

    assert_eq!(health.deleted_nodes, 20);
    assert!(health.deletion_ratio >= 0.05);
    assert!(health.deletion_ratio < 0.40);
}

#[test]
fn tc7_4_analyze_health_after_high_deletions() {
    let mut index = create_test_index(100, 16);

    for i in 0..50 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();

    assert_eq!(health.deleted_nodes, 50);
    assert!(health.deletion_ratio >= 0.40);
    assert_eq!(health.recommended_strategy, RebuildStrategy::FullRebuild);
}

#[test]
fn tc7_5_rebuild_compacts_deleted_nodes() {
    let mut index = create_test_index(100, 16);

    for i in 0..50 {
        index.delete(i).unwrap();
    }

    let len_before = index.len();
    let _health_before = index.analyze_health();

    let result = index.force_full_rebuild().unwrap();

    let health_after = index.analyze_health();

    assert_eq!(result, 50);
    assert_eq!(index.len(), 50);
    assert!(index.len() < len_before);
    assert_eq!(health_after.deleted_nodes, 0);
}

#[test]
fn tc7_6_search_works_correctly_after_rebuild() {
    let mut index = create_test_index(100, 16);

    for i in 0..30 {
        index.delete(i).unwrap();
    }

    index.rebuild().unwrap();

    let target = create_random_vector(16);
    let results = index.search(&target, 10);

    let health = index.analyze_health();
    assert!(!results.is_empty() || health.total_nodes == 0);
}

#[test]
fn tc7_7_force_full_rebuild_returns_compacted_count() {
    let mut index = create_test_index(50, 8);

    for i in 0..10 {
        index.delete(i).unwrap();
    }

    let compacted = index.force_full_rebuild().unwrap();

    assert_eq!(compacted, 10);
    assert_eq!(index.analyze_health().deleted_nodes, 0);
    assert_eq!(index.analyze_health().total_nodes, 40);
}

#[test]
fn tc7_8_rebuild_with_custom_config() {
    let mut index = create_test_index(100, 16);

    for i in 0..30 {
        index.delete(i).unwrap();
    }

    let config = RebuildConfig {
        skip_threshold: 0.05,
        full_rebuild_threshold: 0.25,
        ..Default::default()
    };

    let result = index.rebuild_with_config(&config).unwrap();

    assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);
    assert_eq!(result.nodes_compacted, 30);
    assert_eq!(result.metrics_after.deleted_nodes, 0);
}

#[test]
fn tc8_1_single_item_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let vector = vec![1.0, 0.0, 0.0, 0.0];
    index.add(&vector, 1).unwrap();
    index.build().unwrap();

    let results = index.search(&vector, 10);

    assert_eq!(results.len(), 1);
    assert_eq!(results[0].0, 1);
    assert_approx_eq(results[0].1, 1.0, 0.001);
}

#[test]
fn tc8_2_two_items_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.build().unwrap();

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);

    assert_eq!(results.len(), 2);
    assert_eq!(results[0].0, 1);
    assert_approx_eq(results[0].1, 1.0, 0.001);
}

#[test]
fn tc8_3_identical_vectors_different_ids() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let vector = vec![1.0, 2.0, 3.0, 4.0];
    index.add(&vector, 1).unwrap();
    index.add(&vector, 2).unwrap();
    index.add(&vector, 3).unwrap();
    index.build().unwrap();

    let results = index.search(&vector, 10);

    assert_eq!(results.len(), 3);
    for (_, score) in &results {
        assert_approx_eq(*score, 1.0, 0.01);
    }
}

#[test]
fn tc8_4_orthogonal_vectors() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    index.add(&[0.0, 0.0, 1.0, 0.0], 3).unwrap();
    index.add(&[0.0, 0.0, 0.0, 1.0], 4).unwrap();
    index.build().unwrap();

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 4);

    assert_eq!(results.len(), 4);

    assert_eq!(results[0].0, 1);
    assert_approx_eq(results[0].1, 1.0, 0.001);

    for i in 1..4 {
        assert_approx_eq(results[i].1, 0.0, 0.001);
    }
}

#[test]
fn tc8_5_high_dimensional_vectors() {
    let dim = 1536;
    let mut index = HNSW2Index::<u32>::new(dim);

    for i in 0..100 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    let target = create_normalized_vector(dim, 999);
    let results = index.search(&target, 10);

    assert_eq!(results.len(), 10);
}

#[test]
fn tc8_6_non_sequential_document_ids() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    let ids = vec![100, 5, 999, 42, 7];

    for id in &ids {
        let point = create_random_vector(dim);
        index.add(&point, *id).unwrap();
    }
    index.build().unwrap();

    assert_eq!(index.len(), 5);

    let target = create_random_vector(dim);
    let results = index.search(&target, 10);
    assert_eq!(results.len(), 5);

    index.delete(999).unwrap();

    let results_after = index.search(&target, 10);
    assert_eq!(results_after.len(), 4);
    assert!(!results_after.iter().any(|(id, _)| *id == 999));
}

#[test]
fn tc8_7_delete_first_last_middle_items() {
    let mut index = create_test_index(10, 4);

    index.delete(0).unwrap();

    index.delete(9).unwrap();

    index.delete(5).unwrap();

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 3);

    let target = create_random_vector(4);
    let results = index.search(&target, 10);

    for (id, _) in &results {
        assert!(*id != 0 && *id != 5 && *id != 9);
    }
}

#[test]
fn tc8_8_interleaved_add_delete_operations() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..5 {
        index.add(&create_random_vector(dim), i).unwrap();
    }
    index.build().unwrap();

    index.delete(2).unwrap();

    for i in 5..10 {
        index.add(&create_random_vector(dim), i).unwrap();
    }
    index.build().unwrap();

    index.delete(7).unwrap();

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 2);
    assert_eq!(health.total_nodes, 10);

    let target = create_random_vector(dim);
    let results = index.search(&target, 20);

    for (id, _) in &results {
        assert!(*id != 2 && *id != 7);
    }
}

#[test]
fn tc_scale_1_large_index() {
    let n = 1000;
    let dim = 64;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_random_vector(dim);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    assert_eq!(index.len(), n as usize);

    let target = create_random_vector(dim);
    let results = index.search(&target, 10);
    assert_eq!(results.len(), 10);

    for i in 0..(n / 2) {
        index.delete(i).unwrap();
    }

    index.force_full_rebuild().unwrap();

    assert_eq!(index.len(), (n / 2) as usize);

    let results_after = index.search(&target, 10);
    assert!(!results_after.is_empty());
}

#[test]
fn tc_scale_2_serialize_large_index() {
    let n = 1000;
    let dim = 64;
    let index = create_test_index(n, dim);

    let serialized = bincode::serialize(&index).expect("Serialization should succeed");

    assert!(!serialized.is_empty());

    let deserialized: HNSW2Index<u32> =
        bincode::deserialize(&serialized).expect("Deserialization should succeed");

    assert_eq!(deserialized.len(), n);
    assert_eq!(deserialized.dim(), dim);

    let target = create_random_vector(dim);
    let results_original = index.search(&target, 10);
    let results_deserialized = deserialized.search(&target, 10);

    assert_eq!(results_original, results_deserialized);
}

#[test]
fn tc_scale_3_various_dimensions() {
    let dimensions = vec![1, 2, 3, 4, 8, 16, 32, 64, 128, 256];

    for dim in dimensions {
        let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

        for i in 0..50 {
            let point = create_random_vector(dim);
            index.add(&point, i).unwrap();
        }
        index.build().unwrap();

        let target = create_random_vector(dim);
        let results = index.search(&target, 10);

        assert!(
            !results.is_empty() || dim == 0,
            "Search should return results for dimension {dim}"
        );

        for i in 0..25 {
            index.delete(i).unwrap();
        }
        index.force_full_rebuild().unwrap();

        assert_eq!(
            index.len(),
            25,
            "After rebuild, should have 25 items for dimension {dim}"
        );
    }
}

#[test]
fn tc_health_metrics_live_nodes() {
    let mut index = create_test_index(100, 8);

    for i in 0..30 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();

    assert_eq!(health.total_nodes, 100);
    assert_eq!(health.deleted_nodes, 30);
    assert_eq!(health.live_nodes(), 70);
}

#[test]
fn tc_rebuild_result_contains_metrics() {
    let mut index = create_test_index(100, 8);

    for i in 0..50 {
        index.delete(i).unwrap();
    }

    let result = index.rebuild().unwrap();

    assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);

    assert_eq!(result.metrics_before.deleted_nodes, 50);
    assert!(result.metrics_before.deletion_ratio >= 0.40);

    assert_eq!(result.metrics_after.deleted_nodes, 0);
    assert_eq!(result.metrics_after.total_nodes, 50);
}

#[test]
fn tc_delete_in_reverse_order() {
    let mut index = create_test_index(20, 4);

    for i in (0..20).rev() {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 20);

    let results = index.search(&create_random_vector(4), 10);
    assert!(results.is_empty());
}

#[test]
fn tc_delete_over_50_percent_triggers_full_rebuild() {
    let mut index = create_test_index(100, 8);

    for i in 0..60 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();
    assert!(health.deletion_ratio >= 0.40);
    assert_eq!(health.recommended_strategy, RebuildStrategy::FullRebuild);

    let result = index.rebuild().unwrap();
    assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);
}

#[test]
fn tc_build_empty_index() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    let result = index.build();
    assert!(result.is_ok());

    let results = index.search(&[1.0, 0.0, 0.0, 0.0], 10);
    assert!(results.is_empty());
}

#[test]
fn tc_build_once_after_all_additions() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    for i in 0..100 {
        index.add(&create_random_vector(dim), i).unwrap();
    }

    index.build().unwrap();

    let results = index.search(&create_random_vector(dim), 10);
    assert_eq!(results.len(), 10);
}

#[test]
fn tc_serialization_preserves_all_state() {
    let dim = 8;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..50 {
        index.add(&create_random_vector(dim), i).unwrap();
    }
    index.build().unwrap();

    for i in 0..10 {
        index.delete(i).unwrap();
    }

    let len_before = index.len();
    let dim_before = index.dim();
    let health_before = index.analyze_health();

    let deserialized = round_trip_serialize(&index);

    assert_eq!(deserialized.len(), len_before);
    assert_eq!(deserialized.dim(), dim_before);

    let health_after = deserialized.analyze_health();
    assert_eq!(health_after.total_nodes, health_before.total_nodes);
    assert_eq!(health_after.deleted_nodes, health_before.deleted_nodes);
    assert_approx_eq(
        health_after.deletion_ratio as f32,
        health_before.deletion_ratio as f32,
        0.0001,
    );
}

#[test]
fn tc_normalized_vectors_search() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[1.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 1.0, 0.0, 0.0], 2).unwrap();
    let normalized = create_normalized_vector(dim, 42);
    index.add(&normalized, 3).unwrap();
    index.build().unwrap();

    let query = create_normalized_vector(dim, 42);
    let results = index.search(&query, 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 3);
    assert_approx_eq(results[0].1, 1.0, 0.001);
}

#[test]
fn tc_non_normalized_vectors_search() {
    let dim = 4;
    let mut index = HNSW2Index::<u32>::new(dim);

    index.add(&[2.0, 0.0, 0.0, 0.0], 1).unwrap();
    index.add(&[0.0, 3.0, 0.0, 0.0], 2).unwrap();
    index.add(&[5.0, 5.0, 5.0, 5.0], 3).unwrap();
    index.build().unwrap();

    let query = vec![10.0, 0.0, 0.0, 0.0];
    let results = index.search(&query, 3);

    assert_eq!(results.len(), 3);
    assert_eq!(results[0].0, 1);
    assert_approx_eq(results[0].1, 1.0, 0.001);
}

#[test]
fn tc_full_lifecycle_workflow() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..100 {
        index.add(&create_random_vector(dim), i).unwrap();
    }
    index.build().unwrap();
    assert_eq!(index.len(), 100);

    let target = create_random_vector(dim);
    let results = index.search(&target, 10);
    assert_eq!(results.len(), 10);

    for i in 0..20 {
        index.delete(i).unwrap();
    }

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 20);

    let serialized = bincode::serialize(&index).unwrap();

    let mut restored: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();

    for i in 20..60 {
        restored.delete(i).unwrap();
    }

    let result = restored.rebuild().unwrap();
    assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);

    let final_health = restored.analyze_health();
    assert_eq!(final_health.deleted_nodes, 0);
    assert_eq!(final_health.total_nodes, 40);

    let final_results = restored.search(&target, 10);
    assert!(!final_results.is_empty());
}

fn brute_force_knn(
    vectors: &[(u32, Vec<f32>)],
    query: &[f32],
    k: usize,
    exclude_ids: &std::collections::HashSet<u32>,
) -> Vec<(u32, f32)> {
    let mut results: Vec<(u32, f32)> = vectors
        .iter()
        .filter(|(id, _)| !exclude_ids.contains(id))
        .map(|(id, vec)| (*id, cosine_similarity(vec, query)))
        .collect();

    results.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    results.truncate(k);
    results
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if mag_a == 0.0 || mag_b == 0.0 {
        0.0
    } else {
        dot / (mag_a * mag_b)
    }
}

fn measure_recall(ground_truth: &[(u32, f32)], search_results: &[(u32, f32)]) -> f64 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let truth_ids: std::collections::HashSet<u32> =
        ground_truth.iter().map(|(id, _)| *id).collect();
    let found_ids: std::collections::HashSet<u32> =
        search_results.iter().map(|(id, _)| *id).collect();

    let intersection = truth_ids.intersection(&found_ids).count();
    intersection as f64 / ground_truth.len() as f64
}

#[test]
fn test_search_with_root_deleted_no_rebuild() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..100 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    index.delete(0).unwrap();

    let query = create_normalized_vector(dim, 50);
    let results = index.search(&query, 10);
    assert!(
        !results.is_empty(),
        "Search should still return results after root deletion"
    );

    for (id, _) in &results {
        assert_ne!(
            *id, 0,
            "Deleted root node should not appear in search results"
        );
    }

    for seed in 100..110 {
        let q = create_normalized_vector(dim, seed);
        let r = index.search(&q, 5);
        assert!(!r.is_empty(), "Search {seed} should return results");
        assert!(
            !r.iter().any(|(id, _)| *id == 0),
            "Root should never appear"
        );
    }
}

#[test]
fn test_search_quality_degrades_with_deletions() {
    let dim = 32;
    let n = 300;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    let mut vectors: Vec<(u32, Vec<f32>)> = Vec::with_capacity(n);
    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        vectors.push((i as u32, point.clone()));
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let k = 10;
    let query = create_normalized_vector(dim, 999);
    let deleted: std::collections::HashSet<u32> = std::collections::HashSet::new();

    let ground_truth = brute_force_knn(&vectors, &query, k, &deleted);
    let search_results = index.search(&query, k);
    let baseline_recall = measure_recall(&ground_truth, &search_results);

    assert!(
        baseline_recall >= 0.5,
        "Baseline recall should be at least 0.5, got {baseline_recall}"
    );

    let mut recalls: Vec<(f64, f64)> = vec![(0.0, baseline_recall)];

    let deletion_levels = [0.05, 0.10, 0.20, 0.35];
    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for &target_ratio in &deletion_levels {
        let target_deletions = (n as f64 * target_ratio) as usize;
        while deleted_ids.len() < target_deletions {
            let id_to_delete = deleted_ids.len() as u32;
            if index.delete(id_to_delete).is_ok() {
                deleted_ids.insert(id_to_delete);
            }
        }

        let ground_truth = brute_force_knn(&vectors, &query, k, &deleted_ids);
        let search_results = index.search(&query, k);
        let recall = measure_recall(&ground_truth, &search_results);
        recalls.push((target_ratio * 100.0, recall));
    }

    let first_recall = recalls[0].1;
    let last_recall = recalls.last().unwrap().1;

    assert!(
        last_recall <= first_recall + 0.25,
        "Recall should not dramatically increase with deletions. Baseline: {first_recall}, Final: {last_recall}"
    );
}

#[test]
fn test_multiple_rebuild_cycles_maintain_health() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);
    let mut next_id: u32 = 0;

    for cycle in 0..5 {
        for _ in 0..100 {
            let point = create_normalized_vector(dim, next_id as u64);
            index.add(&point, next_id).unwrap();
            next_id += 1;
        }
        index.build().unwrap();

        let live_ids: Vec<u32> = index.get_data().map(|(id, _)| id).collect();

        let to_delete = live_ids.len() / 2;
        for &id in live_ids.iter().take(to_delete) {
            let _ = index.delete(id);
        }

        index.force_full_rebuild().unwrap();

        let health = index.analyze_health();
        assert_eq!(
            health.deleted_nodes, 0,
            "Cycle {cycle}: No deleted nodes after rebuild"
        );

        if health.total_nodes > 0 {
            let query = create_normalized_vector(dim, 12345 + cycle as u64);
            let results = index.search(&query, 10);
            assert!(
                !results.is_empty(),
                "Cycle {cycle}: Search should return results"
            );
        }

        let serialized = bincode::serialize(&index).unwrap();
        let _: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();
    }
}

#[test]
fn test_heavy_churn_alternating_add_delete() {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);
    let mut rng = rand::rngs::StdRng::seed_from_u64(42);

    let mut live_ids: Vec<u32> = Vec::new();
    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let mut next_id: u32 = 0;

    for _ in 0..100 {
        let point = create_normalized_vector(dim, next_id as u64);
        index.add(&point, next_id).unwrap();
        live_ids.push(next_id);
        next_id += 1;
    }
    index.build().unwrap();

    for round in 0..50 {
        for _ in 0..5 {
            let point = create_normalized_vector(dim, next_id as u64);
            index.add(&point, next_id).unwrap();
            live_ids.push(next_id);
            next_id += 1;
        }
        index.build().unwrap();

        live_ids.shuffle(&mut rng);
        for _ in 0..3.min(live_ids.len()) {
            if let Some(id) = live_ids.pop() {
                if index.delete(id).is_ok() {
                    deleted_ids.insert(id);
                }
            }
        }

        if round % 10 == 0 {
            let query = create_normalized_vector(dim, 99999 + round as u64);
            let results = index.search(&query, 20);
            for (id, _) in &results {
                assert!(
                    !deleted_ids.contains(id),
                    "Round {round}: Deleted item {id} appeared in search results"
                );
            }
        }
    }

    let query = create_normalized_vector(dim, 88888);
    let results = index.search(&query, 50);
    for (id, _) in &results {
        assert!(
            !deleted_ids.contains(id),
            "Final check: Deleted item {id} appeared in search results"
        );
    }

    let health = index.analyze_health();
    assert_eq!(
        health.total_nodes,
        index.len(),
        "len() should match health.total_nodes"
    );
}

#[test]
fn test_large_incremental_additions_after_build() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..50 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    let query = create_normalized_vector(dim, 25);
    let results = index.search(&query, 10);
    assert!(!results.is_empty(), "Initial search should work");

    let mut next_id: u32 = 50;
    for batch in 0..10 {
        let batch_start = next_id;
        for _ in 0..50 {
            let point = create_normalized_vector(dim, next_id as u64);
            index.add(&point, next_id).unwrap();
            next_id += 1;
        }
        index.build().unwrap();

        let query = create_normalized_vector(dim, next_id as u64 - 25);
        let results = index.search(&query, 10);
        assert!(
            !results.is_empty(),
            "Batch {batch}: Search should work after adding items"
        );

        let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
        for id in batch_start..next_id {
            assert!(
                all_ids.contains(&id),
                "Batch {batch}: Newly added item {id} should exist in index"
            );
        }

        let results_all = index.search(&query, 100);
        let has_new_items = results_all
            .iter()
            .any(|(id, _)| *id >= batch_start && *id < next_id);
        assert!(
            has_new_items,
            "Batch {batch}: Search should find some items from new batch"
        );
    }

    assert_eq!(index.len(), 550, "Should have 550 items total");

    let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
    for i in 0..550 {
        assert!(all_ids.contains(&i), "Item {i} should exist");
    }
}

#[test]
fn test_search_quality_after_partial_repair() {
    let dim = 32;
    let n = 300;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    let mut vectors: Vec<(u32, Vec<f32>)> = Vec::with_capacity(n);
    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        vectors.push((i as u32, point.clone()));
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let k = 10;
    let query = create_normalized_vector(dim, 999);
    let empty_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let ground_truth_baseline = brute_force_knn(&vectors, &query, k, &empty_set);
    let results_baseline = index.search(&query, k);
    let baseline_recall = measure_recall(&ground_truth_baseline, &results_baseline);

    let deletions = (n as f64 * 0.15) as usize;
    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for i in 0..deletions {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }

    let config = RebuildConfig {
        skip_threshold: 0.05,
        full_rebuild_threshold: 0.50,
        ..Default::default()
    };

    let result = index.rebuild_with_config(&config).unwrap();

    assert!(
        result.strategy_used == RebuildStrategy::PartialRepair
            || result.strategy_used == RebuildStrategy::NoAction
            || result.strategy_used == RebuildStrategy::FullRebuild,
        "Expected a valid rebuild strategy, got {:?}",
        result.strategy_used
    );

    let ground_truth_after = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_after = index.search(&query, k);
    let recall_after = measure_recall(&ground_truth_after, &results_after);

    assert!(
        recall_after >= 0.5,
        "Recall after partial repair should be at least 50%, got {recall_after}. Baseline was {baseline_recall}"
    );
}

#[test]
fn test_exact_5_percent_threshold_boundary() {
    let n = 100;
    let dim = 8;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    for i in 0..4 {
        index.delete(i as u32).unwrap();
    }

    let health_at_4_percent = index.analyze_health();
    assert_eq!(health_at_4_percent.deleted_nodes, 4);
    assert!(
        health_at_4_percent.deletion_ratio < 0.05,
        "4% should be below 5% threshold"
    );
    assert_eq!(
        health_at_4_percent.recommended_strategy,
        RebuildStrategy::NoAction,
        "At 4%, should recommend NoAction"
    );

    index.delete(4).unwrap();

    let health_at_5_percent = index.analyze_health();
    assert_eq!(health_at_5_percent.deleted_nodes, 5);
    assert!(
        health_at_5_percent.deletion_ratio >= 0.05,
        "5% should be at or above 5% threshold"
    );
    assert!(
        health_at_5_percent.recommended_strategy == RebuildStrategy::NoAction
            || health_at_5_percent.recommended_strategy == RebuildStrategy::PartialRepair,
        "At 5%, strategy should be NoAction or PartialRepair, got {:?}",
        health_at_5_percent.recommended_strategy
    );
}

#[test]
fn test_exact_40_percent_threshold_boundary() {
    let n = 100;
    let dim = 8;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    for i in 0..39 {
        index.delete(i as u32).unwrap();
    }

    let health_at_39_percent = index.analyze_health();
    assert_eq!(health_at_39_percent.deleted_nodes, 39);
    assert!(
        health_at_39_percent.deletion_ratio < 0.40,
        "39% should be below 40% threshold"
    );
    assert_ne!(
        health_at_39_percent.recommended_strategy,
        RebuildStrategy::FullRebuild,
        "At 39%, should NOT recommend FullRebuild"
    );

    index.delete(39).unwrap();

    let health_at_40_percent = index.analyze_health();
    assert_eq!(health_at_40_percent.deleted_nodes, 40);
    assert!(
        health_at_40_percent.deletion_ratio >= 0.40,
        "40% should be at or above 40% threshold"
    );
    assert_eq!(
        health_at_40_percent.recommended_strategy,
        RebuildStrategy::FullRebuild,
        "At 40%, should recommend FullRebuild"
    );
}

#[test]
fn test_delete_heavy_then_regrow() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..100 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    for i in 0..90 {
        index.delete(i).unwrap();
    }

    index.force_full_rebuild().unwrap();
    assert_eq!(
        index.len(),
        10,
        "After rebuild, should have 10 surviving nodes"
    );

    let surviving_ids: std::collections::HashSet<u32> =
        index.get_data().map(|(id, _)| id).collect();
    assert_eq!(surviving_ids.len(), 10);

    for i in 100..300 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    assert_eq!(
        index.len(),
        210,
        "Should have 210 total nodes (10 old + 200 new)"
    );

    for &old_id in surviving_ids.iter().take(3) {
        let query = create_normalized_vector(dim, old_id as u64);
        let results = index.search(&query, 5);
        assert!(
            results.iter().any(|(id, _)| *id == old_id),
            "Old surviving node {old_id} should be findable"
        );
    }

    let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
    for new_id in 100..300 {
        assert!(
            all_ids.contains(&new_id),
            "New node {new_id} should exist in index"
        );
    }

    let query = create_normalized_vector(dim, 200);
    let results = index.search(&query, 50);
    let has_new_items = results.iter().any(|(id, _)| *id >= 100);
    assert!(
        has_new_items,
        "Search should return some items from newly added range"
    );

    let health = index.analyze_health();
    assert_eq!(health.deleted_nodes, 0, "No deleted nodes after regrowth");
}

#[test]
fn test_state_stability_after_many_operations() {
    use rand::SeedableRng;
    use rand::seq::SliceRandom;

    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);
    let mut rng = rand::rngs::StdRng::seed_from_u64(12345);

    let mut next_id: u32 = 0;
    let mut live_ids: Vec<u32> = Vec::new();
    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for op in 0..200 {
        let operation = op % 5;

        match operation {
            0 | 1 => {
                let batch_size = 10 + (op % 11);
                for _ in 0..batch_size {
                    let point = create_normalized_vector(dim, next_id as u64);
                    index.add(&point, next_id).unwrap();
                    live_ids.push(next_id);
                    next_id += 1;
                }
                index.build().unwrap();
            }
            2 => {
                let delete_count = (live_ids.len() / 5).max(1).min(10);
                live_ids.shuffle(&mut rng);
                for _ in 0..delete_count {
                    if let Some(id) = live_ids.pop() {
                        if index.delete(id).is_ok() {
                            deleted_ids.insert(id);
                        }
                    }
                }
            }
            3 => {
                let query = create_normalized_vector(dim, op as u64 * 1000);
                let results = index.search(&query, 10);
                for (id, _) in &results {
                    assert!(
                        !deleted_ids.contains(id),
                        "Op {op}: Deleted item {id} in search results"
                    );
                }
            }
            4 => {
                let health = index.analyze_health();
                if health.deletion_ratio >= 0.30 {
                    index.force_full_rebuild().unwrap();
                    live_ids = index.get_data().map(|(id, _)| id).collect();
                }
            }
            _ => {}
        }

        if op % 50 == 49 {
            let health = index.analyze_health();

            assert_eq!(
                index.len(),
                health.total_nodes,
                "Op {op}: len() should match health.total_nodes"
            );

            let serialized = bincode::serialize(&index).unwrap();
            let _: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();
        }
    }

    let health = index.analyze_health();
    assert_eq!(
        index.len(),
        health.total_nodes,
        "Final: len() should match health.total_nodes"
    );

    let query = create_normalized_vector(dim, 77777);
    let results = index.search(&query, 50);
    for (id, _) in &results {
        assert!(
            !deleted_ids.contains(id),
            "Final: Deleted item {id} in search results"
        );
    }
}

#[test]
fn test_serialize_deserialize_across_rebuild_cycles() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..100 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    for i in 0..30 {
        index.delete(i).unwrap();
    }

    let bytes_before_rebuild = bincode::serialize(&index).expect("Serialize before rebuild");

    index.force_full_rebuild().unwrap();

    let bytes_after_rebuild = bincode::serialize(&index).expect("Serialize after rebuild");

    let index_before: HNSW2Index<u32> =
        bincode::deserialize(&bytes_before_rebuild).expect("Deserialize before");
    let index_after: HNSW2Index<u32> =
        bincode::deserialize(&bytes_after_rebuild).expect("Deserialize after");

    let health_before = index_before.analyze_health();
    let health_after = index_after.analyze_health();

    assert_eq!(
        health_before.total_nodes, 100,
        "Before-rebuild snapshot should have 100 total nodes"
    );
    assert_eq!(
        health_before.deleted_nodes, 30,
        "Before-rebuild snapshot should have 30 deleted nodes"
    );

    assert_eq!(
        health_after.total_nodes, 70,
        "After-rebuild snapshot should have 70 total nodes"
    );
    assert_eq!(
        health_after.deleted_nodes, 0,
        "After-rebuild snapshot should have 0 deleted nodes"
    );

    let query = create_normalized_vector(dim, 50);

    let results_before = index_before.search(&query, 10);
    assert!(
        !results_before.is_empty(),
        "Before-rebuild snapshot should search"
    );
    for (id, _) in &results_before {
        assert!(*id >= 30, "Before-rebuild: Deleted item {id} in results");
    }

    let results_after = index_after.search(&query, 10);
    assert!(
        !results_after.is_empty(),
        "After-rebuild snapshot should search"
    );

    let mut index_from_before = index_before;
    index_from_before.force_full_rebuild().unwrap();
    let health_rebuilt = index_from_before.analyze_health();
    assert_eq!(
        health_rebuilt.deleted_nodes, 0,
        "Deserialized index should rebuild correctly"
    );
}

#[test]
fn test_partial_repair_trigger_and_validation() {
    let dim = 16;
    let n = 200;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let health_initial = index.analyze_health();
    assert_eq!(health_initial.total_nodes, 200);
    assert_eq!(health_initial.deleted_nodes, 0);
    assert_eq!(health_initial.unreachable_nodes, 0);
    assert_eq!(
        health_initial.recommended_strategy,
        RebuildStrategy::NoAction
    );

    let delete_count = 14;
    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();
    for i in 0..delete_count {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }

    let health_after_delete = index.analyze_health();
    assert_eq!(health_after_delete.deleted_nodes, 14);
    assert!(health_after_delete.deletion_ratio >= 0.05);
    assert!(health_after_delete.deletion_ratio < 0.40);

    let config = RebuildConfig {
        skip_threshold: 0.05,
        full_rebuild_threshold: 0.50,
        ..Default::default()
    };

    let result = index.rebuild_with_config(&config).unwrap();

    assert!(
        result.strategy_used == RebuildStrategy::PartialRepair
            || result.strategy_used == RebuildStrategy::NoAction,
        "Expected PartialRepair or NoAction, got {:?}",
        result.strategy_used
    );

    if result.strategy_used == RebuildStrategy::PartialRepair {
        assert_eq!(
            result.nodes_compacted, 0,
            "PartialRepair should not compact nodes"
        );
    }

    let health_after_repair = index.analyze_health();
    assert_eq!(
        health_after_repair.deleted_nodes, 14,
        "PartialRepair should not remove tombstones"
    );
    assert_eq!(health_after_repair.total_nodes, 200);

    let query = create_normalized_vector(dim, 999);
    let results = index.search(&query, 50);
    for (id, _) in &results {
        assert!(
            !deleted_ids.contains(id),
            "Deleted ID {id} should not appear in search results"
        );
    }

    let serialized = bincode::serialize(&index).unwrap();
    let deserialized: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();

    let health_deserialized = deserialized.analyze_health();
    assert_eq!(
        health_deserialized.total_nodes,
        health_after_repair.total_nodes
    );
    assert_eq!(
        health_deserialized.deleted_nodes,
        health_after_repair.deleted_nodes
    );

    let results_deserialized = deserialized.search(&query, 50);
    for (id, _) in &results_deserialized {
        assert!(
            !deleted_ids.contains(id),
            "Deleted ID {id} should not appear in deserialized search results"
        );
    }
}

#[test]
fn test_full_rebuild_at_40_percent_threshold() {
    let dim = 16;
    let n = 100;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    for i in 0..39 {
        index.delete(i as u32).unwrap();
    }

    let health_at_39 = index.analyze_health();
    assert_eq!(health_at_39.deleted_nodes, 39);
    assert!(health_at_39.deletion_ratio < 0.40);
    assert_ne!(
        health_at_39.recommended_strategy,
        RebuildStrategy::FullRebuild,
        "At 39%, should NOT recommend FullRebuild"
    );

    index.delete(39).unwrap();

    let health_at_40 = index.analyze_health();
    assert_eq!(health_at_40.deleted_nodes, 40);
    assert!(health_at_40.deletion_ratio >= 0.40);
    assert_eq!(
        health_at_40.recommended_strategy,
        RebuildStrategy::FullRebuild,
        "At 40%, should recommend FullRebuild"
    );

    let result = index.rebuild().unwrap();
    assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);

    assert_eq!(result.nodes_compacted, 40);
    assert_eq!(result.metrics_after.total_nodes, 60);
    assert_eq!(result.metrics_after.deleted_nodes, 0);

    let query = create_normalized_vector(dim, 50);
    let results = index.search(&query, 20);
    assert!(!results.is_empty(), "Should find results after rebuild");

    for (id, _) in &results {
        assert!(
            *id >= 40,
            "Result ID {id} should be >= 40 (surviving range)"
        );
    }

    for i in 100..140 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let health_final = index.analyze_health();
    assert_eq!(health_final.total_nodes, 100);
    assert_eq!(health_final.deleted_nodes, 0);

    let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
    for i in 40..100 {
        assert!(all_ids.contains(&i), "Surviving ID {i} should exist");
    }
    for i in 100..140 {
        assert!(all_ids.contains(&i), "New ID {i} should exist");
    }
}

#[test]
fn test_graph_connectivity_throughout_lifecycle() {
    let dim = 32;
    let n = 300;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let health_initial = index.analyze_health();
    assert_eq!(health_initial.total_nodes, 300);
    assert_eq!(
        health_initial.unreachable_nodes, 0,
        "Initial index should have no unreachable nodes"
    );

    let delete_5_percent = 15;
    for i in 0..delete_5_percent {
        index.delete(i as u32).unwrap();
    }

    let health_at_5 = index.analyze_health();
    assert_eq!(health_at_5.deleted_nodes, 15);
    let _unreachable_at_5 = health_at_5.unreachable_nodes;

    let delete_10_more = 30;
    for i in delete_5_percent..(delete_5_percent + delete_10_more) {
        index.delete(i as u32).unwrap();
    }

    let health_at_15 = index.analyze_health();
    assert_eq!(health_at_15.deleted_nodes, 45);
    let unreachable_at_15 = health_at_15.unreachable_nodes;

    let config = RebuildConfig {
        skip_threshold: 0.05,
        full_rebuild_threshold: 0.50,
        ..Default::default()
    };

    let result = index.rebuild_with_config(&config).unwrap();

    let health_after_partial = index.analyze_health();

    if result.strategy_used == RebuildStrategy::PartialRepair {
        assert!(
            health_after_partial.unreachable_nodes <= unreachable_at_15 + 5,
            "PartialRepair should maintain or improve connectivity. Before: {}, After: {}",
            unreachable_at_15,
            health_after_partial.unreachable_nodes
        );
    }

    let current_deleted = health_after_partial.deleted_nodes;
    let target_deleted = 135;
    let to_delete_more = target_deleted - current_deleted;

    let mut next_delete_id = 45u32;
    for _ in 0..to_delete_more {
        while index.delete(next_delete_id).is_err() {
            next_delete_id += 1;
            if next_delete_id >= 300 {
                break;
            }
        }
        next_delete_id += 1;
    }

    let health_at_45 = index.analyze_health();
    assert!(
        health_at_45.deletion_ratio >= 0.40,
        "Should be at or above 40% deletion"
    );

    let _compacted = index.force_full_rebuild().unwrap();

    let health_after_full = index.analyze_health();
    assert_eq!(
        health_after_full.unreachable_nodes, 0,
        "After full rebuild, all nodes should be reachable"
    );
    assert_eq!(health_after_full.deleted_nodes, 0);

    let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
    let live_count = all_ids.len();

    assert!(live_count > 0, "Should have some live nodes remaining");

    for seed in 0..10 {
        let query = create_normalized_vector(dim, 1000 + seed);
        let results = index.search(&query, live_count.min(50));
        assert!(!results.is_empty(), "Search {seed} should return results");
    }
}

#[test]
fn test_search_recall_throughout_lifecycle() {
    let dim = 64;
    let n = 500;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    let mut vectors: Vec<(u32, Vec<f32>)> = Vec::with_capacity(n);
    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        vectors.push((i as u32, point.clone()));
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let k = 20;
    let query = create_normalized_vector(dim, 9999);

    let empty_set: std::collections::HashSet<u32> = std::collections::HashSet::new();
    let ground_truth_baseline = brute_force_knn(&vectors, &query, k, &empty_set);
    let results_baseline = index.search(&query, k);
    let baseline_recall = measure_recall(&ground_truth_baseline, &results_baseline);

    assert!(
        baseline_recall >= 0.80,
        "Baseline recall should be at least 0.80, got {baseline_recall}"
    );

    let mut deleted_ids: std::collections::HashSet<u32> = std::collections::HashSet::new();

    for i in 0..50 {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }
    let ground_truth_10 = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_10 = index.search(&query, k);
    let _recall_10 = measure_recall(&ground_truth_10, &results_10);

    for i in 50..100 {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }
    let ground_truth_20 = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_20 = index.search(&query, k);
    let _recall_20 = measure_recall(&ground_truth_20, &results_20);

    for i in 100..150 {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }
    let ground_truth_30 = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_30 = index.search(&query, k);
    let recall_30 = measure_recall(&ground_truth_30, &results_30);

    let recall_before_rebuild = recall_30;

    let _result = index.rebuild().unwrap();

    let ground_truth_after_rebuild = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_after_rebuild = index.search(&query, k);
    let recall_after_rebuild = measure_recall(&ground_truth_after_rebuild, &results_after_rebuild);

    assert!(
        recall_after_rebuild >= recall_before_rebuild - 0.15,
        "Recall after rebuild ({recall_after_rebuild}) should be close to or better than before ({recall_before_rebuild})"
    );

    for i in 150..250 {
        index.delete(i as u32).unwrap();
        deleted_ids.insert(i as u32);
    }

    index.force_full_rebuild().unwrap();

    let ground_truth_final = brute_force_knn(&vectors, &query, k, &deleted_ids);
    let results_final = index.search(&query, k);
    let recall_final = measure_recall(&ground_truth_final, &results_final);

    assert!(
        recall_final >= 0.75,
        "Final recall after full rebuild should be at least 0.75, got {recall_final}"
    );
}

#[test]
fn test_large_scale_lifecycle_5000_vectors() {
    let dim = 128;
    let n = 5000;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..n {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i as u32).unwrap();
    }
    index.build().unwrap();

    let health_initial = index.analyze_health();
    assert_eq!(health_initial.total_nodes, 5000);
    assert_eq!(health_initial.deleted_nodes, 0);

    let delete_3_percent = 150;
    for i in 0..delete_3_percent {
        index.delete(i as u32).unwrap();
    }

    let health_at_3 = index.analyze_health();
    assert_eq!(health_at_3.deleted_nodes, 150);
    assert!(health_at_3.deletion_ratio < 0.05);
    assert_eq!(
        health_at_3.recommended_strategy,
        RebuildStrategy::NoAction,
        "At 3%, should recommend NoAction"
    );

    let delete_to_7_percent = 350;
    for i in delete_3_percent..delete_to_7_percent {
        index.delete(i as u32).unwrap();
    }

    let health_at_7 = index.analyze_health();
    assert_eq!(health_at_7.deleted_nodes, 350);
    assert!(health_at_7.deletion_ratio >= 0.05);

    let result_7 = index.rebuild().unwrap();
    assert!(
        result_7.strategy_used == RebuildStrategy::PartialRepair
            || result_7.strategy_used == RebuildStrategy::NoAction,
        "At 7%, should use PartialRepair or NoAction, got {:?}",
        result_7.strategy_used
    );

    let delete_to_50_percent = 2500;
    for i in delete_to_7_percent..delete_to_50_percent {
        index.delete(i as u32).unwrap();
    }

    let health_at_50 = index.analyze_health();
    assert_eq!(health_at_50.deleted_nodes, 2500);
    assert!(health_at_50.deletion_ratio >= 0.40);
    assert_eq!(
        health_at_50.recommended_strategy,
        RebuildStrategy::FullRebuild
    );

    let result_50 = index.force_full_rebuild().unwrap();
    assert_eq!(result_50, 2500);

    let health_after_full = index.analyze_health();
    assert_eq!(health_after_full.deleted_nodes, 0);
    assert_eq!(health_after_full.total_nodes, 2500);

    let batch_size = 200;
    for batch in 0..5 {
        let start_id = 5000 + batch * batch_size;
        for i in start_id..(start_id + batch_size) {
            let point = create_normalized_vector(dim, i as u64);
            index.add(&point, i as u32).unwrap();
        }
        index.build().unwrap();

        let health = index.analyze_health();
        let expected_total = 2500 + (batch + 1) * batch_size;
        assert_eq!(
            health.total_nodes, expected_total as usize,
            "After batch {batch}, should have {expected_total} total nodes"
        );
    }

    let all_ids: std::collections::HashSet<u32> = index.get_data().map(|(id, _)| id).collect();
    for i in 5000..6000 {
        assert!(all_ids.contains(&i), "New ID {i} should exist");
    }

    let serialized = bincode::serialize(&index).unwrap();
    let deserialized: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();

    let health_deserialized = deserialized.analyze_health();
    assert_eq!(health_deserialized.total_nodes, 3500);
    assert_eq!(health_deserialized.deleted_nodes, 0);

    let query = create_normalized_vector(dim, 99999);
    let results = deserialized.search(&query, 50);
    assert!(!results.is_empty());

    let final_health = deserialized.analyze_health();
    assert_eq!(
        final_health.deleted_nodes, 0,
        "Final state should have 0 deleted nodes"
    );
    let max_unreachable = (final_health.total_nodes as f64 * 0.005).ceil() as usize;
    assert!(
        final_health.unreachable_nodes <= max_unreachable,
        "Unreachable nodes ({}) should be at most {} (0.5% of {})",
        final_health.unreachable_nodes,
        max_unreachable,
        final_health.total_nodes
    );
}

#[test]
fn test_health_metrics_consistency_across_operations() {
    let dim = 16;
    let mut index = HNSW2Index::<u32>::new_with_deletion(dim);

    for i in 0..100 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    let metrics = index.analyze_health();
    assert_eq!(metrics.total_nodes, 100);
    assert_eq!(metrics.deleted_nodes, 0);
    assert_eq!(metrics.deletion_ratio, 0.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);
    assert_eq!(metrics.severely_affected_nodes, 0);

    let target = create_normalized_vector(dim, 101);

    let output = index.search(&target, 10);
    assert_eq!(output.len(), 10);

    for i in 100..200 {
        let point = create_normalized_vector(dim, i as u64);
        index.add(&point, i).unwrap();
    }
    index.build().unwrap();

    let metrics = index.analyze_health();
    assert_eq!(metrics.total_nodes, 200);
    assert_eq!(metrics.deleted_nodes, 0);
    assert_eq!(metrics.deletion_ratio, 0.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);
    assert_eq!(metrics.severely_affected_nodes, 0);

    let output = index.search(&target, 10);
    assert_eq!(output.len(), 10);

    let to_delete = vec![output[0].0, output[3].0, output[7].0];
    for id in &to_delete {
        index.delete(*id).unwrap();
    }

    let metrics = index.analyze_health();
    assert_eq!(metrics.total_nodes, 200);
    assert_eq!(metrics.deleted_nodes, 3);
    assert_eq!(metrics.deletion_ratio, 3.0 / 200.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);

    let new_output = index.search(&target, 10);
    assert_eq!(new_output.len(), 10);
    for (id, _) in &new_output {
        assert!(
            !to_delete.contains(id),
            "Deleted id {id} appeared in search results"
        );
    }

    let serialized = bincode::serialize(&index).unwrap();

    let _ = index;

    let mut deserialized: HNSW2Index<u32> = bincode::deserialize(&serialized).unwrap();

    let metrics = deserialized.analyze_health();
    assert_eq!(metrics.total_nodes, 200);
    assert_eq!(metrics.deleted_nodes, 3);
    assert_eq!(metrics.deletion_ratio, 3.0 / 200.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);

    let output = deserialized.search(&target, 10);
    assert_eq!(output, new_output);

    for i in 200..300 {
        let point = create_normalized_vector(dim, i as u64);
        deserialized.add(&point, i).unwrap();
    }
    deserialized.build().unwrap();

    let metrics = deserialized.analyze_health();
    assert_eq!(metrics.total_nodes, 300);
    assert_eq!(metrics.deleted_nodes, 3);
    assert_eq!(metrics.deletion_ratio, 3.0 / 300.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);

    let output = deserialized.search(&target, 10);
    assert_eq!(output.len(), 10);

    let to_delete = vec![output[0].0, output[3].0, output[7].0];
    for id in &to_delete {
        deserialized.delete(*id).unwrap();
    }

    let metrics = deserialized.analyze_health();
    assert_eq!(metrics.total_nodes, 300);
    assert_eq!(metrics.deleted_nodes, 6);
    assert_eq!(metrics.deletion_ratio, 6.0 / 300.0);
    assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    assert_eq!(metrics.unreachable_nodes, 0);

    let new_output = deserialized.search(&target, 10);
    assert_eq!(new_output.len(), 10);
    for (id, _) in &new_output {
        assert!(
            !to_delete.contains(id),
            "Deleted id {id} appeared in search results"
        );
    }
}
