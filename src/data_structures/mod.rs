pub mod capped_heap;
pub mod fst;
pub mod hnsw;
pub mod hnsw2;
pub mod map;
pub mod ordered_key;
pub mod radix;
pub mod vector_bruteforce;

pub trait ShouldInclude<DocumentId>: Send + Sync {
    fn should_include(&self, doc_id: &DocumentId) -> bool;

    fn should_exclude(&self, doc_id: &DocumentId) -> bool {
        !self.should_include(doc_id)
    }
}
