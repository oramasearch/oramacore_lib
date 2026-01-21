//! HNSW index rebuild functionality
//!
//! This module provides intelligent rebuild strategies for the HNSW index:
//! - **No Action** - When deletion ratio < 5%
//! - **Partial Repair** - When deletion ratio is 5-40%, repair broken connections
//! - **Full Rebuild** - When deletion ratio >= 40%, rebuild from scratch

/// Configuration for the rebuild process
#[derive(Debug, Clone)]
pub struct RebuildConfig {
    /// Deletion ratio below which no action is taken (default: 0.05 = 5%)
    pub skip_threshold: f64,
    /// Deletion ratio at or above which a full rebuild is performed (default: 0.40 = 40%)
    pub full_rebuild_threshold: f64,
    /// Connection loss ratio that marks a node as severely affected (default: 0.15 = 15%)
    pub connection_loss_threshold: f64,
    /// Minimum number of live connections a node should have (default: 2)
    pub min_connections: usize,
    /// Whether to detect unreachable nodes during health analysis (default: true)
    pub detect_unreachable: bool,
}

impl Default for RebuildConfig {
    fn default() -> Self {
        Self {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.40,
            connection_loss_threshold: 0.15,
            min_connections: 2,
            detect_unreachable: true,
        }
    }
}

/// Strategy recommended or used for rebuilding the index
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RebuildStrategy {
    /// No rebuild needed - deletion ratio is too low to warrant action
    NoAction,
    /// Partial repair - fix broken connections without full rebuild
    PartialRepair,
    /// Full rebuild - reconstruct the entire index from scratch
    FullRebuild,
}

/// Health metrics for the HNSW graph
#[derive(Debug, Clone, PartialEq)]
pub struct GraphHealthMetrics {
    /// Total number of nodes in the index (including deleted)
    pub total_nodes: usize,
    /// Number of deleted (tombstoned) nodes
    pub deleted_nodes: usize,
    /// Ratio of deleted nodes to total nodes
    pub deletion_ratio: f64,
    /// Number of nodes with significant connection loss
    pub severely_affected_nodes: usize,
    /// Number of nodes unreachable from the root
    pub unreachable_nodes: usize,
    /// The recommended rebuild strategy based on these metrics
    pub recommended_strategy: RebuildStrategy,
    /// Number of items inserted since the last rebuild
    pub insertions_since_rebuild: usize,
    /// Number of items deleted since the last rebuild
    pub deletions_since_rebuild: usize,
}

impl GraphHealthMetrics {
    /// Returns the number of live (non-deleted) nodes
    pub fn live_nodes(&self) -> usize {
        self.total_nodes.saturating_sub(self.deleted_nodes)
    }
}

/// Result of a rebuild operation
#[derive(Debug, Clone)]
pub struct RebuildResult {
    /// The strategy that was actually used
    pub strategy_used: RebuildStrategy,
    /// Number of nodes whose connections were repaired (for PartialRepair)
    pub nodes_repaired: usize,
    /// Number of nodes compacted/removed (for FullRebuild)
    pub nodes_compacted: usize,
    /// Health metrics before the rebuild
    pub metrics_before: GraphHealthMetrics,
    /// Health metrics after the rebuild
    pub metrics_after: GraphHealthMetrics,
}
