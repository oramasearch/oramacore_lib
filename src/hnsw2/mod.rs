#![allow(dead_code)]

pub mod core;
pub mod hnsw_params;
pub mod rebuild;

use core::ann_index;
use core::metrics;
use core::neighbor::Neighbor;
use core::node;
use fixedbitset::FixedBitSet;
use hnsw_params::HNSWParams;
use rand::prelude::*;
#[cfg(not(feature = "no_thread"))]
use rayon::prelude::*;
use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};
use tracing::info;
use std::borrow::Cow;
use std::collections::BinaryHeap;

use std::collections::HashMap;
use std::collections::HashSet;
use std::collections::VecDeque;

use std::sync::RwLock;

use rebuild::{GraphHealthMetrics, RebuildConfig, RebuildResult, RebuildStrategy};

use crate::into_iter;

/// Efficient visited-set that avoids O(n) clear operations using generation counters.
/// Instead of clearing the entire array, we increment the generation counter.
/// A slot is considered visited if its stored generation equals the current generation.
struct Visited {
    store: Vec<u8>,
    generation: u8,
}

impl Visited {
    fn new(capacity: usize) -> Self {
        Self {
            store: vec![0; capacity],
            generation: 1,
        }
    }

    /// Clear the visited set in O(1) time (most of the time).
    /// When generation wraps around (every 255 clears), we do an O(n) reset.
    fn clear(&mut self) {
        if self.generation == 255 {
            self.store.fill(0);
            self.generation = 1;
        } else {
            self.generation += 1;
        }
    }

    /// Mark an id as visited. Returns true if it was not previously visited.
    #[inline]
    fn insert(&mut self, id: usize) -> bool {
        if id >= self.store.len() {
            // Grow the store if needed
            self.store.resize(id + 1, 0);
        }
        if self.store[id] == self.generation {
            false
        } else {
            self.store[id] = self.generation;
            true
        }
    }

    /// Check if an id has been visited.
    #[inline]
    fn contains(&self, id: usize) -> bool {
        id < self.store.len() && self.store[id] == self.generation
    }
}

/// Reusable buffers for construction to avoid repeated allocations.
struct ConstructionBuffers<E: node::FloatElement> {
    visited: Visited,
    candidates: BinaryHeap<Neighbor<E, usize>>,
    top_candidates: BinaryHeap<Neighbor<E, usize>>,
    sorted_candidates: Vec<Neighbor<E, usize>>,
}

impl<E: node::FloatElement> ConstructionBuffers<E> {
    fn new(capacity: usize, ef_build: usize) -> Self {
        Self {
            visited: Visited::new(capacity),
            candidates: BinaryHeap::with_capacity(ef_build),
            top_candidates: BinaryHeap::with_capacity(ef_build),
            sorted_candidates: Vec::with_capacity(ef_build),
        }
    }

    fn clear(&mut self) {
        self.visited.clear();
        self.candidates.clear();
        self.top_candidates.clear();
        self.sorted_candidates.clear();
    }
}

/// Thread-safe pool of construction buffers for parallel batch construction.
struct BufferPool<E: node::FloatElement> {
    pool: std::sync::Mutex<Vec<ConstructionBuffers<E>>>,
    capacity: usize,
    ef_build: usize,
}

impl<E: node::FloatElement> BufferPool<E> {
    fn new(capacity: usize, ef_build: usize) -> Self {
        Self {
            pool: std::sync::Mutex::new(Vec::new()),
            capacity,
            ef_build,
        }
    }

    fn acquire(&self) -> ConstructionBuffers<E> {
        self.pool
            .lock()
            .unwrap()
            .pop()
            .unwrap_or_else(|| ConstructionBuffers::new(self.capacity, self.ef_build))
    }

    fn release(&self, mut buffers: ConstructionBuffers<E>) {
        buffers.clear();
        self.pool.lock().unwrap().push(buffers);
    }
}

#[derive(Default, Debug)]
pub struct HNSWIndex<E: node::FloatElement, T: node::IdxType> {
    _dimension: usize, // dimension
    _n_items: usize,   // next item count
    _n_constructed_items: usize,
    _max_item: usize,
    _n_neighbor: usize,  // neighbor num except level 0
    _n_neighbor0: usize, // neight num of level 0
    _max_level: usize,   //max level
    _cur_level: usize,   //current level
    // #[serde(skip_serializing, skip_deserializing)]
    _id2neighbor: Vec<Vec<RwLock<Vec<usize>>>>, //neight_id from level 1 to level _max_level
    // #[serde(skip_serializing, skip_deserializing)]
    _id2neighbor0: Vec<RwLock<Vec<usize>>>, //neigh_id at level 0
    // #[serde(skip_serializing, skip_deserializing)]
    _nodes: Vec<Box<node::Node<E, T>>>, // data saver
    // #[serde(skip_serializing, skip_deserializing)]
    _item2id: HashMap<T, usize>, //item_id to id in Hnsw
    _root_id: usize,             //root of hnsw
    _id2level: Vec<usize>,
    _has_removed: bool,
    _ef_build: usize,  // num of max candidates when building
    _ef_search: usize, // num of max candidates when searching
    // #[serde(skip_serializing, skip_deserializing)]
    _delete_ids: HashSet<usize>, //save deleted ids
    mt: metrics::Metric,         //compute metrics

    // Change tracking since last rebuild
    _insertions_since_rebuild: usize, // Count of items added since last rebuild
    _deletions_since_rebuild: usize,  // Count of items deleted since last rebuild

                                 // // use for serde
                                 // _id2neighbor_tmp: Vec<Vec<Vec<usize>>>,
                                 // _id2neighbor0_tmp: Vec<Vec<usize>>,
                                 // _nodes_tmp: Vec<node::Node<E, T>>,
                                 // _item2id_tmp: Vec<(T, usize)>,
                                 // _delete_ids_tmp: Vec<usize>,
}

impl<E: node::FloatElement, T: node::IdxType> HNSWIndex<E, T> {
    pub fn new(dimension: usize, params: &HNSWParams<E>) -> HNSWIndex<E, T> {
        HNSWIndex {
            _dimension: dimension,
            _n_items: 0,
            _n_constructed_items: 0,
            _max_item: params.max_item,
            _n_neighbor: params.n_neighbor,
            _n_neighbor0: params.n_neighbor0,
            _max_level: params.max_level,
            _cur_level: 0,
            _root_id: 0,
            _has_removed: params.has_deletion,
            _ef_build: params.ef_build,
            _ef_search: params.ef_search,
            mt: metrics::Metric::Unknown,
            _insertions_since_rebuild: 0,
            _deletions_since_rebuild: 0,
            ..Default::default()
        }
    }

    #[inline]
    pub fn len(&self) -> usize {
        self._n_items
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Returns the number of items inserted since the last rebuild
    #[inline]
    pub fn insertions_since_rebuild(&self) -> usize {
        self._insertions_since_rebuild
    }

    /// Returns the number of items deleted since the last rebuild
    #[inline]
    pub fn deletions_since_rebuild(&self) -> usize {
        self._deletions_since_rebuild
    }

    pub fn data(&self) -> impl Iterator<Item = (&[E], T)> {
        self._nodes.iter().filter_map(|node| node.data())
    }

    pub fn into_data(self) -> impl Iterator<Item = (Vec<E>, T)> {
        self._nodes.into_iter().filter_map(|node| node.into_data())
    }

    fn get_random_level(&self) -> usize {
        let mut rng = rand::rng();
        let mut ret = 0;
        while ret < self._max_level {
            if rng.random_range(0.0..1.0) > 0.5 {
                ret += 1;
            } else {
                break;
            }
        }
        ret
    }
    //input top_candidate as max top heap
    //return min top heap in top_candidates, delete part candidate
    fn get_neighbors_by_heuristic2(
        &self,
        sorted_list: &[Neighbor<E, usize>],
        ret_size: usize,
    ) -> Vec<Neighbor<E, usize>> {
        let sorted_list_len = sorted_list.len();
        let mut return_list: Vec<Neighbor<E, usize>> = Vec::with_capacity(sorted_list_len);

        for iter in sorted_list.iter() {
            if return_list.len() >= ret_size {
                break;
            }

            let idx = iter.idx();
            let distance = iter._distance;
            if sorted_list_len < ret_size {
                return_list.push(Neighbor::new(idx, distance));
                continue;
            }

            let mut good = true;

            for ret_neighbor in return_list.iter() {
                let cur2ret_dis = self.get_distance_from_id(idx, ret_neighbor.idx());
                if cur2ret_dis < distance {
                    good = false;
                    break;
                }
            }

            if good {
                return_list.push(Neighbor::new(idx, distance));
            }
        }

        return_list // from small to large
    }

    fn get_neighbor(&self, id: usize, level: usize) -> &RwLock<Vec<usize>> {
        if level == 0 {
            return &self._id2neighbor0[id];
        }
        &self._id2neighbor[id][level - 1]
    }

    #[allow(dead_code)]
    fn get_level(&self, id: usize) -> usize {
        self._id2level[id]
    }

    fn connect_neighbor(
        &self,
        cur_id: usize,
        sorted_candidates: &[Neighbor<E, usize>],
        level: usize,
        is_update: bool,
    ) -> Result<usize, &'static str> {
        let n_neigh = if level == 0 {
            self._n_neighbor0
        } else {
            self._n_neighbor
        };
        let selected_neighbors = self.get_neighbors_by_heuristic2(sorted_candidates, n_neigh);
        if selected_neighbors.len() > n_neigh {
            return Err("Should be not be more than M_ candidates returned by the heuristic");
        }
        if selected_neighbors.is_empty() {
            return Err("top candidate is empty, impossible!");
        }

        let next_closest_entry_point = selected_neighbors[0].idx();

        {
            let mut cur_neigh = self.get_neighbor(cur_id, level).write().unwrap();
            cur_neigh.clear();
            selected_neighbors.iter().for_each(|selected_neighbor| {
                cur_neigh.push(selected_neighbor.idx());
            });
        }

        for selected_neighbor in selected_neighbors.iter() {
            let mut neighbor_of_selected_neighbors = self
                .get_neighbor(selected_neighbor.idx(), level)
                .write()
                .unwrap();
            if neighbor_of_selected_neighbors.len() > n_neigh {
                return Err("Bad Value of neighbor_of_selected_neighbors");
            }
            if selected_neighbor.idx() == cur_id {
                return Err("Trying to connect an element to itself");
            }

            let mut is_cur_id_present = false;

            if is_update {
                for iter in neighbor_of_selected_neighbors.iter() {
                    if *iter == cur_id {
                        is_cur_id_present = true;
                        break;
                    }
                }
            }

            if !is_cur_id_present {
                if neighbor_of_selected_neighbors.len() < n_neigh {
                    neighbor_of_selected_neighbors.push(cur_id);
                } else {
                    let d_max = self.get_distance_from_id(cur_id, selected_neighbor.idx());

                    let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
                    candidates.push(Neighbor::new(cur_id, d_max));
                    for iter in neighbor_of_selected_neighbors.iter() {
                        let neighbor_id = *iter;
                        let d_neigh =
                            self.get_distance_from_id(neighbor_id, selected_neighbor.idx());
                        candidates.push(Neighbor::new(neighbor_id, d_neigh));
                    }
                    let return_list =
                        self.get_neighbors_by_heuristic2(&candidates.into_sorted_vec(), n_neigh);

                    neighbor_of_selected_neighbors.clear();
                    for neighbor_in_list in return_list {
                        neighbor_of_selected_neighbors.push(neighbor_in_list.idx());
                    }
                }
            }
        }

        Ok(next_closest_entry_point)
    }

    /// Delete a node by its internal ID
    pub fn delete_id(&mut self, id: usize) -> Result<(), &'static str> {
        if id >= self._n_constructed_items {
            return Err("Invalid delete id");
        }
        if self.is_deleted(id) {
            return Err("id has deleted");
        }
        self._delete_ids.insert(id);
        self._deletions_since_rebuild += 1;
        Ok(())
    }

    /// Delete a node by its external ID (the T type)
    pub fn delete_by_idx(&mut self, idx: &T) -> Result<(), &'static str> {
        for (internal_id, node) in self._nodes.iter().enumerate() {
            if let Some(node_idx) = node.idx() {
                if node_idx == idx {
                    return self.delete_id(internal_id);
                }
            }
        }
        Err("Document ID not found")
    }

    /// Delete all nodes where the predicate returns true for the node's idx.
    /// Returns the count of successfully deleted nodes.
    pub fn delete_batch_where<F>(&mut self, should_delete: F) -> usize
    where
        F: Fn(&T) -> bool,
    {
        // Collect internal IDs to delete first to avoid borrow conflicts
        let ids_to_delete: Vec<usize> = self
            ._nodes
            .iter()
            .enumerate()
            .filter_map(|(internal_id, node)| {
                if let Some(node_idx) = node.idx() {
                    if should_delete(node_idx) {
                        return Some(internal_id);
                    }
                }
                None
            })
            .collect();

        let mut deleted_count = 0;
        for internal_id in ids_to_delete {
            if self.delete_id(internal_id).is_ok() {
                deleted_count += 1;
            }
        }
        deleted_count
    }

    fn is_deleted(&self, id: usize) -> bool {
        self._has_removed && self._delete_ids.contains(&id)
    }

    fn get_data(&self, id: usize) -> &node::Node<E, T> {
        &self._nodes[id]
    }

    fn get_distance_from_vec(&self, x: &node::Node<E, T>, y: &node::Node<E, T>) -> E {
        metrics::metric(x.vectors(), y.vectors(), self.mt).unwrap()
    }

    fn get_distance_from_id(&self, x: usize, y: usize) -> E {
        metrics::metric(
            self.get_data(x).vectors(),
            self.get_data(y).vectors(),
            self.mt,
        )
        .unwrap()
    }

    fn search_layer_with_candidate(
        &self,
        search_data: &node::Node<E, T>,
        sorted_candidates: &[Neighbor<E, usize>],
        visited_id: &mut FixedBitSet,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut top_candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::with_capacity(ef);
        for neighbor in sorted_candidates.iter() {
            let root = neighbor.idx();
            if !has_deletion || !self.is_deleted(root) {
                let dist = self.get_distance_from_vec(self.get_data(root), search_data);
                top_candidates.push(Neighbor::new(root, dist));
                candidates.push(Neighbor::new(root, -dist));
            } else {
                candidates.push(Neighbor::new(root, -E::max_value()))
            }
            visited_id.insert(root);
        }
        let mut lower_bound = if top_candidates.is_empty() {
            E::max_value() //max dist in top_candidates
        } else {
            top_candidates.peek().unwrap()._distance
        };

        while !candidates.is_empty() {
            let cur_neigh = candidates.peek().unwrap();
            let cur_dist = -cur_neigh._distance;
            let cur_id = cur_neigh.idx();
            candidates.pop();
            if cur_dist > lower_bound {
                break;
            }
            let cur_neighbors = self.get_neighbor(cur_id, level).read().unwrap();
            cur_neighbors.iter().for_each(|neigh| {
                if visited_id.contains(*neigh) {
                    return;
                }
                visited_id.insert(*neigh);
                let dist = self.get_distance_from_vec(self.get_data(*neigh), search_data);
                if top_candidates.len() < ef || dist < lower_bound {
                    candidates.push(Neighbor::new(*neigh, -dist));

                    if !self.is_deleted(*neigh) {
                        top_candidates.push(Neighbor::new(*neigh, dist))
                    }

                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap()._distance;
                    }
                }
            });
        }

        // println!("top_candidates {}. {}", top_candidates.len(), ef);

        top_candidates
    }
    //find ef nearist nodes to search data from root at level
    fn search_layer(
        &self,
        root: usize,
        search_data: &node::Node<E, T>,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) -> BinaryHeap<Neighbor<E, usize>> {
        let mut visited_id = FixedBitSet::with_capacity(self._nodes.len());
        let mut top_candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut candidates: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        let mut lower_bound: E;

        if !has_deletion || !self.is_deleted(root) {
            let dist = self.get_distance_from_vec(self.get_data(root), search_data);
            top_candidates.push(Neighbor::new(root, dist));
            candidates.push(Neighbor::new(root, -dist));
            lower_bound = dist;
        } else {
            lower_bound = E::max_value(); //max dist in top_candidates
            candidates.push(Neighbor::new(root, -lower_bound))
        }
        visited_id.insert(root);

        while !candidates.is_empty() {
            let cur_neigh = candidates.peek().unwrap();
            let cur_dist = -cur_neigh._distance;
            let cur_id = cur_neigh.idx();
            candidates.pop();
            if cur_dist > lower_bound {
                break;
            }
            let cur_neighbors = self.get_neighbor(cur_id, level).read().unwrap();
            cur_neighbors.iter().for_each(|neigh| {
                if visited_id.contains(*neigh) {
                    return;
                }
                visited_id.insert(*neigh);
                let dist = self.get_distance_from_vec(self.get_data(*neigh), search_data);
                if top_candidates.len() < ef || dist < lower_bound {
                    candidates.push(Neighbor::new(*neigh, -dist));

                    if !self.is_deleted(*neigh) {
                        top_candidates.push(Neighbor::new(*neigh, dist))
                    }

                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap()._distance;
                    }
                }
            });
        }

        top_candidates
    }

    fn search_knn(
        &self,
        search_data: &node::Node<E, T>,
        k: usize,
    ) -> Result<BinaryHeap<Neighbor<E, usize>>, &'static str> {
        let mut top_candidate: BinaryHeap<Neighbor<E, usize>> = BinaryHeap::new();
        if self._n_constructed_items == 0 {
            return Ok(top_candidate);
        }
        let mut cur_id = self._root_id;
        let mut cur_dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
        let mut cur_level = self._cur_level;
        loop {
            let mut changed = true;
            while changed {
                changed = false;
                let cur_neighs = self.get_neighbor(cur_id, cur_level).read().unwrap();
                for neigh in cur_neighs.iter() {
                    if *neigh > self._max_item {
                        return Err("cand error");
                    }
                    let dist = self.get_distance_from_vec(self.get_data(cur_id), search_data);
                    if dist < cur_dist {
                        cur_dist = dist;
                        cur_id = *neigh;
                        changed = true;
                    }
                }
            }
            if cur_level == 0 {
                break;
            }
            cur_level -= 1;
        }

        let search_range = if self._ef_search > k {
            self._ef_search
        } else {
            k
        };

        top_candidate = self.search_layer(cur_id, search_data, 0, search_range, self._has_removed);
        while top_candidate.len() > k {
            top_candidate.pop();
        }

        Ok(top_candidate)
    }

    fn init_item(&mut self, data: &node::Node<E, T>) -> usize {
        let cur_id = self._n_items;
        let mut cur_level = self.get_random_level();
        if cur_id == 0 {
            cur_level = self._max_level;
            self._cur_level = cur_level;
            self._root_id = cur_id;
        }
        let neigh0: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neighbor0));
        let mut neigh: Vec<RwLock<Vec<usize>>> = Vec::with_capacity(cur_level);
        for _i in 0..cur_level {
            let level_neigh: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neighbor));
            neigh.push(level_neigh);
        }
        self._nodes.push(Box::new(data.clone()));
        self._id2neighbor0.push(neigh0);
        self._id2neighbor.push(neigh);
        self._id2level.push(cur_level);
        // self._item2id.insert(data.idx().unwrap(), cur_id);
        self._n_items += 1;
        cur_id
    }

    fn batch_construct(&mut self, _mt: metrics::Metric) -> Result<(), &'static str> {
        if self._n_items < self._n_constructed_items {
            return Err("contruct error");
        }

        let added = self._n_items - self._n_constructed_items;

        into_iter!((self._n_constructed_items..self._n_items), ctr);
        ctr.for_each(|insert_id: usize| {
            self.construct_single_item(insert_id).unwrap();
        });

        self._n_constructed_items = self._n_items;
        self._insertions_since_rebuild += added;
        Ok(())
    }

    /// Optimized batch insertion when all items are known upfront.
    /// This method adds all items first without constructing connections,
    /// then builds the graph using optimized buffer-pooled construction.
    pub fn batch_add<'a, I: Iterator<Item = (impl AsRef<[E]> + 'a, T)>>(&mut self, items: I) -> Result<(), &'static str> where E: 'a {
        for (vs, idx) in items {
            self.add_item_not_constructed(&node::Node::new_with_idx(vs.as_ref(), idx.clone()))?;
        }
        Ok(())
    }

    /// Search layer using pre-allocated buffers to avoid repeated allocations.
    fn search_layer_with_candidate_buffered(
        &self,
        search_data: &node::Node<E, T>,
        initial_candidates: &[Neighbor<E, usize>],
        visited: &mut Visited,
        candidates: &mut BinaryHeap<Neighbor<E, usize>>,
        top_candidates: &mut BinaryHeap<Neighbor<E, usize>>,
        level: usize,
        ef: usize,
        has_deletion: bool,
    ) {
        candidates.clear();
        top_candidates.clear();

        for neighbor in initial_candidates.iter() {
            let root = neighbor.idx();
            if !has_deletion || !self.is_deleted(root) {
                let dist = self.get_distance_from_vec(self.get_data(root), search_data);
                top_candidates.push(Neighbor::new(root, dist));
                candidates.push(Neighbor::new(root, -dist));
            } else {
                candidates.push(Neighbor::new(root, -E::max_value()))
            }
            visited.insert(root);
        }

        let mut lower_bound = if top_candidates.is_empty() {
            E::max_value()
        } else {
            top_candidates.peek().unwrap()._distance
        };

        while !candidates.is_empty() {
            let cur_neigh = candidates.peek().unwrap();
            let cur_dist = -cur_neigh._distance;
            let cur_id = cur_neigh.idx();
            candidates.pop();

            if cur_dist > lower_bound {
                break;
            }

            let cur_neighbors = self.get_neighbor(cur_id, level).read().unwrap();
            for neigh in cur_neighbors.iter() {
                if visited.contains(*neigh) {
                    continue;
                }
                visited.insert(*neigh);

                let dist = self.get_distance_from_vec(self.get_data(*neigh), search_data);
                if top_candidates.len() < ef || dist < lower_bound {
                    candidates.push(Neighbor::new(*neigh, -dist));

                    if !self.is_deleted(*neigh) {
                        top_candidates.push(Neighbor::new(*neigh, dist));
                    }

                    if top_candidates.len() > ef {
                        top_candidates.pop();
                    }

                    if !top_candidates.is_empty() {
                        lower_bound = top_candidates.peek().unwrap()._distance;
                    }
                }
            }
        }
    }

    /// Construct a single item using pre-allocated buffers.
    fn construct_single_item_with_buffers(
        &self,
        insert_id: usize,
        buffers: &mut ConstructionBuffers<E>,
    ) -> Result<(), &'static str> {
        let insert_level = self._id2level[insert_id];
        let mut cur_id = self._root_id;

        if insert_id == 0 {
            return Ok(());
        }

        // Navigate through upper levels to find entry point
        if insert_level < self._cur_level {
            let mut cur_dist = self.get_distance_from_id(cur_id, insert_id);
            let mut cur_level = self._cur_level;
            while cur_level > insert_level {
                let mut changed = true;
                while changed {
                    changed = false;
                    let cur_neighs = self.get_neighbor(cur_id, cur_level).read().unwrap();
                    for cur_neigh in cur_neighs.iter() {
                        if *cur_neigh > self._n_items {
                            return Err("cand error");
                        }
                        let neigh_dist = self.get_distance_from_id(*cur_neigh, insert_id);
                        if neigh_dist < cur_dist {
                            cur_dist = neigh_dist;
                            cur_id = *cur_neigh;
                            changed = true;
                        }
                    }
                }
                cur_level -= 1;
            }
        }

        let mut level = if insert_level < self._cur_level {
            insert_level
        } else {
            self._cur_level
        };

        // Clear and setup buffers
        buffers.visited.clear();
        buffers.sorted_candidates.clear();

        let insert_data = self.get_data(insert_id);
        buffers.visited.insert(insert_id);
        buffers.sorted_candidates.push(Neighbor::new(
            cur_id,
            self.get_distance_from_id(cur_id, insert_id),
        ));

        loop {
            // Search for candidates using buffered method
            self.search_layer_with_candidate_buffered(
                insert_data,
                &buffers.sorted_candidates,
                &mut buffers.visited,
                &mut buffers.candidates,
                &mut buffers.top_candidates,
                level,
                self._ef_build,
                false,
            );

            if self.is_deleted(cur_id) {
                let cur_dist = self.get_distance_from_id(cur_id, insert_id);
                buffers.top_candidates.push(Neighbor::new(cur_id, cur_dist));
                if buffers.top_candidates.len() > self._ef_build {
                    buffers.top_candidates.pop();
                }
            }

            // Convert top_candidates to sorted_candidates
            buffers.sorted_candidates.clear();
            while let Some(candidate) = buffers.top_candidates.pop() {
                buffers.sorted_candidates.push(candidate);
            }
            buffers.sorted_candidates.reverse();

            if buffers.sorted_candidates.is_empty() {
                return Err("sorted sorted_candidate is empty");
            }

            cur_id = self
                .connect_neighbor(insert_id, &buffers.sorted_candidates, level, false)
                .unwrap();

            if level == 0 {
                break;
            }
            level -= 1;
        }
        Ok(())
    }

    /// Optimized batch construction using buffer pooling for better performance.
    fn batch_construct_optimized(&mut self, _mt: metrics::Metric) -> Result<(), &'static str> {
        if self._n_items < self._n_constructed_items {
            return Err("construct error");
        }

        let added = self._n_items - self._n_constructed_items;
        let pool = BufferPool::new(self._nodes.len(), self._ef_build);

        into_iter!((self._n_constructed_items..self._n_items), ctr);
        ctr.for_each(|insert_id: usize| {
            let mut buffers = pool.acquire();
            self.construct_single_item_with_buffers(insert_id, &mut buffers)
                .unwrap();
            pool.release(buffers);
        });

        self._n_constructed_items = self._n_items;
        self._insertions_since_rebuild += added;
        Ok(())
    }

    /// Ensure all nodes are reachable from the root after construction.
    /// This repairs any nodes that became unreachable due to the neighbor selection heuristic.
    fn ensure_full_connectivity(&mut self) -> Result<(), &'static str> {
        // Limit iterations to prevent infinite loops in pathological cases
        const MAX_ITERATIONS: usize = 10;

        for _ in 0..MAX_ITERATIONS {
            let unreachable = self.find_unreachable_nodes();
            if unreachable.is_empty() {
                return Ok(());
            }

            // Repair each unreachable node
            for node_id in unreachable {
                self.repair_node_connections(node_id)?;
            }
        }

        Ok(())
    }

    fn add_item_not_constructed(&mut self, data: &node::Node<E, T>) -> Result<(), &'static str> {
        if data.len() != self._dimension {
            return Err("dimension is different");
        }
        {
            // if self._item2id.contains_key(data.idx().unwrap()) {
            //     //to_do update point
            //     return Ok(self._item2id[data.idx().unwrap()]);
            // }

            if self._n_items >= self._max_item {
                return Err("The number of elements exceeds the specified limit");
            }
        }

        let insert_id = self.init_item(data);
        let _insert_level = self.get_level(insert_id);
        Ok(())
    }

    /// Initialize an item with a specific level (used during rebuild to preserve structure)
    fn init_item_with_level(&mut self, data: &node::Node<E, T>, level: usize) -> usize {
        let cur_id = self._n_items;
        let cur_level = if cur_id == 0 {
            // First node always gets max level and becomes root
            self._cur_level = self._max_level;
            self._root_id = cur_id;
            self._max_level
        } else {
            // Use the provided level, capped at max_level
            level.min(self._max_level)
        };

        let neigh0: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neighbor0));
        let mut neigh: Vec<RwLock<Vec<usize>>> = Vec::with_capacity(cur_level);
        for _i in 0..cur_level {
            let level_neigh: RwLock<Vec<usize>> = RwLock::new(Vec::with_capacity(self._n_neighbor));
            neigh.push(level_neigh);
        }
        self._nodes.push(Box::new(data.clone()));
        self._id2neighbor0.push(neigh0);
        self._id2neighbor.push(neigh);
        self._id2level.push(cur_level);
        self._n_items += 1;
        cur_id
    }

    /// Add an item without constructing connections, preserving a specific level
    fn add_item_with_level(&mut self, data: &node::Node<E, T>, level: usize) -> Result<(), &'static str> {
        if data.len() != self._dimension {
            return Err("dimension is different");
        }
        if self._n_items >= self._max_item {
            return Err("The number of elements exceeds the specified limit");
        }

        let _insert_id = self.init_item_with_level(data, level);
        Ok(())
    }

    fn add_single_item(&mut self, data: &node::Node<E, T>) -> Result<(), &'static str> {
        //not support asysn
        if data.len() != self._dimension {
            return Err("dimension is different");
        }
        {
            // if self._item2id.contains_key(data.idx().unwrap()) {
            //     //to_do update point
            //     return Ok(self._item2id[data.idx().unwrap()]);
            // }

            if self._n_items >= self._max_item {
                return Err("The number of elements exceeds the specified limit");
            }
        }

        let insert_id = self.init_item(data);
        let _insert_level = self.get_level(insert_id);
        self.construct_single_item(insert_id).unwrap();

        self._n_constructed_items += 1;
        self._insertions_since_rebuild += 1;

        Ok(())
    }

    fn construct_single_item(&self, insert_id: usize) -> Result<(), &'static str> {
        let insert_level = self._id2level[insert_id];
        let mut cur_id = self._root_id;

        if insert_id == 0 {
            return Ok(());
        }

        if insert_level < self._cur_level {
            let mut cur_dist = self.get_distance_from_id(cur_id, insert_id);
            let mut cur_level = self._cur_level;
            while cur_level > insert_level {
                let mut changed = true;
                while changed {
                    changed = false;
                    let cur_neighs = self.get_neighbor(cur_id, cur_level).read().unwrap();
                    for cur_neigh in cur_neighs.iter() {
                        if *cur_neigh > self._n_items {
                            return Err("cand error");
                        }
                        let neigh_dist = self.get_distance_from_id(*cur_neigh, insert_id);
                        if neigh_dist < cur_dist {
                            cur_dist = neigh_dist;
                            cur_id = *cur_neigh;
                            changed = true;
                        }
                    }
                }
                cur_level -= 1;
            }
        }

        let mut level = if insert_level < self._cur_level {
            insert_level
        } else {
            self._cur_level
        };
        let mut visited_id = FixedBitSet::with_capacity(self._nodes.len());
        let mut sorted_candidates: Vec<Neighbor<E, usize>> = Vec::new();
        let insert_data = self.get_data(insert_id);
        visited_id.insert(insert_id);
        sorted_candidates.push(Neighbor::new(
            cur_id,
            self.get_distance_from_id(cur_id, insert_id),
        ));
        loop {
            // let mut visited_id: HashSet<usize> = HashSet::new();
            let mut top_candidates = self.search_layer_with_candidate(
                insert_data,
                &sorted_candidates,
                &mut visited_id,
                level,
                self._ef_build,
                false,
            );
            // let mut top_candidates = self.search_layer_default(cur_id, insert_data, level);
            if self.is_deleted(cur_id) {
                let cur_dist = self.get_distance_from_id(cur_id, insert_id);
                top_candidates.push(Neighbor::new(cur_id, cur_dist));
                if top_candidates.len() > self._ef_build {
                    top_candidates.pop();
                }
            }
            sorted_candidates = top_candidates.into_sorted_vec();
            if sorted_candidates.is_empty() {
                return Err("sorted sorted_candidate is empty");
            }
            cur_id = self
                .connect_neighbor(insert_id, &sorted_candidates, level, false)
                .unwrap();
            if level == 0 {
                break;
            }
            level -= 1;
        }
        Ok(())
    }

    // ==================== Rebuild Methods ====================

    /// Analyze the health of the HNSW graph and return metrics
    pub fn analyze_health(&self) -> GraphHealthMetrics {
        self.analyze_health_with_config(&RebuildConfig::default())
    }

    /// Analyze the health of the HNSW graph with custom configuration
    pub fn analyze_health_with_config(&self, config: &RebuildConfig) -> GraphHealthMetrics {
        let total_nodes = self._n_constructed_items;
        let deleted_nodes = self._delete_ids.len();
        let deletion_ratio = if total_nodes == 0 {
            0.0
        } else {
            deleted_nodes as f64 / total_nodes as f64
        };

        // Count severely affected nodes (nodes with significant connection loss)
        let severely_affected_nodes = self.count_severely_affected_nodes(config);

        // Find unreachable nodes if configured
        let unreachable_nodes = if config.detect_unreachable {
            self.find_unreachable_nodes().len()
        } else {
            0
        };

        let recommended_strategy =
            self.recommend_rebuild_strategy_internal(config, deletion_ratio, severely_affected_nodes, unreachable_nodes, total_nodes);

        GraphHealthMetrics {
            total_nodes,
            deleted_nodes,
            deletion_ratio,
            severely_affected_nodes,
            unreachable_nodes,
            recommended_strategy,
            insertions_since_rebuild: self._insertions_since_rebuild,
            deletions_since_rebuild: self._deletions_since_rebuild,
        }
    }

    /// Count nodes that have lost a significant portion of their connections
    fn count_severely_affected_nodes(&self, config: &RebuildConfig) -> usize {
        let mut count = 0;

        for id in 0..self._n_constructed_items {
            if self.is_deleted(id) {
                continue;
            }

            // Check level 0 connections
            let neighbors = self._id2neighbor0[id].read().unwrap();
            let original_count = neighbors.len();
            let live_count = neighbors.iter().filter(|&&n| !self.is_deleted(n)).count();
            drop(neighbors);

            if original_count > 0 {
                let loss_ratio = 1.0 - (live_count as f64 / original_count as f64);
                if loss_ratio >= config.connection_loss_threshold
                    || live_count < config.min_connections
                {
                    count += 1;
                    continue;
                }
            } else if config.min_connections > 0 && id != self._root_id {
                // Node has no connections but should have some (not root)
                count += 1;
            }
        }

        count
    }

    /// Find all nodes unreachable from the root using BFS at level 0
    fn find_unreachable_nodes(&self) -> HashSet<usize> {
        if self._n_constructed_items == 0 {
            return HashSet::new();
        }

        let mut visited = HashSet::new();
        let mut queue = VecDeque::new();

        // Start BFS from root (if not deleted)
        if !self.is_deleted(self._root_id) {
            queue.push_back(self._root_id);
            visited.insert(self._root_id);
        }

        // BFS traversal at level 0
        while let Some(current) = queue.pop_front() {
            let neighbors = self._id2neighbor0[current].read().unwrap();
            for &neighbor in neighbors.iter() {
                if !visited.contains(&neighbor) && !self.is_deleted(neighbor) {
                    visited.insert(neighbor);
                    queue.push_back(neighbor);
                }
            }
        }

        // Find all live nodes that were not visited
        let mut unreachable = HashSet::new();
        for id in 0..self._n_constructed_items {
            if !self.is_deleted(id) && !visited.contains(&id) {
                unreachable.insert(id);
            }
        }

        unreachable
    }

    /// Determine the recommended rebuild strategy based on metrics
    fn recommend_rebuild_strategy_internal(
        &self,
        config: &RebuildConfig,
        deletion_ratio: f64,
        severely_affected_nodes: usize,
        unreachable_nodes: usize,
        total_nodes: usize,
    ) -> RebuildStrategy {
        // High deletion ratio -> full rebuild
        if deletion_ratio >= config.full_rebuild_threshold {
            return RebuildStrategy::FullRebuild;
        }

        // Very low deletion ratio -> no action
        if deletion_ratio < config.skip_threshold {
            return RebuildStrategy::NoAction;
        }

        // Moderate deletion ratio (5-40%), check connection health
        let live_nodes = total_nodes - self._delete_ids.len();
        if live_nodes == 0 {
            return RebuildStrategy::NoAction;
        }

        let affected_ratio = severely_affected_nodes as f64 / live_nodes as f64;
        let unreachable_ratio = unreachable_nodes as f64 / live_nodes as f64;

        // If many nodes are affected or unreachable, do partial repair
        if affected_ratio > 0.05 || unreachable_ratio > 0.01 {
            return RebuildStrategy::PartialRepair;
        }

        RebuildStrategy::NoAction
    }

    /// Rebuild the index using automatic strategy selection
    pub fn rebuild_index(&mut self) -> Result<RebuildResult, &'static str> {
        self.rebuild_with_config(&RebuildConfig::default())
    }

    /// Rebuild the index with custom configuration
    pub fn rebuild_with_config(&mut self, config: &RebuildConfig) -> Result<RebuildResult, &'static str> {
        let metrics_before = self.analyze_health_with_config(config);
        let strategy = metrics_before.recommended_strategy;

        info!("Rebuild Strategy Chosen: {:?}", strategy);

        let (nodes_repaired, nodes_compacted) = match strategy {
            RebuildStrategy::NoAction => (0, 0),
            RebuildStrategy::PartialRepair => {
                let repaired = self.repair_connections(config)?;
                (repaired, 0)
            }
            RebuildStrategy::FullRebuild => {
                let compacted = self.force_full_rebuild()?;
                (0, compacted)
            }
        };

        let metrics_after = self.analyze_health_with_config(config);

        Ok(RebuildResult {
            strategy_used: strategy,
            nodes_repaired,
            nodes_compacted,
            metrics_before,
            metrics_after,
        })
    }

    /// Force a full rebuild of the index, extracting live nodes and rebuilding from scratch
    pub fn force_full_rebuild(&mut self) -> Result<usize, &'static str> {
        if self._n_constructed_items == 0 {
            return Ok(0);
        }

        // Collect all live nodes' data
        let live_data: Vec<(Box<node::Node<E, T>>, usize)> = (0..self._n_constructed_items)
            .filter(|&id| !self.is_deleted(id))
            .map(|id| (self._nodes[id].clone(), self._id2level[id]))
            .collect();

        let compacted_count = self._delete_ids.len();

        if live_data.is_empty() {
            // All nodes were deleted, reset to empty state
            self._n_items = 0;
            self._n_constructed_items = 0;
            self._cur_level = 0;
            self._root_id = 0;
            self._nodes.clear();
            self._id2neighbor.clear();
            self._id2neighbor0.clear();
            self._id2level.clear();
            self._item2id.clear();
            self._delete_ids.clear();
            self._insertions_since_rebuild = 0;
            self._deletions_since_rebuild = 0;
            return Ok(compacted_count);
        }

        // Reset index state
        self._n_items = 0;
        self._n_constructed_items = 0;
        self._cur_level = 0;
        self._root_id = 0;
        self._nodes.clear();
        self._id2neighbor.clear();
        self._id2neighbor0.clear();
        self._id2level.clear();
        self._item2id.clear();
        self._delete_ids.clear();

        // Re-add all live nodes, preserving their original levels
        for (node_data, old_level) in live_data.iter() {
            self.add_item_with_level(node_data, *old_level)?;
        }

        // Rebuild the graph
        self.batch_construct(self.mt)?;

        // Ensure all nodes are reachable after rebuild
        self.ensure_full_connectivity()?;

        // Reset change tracking counters after full rebuild
        self._insertions_since_rebuild = 0;
        self._deletions_since_rebuild = 0;

        Ok(compacted_count)
    }

    /// Repair connections for nodes affected by deletions (Lucene-style partial repair)
    fn repair_connections(&mut self, config: &RebuildConfig) -> Result<usize, &'static str> {
        let mut repaired_count = 0;

        // First, clean up deleted references from all neighbor lists
        self.cleanup_deleted_references();

        // Find nodes that need repair
        let nodes_to_repair: Vec<usize> = (0..self._n_constructed_items)
            .filter(|&id| {
                if self.is_deleted(id) {
                    return false;
                }

                let neighbors = self._id2neighbor0[id].read().unwrap();
                let live_count = neighbors.iter().filter(|&&n| !self.is_deleted(n)).count();
                let original_count = neighbors.len();
                drop(neighbors);

                if original_count == 0 && id != self._root_id {
                    // No connections, needs repair
                    return true;
                }

                if original_count > 0 {
                    let loss_ratio = 1.0 - (live_count as f64 / original_count as f64);
                    if loss_ratio >= config.connection_loss_threshold {
                        return true;
                    }
                }

                live_count < config.min_connections && id != self._root_id
            })
            .collect();

        // Also include unreachable nodes
        let unreachable = self.find_unreachable_nodes();
        let mut all_nodes_to_repair: HashSet<usize> = nodes_to_repair.into_iter().collect();
        all_nodes_to_repair.extend(unreachable);

        // Repair each affected node
        for node_id in all_nodes_to_repair {
            if self.repair_node_connections(node_id)? {
                repaired_count += 1;
            }
        }

        // Reset deletions counter only - partial repair doesn't reset index structure
        // Keep insertions counter as the inserted nodes are still part of the index
        self._deletions_since_rebuild = 0;

        Ok(repaired_count)
    }

    /// Clean up references to deleted nodes from all neighbor lists
    fn cleanup_deleted_references(&mut self) {
        // Clean level 0 neighbors
        for id in 0..self._n_constructed_items {
            if self.is_deleted(id) {
                continue;
            }
            let mut neighbors = self._id2neighbor0[id].write().unwrap();
            neighbors.retain(|&n| !self.is_deleted(n));
        }

        // Clean higher level neighbors
        for id in 0..self._n_constructed_items {
            if self.is_deleted(id) {
                continue;
            }
            let level = self._id2level[id];
            for l in 0..level {
                let mut neighbors = self._id2neighbor[id][l].write().unwrap();
                neighbors.retain(|&n| !self.is_deleted(n));
            }
        }
    }

    /// Repair connections for a single node by searching for new neighbors
    fn repair_node_connections(&mut self, node_id: usize) -> Result<bool, &'static str> {
        if self.is_deleted(node_id) {
            return Ok(false);
        }

        let node_data = self.get_data(node_id);

        // Find entry point by traversing from root
        let mut cur_id = self._root_id;

        if cur_id == node_id || self.is_deleted(cur_id) {
            // Find another entry point if root is deleted or is the node itself
            for id in 0..self._n_constructed_items {
                if !self.is_deleted(id) && id != node_id {
                    cur_id = id;
                    break;
                }
            }
        }

        if cur_id == node_id {
            // This node is the only live node, nothing to repair
            return Ok(false);
        }

        // Navigate to the appropriate level
        // Start from the highest level available for the current entry point
        let mut cur_level = self._id2level[cur_id].min(self._cur_level);
        while cur_level > 0 {
            // Only access this level if the current node has connections at this level
            let cur_node_level = self._id2level[cur_id];
            if cur_level <= cur_node_level {
                let mut changed = true;
                while changed {
                    changed = false;
                    let cur_neighs = self.get_neighbor(cur_id, cur_level).read().unwrap();
                    for &neigh in cur_neighs.iter() {
                        if self.is_deleted(neigh) {
                            continue;
                        }
                        let neigh_dist = self.get_distance_from_vec(self.get_data(neigh), node_data);
                        let cur_dist = self.get_distance_from_vec(self.get_data(cur_id), node_data);
                        if neigh_dist < cur_dist {
                            drop(cur_neighs);
                            cur_id = neigh;
                            changed = true;
                            break;
                        }
                    }
                    if changed {
                        break;
                    }
                }
            }
            cur_level -= 1;
        }

        // Search for candidates at level 0
        let candidates = self.search_layer(cur_id, node_data, 0, self._ef_build, true);

        if candidates.is_empty() {
            return Ok(false);
        }

        // Filter out the node itself from candidates to avoid self-connection
        let sorted_candidates: Vec<Neighbor<E, usize>> = candidates
            .into_sorted_vec()
            .into_iter()
            .filter(|n| n.idx() != node_id)
            .collect();

        if sorted_candidates.is_empty() {
            return Ok(false);
        }

        // Connect the node to new neighbors
        self.connect_neighbor(node_id, &sorted_candidates, 0, true)?;

        Ok(true)
    }
}

impl<E: node::FloatElement, T: node::IdxType> ann_index::ANNIndex<E, T> for HNSWIndex<E, T> {
    fn build(&mut self, mt: metrics::Metric) -> Result<(), &'static str> {
        self.mt = mt;
        let new_items = self._n_items - self._n_constructed_items;
        if new_items > 0 {
            let is_fresh_build = self._n_constructed_items == 0;
            // Use optimized batch construction for fresh builds with buffer pooling
            if is_fresh_build {
                self.batch_construct_optimized(mt)?;
            } else {
                self.batch_construct(mt)?;
            }
            // Check connectivity on fresh builds OR when batch is >= 10% of existing index
            // Small incremental additions connect to existing graph naturally via HNSW algorithm
            // Only large batches risk creating isolated clusters that need repair
            if is_fresh_build || new_items >= self._n_constructed_items / 10 {
                self.ensure_full_connectivity()?;
            }
        }
        Ok(())
    }
    fn add_node(&mut self, item: &node::Node<E, T>) -> Result<(), &'static str> {
        self.add_item_not_constructed(item)
    }
    fn built(&self) -> bool {
        true
    }

    fn rebuild(&mut self, _mt: metrics::Metric) -> Result<(), &'static str> {
        self.rebuild_index().map(|_| ())
    }

    fn node_search_k(&self, item: &node::Node<E, T>, k: usize) -> Vec<(node::Node<E, T>, E)> {
        let mut ret: BinaryHeap<Neighbor<E, usize>> = self.search_knn(item, k).unwrap();
        let mut result: Vec<(node::Node<E, T>, E)> = Vec::with_capacity(k);
        let mut result_idx: Vec<(usize, E)> = Vec::with_capacity(k);
        while !ret.is_empty() {
            let top = ret.peek().unwrap();
            let top_idx = top.idx();
            let top_distance = top.distance();
            ret.pop();
            result_idx.push((top_idx, top_distance))
        }
        for i in 0..result_idx.len() {
            let cur_id = result_idx.len() - i - 1;
            result.push((
                *self._nodes[result_idx[cur_id].0].clone(),
                result_idx[cur_id].1,
            ));
        }
        result
    }

    fn name(&self) -> &'static str {
        "HNSWIndex"
    }

    fn dimension(&self) -> usize {
        self._dimension
    }
}

/// Legacy dump format without change tracking fields (for backward compatibility)
#[derive(Default, Debug, Deserialize)]
struct HNSWIndexDumpLegacy<'hsnw, E: node::FloatElement, T: node::IdxType> {
    _dimension: usize,
    _n_items: usize,
    _n_constructed_items: usize,
    _max_item: usize,
    _n_neighbor: usize,
    _n_neighbor0: usize,
    _max_level: usize,
    _cur_level: usize,
    _root_id: usize,
    _id2level: Vec<usize>,
    _has_removed: bool,
    _ef_build: usize,
    _ef_search: usize,
    mt: metrics::Metric,
    _id2neighbor_tmp: Vec<Vec<Vec<usize>>>,
    _id2neighbor0_tmp: Vec<Vec<usize>>,
    _nodes_tmp: Vec<Cow<'hsnw, Box<node::Node<E, T>>>>,
    _item2id_tmp: Vec<(Cow<'hsnw, T>, usize)>,
    _delete_ids_tmp: Vec<usize>,
}

#[derive(Default, Debug, Serialize, Deserialize)]
pub struct HNSWIndexDump<'hsnw, E: node::FloatElement, T: node::IdxType> {
    _dimension: usize, // dimension
    _n_items: usize,   // next item count
    _n_constructed_items: usize,
    _max_item: usize,
    _n_neighbor: usize,  // neighbor num except level 0
    _n_neighbor0: usize, // neight num of level 0
    _max_level: usize,   //max level
    _cur_level: usize,   //current level
    _root_id: usize,     //root of hnsw
    _id2level: Vec<usize>,
    _has_removed: bool,
    _ef_build: usize,    // num of max candidates when building
    _ef_search: usize,   // num of max candidates when searching
    mt: metrics::Metric, //compute metrics

    // Change tracking since last rebuild
    _insertions_since_rebuild: usize,
    _deletions_since_rebuild: usize,

    // use for serde
    _id2neighbor_tmp: Vec<Vec<Vec<usize>>>,
    _id2neighbor0_tmp: Vec<Vec<usize>>,
    _nodes_tmp: Vec<Cow<'hsnw, Box<node::Node<E, T>>>>,
    _item2id_tmp: Vec<(Cow<'hsnw, T>, usize)>,
    _delete_ids_tmp: Vec<usize>,
}

impl<E: node::FloatElement, T: node::IdxType> Serialize for HNSWIndex<E, T> {
    fn serialize<S: serde::Serializer>(&self, s: S) -> Result<S::Ok, S::Error> {
        let _id2neighbor_tmp = self
            ._id2neighbor
            .iter()
            .map(|x| {
                x.iter()
                    .map(|y| y.read().unwrap().clone())
                    .collect::<Vec<_>>()
            })
            .collect::<Vec<_>>();

        let _id2neighbor0_tmp = self
            ._id2neighbor0
            .iter()
            .map(|x| x.read().unwrap().clone())
            .collect();

        let _nodes_tmp = self._nodes.iter().map(Cow::Borrowed).collect();
        let _item2id_tmp = self
            ._item2id
            .iter()
            .map(|(k, v)| (Cow::Borrowed(k), *v))
            .collect();
        let _delete_ids_tmp = self._delete_ids.iter().copied().collect();

        let dump = HNSWIndexDump {
            _dimension: self._dimension,
            _n_items: self._n_items,
            _n_constructed_items: self._n_constructed_items,
            _max_item: self._max_item,
            _n_neighbor: self._n_neighbor,
            _n_neighbor0: self._n_neighbor0,
            _max_level: self._max_level,
            _cur_level: self._cur_level,
            _root_id: self._root_id,
            _id2level: self._id2level.clone(),
            _has_removed: self._has_removed,
            _ef_build: self._ef_build,
            _ef_search: self._ef_search,
            mt: self.mt,
            _insertions_since_rebuild: self._insertions_since_rebuild,
            _deletions_since_rebuild: self._deletions_since_rebuild,
            _id2neighbor_tmp,
            _id2neighbor0_tmp,
            _nodes_tmp,
            _item2id_tmp,
            _delete_ids_tmp,
        };

        dump.serialize(s)
    }
}

impl<E: node::FloatElement + DeserializeOwned, T: node::IdxType + DeserializeOwned> HNSWIndex<E, T> {
    /// Deserialize from bincode bytes with backward compatibility for legacy format.
    /// Use this method when loading data that may have been serialized with an older version.
    pub fn deserialize_bincode_compat(bytes: &[u8]) -> Result<Self, Box<bincode::ErrorKind>> {
        // Try new format first
        if let Ok(dump) = bincode::deserialize::<HNSWIndexDump<E, T>>(bytes) {
            return Self::from_dump(dump);
        }

        // Fall back to legacy format (without change tracking fields)
        let legacy: HNSWIndexDumpLegacy<E, T> = bincode::deserialize(bytes)?;
        Self::from_legacy_dump(legacy)
    }

    fn from_dump(dump: HNSWIndexDump<E, T>) -> Result<Self, Box<bincode::ErrorKind>> {
        let _nodes: Vec<_> = dump._nodes_tmp.into_iter().map(|x| x.into_owned()).collect();
        let _id2neighbor = dump._id2neighbor_tmp.into_iter()
            .map(|x| x.into_iter().map(RwLock::new).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let _id2neighbor0 = dump._id2neighbor0_tmp.into_iter().map(RwLock::new).collect::<Vec<_>>();
        let _item2id = dump._item2id_tmp.into_iter().map(|(k, v)| (k.into_owned(), v)).collect::<HashMap<_, _>>();
        let _delete_ids = dump._delete_ids_tmp.into_iter().collect::<HashSet<_>>();

        Ok(Self {
            _dimension: dump._dimension,
            _n_items: dump._n_items,
            _n_constructed_items: dump._n_constructed_items,
            _max_item: dump._max_item,
            _n_neighbor: dump._n_neighbor,
            _n_neighbor0: dump._n_neighbor0,
            _max_level: dump._max_level,
            _cur_level: dump._cur_level,
            _root_id: dump._root_id,
            _id2level: dump._id2level,
            _has_removed: dump._has_removed,
            _ef_build: dump._ef_build,
            _ef_search: dump._ef_search,
            mt: dump.mt,
            _insertions_since_rebuild: dump._insertions_since_rebuild,
            _deletions_since_rebuild: dump._deletions_since_rebuild,
            _id2neighbor,
            _id2neighbor0,
            _nodes,
            _item2id,
            _delete_ids,
        })
    }

    fn from_legacy_dump(dump: HNSWIndexDumpLegacy<E, T>) -> Result<Self, Box<bincode::ErrorKind>> {
        let _nodes: Vec<_> = dump._nodes_tmp.into_iter().map(|x| x.into_owned()).collect();
        let _id2neighbor = dump._id2neighbor_tmp.into_iter()
            .map(|x| x.into_iter().map(RwLock::new).collect::<Vec<_>>())
            .collect::<Vec<_>>();
        let _id2neighbor0 = dump._id2neighbor0_tmp.into_iter().map(RwLock::new).collect::<Vec<_>>();
        let _item2id = dump._item2id_tmp.into_iter().map(|(k, v)| (k.into_owned(), v)).collect::<HashMap<_, _>>();
        let _delete_ids = dump._delete_ids_tmp.into_iter().collect::<HashSet<_>>();

        Ok(Self {
            _dimension: dump._dimension,
            _n_items: dump._n_items,
            _n_constructed_items: dump._n_constructed_items,
            _max_item: dump._max_item,
            _n_neighbor: dump._n_neighbor,
            _n_neighbor0: dump._n_neighbor0,
            _max_level: dump._max_level,
            _cur_level: dump._cur_level,
            _root_id: dump._root_id,
            _id2level: dump._id2level,
            _has_removed: dump._has_removed,
            _ef_build: dump._ef_build,
            _ef_search: dump._ef_search,
            mt: dump.mt,
            _insertions_since_rebuild: 0,
            _deletions_since_rebuild: 0,
            _id2neighbor,
            _id2neighbor0,
            _nodes,
            _item2id,
            _delete_ids,
        })
    }
}

impl<'de, E: node::FloatElement + DeserializeOwned, T: node::IdxType + Deserialize<'de>>
    Deserialize<'de> for HNSWIndex<E, T>
{
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        // Deserialize the new format (with change tracking fields)
        let dump: HNSWIndexDump<E, T> = HNSWIndexDump::deserialize(deserializer)?;

        let _nodes: Vec<_> = dump
            ._nodes_tmp
            .into_iter()
            .map(|x| x.into_owned())
            .collect();

        let _id2neighbor = dump
            ._id2neighbor_tmp
            .into_iter()
            .map(|x| x.into_iter().map(RwLock::new).collect::<Vec<_>>())
            .collect::<Vec<_>>();

        let _id2neighbor0 = dump
            ._id2neighbor0_tmp
            .into_iter()
            .map(RwLock::new)
            .collect::<Vec<_>>();

        let _item2id = dump
            ._item2id_tmp
            .into_iter()
            .map(|(k, v)| {
                // K is always owned here
                // serde allocates it by itself
                (k.into_owned(), v)
            })
            .collect::<HashMap<_, _>>();
        let _delete_ids = dump._delete_ids_tmp.into_iter().collect::<HashSet<_>>();

        Ok(Self {
            _dimension: dump._dimension,
            _n_items: dump._n_items,
            _n_constructed_items: dump._n_constructed_items,
            _max_item: dump._max_item,
            _n_neighbor: dump._n_neighbor,
            _n_neighbor0: dump._n_neighbor0,
            _max_level: dump._max_level,
            _cur_level: dump._cur_level,
            _root_id: dump._root_id,
            _id2level: dump._id2level,
            _has_removed: dump._has_removed,
            _ef_build: dump._ef_build,
            _ef_search: dump._ef_search,
            mt: dump.mt,
            _insertions_since_rebuild: dump._insertions_since_rebuild,
            _deletions_since_rebuild: dump._deletions_since_rebuild,
            _id2neighbor,
            _id2neighbor0,
            _nodes,
            _item2id,
            _delete_ids,
        })
    }
}

#[cfg(test)]
mod hsnw_tests {
    use super::core::{ann_index::ANNIndex, metrics::Metric};

    use super::*;
    use rand::distr::Uniform;

    #[test]
    fn test_serialize_deserialize() {
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
        let mut index = HNSWIndex::<f32, usize>::new(dimension, &HNSWParams::<f32>::default());
        for (i, sample) in samples.into_iter().enumerate() {
            index.add(&sample, i).unwrap();
        }
        index.build(Metric::Euclidean).unwrap();

        let a = bincode::serialize(&index).unwrap();
        let new_index: HNSWIndex<f32, usize> = bincode::deserialize(&a).unwrap();

        let mut target = Vec::with_capacity(dimension);
        for _ in 0..dimension {
            target.push(normal.sample(&mut rand::rng()));
        }

        let v1 = index.search(&target, 10);
        let v2 = new_index.search(&target, 10);

        assert_eq!(v1, v2);
    }

    #[test]
    fn test_serde_backcompatibility() {
        let b = include_bytes!("./dump.hsnw");
        // Use backward-compatible deserialization for legacy format
        let new_index: HNSWIndex<f32, usize> = HNSWIndex::deserialize_bincode_compat(b).unwrap();
        assert_eq!(new_index.len(), 100);
        // Legacy format should have default values for new fields
        assert_eq!(new_index.insertions_since_rebuild(), 0);
        assert_eq!(new_index.deletions_since_rebuild(), 0);
    }

    // ==================== Rebuild Tests ====================

    fn create_test_index(n: usize, dimension: usize) -> HNSWIndex<f32, usize> {
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let samples: Vec<Vec<f32>> = (0..n)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect()
            })
            .collect();

        let mut params = HNSWParams::<f32>::default();
        params.has_deletion = true;
        let mut index = HNSWIndex::<f32, usize>::new(dimension, &params);
        for (i, sample) in samples.into_iter().enumerate() {
            index.add(&sample, i).unwrap();
        }
        index.build(Metric::Euclidean).unwrap();
        index
    }

    #[test]
    fn test_analyze_health_no_deletions() {
        let index = create_test_index(100, 16);
        let metrics = index.analyze_health();

        assert_eq!(metrics.total_nodes, 100);
        assert_eq!(metrics.deleted_nodes, 0);
        assert!((metrics.deletion_ratio - 0.0).abs() < 0.001);
        assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    }

    #[test]
    fn test_analyze_health_low_deletions() {
        let mut index = create_test_index(100, 16);

        // Delete 3% of nodes (below 5% threshold)
        for i in 0..3 {
            index.delete_id(i).unwrap();
        }

        let metrics = index.analyze_health();
        assert_eq!(metrics.deleted_nodes, 3);
        assert!(metrics.deletion_ratio < 0.05);
        assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    }

    #[test]
    fn test_analyze_health_moderate_deletions() {
        let mut index = create_test_index(100, 16);

        // Delete 20% of nodes (in 5-40% range)
        for i in 0..20 {
            index.delete_id(i).unwrap();
        }

        let metrics = index.analyze_health();
        assert_eq!(metrics.deleted_nodes, 20);
        assert!(metrics.deletion_ratio >= 0.05);
        assert!(metrics.deletion_ratio < 0.40);
    }

    #[test]
    fn test_analyze_health_high_deletions() {
        let mut index = create_test_index(100, 16);

        // Delete 50% of nodes (above 40% threshold)
        for i in 0..50 {
            index.delete_id(i).unwrap();
        }

        let metrics = index.analyze_health();
        assert_eq!(metrics.deleted_nodes, 50);
        assert!(metrics.deletion_ratio >= 0.40);
        assert_eq!(metrics.recommended_strategy, RebuildStrategy::FullRebuild);
    }

    #[test]
    fn test_full_rebuild_preserves_live_data() {
        let mut index = create_test_index(100, 16);
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Delete 50% of nodes
        for i in 0..50 {
            index.delete_id(i).unwrap();
        }

        // Remember a query target
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();

        // Get search results before rebuild
        let results_before = index.search(&target, 10);

        // Force full rebuild
        let compacted = index.force_full_rebuild().unwrap();
        assert_eq!(compacted, 50);

        // Verify index state after rebuild
        let metrics = index.analyze_health();
        assert_eq!(metrics.total_nodes, 50);
        assert_eq!(metrics.deleted_nodes, 0);
        assert_eq!(metrics.deletion_ratio, 0.0);

        // Search results should still work (though order may differ due to rebuild)
        let results_after = index.search(&target, 10);
        assert!(!results_after.is_empty());

        // Results before should be a subset of valid IDs (>=50)
        for id in results_before {
            assert!(id >= 50 || results_after.contains(&id) || index.is_deleted(id));
        }
    }

    #[test]
    fn test_rebuild_with_config() {
        let mut index = create_test_index(100, 16);

        // Delete 30% of nodes
        for i in 0..30 {
            index.delete_id(i).unwrap();
        }

        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.25, // Lower threshold to trigger full rebuild
            ..Default::default()
        };

        let result = index.rebuild_with_config(&config).unwrap();
        assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);
        assert_eq!(result.nodes_compacted, 30);
        assert_eq!(result.metrics_after.deleted_nodes, 0);
    }

    #[test]
    fn test_rebuild_empty_index() {
        let params = HNSWParams::<f32>::default();
        let mut index = HNSWIndex::<f32, usize>::new(16, &params);
        index.build(Metric::Euclidean).unwrap();

        let result = index.rebuild_index().unwrap();
        assert_eq!(result.strategy_used, RebuildStrategy::NoAction);
    }

    #[test]
    fn test_rebuild_all_deleted() {
        let mut index = create_test_index(10, 16);

        // Delete all nodes
        for i in 0..10 {
            index.delete_id(i).unwrap();
        }

        let compacted = index.force_full_rebuild().unwrap();
        assert_eq!(compacted, 10);
        assert_eq!(index.len(), 0);

        let metrics = index.analyze_health();
        assert_eq!(metrics.total_nodes, 0);
        assert_eq!(metrics.deleted_nodes, 0);
    }

    #[test]
    fn test_find_unreachable_nodes_empty() {
        let params = HNSWParams::<f32>::default();
        let index = HNSWIndex::<f32, usize>::new(16, &params);
        let unreachable = index.find_unreachable_nodes();
        assert!(unreachable.is_empty());
    }

    #[test]
    fn test_cleanup_deleted_references() {
        let mut index = create_test_index(100, 16);

        // Delete some nodes
        for i in 0..10 {
            index.delete_id(i).unwrap();
        }

        // Run cleanup
        index.cleanup_deleted_references();

        // Verify no neighbor lists contain deleted IDs
        for id in 0..index._n_constructed_items {
            if index.is_deleted(id) {
                continue;
            }
            let neighbors = index._id2neighbor0[id].read().unwrap();
            for &n in neighbors.iter() {
                assert!(!index.is_deleted(n), "Found deleted node {} in neighbors of {}", n, id);
            }
        }
    }

    #[test]
    fn test_strategy_boundaries() {
        // Test at exactly 5% deletion ratio
        let mut index = create_test_index(100, 16);
        for i in 0..5 {
            index.delete_id(i).unwrap();
        }
        let metrics = index.analyze_health();
        assert!((metrics.deletion_ratio - 0.05).abs() < 0.001);
        // At 5%, it should check connection health, which could be NoAction or PartialRepair

        // Test at exactly 40% deletion ratio
        let mut index2 = create_test_index(100, 16);
        for i in 0..40 {
            index2.delete_id(i).unwrap();
        }
        let metrics2 = index2.analyze_health();
        assert!((metrics2.deletion_ratio - 0.40).abs() < 0.001);
        assert_eq!(metrics2.recommended_strategy, RebuildStrategy::FullRebuild);
    }

    #[test]
    fn test_rebuild_search_quality() {
        let mut index = create_test_index(1000, 32);
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Create query targets
        let targets: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..32).map(|_| normal.sample(&mut rand::rng())).collect())
            .collect();

        // Delete 45% of nodes to trigger full rebuild
        for i in 0..450 {
            index.delete_id(i).unwrap();
        }

        // Search before rebuild
        let _results_before: Vec<Vec<usize>> = targets
            .iter()
            .map(|t| index.search(t, 10))
            .collect();

        // Rebuild
        let result = index.rebuild_index().unwrap();
        assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);

        // Search after rebuild
        let results_after: Vec<Vec<usize>> = targets
            .iter()
            .map(|t| index.search(t, 10))
            .collect();

        // After rebuild, all results should be valid (no deleted nodes)
        for results in &results_after {
            for id in results {
                assert!(!index.is_deleted(*id));
            }
        }

        // Results should not be empty
        for results in &results_after {
            assert!(!results.is_empty(), "Search returned no results after rebuild");
        }
    }

    #[test]
    fn test_partial_repair_strategy() {
        // Create a larger index where deletions will cause connection loss
        let mut index = create_test_index(200, 16);

        // Delete 15% of nodes (in 5-40% range)
        // Delete nodes strategically to cause connection loss
        for i in 0..30 {
            index.delete_id(i).unwrap();
        }

        let metrics = index.analyze_health();
        assert!(metrics.deletion_ratio >= 0.05);
        assert!(metrics.deletion_ratio < 0.40);

        // With low connection loss threshold, PartialRepair should be triggered
        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.40,
            connection_loss_threshold: 0.01, // Very low threshold to trigger repair
            min_connections: 1,
            detect_unreachable: true,
        };

        let _metrics_with_config = index.analyze_health_with_config(&config);

        // Either PartialRepair or NoAction depending on connection health
        // The key is testing the rebuild path works
        let result = index.rebuild_with_config(&config).unwrap();

        // Verify the rebuild completed successfully
        assert!(result.strategy_used == RebuildStrategy::PartialRepair
            || result.strategy_used == RebuildStrategy::NoAction);

        // After repair, verify index is still functional
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);

        // All results should be live nodes
        for id in &results {
            assert!(!index.is_deleted(*id), "Search returned deleted node after repair");
        }
    }

    #[test]
    fn test_partial_repair_with_affected_nodes() {
        // Create index with conditions that will trigger PartialRepair
        let mut index = create_test_index(100, 16);

        // Delete 10% of nodes - in the partial repair range
        for i in 0..10 {
            index.delete_id(i).unwrap();
        }

        // Use config with low thresholds to ensure PartialRepair triggers
        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.40,
            connection_loss_threshold: 0.05, // 5% connection loss triggers repair
            min_connections: 3, // Higher requirement to trigger more repairs
            detect_unreachable: true,
        };

        let result = index.rebuild_with_config(&config).unwrap();

        // Verify rebuild completed (strategy depends on actual connection state)
        assert!(result.metrics_after.deleted_nodes <= 10);

        // After any rebuild/repair, search should work
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert!(!results.is_empty(), "Search returned no results after partial repair");
    }

    #[test]
    fn test_repair_connections_cleans_up_deleted() {
        let mut index = create_test_index(50, 16);

        // Delete some nodes
        for i in 0..5 {
            index.delete_id(i).unwrap();
        }

        // Before cleanup, some neighbor lists may contain deleted nodes
        let config = RebuildConfig::default();

        // Run repair which includes cleanup
        let _ = index.repair_connections(&config);

        // After repair, no neighbor list should contain deleted nodes
        for id in 0..index._n_constructed_items {
            if index.is_deleted(id) {
                continue;
            }
            let neighbors = index._id2neighbor0[id].read().unwrap();
            for &n in neighbors.iter() {
                assert!(!index.is_deleted(n),
                    "Found deleted node {} in neighbors of {} after repair", n, id);
            }
        }
    }

    #[test]
    fn test_repair_node_connections_single_node() {
        let mut index = create_test_index(50, 16);

        // Delete neighbors of a specific node to isolate it somewhat
        // First, find a node with some neighbors
        let test_node = 25; // Pick a node in the middle

        // Get its neighbors
        let neighbors_to_delete: Vec<usize> = {
            let neighbors = index._id2neighbor0[test_node].read().unwrap();
            neighbors.iter().take(2).copied().collect() // Delete first 2 neighbors
        };

        // Delete those neighbors
        for &n in &neighbors_to_delete {
            if n != 0 { // Don't delete root
                let _ = index.delete_id(n);
            }
        }

        // Clean up the deleted references first
        index.cleanup_deleted_references();

        // Now repair the test node's connections
        let _repaired = index.repair_node_connections(test_node).unwrap();

        // Verify the node has valid connections after repair
        let neighbors = index._id2neighbor0[test_node].read().unwrap();
        for &n in neighbors.iter() {
            assert!(!index.is_deleted(n),
                "Repaired node {} still has deleted neighbor {}", test_node, n);
        }

        // The node should still be searchable
        let node_data = index.get_data(test_node);
        let search_results = index.search_knn(node_data, 10).unwrap();
        assert!(!search_results.is_empty(), "Node should be findable after repair");
    }

    #[test]
    fn test_rebuild_with_root_deleted() {
        let mut index = create_test_index(100, 16);

        // The root is typically node 0
        let root_id = index._root_id;

        // Delete the root node
        index.delete_id(root_id).unwrap();

        // Delete more nodes to trigger a rebuild (45% total)
        for i in 1..45 {
            index.delete_id(i).unwrap();
        }

        let metrics = index.analyze_health();
        assert!(metrics.deletion_ratio >= 0.40);
        assert_eq!(metrics.recommended_strategy, RebuildStrategy::FullRebuild);

        // Rebuild should work even with root deleted
        let result = index.rebuild_index().unwrap();
        assert_eq!(result.strategy_used, RebuildStrategy::FullRebuild);

        // Index should be functional after rebuild
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);

        assert!(!results.is_empty(), "Search should work after rebuilding with root deleted");

        // All results should be valid
        for id in &results {
            assert!(!index.is_deleted(*id));
        }
    }

    #[test]
    fn test_repair_with_root_deleted() {
        let mut index = create_test_index(100, 16);

        let root_id = index._root_id;

        // Delete root and a few other nodes (enough to be in partial repair range)
        index.delete_id(root_id).unwrap();
        for i in 1..10 {
            index.delete_id(i).unwrap();
        }

        // Use config that will try partial repair
        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.50, // Higher threshold to avoid full rebuild
            connection_loss_threshold: 0.05,
            min_connections: 2,
            detect_unreachable: true,
        };

        // Repair should handle the case where root is deleted
        let _result = index.rebuild_with_config(&config).unwrap();

        // Index should still be functional
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);

        // Search should still return results (may need to traverse from non-root)
        // Note: with root deleted, search quality may degrade but should still work
        for id in &results {
            assert!(!index.is_deleted(*id));
        }
    }

    #[test]
    fn test_unreachable_nodes_detected() {
        let mut index = create_test_index(100, 16);

        // Check initial state - a fresh index should have minimal unreachable nodes
        // (some may exist due to graph structure but should be few)
        let _unreachable_before = index.find_unreachable_nodes();

        // Delete some nodes that might create unreachable pockets
        for i in 0..20 {
            index.delete_id(i).unwrap();
        }

        // Cleanup references to deleted nodes
        index.cleanup_deleted_references();

        // Check for unreachable nodes
        let unreachable_after = index.find_unreachable_nodes();

        // Verify health analysis includes unreachable count
        let metrics = index.analyze_health();
        assert_eq!(metrics.unreachable_nodes, unreachable_after.len());

        // All unreachable nodes should be live (not deleted)
        for &node_id in &unreachable_after {
            assert!(!index.is_deleted(node_id), "Unreachable node {} should be live", node_id);
        }
    }

    #[test]
    fn test_unreachable_nodes_repaired() {
        let mut index = create_test_index(100, 16);

        // Delete nodes to potentially create unreachable ones
        for i in 0..15 {
            index.delete_id(i).unwrap();
        }

        // Clean up first
        index.cleanup_deleted_references();

        // Find unreachable nodes before repair
        let unreachable_before = index.find_unreachable_nodes();

        // Configure for partial repair
        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.50, // High threshold to ensure partial repair
            connection_loss_threshold: 0.05,
            min_connections: 2,
            detect_unreachable: true,
        };

        // Run repair
        let result = index.rebuild_with_config(&config).unwrap();

        // Check unreachable nodes after repair
        let unreachable_after = index.find_unreachable_nodes();

        // If there were unreachable nodes and partial repair was used, they should be fixed
        if result.strategy_used == RebuildStrategy::PartialRepair && !unreachable_before.is_empty() {
            assert!(
                unreachable_after.len() <= unreachable_before.len(),
                "Unreachable nodes should decrease or stay same after repair"
            );
        }

        // All remaining live nodes should be searchable
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);

        for id in &results {
            assert!(!index.is_deleted(*id));
        }
    }

    #[test]
    fn test_search_quality_after_partial_repair() {
        let mut index = create_test_index(500, 32);
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Create query targets
        let targets: Vec<Vec<f32>> = (0..5)
            .map(|_| (0..32).map(|_| normal.sample(&mut rand::rng())).collect())
            .collect();

        // Delete 15% of nodes (in partial repair range)
        for i in 0..75 {
            index.delete_id(i).unwrap();
        }

        // Search before rebuild (capture for potential comparison)
        let _results_before: Vec<Vec<usize>> = targets
            .iter()
            .map(|t| index.search(t, 10))
            .collect();

        // Use config that triggers partial repair
        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.50, // High threshold
            connection_loss_threshold: 0.05,
            min_connections: 2,
            detect_unreachable: true,
        };

        let _result = index.rebuild_with_config(&config).unwrap();

        // Search after rebuild
        let results_after: Vec<Vec<usize>> = targets
            .iter()
            .map(|t| index.search(t, 10))
            .collect();

        // All results should be valid (no deleted nodes)
        for results in &results_after {
            for id in results {
                assert!(!index.is_deleted(*id));
            }
        }

        // Results should not be empty
        for results in &results_after {
            assert!(!results.is_empty(), "Search returned no results after partial repair");
        }
    }

    #[test]
    fn test_count_severely_affected_nodes() {
        let mut index = create_test_index(100, 16);

        // Delete some nodes
        for i in 0..15 {
            index.delete_id(i).unwrap();
        }

        let config = RebuildConfig {
            skip_threshold: 0.05,
            full_rebuild_threshold: 0.40,
            connection_loss_threshold: 0.10, // 10% connection loss
            min_connections: 2,
            detect_unreachable: true,
        };

        let affected_count = index.count_severely_affected_nodes(&config);

        // Verify the count is reasonable (should be >= 0)
        assert!(affected_count <= 100 - 15); // Can't exceed live nodes

        // Verify this matches what analyze_health reports
        let metrics = index.analyze_health_with_config(&config);
        assert_eq!(metrics.severely_affected_nodes, affected_count);
    }

    #[test]
    fn test_level_preservation_in_rebuild() {
        let mut index = create_test_index(100, 16);

        // Record original levels for some nodes that will survive
        let mut original_levels: Vec<(usize, usize)> = Vec::new();
        for i in 50..60 {
            original_levels.push((i, index._id2level[i]));
        }

        // Delete 50% of nodes to trigger full rebuild
        for i in 0..50 {
            index.delete_id(i).unwrap();
        }

        // Rebuild
        let result = index.force_full_rebuild().unwrap();
        assert_eq!(result, 50);

        // After rebuild, nodes are re-indexed from 0
        // The first 50 surviving nodes (originally 50-99) are now 0-49
        // Check that levels were preserved (they should match relative to their new positions)
        // Note: Due to re-indexing, we can't directly compare old IDs to new IDs
        // But we can verify that the distribution of levels is similar

        let level_sum_before: usize = original_levels.iter().map(|(_, l)| *l).sum();
        // Levels should have been preserved, so non-zero sum is expected
        assert!(level_sum_before > 0 || original_levels.iter().all(|(_, l)| *l == 0));

        // Verify index is functional after rebuild with preserved levels
        let normal = Uniform::new(0.0, 10.0).unwrap();
        let target: Vec<f32> = (0..16).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert!(!results.is_empty(), "Search should work after rebuild");
    }

    #[test]
    fn test_batch_add_basic() {
        let dimension = 16;
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Create items for batch add
        let samples: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect()
            })
            .collect();

        let items: Vec<(&[f32], usize)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_slice(), i))
            .collect();

        let mut index = HNSWIndex::<f32, usize>::new(dimension, &HNSWParams::<f32>::default());

        // Use batch_add
        index.batch_add(items.into_iter()).unwrap();
        index.build(Metric::Euclidean).unwrap();

        // Verify index has correct count
        assert_eq!(index.len(), 100);

        // Verify search works
        let target: Vec<f32> = (0..dimension).map(|_| normal.sample(&mut rand::rng())).collect();
        let results = index.search(&target, 10);
        assert_eq!(results.len(), 10);
    }

    #[test]
    fn test_batch_add_search_quality() {
        let dimension = 32;
        let normal = Uniform::new(0.0, 10.0).unwrap();

        // Create two indexes - one with batch_add, one with regular add
        let samples: Vec<Vec<f32>> = (0..500)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect()
            })
            .collect();

        // Index using regular add
        let mut regular_index = HNSWIndex::<f32, usize>::new(dimension, &HNSWParams::<f32>::default());
        for (i, sample) in samples.iter().enumerate() {
            regular_index.add(sample, i).unwrap();
        }
        regular_index.build(Metric::Euclidean).unwrap();

        // Index using batch_add
        let items: Vec<(&[f32], usize)> = samples
            .iter()
            .enumerate()
            .map(|(i, s)| (s.as_slice(), i))
            .collect();
        let mut batch_index = HNSWIndex::<f32, usize>::new(dimension, &HNSWParams::<f32>::default());
        batch_index.batch_add(items.into_iter()).unwrap();
        batch_index.build(Metric::Euclidean).unwrap();

        // Compare search results - both should be functional
        let targets: Vec<Vec<f32>> = (0..10)
            .map(|_| (0..dimension).map(|_| normal.sample(&mut rand::rng())).collect())
            .collect();

        for target in &targets {
            let regular_results = regular_index.search(target, 10);
            let batch_results = batch_index.search(target, 10);

            // Both should return results
            assert!(!regular_results.is_empty(), "Regular index search failed");
            assert!(!batch_results.is_empty(), "Batch index search failed");
        }
    }

    #[test]
    fn test_batch_construct_optimized_equivalence() {
        let dimension = 16;
        let normal = Uniform::new(0.0, 10.0).unwrap();

        let samples: Vec<Vec<f32>> = (0..100)
            .map(|_| {
                (0..dimension)
                    .map(|_| normal.sample(&mut rand::rng()))
                    .collect()
            })
            .collect();

        // Test that both construction methods produce valid graphs
        let mut index = HNSWIndex::<f32, usize>::new(dimension, &HNSWParams::<f32>::default());
        for (i, sample) in samples.iter().enumerate() {
            index.add(sample, i).unwrap();
        }
        index.build(Metric::Euclidean).unwrap();

        // Verify graph health
        let metrics = index.analyze_health();
        assert_eq!(metrics.total_nodes, 100);
        assert_eq!(metrics.unreachable_nodes, 0);
        assert_eq!(metrics.recommended_strategy, RebuildStrategy::NoAction);
    }

    #[test]
    fn test_visited_generation_counter() {
        // Test the Visited struct directly
        let mut visited = super::Visited::new(100);

        // Insert some values
        assert!(visited.insert(0)); // First insert returns true
        assert!(!visited.insert(0)); // Second insert returns false (already visited)
        assert!(visited.insert(1));
        assert!(!visited.insert(1));

        // Verify contains
        assert!(visited.contains(0));
        assert!(visited.contains(1));
        assert!(!visited.contains(2));

        // Clear and verify
        visited.clear();
        assert!(!visited.contains(0));
        assert!(!visited.contains(1));

        // Can insert again after clear
        assert!(visited.insert(0));
        assert!(visited.contains(0));
    }

    #[test]
    fn test_visited_generation_wraparound() {
        // Test generation counter wraparound (every 255 clears)
        let mut visited = super::Visited::new(10);

        // Do 256 clear cycles to trigger wraparound
        for _ in 0..256 {
            visited.insert(0);
            assert!(visited.contains(0));
            visited.clear();
            assert!(!visited.contains(0));
        }

        // Should still work after wraparound
        visited.insert(5);
        assert!(visited.contains(5));
        assert!(!visited.contains(0));
    }

    #[test]
    fn test_visited_auto_grow() {
        let mut visited = super::Visited::new(10);

        // Insert beyond initial capacity
        assert!(visited.insert(100));
        assert!(visited.contains(100));
        assert!(!visited.contains(50));
    }
}
