use std::collections::HashMap;
use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::fs;
use tokio::sync::RwLock;

use crate::compute::enforce_active_expert_bound;
use crate::config::{fnv1a64, RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet};
use crate::error::{RuntimeError, RuntimeResult};
use crate::routing::{ExpertKey, Route};

/// Serialisable representation of a [`Route`].
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct AsyncRoute {
    pub tier: u16,
    pub group: u32,
    pub expert: u32,
    pub weight: f32,
    pub priority: u16,
}

impl From<Route> for AsyncRoute {
    fn from(route: Route) -> Self {
        Self {
            tier: route.expert.tier.0,
            group: route.expert.group,
            expert: route.expert.expert,
            weight: route.weight,
            priority: route.priority,
        }
    }
}

impl From<AsyncRoute> for Route {
    fn from(route: AsyncRoute) -> Self {
        Self {
            expert: ExpertKey {
                tier: TierId(route.tier),
                group: route.group,
                expert: route.expert,
            },
            weight: route.weight,
            priority: route.priority,
        }
    }
}

/// Persistable routing table entry keyed by (layer_index, token_index).
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTableEntry {
    pub layer_index: u32,
    pub token_index: u32,
    pub routes: Vec<AsyncRoute>,
}

/// Full routing table snapshot for persistence.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RoutingTableSnapshot {
    pub seed: u128,
    pub entries: Vec<RoutingTableEntry>,
    pub tier_catalog: Vec<TierConfig>,
}

/// Async routing table that supports concurrent reads and writes with file-based persistence.
pub struct AsyncRoutingTable {
    inner: RwLock<HashMap<(u32, u32), Vec<Route>>>,
    seed: RoutingSeed,
    tier_catalog: Vec<TierConfig>,
    quantization_scale: f32,
}

impl AsyncRoutingTable {
    pub fn new(seed: RoutingSeed, tier_catalog: Vec<TierConfig>) -> Self {
        Self {
            inner: RwLock::new(HashMap::new()),
            seed,
            tier_catalog,
            quantization_scale: 1_000_000.0,
        }
    }

    pub fn with_quantization_scale(mut self, quantization_scale: f32) -> Self {
        self.quantization_scale = quantization_scale.max(1.0);
        self
    }

    /// Compute routes for a token and store them in the table.
    pub async fn compute_and_store(
        &self,
        token_state: &[f32],
        layer_index: u32,
        token_index: u32,
        tiers: &TierSet,
        cfg: RoutingConfig,
    ) -> RuntimeResult<Vec<Route>> {
        let routes = self.compute_routes(token_state, layer_index, token_index, tiers, cfg)?;
        let key = (layer_index, token_index);
        {
            let mut map = self.inner.write().await;
            map.insert(key, routes.clone());
        }
        Ok(routes)
    }

    /// Retrieve cached routes for a (layer, token) pair.
    pub async fn get_routes(&self, layer_index: u32, token_index: u32) -> Option<Vec<Route>> {
        let map = self.inner.read().await;
        map.get(&(layer_index, token_index)).cloned()
    }

    /// Return the number of cached entries.
    pub async fn len(&self) -> usize {
        let map = self.inner.read().await;
        map.len()
    }

    /// Check if the routing table is empty.
    pub async fn is_empty(&self) -> bool {
        let map = self.inner.read().await;
        map.is_empty()
    }

    /// Clear all cached routing entries.
    pub async fn clear(&self) {
        let mut map = self.inner.write().await;
        map.clear();
    }

    /// Persist the current routing table to a JSON file.
    pub async fn save_to_file(&self, path: impl AsRef<Path>) -> RuntimeResult<()> {
        let map = self.inner.read().await;
        let snapshot = RoutingTableSnapshot {
            seed: self.seed.base,
            entries: map
                .iter()
                .map(|(&(layer_index, token_index), routes)| RoutingTableEntry {
                    layer_index,
                    token_index,
                    routes: routes.iter().copied().map(AsyncRoute::from).collect(),
                })
                .collect(),
            tier_catalog: self.tier_catalog.clone(),
        };
        drop(map);

        let json = serde_json::to_string_pretty(&snapshot).map_err(|e| {
            RuntimeError::serialization_error(format!("failed to serialize routing table: {e}"))
        })?;

        fs::write(path, json)
            .await
            .map_err(|e| RuntimeError::io_error(format!("failed to write routing table: {e}")))
    }

    /// Load a routing table snapshot from a JSON file and restore cached entries.
    pub async fn load_from_file(path: impl AsRef<Path>) -> RuntimeResult<Self> {
        let path_ref = path.as_ref();
        let text = fs::read_to_string(path_ref)
            .await
            .map_err(|e| {
                RuntimeError::io_error(format!(
                    "failed to read routing table file {}: {}",
                    path_ref.display(),
                    e
                ))
            })?;

        let snapshot: RoutingTableSnapshot = serde_json::from_str(&text).map_err(|e| {
            RuntimeError::serialization_error(format!(
                "failed to deserialize routing table: {e}"
            ))
        })?;

        let mut map = HashMap::new();
        for entry in &snapshot.entries {
            map.insert(
                (entry.layer_index, entry.token_index),
                entry
                    .routes
                    .iter()
                    .cloned()
                    .map(Route::from)
                    .collect(),
            );
        }

        Ok(Self {
            inner: RwLock::new(map),
            seed: RoutingSeed::new(snapshot.seed),
            tier_catalog: snapshot.tier_catalog,
            quantization_scale: 1_000_000.0,
        })
    }

    /// Internal route computation (same logic as [`crate::routing::DeterministicRouter`]).
    fn compute_routes(
        &self,
        token_state: &[f32],
        layer_index: u32,
        token_index: u32,
        tiers: &TierSet,
        cfg: RoutingConfig,
    ) -> RuntimeResult<Vec<Route>> {
        cfg.validate()?;

        let allowed_tiers = self.allowed_tiers(tiers);
        if allowed_tiers.is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }

        let layer_seed = self.seed.layer_seed(layer_index);
        let token_seed = self.seed.token_seed(layer_index, token_index);

        let tier_scores = self.tier_scores(token_state, layer_seed, &allowed_tiers);
        let tier_indices = crate::routing::stable_top_k(
            &tier_scores,
            cfg.k_tier.min(allowed_tiers.len()),
            token_seed,
            self.quantization_scale,
        );

        let mut routed = Vec::new();
        let mut raw_scores = Vec::new();

        for tier_idx in tier_indices {
            let tier_id = allowed_tiers[tier_idx];
            let tier_cfg = self
                .find_tier_config(tier_id)
                .ok_or(RuntimeError::UnknownTier(tier_id.0))?;
            let groups = tier_cfg.groups as usize;
            if groups == 0 {
                return Err(RuntimeError::invalid_routing_config(
                    "tier groups must be greater than zero",
                ));
            }

            let group_scores = self.group_scores(token_state, token_seed, tier_id, groups);
            let group_indices = crate::routing::stable_top_k(
                &group_scores,
                cfg.k_group.min(groups),
                token_seed ^ tier_id.0 as u64,
                self.quantization_scale,
            );

            for group_idx in group_indices {
                let experts = tier_cfg.experts_per_group as usize;
                if experts == 0 {
                    return Err(RuntimeError::invalid_routing_config(
                        "experts_per_group must be greater than zero",
                    ));
                }

                let expert_scores = self.expert_scores(
                    token_state,
                    token_seed,
                    tier_id,
                    group_idx as u32,
                    experts,
                );
                let expert_indices = crate::routing::stable_top_k(
                    &expert_scores,
                    cfg.k_expert.min(experts),
                    token_seed ^ ((tier_id.0 as u64) << 16) ^ group_idx as u64,
                    self.quantization_scale,
                );

                for expert_idx in expert_indices {
                    raw_scores.push(expert_scores[expert_idx]);
                    routed.push(Route {
                        expert: ExpertKey {
                            tier: tier_id,
                            group: group_idx as u32,
                            expert: expert_idx as u32,
                        },
                        weight: 0.0,
                        priority: routed.len() as u16,
                    });
                }
            }
        }

        enforce_active_expert_bound(&routed, cfg)?;

        if routed.is_empty() {
            return Ok(routed);
        }

        let max_score = raw_scores
            .iter()
            .fold(f32::NEG_INFINITY, |acc, score| acc.max(*score));
        let exp_scores: Vec<f32> = raw_scores
            .iter()
            .map(|score| (*score - max_score).exp())
            .collect();
        let normalizer: f32 = exp_scores.iter().sum();

        for (route, score) in routed.iter_mut().zip(exp_scores.iter()) {
            route.weight = *score / normalizer;
        }

        Ok(routed)
    }

    fn find_tier_config(&self, id: TierId) -> Option<&TierConfig> {
        self.tier_catalog.iter().find(|tier| tier.id == id)
    }

    fn allowed_tiers(&self, tiers: &TierSet) -> Vec<TierId> {
        let mut selected = tiers.resolve(&self.tier_catalog);
        selected.sort_unstable();
        selected
    }

    fn tier_scores(&self, token_state: &[f32], layer_seed: u64, tiers: &[TierId]) -> Vec<f32> {
        tiers
            .iter()
            .map(|tier| {
                projection_score(token_state, layer_seed, 0x01, tier.0 as u64)
            })
            .collect()
    }

    fn group_scores(
        &self,
        token_state: &[f32],
        token_seed: u64,
        tier: TierId,
        groups: usize,
    ) -> Vec<f32> {
        let mut scores = Vec::with_capacity(groups);
        for group in 0..groups {
            scores.push(projection_score(
                token_state,
                token_seed,
                tier.0 as u64,
                group as u64,
            ));
        }
        scores
    }

    fn expert_scores(
        &self,
        token_state: &[f32],
        token_seed: u64,
        tier: TierId,
        group: u32,
        experts: usize,
    ) -> Vec<f32> {
        let mut scores = Vec::with_capacity(experts);
        for expert in 0..experts {
            scores.push(projection_score(
                token_state,
                token_seed,
                (tier.0 as u64) << 32 | u64::from(group),
                expert as u64,
            ));
        }
        scores
    }
}

fn projection_score(token_state: &[f32], seed: u64, key_a: u64, key_b: u64) -> f32 {
    if token_state.is_empty() {
        return 0.0;
    }

    let mut score = 0.0_f32;
    for (idx, value) in token_state.iter().enumerate() {
        let hash = seeded_index_hash(seed, key_a ^ key_b ^ idx as u64);
        let weight = hash_to_signed_unit(hash);
        score += *value * weight;
    }
    score
}

fn seeded_index_hash(seed: u64, index: u64) -> u64 {
    let mut payload = [0u8; 16];
    payload[..8].copy_from_slice(&seed.to_le_bytes());
    payload[8..16].copy_from_slice(&index.to_le_bytes());
    fnv1a64(0xcbf29ce484222325, &payload)
}

fn hash_to_signed_unit(hash: u64) -> f32 {
    ((hash as f64 / u64::MAX as f64) * 2.0 - 1.0) as f32
}

#[cfg(test)]
mod tests {
    use crate::config::{Placement, RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet};

    use super::{AsyncRoute, AsyncRoutingTable, RoutingTableSnapshot};

    fn sample_tier_catalog() -> Vec<TierConfig> {
        vec![
            TierConfig {
                id: TierId(1),
                groups: 4,
                experts_per_group: 4,
                placement: Placement::Hot,
            },
            TierConfig {
                id: TierId(10),
                groups: 4,
                experts_per_group: 4,
                placement: Placement::Warm,
            },
        ]
    }

    fn sample_routing_table() -> AsyncRoutingTable {
        AsyncRoutingTable::new(RoutingSeed::new(99), sample_tier_catalog())
    }

    #[tokio::test]
    async fn async_compute_and_store_produces_routes() {
        let table = sample_routing_table();
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 2,
            k_expert: 2,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1), TierId(10)],
            cumulative: false,
        };
        let token = vec![0.1, -0.2, 0.9, 0.05, -0.7];

        let routes = table
            .compute_and_store(&token, 3, 17, &tiers, cfg)
            .await
            .expect("compute_and_store should succeed");

        assert_eq!(routes.len(), cfg.max_active_experts());
        assert_eq!(table.len().await, 1);
    }

    #[tokio::test]
    async fn async_get_routes_returns_cached_value() {
        let table = sample_routing_table();
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 1,
            k_expert: 1,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1)],
            cumulative: false,
        };
        let token = vec![1.0, 2.0, 3.0];

        let first = table
            .compute_and_store(&token, 0, 0, &tiers, cfg)
            .await
            .expect("first compute should succeed");
        let cached = table
            .get_routes(0, 0)
            .await
            .expect("get_routes should return cached value");
        assert_eq!(first, cached);
    }

    #[tokio::test]
    async fn async_clear_removes_all_entries() {
        let table = sample_routing_table();
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 1,
            k_expert: 1,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1)],
            cumulative: false,
        };

        table
            .compute_and_store(&[1.0], 0, 0, &tiers, cfg)
            .await
            .expect("compute should succeed");
        table
            .compute_and_store(&[1.0], 1, 0, &tiers, cfg)
            .await
            .expect("compute should succeed");
        assert_eq!(table.len().await, 2);

        table.clear().await;
        assert_eq!(table.len().await, 0);
        assert!(table.is_empty().await);
    }

    #[tokio::test]
    async fn async_routing_table_file_roundtrip() {
        let table = sample_routing_table();
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 2,
            k_expert: 2,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1)],
            cumulative: false,
        };

        table
            .compute_and_store(&[0.5, -0.3, 0.8], 5, 10, &tiers, cfg)
            .await
            .expect("compute should succeed");

        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let path = temp_dir.path().join("routing_table.json");

        table
            .save_to_file(&path)
            .await
            .expect("save should succeed");

        let loaded = AsyncRoutingTable::load_from_file(&path)
            .await
            .expect("load should succeed");

        assert_eq!(loaded.len().await, 1);
        let cached = loaded
            .get_routes(5, 10)
            .await
            .expect("cached routes should exist");
        assert_eq!(cached.len(), cfg.max_active_experts());
    }

    #[tokio::test]
    async fn async_routing_table_empty_on_new() {
        let table = sample_routing_table();
        assert!(table.is_empty().await);
        assert_eq!(table.len().await, 0);
    }

    #[tokio::test]
    async fn async_route_serialization_roundtrip() {
        let route = crate::routing::Route {
            expert: crate::routing::ExpertKey {
                tier: TierId(5),
                group: 2,
                expert: 3,
            },
            weight: 0.42,
            priority: 1,
        };
        let async_route = AsyncRoute::from(route);
        let json = serde_json::to_string(&async_route).expect("serialize should succeed");
        let restored: AsyncRoute = serde_json::from_str(&json).expect("deserialize should succeed");
        assert_eq!(async_route, restored);
    }

    #[tokio::test]
    async fn async_routing_table_snapshot_serialization() {
        let snapshot = RoutingTableSnapshot {
            seed: 42,
            entries: vec![],
            tier_catalog: sample_tier_catalog(),
        };
        let json = serde_json::to_string(&snapshot).expect("serialize should succeed");
        let restored: RoutingTableSnapshot =
            serde_json::from_str(&json).expect("deserialize should succeed");
        assert_eq!(snapshot.seed, restored.seed);
        assert_eq!(snapshot.tier_catalog, restored.tier_catalog);
    }

    #[tokio::test]
    async fn async_routing_table_deterministic_replay() {
        let table = sample_routing_table();
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 2,
            k_expert: 2,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1), TierId(10)],
            cumulative: false,
        };
        let token = vec![0.1, -0.2, 0.9, 0.05, -0.7];

        let first = table
            .compute_and_store(&token, 3, 17, &tiers, cfg)
            .await
            .expect("first compute should succeed");

        let table2 = sample_routing_table();
        let second = table2
            .compute_and_store(&token, 3, 17, &tiers, cfg)
            .await
            .expect("second compute should succeed");

        assert_eq!(first, second);
    }
}
