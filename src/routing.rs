use crate::compute::enforce_active_expert_bound;
use crate::config::{fnv1a64, RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet};
use crate::error::{RuntimeError, RuntimeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub struct ExpertKey {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    pub expert: ExpertKey,
    pub weight: f32,
    pub priority: u16,
}

pub trait Router {
    fn route(
        &self,
        token_state: &[f32],
        layer_index: u32,
        token_index: u32,
        tiers: &TierSet,
        cfg: RoutingConfig,
    ) -> RuntimeResult<Vec<Route>>;
}

#[derive(Debug, Clone)]
pub struct DeterministicRouter {
    seed: RoutingSeed,
    tier_catalog: Vec<TierConfig>,
    quantization_scale: f32,
}

impl DeterministicRouter {
    pub fn new(seed: RoutingSeed, tier_catalog: Vec<TierConfig>) -> Self {
        Self {
            seed,
            tier_catalog,
            quantization_scale: 1_000_000.0,
        }
    }

    pub fn with_quantization_scale(mut self, quantization_scale: f32) -> Self {
        self.quantization_scale = quantization_scale.max(1.0);
        self
    }

    fn find_tier_config(&self, id: TierId) -> Option<&TierConfig> {
        self.tier_catalog.iter().find(|tier| tier.id == id)
    }

    fn allowed_tiers(&self, tiers: &TierSet) -> Vec<TierId> {
        let mut selected = tiers.resolve(&self.tier_catalog);
        selected.sort_unstable();
        selected
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

    fn tier_scores(&self, token_state: &[f32], layer_seed: u64, tiers: &[TierId]) -> Vec<f32> {
        tiers
            .iter()
            .map(|tier| Self::projection_score(token_state, layer_seed, 0x01, tier.0 as u64))
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
            scores.push(Self::projection_score(
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
            scores.push(Self::projection_score(
                token_state,
                token_seed,
                (tier.0 as u64) << 32 | u64::from(group),
                expert as u64,
            ));
        }
        scores
    }
}

impl Default for DeterministicRouter {
    fn default() -> Self {
        Self::new(RoutingSeed::new(0), Vec::new())
    }
}

impl Router for DeterministicRouter {
    fn route(
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
        let tier_indices = stable_top_k(
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
                return Err(RuntimeError::InvalidRoutingConfig(
                    "tier groups must be greater than zero",
                ));
            }

            let group_scores = self.group_scores(token_state, token_seed, tier_id, groups);
            let group_indices = stable_top_k(
                &group_scores,
                cfg.k_group.min(groups),
                token_seed ^ tier_id.0 as u64,
                self.quantization_scale,
            );

            for group_idx in group_indices {
                let experts = tier_cfg.experts_per_group as usize;
                if experts == 0 {
                    return Err(RuntimeError::InvalidRoutingConfig(
                        "experts_per_group must be greater than zero",
                    ));
                }

                let expert_scores =
                    self.expert_scores(token_state, token_seed, tier_id, group_idx as u32, experts);
                let expert_indices = stable_top_k(
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
}

pub fn stable_top_k(scores: &[f32], k: usize, seed: u64, scale: f32) -> Vec<usize> {
    if scores.is_empty() || k == 0 {
        return Vec::new();
    }

    let mut ranked: Vec<(usize, i64, u64)> = scores
        .iter()
        .enumerate()
        .map(|(idx, score)| {
            let quantized = quantize(*score, scale);
            let tie = seeded_index_hash(seed, idx as u64);
            (idx, quantized, tie)
        })
        .collect();

    // Stable sort ensures deterministic order even when all comparator keys match.
    ranked.sort_by(|a, b| b.1.cmp(&a.1).then(a.2.cmp(&b.2)).then(a.0.cmp(&b.0)));

    ranked
        .into_iter()
        .take(k.min(scores.len()))
        .map(|entry| entry.0)
        .collect()
}

fn quantize(value: f32, scale: f32) -> i64 {
    let scaled = (value as f64 * scale as f64).round();
    if scaled > i64::MAX as f64 {
        i64::MAX
    } else if scaled < i64::MIN as f64 {
        i64::MIN
    } else {
        scaled as i64
    }
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
    use super::{stable_top_k, DeterministicRouter, Router};
    use crate::config::{Placement, RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet};

    fn sample_router(seed: u128) -> DeterministicRouter {
        let tiers = vec![
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
        ];
        DeterministicRouter::new(RoutingSeed::new(seed), tiers)
    }

    #[test]
    fn stable_top_k_uses_seeded_tie_break() {
        let scores = [1.0, 1.0, 1.0, 0.5];
        let first = stable_top_k(&scores, 3, 42, 1_000_000.0);
        let second = stable_top_k(&scores, 3, 42, 1_000_000.0);
        assert_eq!(first, second);
    }

    #[test]
    fn deterministic_router_replays_identically() {
        let router = sample_router(99);
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

        let first = router
            .route(&token, 3, 17, &tiers, cfg)
            .expect("routing should succeed");
        let second = router
            .route(&token, 3, 17, &tiers, cfg)
            .expect("routing should succeed");

        assert_eq!(first, second);
        assert_eq!(first.len(), cfg.max_active_experts());
    }

    #[test]
    fn deterministic_router_respects_compute_bound() {
        let router = sample_router(7);
        let cfg = RoutingConfig {
            k_tier: 1,
            k_group: 1,
            k_expert: 1,
        };
        let tiers = TierSet {
            tiers: vec![TierId(1)],
            cumulative: false,
        };

        let routes = router
            .route(&[1.0, 2.0, 3.0], 0, 0, &tiers, cfg)
            .expect("routing should succeed");

        assert!(routes.len() <= cfg.max_active_experts());
    }
}
