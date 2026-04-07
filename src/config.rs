use crate::error::{RuntimeError, RuntimeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId(pub u16);

impl TierId {
    pub const fn new(id: u16) -> Self {
        Self(id)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Hot,
    Warm,
    Cold,
    Archive,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TierConfig {
    pub id: TierId,
    pub groups: u32,
    pub experts_per_group: u32,
    pub placement: Placement,
}

impl TierConfig {
    pub fn total_experts(&self) -> usize {
        self.groups as usize * self.experts_per_group as usize
    }
}

#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct TierSet {
    pub tiers: Vec<TierId>,
    pub cumulative: bool,
}

impl TierSet {
    pub fn new(tiers: Vec<TierId>, cumulative: bool) -> Self {
        Self { tiers, cumulative }
    }

    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    pub fn contains(&self, tier: TierId) -> bool {
        if self.cumulative {
            self.tiers
                .iter()
                .max()
                .map(|max| tier <= *max)
                .unwrap_or(false)
        } else {
            self.tiers.contains(&tier)
        }
    }

    pub fn resolve(&self, available_tiers: &[TierConfig]) -> Vec<TierId> {
        let mut resolved = Vec::new();
        for tier in available_tiers {
            if self.contains(tier.id) {
                resolved.push(tier.id);
            }
        }
        resolved
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingConfig {
    pub k_tier: usize,
    pub k_group: usize,
    pub k_expert: usize,
}

impl RoutingConfig {
    pub fn validate(self) -> RuntimeResult<()> {
        if self.k_tier == 0 {
            return Err(RuntimeError::InvalidRoutingConfig(
                "k_tier must be greater than zero",
            ));
        }
        if self.k_group == 0 {
            return Err(RuntimeError::InvalidRoutingConfig(
                "k_group must be greater than zero",
            ));
        }
        if self.k_expert == 0 {
            return Err(RuntimeError::InvalidRoutingConfig(
                "k_expert must be greater than zero",
            ));
        }
        Ok(())
    }

    pub fn max_active_experts(self) -> usize {
        self.k_tier
            .saturating_mul(self.k_group)
            .saturating_mul(self.k_expert)
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RoutingSeed {
    pub base: u128,
}

impl RoutingSeed {
    pub const fn new(base: u128) -> Self {
        Self { base }
    }

    pub fn layer_seed(self, layer: u32) -> u64 {
        let mut payload = [0u8; 20];
        payload[..16].copy_from_slice(&self.base.to_le_bytes());
        payload[16..20].copy_from_slice(&layer.to_le_bytes());
        fnv1a64(0xcbf29ce484222325, &payload)
    }

    pub fn token_seed(self, layer: u32, token: u32) -> u64 {
        let layer_seed = self.layer_seed(layer);
        let mut payload = [0u8; 12];
        payload[..8].copy_from_slice(&layer_seed.to_le_bytes());
        payload[8..12].copy_from_slice(&token.to_le_bytes());
        fnv1a64(0xcbf29ce484222325, &payload)
    }
}

pub(crate) fn fnv1a64(seed: u64, payload: &[u8]) -> u64 {
    let mut hash = seed;
    for byte in payload {
        hash ^= u64::from(*byte);
        hash = hash.wrapping_mul(0x100000001b3);
    }
    hash
}

#[cfg(test)]
mod tests {
    use super::{RoutingConfig, RoutingSeed, TierConfig, TierId, TierSet};
    use crate::config::Placement;

    #[test]
    fn tierset_resolve_handles_cumulative_mode() {
        let available = vec![
            TierConfig {
                id: TierId(1),
                groups: 2,
                experts_per_group: 2,
                placement: Placement::Hot,
            },
            TierConfig {
                id: TierId(10),
                groups: 2,
                experts_per_group: 2,
                placement: Placement::Warm,
            },
            TierConfig {
                id: TierId(100),
                groups: 2,
                experts_per_group: 2,
                placement: Placement::Cold,
            },
        ];

        let selected = TierSet::new(vec![TierId(10)], true).resolve(&available);
        assert_eq!(selected, vec![TierId(1), TierId(10)]);
    }

    #[test]
    fn routing_seed_is_deterministic() {
        let seed = RoutingSeed::new(42);
        assert_eq!(seed.layer_seed(7), seed.layer_seed(7));
        assert_eq!(seed.token_seed(7, 3), seed.token_seed(7, 3));
        assert_ne!(seed.token_seed(7, 3), seed.token_seed(7, 4));
    }

    #[test]
    fn routing_config_requires_positive_k() {
        let invalid = RoutingConfig {
            k_tier: 0,
            k_group: 1,
            k_expert: 1,
        };
        assert!(invalid.validate().is_err());
    }
}
