#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord)]
pub struct TierId(pub u16);

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Placement {
    Hot,
    Warm,
    Cold,
    Archive,
}

#[derive(Debug, Clone)]
pub struct TierConfig {
    pub id: TierId,
    pub groups: u32,
    pub experts_per_group: u32,
    pub placement: Placement,
}

#[derive(Debug, Clone, Default)]
pub struct TierSet {
    pub tiers: Vec<TierId>,
    pub cumulative: bool,
}

impl TierSet {
    pub fn contains(&self, tier: TierId) -> bool {
        if self.cumulative {
            self.tiers.iter().max().map(|max| tier <= *max).unwrap_or(false)
        } else {
            self.tiers.contains(&tier)
        }
    }
}

#[derive(Debug, Clone, Copy)]
pub struct RoutingConfig {
    pub k_tier: usize,
    pub k_group: usize,
    pub k_expert: usize,
}

impl RoutingConfig {
    pub fn max_active_experts(&self) -> usize {
        self.k_tier * self.k_group * self.k_expert
    }
}

