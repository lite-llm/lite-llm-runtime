use crate::config::{RoutingConfig, TierId, TierSet};

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Route {
    pub tier: TierId,
    pub group: u32,
    pub expert: u32,
    pub weight: f32,
}

pub trait Router {
    fn route(&self, token_state: &[f32], tiers: &TierSet, cfg: RoutingConfig) -> Vec<Route>;
}

#[derive(Debug, Default)]
pub struct DeterministicRouter;

impl Router for DeterministicRouter {
    fn route(&self, _token_state: &[f32], tiers: &TierSet, cfg: RoutingConfig) -> Vec<Route> {
        let mut routes = Vec::new();
        let Some(first_tier) = tiers.tiers.first().copied() else {
            return routes;
        };

        let active = cfg.max_active_experts();
        for idx in 0..active {
            routes.push(Route {
                tier: first_tier,
                group: (idx % cfg.k_group.max(1)) as u32,
                expert: (idx % cfg.k_expert.max(1)) as u32,
                weight: 1.0 / active.max(1) as f32,
            });
        }
        routes
    }
}

