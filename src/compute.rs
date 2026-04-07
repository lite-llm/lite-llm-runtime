use crate::config::RoutingConfig;
use crate::error::{RuntimeError, RuntimeResult};
use crate::routing::Route;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ComputeBoundModel {
    pub layers: usize,
    pub dense_flops_per_layer: u64,
    pub expert_flops_per_invocation: u64,
    pub dense_parameters_per_layer: u64,
    pub max_expert_parameters: u64,
}

impl ComputeBoundModel {
    pub fn active_parameter_upper_bound(&self, routing: RoutingConfig) -> u128 {
        let dense = self.layers as u128 * self.dense_parameters_per_layer as u128;
        let sparse = self.layers as u128
            * routing.max_active_experts() as u128
            * self.max_expert_parameters as u128;
        dense + sparse
    }

    pub fn flops_per_token_bound(&self, routing: RoutingConfig) -> u128 {
        self.layers as u128
            * (self.dense_flops_per_layer as u128
                + routing.max_active_experts() as u128 * self.expert_flops_per_invocation as u128)
    }
}

pub fn enforce_active_expert_bound(routes: &[Route], routing: RoutingConfig) -> RuntimeResult<()> {
    let max = routing.max_active_experts();
    if routes.len() > max {
        return Err(RuntimeError::ComputeBoundExceeded {
            actual: routes.len(),
            max,
        });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::{enforce_active_expert_bound, ComputeBoundModel};
    use crate::config::{RoutingConfig, TierId};
    use crate::routing::{ExpertKey, Route};

    #[test]
    fn compute_formula_matches_spec_shape() {
        let model = ComputeBoundModel {
            layers: 128,
            dense_flops_per_layer: 100,
            expert_flops_per_invocation: 200,
            dense_parameters_per_layer: 1_000,
            max_expert_parameters: 8_000_000,
        };
        let routing = RoutingConfig {
            k_tier: 1,
            k_group: 2,
            k_expert: 2,
        };

        assert_eq!(routing.max_active_experts(), 4);
        assert!(model.active_parameter_upper_bound(routing) > 0);
        assert!(model.flops_per_token_bound(routing) > 0);
    }

    #[test]
    fn enforce_bound_rejects_overflow_routes() {
        let routing = RoutingConfig {
            k_tier: 1,
            k_group: 1,
            k_expert: 1,
        };
        let routes = vec![
            Route {
                expert: ExpertKey {
                    tier: TierId(1),
                    group: 0,
                    expert: 0,
                },
                weight: 0.5,
                priority: 0,
            },
            Route {
                expert: ExpertKey {
                    tier: TierId(1),
                    group: 0,
                    expert: 1,
                },
                weight: 0.5,
                priority: 1,
            },
        ];

        assert!(enforce_active_expert_bound(&routes, routing).is_err());
    }
}
