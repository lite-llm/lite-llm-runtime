use std::collections::HashSet;

use crate::config::{RoutingSeed, TierConfig, TierId, TierSet};
use crate::error::{RuntimeError, RuntimeResult};
use crate::state_machine::{RuntimeState, RuntimeStateMachine};

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub routing_seed: RoutingSeed,
    pub available_tiers: Vec<TierConfig>,
    pub expected_manifest_version: u32,
    pub training_mode: bool,
}

impl RuntimeOptions {
    pub fn validate(&self) -> RuntimeResult<()> {
        if self.available_tiers.is_empty() {
            return Err(RuntimeError::invalid_manifest(
                "runtime must define at least one available tier",
            ));
        }
        if self.expected_manifest_version == 0 {
            return Err(RuntimeError::invalid_manifest(
                "expected manifest version must be greater than zero",
            ));
        }

        let mut seen = HashSet::new();
        for tier in &self.available_tiers {
            if !seen.insert(tier.id.0) {
                return Err(RuntimeError::invalid_manifest(
                    "duplicate tier id in runtime",
                ));
            }
            if tier.groups == 0 || tier.experts_per_group == 0 {
                return Err(RuntimeError::invalid_manifest(
                    "tier groups and experts_per_group must be > 0",
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ManifestShard {
    pub shard_id: String,
    pub checksum_hex: String,
    pub bytes: u64,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct CheckpointManifest {
    pub version: u32,
    pub tiers: TierSet,
    pub base_parameters_checksum: String,
    pub router_checksum: String,
    pub optimizer_checksum: Option<String>,
    pub shards: Vec<ManifestShard>,
}

impl CheckpointManifest {
    pub fn parse(text: &str) -> RuntimeResult<Self> {
        let mut version = None;
        let mut tiers = None;
        let mut cumulative = false;
        let mut base_parameters_checksum = None;
        let mut router_checksum = None;
        let mut optimizer_checksum = None;
        let mut shards: Vec<ManifestShard> = Vec::new();

        for raw_line in text.lines() {
            let line = raw_line.trim();
            if line.is_empty() || line.starts_with('#') {
                continue;
            }

            let mut parts = line.splitn(2, '=');
            let key = parts
                .next()
                .ok_or_else(|| RuntimeError::invalid_manifest("malformed manifest line"))?;
            let value = parts
                .next()
                .ok_or_else(|| RuntimeError::invalid_manifest("manifest line missing '='"))?;

            match key {
                "version" => {
                    version = Some(value.parse::<u32>().map_err(|_| {
                        RuntimeError::invalid_manifest("version must be an unsigned integer")
                    })?);
                }
                "tiers" => {
                    let mut parsed = Vec::new();
                    for token in value.split(',') {
                        let trimmed = token.trim();
                        if trimmed.is_empty() {
                            continue;
                        }
                        parsed.push(TierId(trimmed.parse::<u16>().map_err(|_| {
                            RuntimeError::invalid_manifest(
                                "tiers must contain comma-separated u16 values",
                            )
                        })?));
                    }
                    tiers = Some(parsed);
                }
                "cumulative" => {
                    cumulative = match value {
                        "true" => true,
                        "false" => false,
                        _ => {
                            return Err(RuntimeError::invalid_manifest(
                                "cumulative must be true or false",
                            ));
                        }
                    }
                }
                "base_checksum" => {
                    base_parameters_checksum = Some(value.to_owned());
                }
                "router_checksum" => {
                    router_checksum = Some(value.to_owned());
                }
                "optimizer_checksum" => {
                    optimizer_checksum = Some(value.to_owned());
                }
                "shard" => {
                    let mut shard_parts = value.split('|');
                    let shard_id = shard_parts
                        .next()
                        .ok_or_else(|| RuntimeError::invalid_manifest("shard id missing"))?;
                    let checksum_hex = shard_parts
                        .next()
                        .ok_or_else(|| RuntimeError::invalid_manifest("shard checksum missing"))?;
                    let bytes = shard_parts
                        .next()
                        .ok_or_else(|| RuntimeError::invalid_manifest("shard byte size missing"))?
                        .parse::<u64>()
                        .map_err(|_| {
                            RuntimeError::invalid_manifest("shard byte size must be unsigned")
                        })?;

                    shards.push(ManifestShard {
                        shard_id: shard_id.to_owned(),
                        checksum_hex: checksum_hex.to_owned(),
                        bytes,
                    });
                }
                _ => {
                    return Err(RuntimeError::invalid_manifest(
                        "unknown manifest field encountered",
                    ));
                }
            }
        }

        Ok(Self {
            version: version.ok_or_else(|| RuntimeError::invalid_manifest("version missing"))?,
            tiers: TierSet {
                tiers: tiers.ok_or_else(|| RuntimeError::invalid_manifest("tiers missing"))?,
                cumulative,
            },
            base_parameters_checksum: base_parameters_checksum
                .ok_or_else(|| RuntimeError::invalid_manifest("base_checksum missing"))?,
            router_checksum: router_checksum
                .ok_or_else(|| RuntimeError::invalid_manifest("router_checksum missing"))?,
            optimizer_checksum,
            shards,
        })
    }

    pub fn validate(&self, expected_version: u32) -> RuntimeResult<()> {
        if self.version != expected_version {
            return Err(RuntimeError::UnsupportedManifestVersion {
                expected: expected_version,
                found: self.version,
            });
        }
        if self.tiers.is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }
        if self.base_parameters_checksum.is_empty() || self.router_checksum.is_empty() {
            return Err(RuntimeError::invalid_manifest(
                "base/router checksums must be present",
            ));
        }
        if self.shards.is_empty() {
            return Err(RuntimeError::invalid_manifest(
                "manifest must include at least one shard",
            ));
        }

        for shard in &self.shards {
            if shard.shard_id.is_empty() || shard.checksum_hex.is_empty() || shard.bytes == 0 {
                return Err(RuntimeError::invalid_manifest(
                    "each shard must include id, checksum, and positive byte size",
                ));
            }
        }

        Ok(())
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum BootStage {
    Init,
    ConfigurationLoaded,
    ResourcesAllocated,
    SeedInitialized,
    LoggingStarted,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ModelLoadStage {
    NotStarted,
    ManifestParsed,
    BaseParametersLoaded,
    ExpertsRegistered,
    RouterParametersLoaded,
    OptimizerStateLoaded,
    Complete,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct RuntimeStatus {
    pub state: RuntimeState,
    pub boot_stage: BootStage,
    pub model_load_stage: ModelLoadStage,
    pub logging_enabled: bool,
    pub active_tiers: TierSet,
    pub recovery_count: u64,
}

#[derive(Debug, Clone)]
pub struct RuntimeLifecycle {
    options: RuntimeOptions,
    state_machine: RuntimeStateMachine,
    boot_stage: BootStage,
    model_load_stage: ModelLoadStage,
    manifest: Option<CheckpointManifest>,
    logging_enabled: bool,
    resources_allocated: bool,
    active_tiers: TierSet,
    recovery_count: u64,
    accepting_requests: bool,
}

impl RuntimeLifecycle {
    pub fn new(options: RuntimeOptions) -> RuntimeResult<Self> {
        options.validate()?;
        Ok(Self {
            options,
            state_machine: RuntimeStateMachine::new(),
            boot_stage: BootStage::Init,
            model_load_stage: ModelLoadStage::NotStarted,
            manifest: None,
            logging_enabled: false,
            resources_allocated: false,
            active_tiers: TierSet::default(),
            recovery_count: 0,
            accepting_requests: false,
        })
    }

    pub fn status(&self) -> RuntimeStatus {
        RuntimeStatus {
            state: self.state_machine.state(),
            boot_stage: self.boot_stage,
            model_load_stage: self.model_load_stage,
            logging_enabled: self.logging_enabled,
            active_tiers: self.active_tiers.clone(),
            recovery_count: self.recovery_count,
        }
    }

    pub fn routing_seed(&self) -> RoutingSeed {
        self.options.routing_seed
    }

    pub fn boot(&mut self) -> RuntimeResult<()> {
        if self.state_machine.state() != RuntimeState::Init {
            return Err(RuntimeError::invalid_boot_order(
                "boot is only valid when runtime is in Init state",
            ));
        }

        self.boot_stage = BootStage::ConfigurationLoaded;
        self.boot_stage = BootStage::ResourcesAllocated;
        self.resources_allocated = true;
        self.boot_stage = BootStage::SeedInitialized;
        self.boot_stage = BootStage::LoggingStarted;
        self.logging_enabled = true;

        Ok(())
    }

    pub fn parse_manifest(&mut self, manifest_text: &str) -> RuntimeResult<()> {
        self.ensure_boot_complete()?;
        self.ensure_model_stage(ModelLoadStage::NotStarted)?;

        let manifest = CheckpointManifest::parse(manifest_text)?;
        manifest.validate(self.options.expected_manifest_version)?;
        self.ensure_tiers_known(&manifest.tiers)?;

        self.manifest = Some(manifest);
        self.model_load_stage = ModelLoadStage::ManifestParsed;
        Ok(())
    }

    pub fn load_base_parameters(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::ManifestParsed)?;
        self.model_load_stage = ModelLoadStage::BaseParametersLoaded;
        Ok(())
    }

    pub fn register_experts(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::BaseParametersLoaded)?;
        self.model_load_stage = ModelLoadStage::ExpertsRegistered;
        Ok(())
    }

    pub fn load_router_parameters(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::ExpertsRegistered)?;
        self.model_load_stage = ModelLoadStage::RouterParametersLoaded;
        Ok(())
    }

    pub fn load_optimizer_state(&mut self) -> RuntimeResult<()> {
        if !self.options.training_mode {
            return Ok(());
        }
        self.ensure_model_stage(ModelLoadStage::RouterParametersLoaded)?;
        self.model_load_stage = ModelLoadStage::OptimizerStateLoaded;
        Ok(())
    }

    pub fn complete_model_load(&mut self) -> RuntimeResult<()> {
        if self.options.training_mode {
            self.ensure_model_stage(ModelLoadStage::OptimizerStateLoaded)?;
        } else {
            self.ensure_model_stage(ModelLoadStage::RouterParametersLoaded)?;
        }

        self.model_load_stage = ModelLoadStage::Complete;
        self.state_machine.transition(RuntimeState::Warm)?;

        if self.active_tiers.is_empty() {
            if let Some(manifest) = &self.manifest {
                self.active_tiers = manifest.tiers.clone();
            }
        }

        Ok(())
    }

    pub fn activate_tiers(&mut self, tiers: TierSet) -> RuntimeResult<()> {
        if tiers.is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }
        self.ensure_tiers_known(&tiers)?;

        let state = self.state_machine.state();
        if !matches!(
            state,
            RuntimeState::Warm | RuntimeState::Active | RuntimeState::Recovering
        ) {
            return Err(RuntimeError::invalid_load_order(
                "tiers can only be activated in Warm, Active, or Recovering",
            ));
        }

        self.active_tiers = tiers;
        Ok(())
    }

    pub fn start_serving(&mut self) -> RuntimeResult<()> {
        if self.active_tiers.is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }
        self.state_machine.transition(RuntimeState::Active)?;
        self.accepting_requests = true;
        Ok(())
    }

    pub fn graceful_shutdown(&mut self) -> RuntimeResult<()> {
        if self.state_machine.state() == RuntimeState::Active {
            self.state_machine.transition(RuntimeState::Frozen)?;
        }

        self.accepting_requests = false;
        self.logging_enabled = false;
        self.resources_allocated = false;
        self.manifest = None;
        self.model_load_stage = ModelLoadStage::NotStarted;
        self.boot_stage = BootStage::Init;
        self.active_tiers = TierSet::default();
        self.state_machine.reset_to_init();

        Ok(())
    }

    pub fn begin_recovery(&mut self) -> RuntimeResult<()> {
        self.state_machine.transition(RuntimeState::Recovering)?;
        self.recovery_count = self.recovery_count.saturating_add(1);
        self.accepting_requests = false;
        Ok(())
    }

    pub fn restore_after_crash(
        &mut self,
        manifest_text: &str,
        resume_active: bool,
    ) -> RuntimeResult<()> {
        if self.state_machine.state() != RuntimeState::Recovering {
            return Err(RuntimeError::recovery_failed(
                "runtime must be in Recovering state before restore",
            ));
        }

        self.model_load_stage = ModelLoadStage::NotStarted;
        self.parse_manifest(manifest_text)?;
        self.load_base_parameters()?;
        self.register_experts()?;
        self.load_router_parameters()?;
        self.load_optimizer_state()?;
        self.complete_model_load()?;

        let manifest_tiers = self
            .manifest
            .as_ref()
            .ok_or_else(|| RuntimeError::recovery_failed("manifest missing during recovery"))?
            .tiers
            .clone();
        self.activate_tiers(manifest_tiers)?;

        if resume_active {
            self.start_serving()?;
        }

        Ok(())
    }

    fn ensure_boot_complete(&self) -> RuntimeResult<()> {
        if self.boot_stage != BootStage::LoggingStarted {
            return Err(RuntimeError::invalid_boot_order(
                "boot sequence must complete before model loading",
            ));
        }
        Ok(())
    }

    fn ensure_model_stage(&self, expected: ModelLoadStage) -> RuntimeResult<()> {
        if self.model_load_stage != expected {
            return Err(RuntimeError::invalid_load_order(
                "model loading phases must execute in order",
            ));
        }
        Ok(())
    }

    fn ensure_tiers_known(&self, tiers: &TierSet) -> RuntimeResult<()> {
        let available: HashSet<u16> = self
            .options
            .available_tiers
            .iter()
            .map(|tier| tier.id.0)
            .collect();

        for tier in &tiers.tiers {
            if !available.contains(&tier.0) {
                return Err(RuntimeError::UnknownTier(tier.0));
            }
        }

        if tiers.resolve(&self.options.available_tiers).is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use crate::config::{Placement, TierConfig};

    use super::{BootStage, ModelLoadStage, RuntimeLifecycle, RuntimeOptions, RuntimeStatus};
    use crate::config::{RoutingSeed, TierId, TierSet};
    use crate::state_machine::RuntimeState;

    fn sample_options(training_mode: bool) -> RuntimeOptions {
        RuntimeOptions {
            routing_seed: RoutingSeed::new(77),
            available_tiers: vec![
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
            ],
            expected_manifest_version: 1,
            training_mode,
        }
    }

    fn sample_manifest() -> &'static str {
        "version=1\n\
        tiers=1,10\n\
        cumulative=false\n\
        base_checksum=abc123\n\
        router_checksum=def456\n\
        optimizer_checksum=fff999\n\
        shard=base|aa11|1024\n\
        shard=exp_1|bb22|2048\n"
    }

    fn assert_state(status: RuntimeStatus, expected: RuntimeState) {
        assert_eq!(status.state, expected);
    }

    #[test]
    fn full_boot_load_activate_flow() {
        let mut lifecycle = RuntimeLifecycle::new(sample_options(true)).expect("valid options");

        lifecycle.boot().expect("boot should succeed");
        assert_eq!(lifecycle.status().boot_stage, BootStage::LoggingStarted);

        lifecycle
            .parse_manifest(sample_manifest())
            .expect("manifest parse should succeed");
        lifecycle
            .load_base_parameters()
            .expect("base load should succeed");
        lifecycle
            .register_experts()
            .expect("expert registration should succeed");
        lifecycle
            .load_router_parameters()
            .expect("router load should succeed");
        lifecycle
            .load_optimizer_state()
            .expect("optimizer load should succeed");
        lifecycle
            .complete_model_load()
            .expect("complete model load should succeed");

        assert_eq!(
            lifecycle.status().model_load_stage,
            ModelLoadStage::Complete
        );
        assert_state(lifecycle.status(), RuntimeState::Warm);

        lifecycle
            .activate_tiers(TierSet::new(vec![TierId(10)], true))
            .expect("tier activation should succeed");
        lifecycle
            .start_serving()
            .expect("start serving should succeed");
        assert_state(lifecycle.status(), RuntimeState::Active);
    }

    #[test]
    fn recovery_restores_to_active() {
        let mut lifecycle = RuntimeLifecycle::new(sample_options(true)).expect("valid options");

        lifecycle.boot().expect("boot should succeed");
        lifecycle
            .parse_manifest(sample_manifest())
            .expect("manifest parse should succeed");
        lifecycle
            .load_base_parameters()
            .expect("base load should succeed");
        lifecycle
            .register_experts()
            .expect("expert registration should succeed");
        lifecycle
            .load_router_parameters()
            .expect("router load should succeed");
        lifecycle
            .load_optimizer_state()
            .expect("optimizer load should succeed");
        lifecycle
            .complete_model_load()
            .expect("complete model load should succeed");
        lifecycle
            .activate_tiers(TierSet::new(vec![TierId(1), TierId(10)], false))
            .expect("tier activation should succeed");
        lifecycle
            .start_serving()
            .expect("start serving should succeed");

        let stable_seed = lifecycle.routing_seed();

        lifecycle
            .begin_recovery()
            .expect("transition to recovery should succeed");
        lifecycle
            .restore_after_crash(sample_manifest(), true)
            .expect("restore should succeed");

        assert_state(lifecycle.status(), RuntimeState::Active);
        assert_eq!(stable_seed, lifecycle.routing_seed());
    }

    #[test]
    fn parse_manifest_requires_boot() {
        let mut lifecycle = RuntimeLifecycle::new(sample_options(false)).expect("valid options");
        assert!(lifecycle.parse_manifest(sample_manifest()).is_err());
    }
}
