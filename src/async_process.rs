//! Async runtime lifecycle and manifest management.
//!
//! Provides async-capable versions of the synchronous process types with
//! serde serialization support, enabling persistent checkpoint and manifest
//! operations across restarts.
//!
//! # Key Types
//!
//! - [`AsyncCheckpointManifest`] -- serializable manifest with tier and shard metadata
//! - [`AsyncManifestShard`] -- individual shard entry with checksum
//! - [`AsyncTierSet`] -- serializable tier set with cumulative resolution
//! - [`AsyncRuntimeLifecycle`] -- async boot, load, activate, shutdown, and recovery flow
//!
//! # Async Operations
//!
//! All model loading, tier activation, and recovery operations are async,
//! allowing non-blocking I/O for manifest reads, checkpoint persistence,
//! and file-based state management.

use std::path::Path;

use serde::{Deserialize, Serialize};
use tokio::fs;

use crate::config::{RoutingSeed, TierConfig, TierId, TierSet};
use crate::error::{RuntimeError, RuntimeResult};
use crate::process::{BootStage, CheckpointManifest, ManifestShard, ModelLoadStage, RuntimeOptions};
use crate::state_machine::{RuntimeState, RuntimeStateMachine};

/// Async-capable version of [`CheckpointManifest`] with serde serialization support.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsyncCheckpointManifest {
    pub version: u32,
    pub tiers: AsyncTierSet,
    pub base_parameters_checksum: String,
    pub router_checksum: String,
    pub optimizer_checksum: Option<String>,
    pub shards: Vec<AsyncManifestShard>,
}

/// Async-capable version of [`ManifestShard`] with serde serialization.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsyncManifestShard {
    pub shard_id: String,
    pub checksum_hex: String,
    pub bytes: u64,
}

/// Async-capable version of [`TierSet`] with serde serialization.
#[derive(Debug, Clone, Default, Serialize, Deserialize, PartialEq, Eq)]
pub struct AsyncTierSet {
    pub tiers: Vec<TierId>,
    pub cumulative: bool,
}

impl AsyncTierSet {
    /// Create a new async tier set.
    pub fn new(tiers: Vec<TierId>, cumulative: bool) -> Self {
        Self { tiers, cumulative }
    }

    /// Check if this tier set has no tiers.
    pub fn is_empty(&self) -> bool {
        self.tiers.is_empty()
    }

    /// Check if the given tier is contained in this set (with cumulative resolution).
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

    /// Resolve this tier set against the available tiers catalog, returning matched tier IDs.
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

// ---------- conversions between sync and async types ----------

impl From<CheckpointManifest> for AsyncCheckpointManifest {
    fn from(manifest: CheckpointManifest) -> Self {
        Self {
            version: manifest.version,
            tiers: AsyncTierSet {
                tiers: manifest.tiers.tiers,
                cumulative: manifest.tiers.cumulative,
            },
            base_parameters_checksum: manifest.base_parameters_checksum,
            router_checksum: manifest.router_checksum,
            optimizer_checksum: manifest.optimizer_checksum,
            shards: manifest
                .shards
                .into_iter()
                .map(|s| AsyncManifestShard {
                    shard_id: s.shard_id,
                    checksum_hex: s.checksum_hex,
                    bytes: s.bytes,
                })
                .collect(),
        }
    }
}

impl From<AsyncCheckpointManifest> for CheckpointManifest {
    fn from(manifest: AsyncCheckpointManifest) -> Self {
        Self {
            version: manifest.version,
            tiers: TierSet {
                tiers: manifest.tiers.tiers,
                cumulative: manifest.tiers.cumulative,
            },
            base_parameters_checksum: manifest.base_parameters_checksum,
            router_checksum: manifest.router_checksum,
            optimizer_checksum: manifest.optimizer_checksum,
            shards: manifest
                .shards
                .into_iter()
                .map(|s| ManifestShard {
                    shard_id: s.shard_id,
                    checksum_hex: s.checksum_hex,
                    bytes: s.bytes,
                })
                .collect(),
        }
    }
}

impl AsyncCheckpointManifest {
    /// Validate this manifest against an expected version number.
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

    /// Serialize this manifest to a JSON string.
    pub fn to_json(&self) -> RuntimeResult<String> {
        serde_json::to_string_pretty(self)
            .map_err(|e| RuntimeError::serialization_error(format!("failed to serialize manifest: {e}")))
    }

    /// Deserialize a manifest from a JSON string.
    pub fn from_json(text: &str) -> RuntimeResult<Self> {
        serde_json::from_str(text)
            .map_err(|e| RuntimeError::serialization_error(format!("failed to deserialize manifest: {e}")))
    }
}

/// Read a manifest from a file path as a JSON document.
pub async fn read_manifest_from_file(path: impl AsRef<Path>) -> RuntimeResult<AsyncCheckpointManifest> {
    let path = path.as_ref();
    let text = fs::read_to_string(path)
        .await
        .map_err(|e| RuntimeError::io_error(format!("failed to read manifest file {}: {}", path.display(), e)))?;
    AsyncCheckpointManifest::from_json(&text)
}

/// Write a manifest to a file path as a JSON document.
pub async fn write_manifest_to_file(
    path: impl AsRef<Path>,
    manifest: &AsyncCheckpointManifest,
) -> RuntimeResult<()> {
    let path = path.as_ref();
    let json = manifest.to_json()?;
    fs::write(path, json)
        .await
        .map_err(|e| RuntimeError::io_error(format!("failed to write manifest file {}: {}", path.display(), e)))
}

/// Async runtime lifecycle manager that mirrors [`crate::process::RuntimeLifecycle`].
///
/// Provides async variants of boot, manifest loading, and model-loading operations.
#[derive(Debug)]
pub struct AsyncRuntimeLifecycle {
    options: RuntimeOptions,
    state_machine: RuntimeStateMachine,
    boot_stage: BootStage,
    model_load_stage: ModelLoadStage,
    manifest: Option<AsyncCheckpointManifest>,
    logging_enabled: bool,
    resources_allocated: bool,
    active_tiers: TierSet,
    recovery_count: u64,
    accepting_requests: bool,
}

impl AsyncRuntimeLifecycle {
    /// Create a new async runtime lifecycle manager with the given options.
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

    /// Get the routing seed from the runtime options.
    pub fn routing_seed(&self) -> RoutingSeed {
        self.options.routing_seed
    }

    /// Get the current boot stage.
    pub fn boot_stage(&self) -> BootStage {
        self.boot_stage
    }

    /// Get the current model load stage.
    pub fn model_load_stage(&self) -> ModelLoadStage {
        self.model_load_stage
    }

    /// Get the current runtime state.
    pub fn state(&self) -> RuntimeState {
        self.state_machine.state()
    }

    /// Get the currently active tier set.
    pub fn active_tiers(&self) -> &TierSet {
        &self.active_tiers
    }

    /// Get the loaded manifest, if any.
    pub fn manifest(&self) -> Option<&AsyncCheckpointManifest> {
        self.manifest.as_ref()
    }

    /// Get the number of recoveries performed.
    pub fn recovery_count(&self) -> u64 {
        self.recovery_count
    }

    /// Async boot sequence.
    pub async fn boot(&mut self) -> RuntimeResult<()> {
        if self.state_machine.state() != RuntimeState::Init {
            return Err(RuntimeError::invalid_boot_order(
                "boot is only valid when runtime is in Init state",
            ));
        }

        // Simulate async I/O work for each boot stage.
        self.boot_stage = BootStage::ConfigurationLoaded;
        tokio::task::yield_now().await;

        self.boot_stage = BootStage::ResourcesAllocated;
        self.resources_allocated = true;
        tokio::task::yield_now().await;

        self.boot_stage = BootStage::SeedInitialized;
        tokio::task::yield_now().await;

        self.boot_stage = BootStage::LoggingStarted;
        self.logging_enabled = true;

        Ok(())
    }

    /// Parse a manifest from a text string (sync-compatible helper).
    pub async fn parse_manifest(&mut self, manifest_text: &str) -> RuntimeResult<()> {
        self.ensure_boot_complete()?;
        self.ensure_model_stage(ModelLoadStage::NotStarted)?;

        let manifest = CheckpointManifest::parse(manifest_text)?;
        manifest.validate(self.options.expected_manifest_version)?;
        self.ensure_tiers_known(&manifest.tiers)?;

        self.manifest = Some(manifest.into());
        self.model_load_stage = ModelLoadStage::ManifestParsed;
        Ok(())
    }

    /// Read and parse a manifest from a file.
    pub async fn load_manifest_from_file(
        &mut self,
        path: impl AsRef<Path>,
    ) -> RuntimeResult<()> {
        self.ensure_boot_complete()?;
        self.ensure_model_stage(ModelLoadStage::NotStarted)?;

        let manifest = read_manifest_from_file(path).await?;
        manifest.validate(self.options.expected_manifest_version)?;
        self.ensure_tiers_known(&TierSet {
            tiers: manifest.tiers.tiers.clone(),
            cumulative: manifest.tiers.cumulative,
        })?;

        self.manifest = Some(manifest);
        self.model_load_stage = ModelLoadStage::ManifestParsed;
        Ok(())
    }

    /// Persist the current manifest to a JSON file.
    pub async fn save_manifest_to_file(
        &self,
        path: impl AsRef<Path>,
    ) -> RuntimeResult<()> {
        let manifest = self
            .manifest
            .as_ref()
            .ok_or_else(|| RuntimeError::invalid_manifest("no manifest loaded to persist"))?;
        write_manifest_to_file(path, manifest).await
    }

    /// Async base parameters loading (simulates async I/O).
    pub async fn load_base_parameters(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::ManifestParsed)?;
        tokio::task::yield_now().await;
        self.model_load_stage = ModelLoadStage::BaseParametersLoaded;
        Ok(())
    }

    /// Async expert registration (simulates async I/O).
    pub async fn register_experts(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::BaseParametersLoaded)?;
        tokio::task::yield_now().await;
        self.model_load_stage = ModelLoadStage::ExpertsRegistered;
        Ok(())
    }

    /// Async router parameters loading.
    pub async fn load_router_parameters(&mut self) -> RuntimeResult<()> {
        self.ensure_model_stage(ModelLoadStage::ExpertsRegistered)?;
        tokio::task::yield_now().await;
        self.model_load_stage = ModelLoadStage::RouterParametersLoaded;
        Ok(())
    }

    /// Async optimizer state loading (only in training mode).
    pub async fn load_optimizer_state(&mut self) -> RuntimeResult<()> {
        if !self.options.training_mode {
            return Ok(());
        }
        self.ensure_model_stage(ModelLoadStage::RouterParametersLoaded)?;
        tokio::task::yield_now().await;
        self.model_load_stage = ModelLoadStage::OptimizerStateLoaded;
        Ok(())
    }

    /// Complete the model loading sequence and transition to Warm state.
    pub async fn complete_model_load(&mut self) -> RuntimeResult<()> {
        if self.options.training_mode {
            self.ensure_model_stage(ModelLoadStage::OptimizerStateLoaded)?;
        } else {
            self.ensure_model_stage(ModelLoadStage::RouterParametersLoaded)?;
        }

        self.model_load_stage = ModelLoadStage::Complete;
        self.state_machine.transition(RuntimeState::Warm)?;

        if self.active_tiers.is_empty() {
            if let Some(manifest) = &self.manifest {
                self.active_tiers = TierSet {
                    tiers: manifest.tiers.tiers.clone(),
                    cumulative: manifest.tiers.cumulative,
                };
            }
        }

        Ok(())
    }

    /// Activate a set of tiers for serving.
    pub async fn activate_tiers(&mut self, tiers: TierSet) -> RuntimeResult<()> {
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
        tokio::task::yield_now().await;
        Ok(())
    }

    /// Transition to Active and begin accepting requests.
    pub async fn start_serving(&mut self) -> RuntimeResult<()> {
        if self.active_tiers.is_empty() {
            return Err(RuntimeError::EmptyTierSet);
        }
        self.state_machine.transition(RuntimeState::Active)?;
        self.accepting_requests = true;
        Ok(())
    }

    /// Gracefully shut down the runtime.
    pub async fn graceful_shutdown(&mut self) -> RuntimeResult<()> {
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

    /// Begin recovery transition.
    pub async fn begin_recovery(&mut self) -> RuntimeResult<()> {
        self.state_machine.transition(RuntimeState::Recovering)?;
        self.recovery_count = self.recovery_count.saturating_add(1);
        self.accepting_requests = false;
        Ok(())
    }

    /// Restore after a crash using a file-based manifest.
    pub async fn restore_after_crash_from_file(
        &mut self,
        path: impl AsRef<Path>,
        resume_active: bool,
    ) -> RuntimeResult<()> {
        if self.state_machine.state() != RuntimeState::Recovering {
            return Err(RuntimeError::recovery_failed(
                "runtime must be in Recovering state before restore",
            ));
        }

        self.model_load_stage = ModelLoadStage::NotStarted;
        self.load_manifest_from_file(path).await?;
        self.load_base_parameters().await?;
        self.register_experts().await?;
        self.load_router_parameters().await?;
        self.load_optimizer_state().await?;
        self.complete_model_load().await?;

        let manifest_tiers = self
            .manifest
            .as_ref()
            .ok_or_else(|| RuntimeError::recovery_failed("manifest missing during recovery"))?
            .tiers
            .clone();
        let tiers = TierSet {
            tiers: manifest_tiers.tiers,
            cumulative: manifest_tiers.cumulative,
        };
        self.activate_tiers(tiers).await?;

        if resume_active {
            self.start_serving().await?;
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
        let available: std::collections::HashSet<u16> = self
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
    use crate::config::{Placement, RoutingSeed, TierConfig, TierId, TierSet};

    use super::{AsyncCheckpointManifest, AsyncManifestShard, AsyncRuntimeLifecycle, AsyncTierSet, RuntimeOptions};
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

    fn sample_async_manifest() -> AsyncCheckpointManifest {
        AsyncCheckpointManifest {
            version: 1,
            tiers: AsyncTierSet {
                tiers: vec![TierId(1), TierId(10)],
                cumulative: false,
            },
            base_parameters_checksum: "abc123".to_string(),
            router_checksum: "def456".to_string(),
            optimizer_checksum: Some("fff999".to_string()),
            shards: vec![
                AsyncManifestShard {
                    shard_id: "base".to_string(),
                    checksum_hex: "aa11".to_string(),
                    bytes: 1024,
                },
                AsyncManifestShard {
                    shard_id: "exp_1".to_string(),
                    checksum_hex: "bb22".to_string(),
                    bytes: 2048,
                },
            ],
        }
    }

    #[tokio::test]
    async fn async_full_boot_load_activate_flow() {
        let mut lifecycle = AsyncRuntimeLifecycle::new(sample_options(true)).expect("valid options");

        lifecycle.boot().await.expect("boot should succeed");
        assert_eq!(lifecycle.boot_stage(), crate::process::BootStage::LoggingStarted);

        lifecycle
            .parse_manifest(
                "version=1\n\
                tiers=1,10\n\
                cumulative=false\n\
                base_checksum=abc123\n\
                router_checksum=def456\n\
                optimizer_checksum=fff999\n\
                shard=base|aa11|1024\n\
                shard=exp_1|bb22|2048\n",
            )
            .await
            .expect("manifest parse should succeed");

        lifecycle
            .load_base_parameters()
            .await
            .expect("base load should succeed");
        lifecycle
            .register_experts()
            .await
            .expect("expert registration should succeed");
        lifecycle
            .load_router_parameters()
            .await
            .expect("router load should succeed");
        lifecycle
            .load_optimizer_state()
            .await
            .expect("optimizer load should succeed");
        lifecycle
            .complete_model_load()
            .await
            .expect("complete model load should succeed");

        assert_eq!(
            lifecycle.model_load_stage(),
            crate::process::ModelLoadStage::Complete
        );
        assert_eq!(lifecycle.state(), RuntimeState::Warm);

        lifecycle
            .activate_tiers(TierSet::new(vec![TierId(10)], true))
            .await
            .expect("tier activation should succeed");
        lifecycle
            .start_serving()
            .await
            .expect("start serving should succeed");
        assert_eq!(lifecycle.state(), RuntimeState::Active);
    }

    #[tokio::test]
    async fn async_manifest_serialization_roundtrip() {
        let manifest = sample_async_manifest();
        let json = manifest.to_json().expect("serialization should succeed");
        let restored =
            AsyncCheckpointManifest::from_json(&json).expect("deserialization should succeed");
        assert_eq!(manifest, restored);
    }

    #[tokio::test]
    async fn async_manifest_file_roundtrip() {
        let manifest = sample_async_manifest();
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let path = temp_dir.path().join("manifest.json");

        super::write_manifest_to_file(&path, &manifest)
            .await
            .expect("write should succeed");

        let loaded = super::read_manifest_from_file(&path)
            .await
            .expect("read should succeed");

        assert_eq!(manifest, loaded);
    }

    #[tokio::test]
    async fn async_graceful_shutdown_resets_state() {
        let mut lifecycle = AsyncRuntimeLifecycle::new(sample_options(false)).expect("valid options");
        lifecycle.boot().await.expect("boot should succeed");
        lifecycle
            .parse_manifest(
                "version=1\n\
                tiers=1,10\n\
                cumulative=false\n\
                base_checksum=abc123\n\
                router_checksum=def456\n\
                shard=base|aa11|1024\n",
            )
            .await
            .expect("manifest parse should succeed");
        lifecycle.load_base_parameters().await.expect("base load should succeed");
        lifecycle.register_experts().await.expect("expert registration should succeed");
        lifecycle
            .load_router_parameters()
            .await
            .expect("router load should succeed");
        lifecycle
            .complete_model_load()
            .await
            .expect("complete model load should succeed");
        lifecycle
            .activate_tiers(TierSet::new(vec![TierId(1)], false))
            .await
            .expect("tier activation should succeed");
        lifecycle.start_serving().await.expect("start serving should succeed");
        assert_eq!(lifecycle.state(), RuntimeState::Active);

        lifecycle
            .graceful_shutdown()
            .await
            .expect("shutdown should succeed");
        assert_eq!(lifecycle.state(), RuntimeState::Init);
        assert_eq!(lifecycle.boot_stage(), crate::process::BootStage::Init);
        assert_eq!(
            lifecycle.model_load_stage(),
            crate::process::ModelLoadStage::NotStarted
        );
    }

    #[tokio::test]
    async fn async_parse_manifest_requires_boot() {
        let mut lifecycle = AsyncRuntimeLifecycle::new(sample_options(false)).expect("valid options");
        let result = lifecycle.parse_manifest("version=1\ntiers=1,10\n").await;
        assert!(result.is_err());
    }

    #[tokio::test]
    async fn async_recovery_restores_to_active() {
        let mut lifecycle = AsyncRuntimeLifecycle::new(sample_options(true)).expect("valid options");
        lifecycle.boot().await.expect("boot should succeed");
        lifecycle
            .parse_manifest(
                "version=1\n\
                tiers=1,10\n\
                cumulative=false\n\
                base_checksum=abc123\n\
                router_checksum=def456\n\
                optimizer_checksum=fff999\n\
                shard=base|aa11|1024\n\
                shard=exp_1|bb22|2048\n",
            )
            .await
            .expect("manifest parse should succeed");
        lifecycle.load_base_parameters().await.expect("base load should succeed");
        lifecycle.register_experts().await.expect("expert registration should succeed");
        lifecycle
            .load_router_parameters()
            .await
            .expect("router load should succeed");
        lifecycle
            .load_optimizer_state()
            .await
            .expect("optimizer load should succeed");
        lifecycle
            .complete_model_load()
            .await
            .expect("complete model load should succeed");
        lifecycle
            .activate_tiers(TierSet::new(vec![TierId(1), TierId(10)], false))
            .await
            .expect("tier activation should succeed");
        lifecycle.start_serving().await.expect("start serving should succeed");

        let stable_seed = lifecycle.routing_seed();

        lifecycle
            .begin_recovery()
            .await
            .expect("transition to recovery should succeed");

        // Use file-based restore for recovery
        let temp_dir = tempfile::tempdir().expect("create temp dir");
        let path = temp_dir.path().join("recovery_manifest.json");
        let manifest = sample_async_manifest();
        super::write_manifest_to_file(&path, &manifest)
            .await
            .expect("write recovery manifest should succeed");

        lifecycle
            .restore_after_crash_from_file(&path, true)
            .await
            .expect("restore should succeed");

        assert_eq!(lifecycle.state(), RuntimeState::Active);
        assert_eq!(stable_seed, lifecycle.routing_seed());
    }

    #[tokio::test]
    async fn async_tierset_resolve_cumulative() {
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

        let selected = AsyncTierSet::new(vec![TierId(10)], true).resolve(&available);
        assert_eq!(selected, vec![TierId(1), TierId(10)]);
    }

    #[tokio::test]
    async fn async_invalid_manifest_serialization() {
        let bad_json = "not valid json at all";
        let result = AsyncCheckpointManifest::from_json(bad_json);
        assert!(result.is_err());
    }
}
