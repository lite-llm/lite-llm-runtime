# lite-llm-runtime

Core runtime crate for Lite LLM (`SPEC-001` to `SPEC-010`).

## Overview
Implements deterministic runtime primitives including lifecycle management, routing, compute bounds, configuration, state machine, and async process management with file-based persistence.

This crate provides the foundational runtime layer: boot/load/activation/recovery flow with async manifest management, deterministic routing engine with stable top-k behavior and quantized weight storage, TierSet and routing seed contracts, active compute-bound invariants, runtime state machine with transition validation, async-capable checkpoint manifest types with serde serialization, and concurrent async routing table with file-based persistence.

## Features

### Feature Flag: `default` (empty)
All features enabled by default. No optional feature flags.

## Dependencies
| Crate | Version | Purpose |
|-------|---------|---------|
| tokio | 1 | Async runtime for file I/O and concurrent access |
| serde | 1.0 | Serialization for manifests and routing tables |
| serde_json | 1.0 | JSON encoding for persistence |

## Key Modules
- `process` — boot model, manifest parsing/validation, load phases, recovery
- `async_process` — async runtime lifecycle, serde-serializable manifest types
- `routing` — deterministic router, stable tie-break ordering, expert key routing
- `async_routing` — concurrent routingTable with compute, cache, and JSON persistence
- `config` — Tier and routing configuration primitives (`TierConfig`, `TierSet`, `RoutingSeed`)
- `compute` — compute-bound enforcement and model helpers
- `state_machine` — runtime transition model with state validation
- `error` — runtime error model (boot, routing, I/O, serialization)

## Public API
### Core Types
- `RuntimeLifecycle` — synchronous runtime boot and model loading
- `AsyncRuntimeLifecycle` — async runtime lifecycle with manifest persistence
- `AsyncCheckpointManifest` — serializable manifest with tier and shard metadata
- `AsyncManifestShard` — individual shard entry with checksum
- `AsyncTierSet` — serializable tier set with cumulative resolution
- `DeterministicRouter` / `Router` — deterministic routing with stable top-k
- `AsyncRoutingTable` — concurrent routing table with file persistence
- `AsyncRoute` — serializable route representation
- `RoutingTableSnapshot` — full table snapshot for persistence
- `RuntimeStateMachine` — state transition validator (Init → Warm → Active → Frozen)
- `CheckpointManifest` / `ManifestShard` — sync manifest types
- `RuntimeOptions` — boot configuration with seed, tiers, manifest version
- `TierConfig` / `TierSet` / `TierId` — tier definitions and sets
- `RoutingSeed` / `RoutingConfig` — seeding and routing configuration
- `Route` / `ExpertKey` — routing decision and expert identification
- `BootStage` / `ModelLoadStage` — boot and load phase enumerations
- `RuntimeState` — runtime state enumeration
- `ComputeBoundModel` — compute-bound enforcement wrapper

### Core Functions
- `stable_top_k()` — stable top-k selection with seeded tie-breaking
- `enforce_active_expert_bound()` — active expert bound validation
- `read_manifest_from_file()` / `write_manifest_to_file()` — async manifest I/O
- `fnv1a64()` / `fnv64_hex()` — FNV-1a hash utilities

### Traits
- None (concrete implementations only)

## Quick Start
```rust
use lite_llm_runtime::{
    AsyncRuntimeLifecycle, RuntimeOptions, RoutingSeed,
    TierConfig, TierId, Placement, TierSet,
};

// Configure runtime options
let options = RuntimeOptions {
    routing_seed: RoutingSeed::new(42),
    available_tiers: vec![
        TierConfig {
            id: TierId(1), groups: 4, experts_per_group: 4,
            placement: Placement::Hot,
        },
        TierConfig {
            id: TierId(2), groups: 4, experts_per_group: 4,
            placement: Placement::Warm,
        },
    ],
    expected_manifest_version: 1,
    training_mode: false,
};

// Boot and load manifest
let mut lifecycle = AsyncRuntimeLifecycle::new(options)?;
lifecycle.boot().await?;

lifecycle.parse_manifest(
    "version=1\ntiers=1,2\ncumulative=false\n\
     base_checksum=abc123\nrouter_checksum=def456\n\
     shard=base|aa11|1024\n"
).await?;

lifecycle.load_base_parameters().await?;
lifecycle.register_experts().await?;
lifecycle.load_router_parameters().await?;
lifecycle.complete_model_load().await?;

// Activate and serve
lifecycle.activate_tiers(TierSet::new(vec![TierId(1)], false)).await?;
lifecycle.start_serving().await?;
```

## Running Tests
```bash
cargo fmt
cargo test
```

## Architecture
This crate implements the core runtime layer for the lite-llm platform. The async process module (`async_process`) provides file-based manifest persistence for crash recovery. The async routing module (`async_routing`) enables concurrent route computation with JSON serialization for routing state replay. It serves as the foundation for `lite-llm-distributed`, `lite-llm-storage`, `lite-llm-training`, and `lite-llm-inference` crates.

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
