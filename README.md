# lite-llm-runtime

Core runtime crate for Lite LLM (`SPEC-001` to `SPEC-010`).

## Scope
Implements deterministic runtime primitives:

- runtime lifecycle boot/load/activation/recovery flow
- deterministic routing engine with stable top-k behavior
- TierSet and routing seed contracts
- active compute-bound invariants
- runtime error model and state machine

## Modules
- `src/process.rs`: boot model, manifest parsing/validation, load phases, recovery
- `src/routing.rs`: deterministic router, stable tie-break ordering
- `src/config.rs`: Tier and routing configuration primitives
- `src/compute.rs`: compute-bound enforcement and model helpers
- `src/state_machine.rs`: runtime transition model
- `src/error.rs`: runtime error model

## Build and Test
```bash
cargo fmt
cargo test
```

## Documentation
- System docs: `../lite-llm-docs/README.md`
- Architecture: `../lite-llm-docs/architecture/system-architecture.md`

## Changelog
See `CHANGELOG.md`.

## License
See `LICENSE`.
