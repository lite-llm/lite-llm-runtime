# Changelog

All notable changes to `lite-llm-runtime` are documented in this file.

## [0.1.0] - 2026-03-01
### Added
- Runtime lifecycle implementation for boot, manifest parse, load phases, tier activation, and recovery.
- Deterministic router with stable top-k and seeded tie-break behavior.
- Compute-bound enforcement helpers and invariants.
- Runtime state machine and runtime error model.
- Deterministic unit tests covering lifecycle and routing replay behavior.
