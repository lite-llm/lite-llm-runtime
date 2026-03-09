use std::error::Error;
use std::fmt;

use crate::state_machine::RuntimeState;

pub type RuntimeResult<T> = Result<T, RuntimeError>;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RuntimeError {
    InvalidStateTransition {
        from: RuntimeState,
        to: RuntimeState,
    },
    InvalidBootOrder(&'static str),
    InvalidLoadOrder(&'static str),
    InvalidManifest(&'static str),
    UnsupportedManifestVersion {
        expected: u32,
        found: u32,
    },
    UnknownTier(u16),
    EmptyTierSet,
    InvalidRoutingConfig(&'static str),
    ComputeBoundExceeded {
        actual: usize,
        max: usize,
    },
    RecoveryFailed(&'static str),
}

impl fmt::Display for RuntimeError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidStateTransition { from, to } => {
                write!(f, "invalid state transition: {:?} -> {:?}", from, to)
            }
            Self::InvalidBootOrder(msg) => write!(f, "invalid boot order: {msg}"),
            Self::InvalidLoadOrder(msg) => write!(f, "invalid model loading order: {msg}"),
            Self::InvalidManifest(msg) => write!(f, "invalid manifest: {msg}"),
            Self::UnsupportedManifestVersion { expected, found } => write!(
                f,
                "unsupported manifest version: expected {expected}, found {found}"
            ),
            Self::UnknownTier(tier) => write!(f, "unknown tier id: {tier}"),
            Self::EmptyTierSet => write!(f, "tierset cannot be empty"),
            Self::InvalidRoutingConfig(msg) => write!(f, "invalid routing config: {msg}"),
            Self::ComputeBoundExceeded { actual, max } => write!(
                f,
                "active experts exceeded compute bound: actual {actual}, max {max}"
            ),
            Self::RecoveryFailed(msg) => write!(f, "recovery failed: {msg}"),
        }
    }
}

impl Error for RuntimeError {}
