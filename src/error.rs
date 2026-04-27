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
    InvalidBootOrder(String),
    InvalidLoadOrder(String),
    InvalidManifest(String),
    UnsupportedManifestVersion {
        expected: u32,
        found: u32,
    },
    UnknownTier(u16),
    EmptyTierSet,
    InvalidRoutingConfig(String),
    ComputeBoundExceeded {
        actual: usize,
        max: usize,
    },
    RecoveryFailed(String),
    IoError(String),
    SerializationError(String),
}

impl RuntimeError {
    pub fn invalid_boot_order(msg: impl Into<String>) -> Self {
        Self::InvalidBootOrder(msg.into())
    }

    pub fn invalid_load_order(msg: impl Into<String>) -> Self {
        Self::InvalidLoadOrder(msg.into())
    }

    pub fn invalid_manifest(msg: impl Into<String>) -> Self {
        Self::InvalidManifest(msg.into())
    }

    pub fn invalid_routing_config(msg: impl Into<String>) -> Self {
        Self::InvalidRoutingConfig(msg.into())
    }

    pub fn recovery_failed(msg: impl Into<String>) -> Self {
        Self::RecoveryFailed(msg.into())
    }

    pub fn io_error(msg: impl Into<String>) -> Self {
        Self::IoError(msg.into())
    }

    pub fn serialization_error(msg: impl Into<String>) -> Self {
        Self::SerializationError(msg.into())
    }
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
            Self::IoError(msg) => write!(f, "io error: {msg}"),
            Self::SerializationError(msg) => write!(f, "serialization error: {msg}"),
        }
    }
}

impl Error for RuntimeError {
    fn source(&self) -> Option<&(dyn Error + 'static)> {
        None
    }
}
