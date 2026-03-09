use std::fmt;

use crate::error::{RuntimeError, RuntimeResult};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RuntimeState {
    Init,
    Warm,
    Active,
    Expanding,
    Frozen,
    Recovering,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeTransitionError {
    pub from: RuntimeState,
    pub to: RuntimeState,
}

impl fmt::Display for RuntimeTransitionError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "invalid transition: {:?} -> {:?}", self.from, self.to)
    }
}

impl RuntimeState {
    pub fn can_transition(self, to: RuntimeState) -> bool {
        matches!(
            (self, to),
            (RuntimeState::Init, RuntimeState::Warm)
                | (RuntimeState::Warm, RuntimeState::Active)
                | (RuntimeState::Active, RuntimeState::Expanding)
                | (RuntimeState::Expanding, RuntimeState::Active)
                | (RuntimeState::Active, RuntimeState::Frozen)
                | (RuntimeState::Frozen, RuntimeState::Warm)
                | (RuntimeState::Frozen, RuntimeState::Active)
                | (_, RuntimeState::Recovering)
                | (RuntimeState::Recovering, RuntimeState::Warm)
                | (RuntimeState::Recovering, RuntimeState::Active)
        )
    }

    pub fn transition(self, to: RuntimeState) -> Result<RuntimeState, RuntimeTransitionError> {
        if self.can_transition(to) {
            Ok(to)
        } else {
            Err(RuntimeTransitionError { from: self, to })
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct RuntimeStateMachine {
    state: RuntimeState,
}

impl Default for RuntimeStateMachine {
    fn default() -> Self {
        Self::new()
    }
}

impl RuntimeStateMachine {
    pub fn new() -> Self {
        Self {
            state: RuntimeState::Init,
        }
    }

    pub fn state(self) -> RuntimeState {
        self.state
    }

    pub fn transition(&mut self, to: RuntimeState) -> RuntimeResult<()> {
        self.state = self
            .state
            .transition(to)
            .map_err(|err| RuntimeError::InvalidStateTransition {
                from: err.from,
                to: err.to,
            })?;
        Ok(())
    }

    pub fn reset_to_init(&mut self) {
        self.state = RuntimeState::Init;
    }
}

#[cfg(test)]
mod tests {
    use super::{RuntimeState, RuntimeStateMachine};

    fn replay(sequence: &[RuntimeState]) -> Vec<RuntimeState> {
        let mut machine = RuntimeStateMachine::new();
        let mut states = vec![machine.state()];

        for state in sequence {
            machine
                .transition(*state)
                .expect("sequence should be valid");
            states.push(machine.state());
        }

        states
    }

    #[test]
    fn transition_rules_match_spec() {
        let mut machine = RuntimeStateMachine::new();
        machine
            .transition(RuntimeState::Warm)
            .expect("init -> warm should succeed");
        machine
            .transition(RuntimeState::Active)
            .expect("warm -> active should succeed");
        assert!(machine.transition(RuntimeState::Init).is_err());
    }

    #[test]
    fn replay_is_deterministic_for_state_transitions() {
        let sequence = [
            RuntimeState::Warm,
            RuntimeState::Active,
            RuntimeState::Expanding,
            RuntimeState::Active,
            RuntimeState::Frozen,
            RuntimeState::Recovering,
            RuntimeState::Warm,
            RuntimeState::Active,
        ];

        let first = replay(&sequence);
        let second = replay(&sequence);
        assert_eq!(first, second);
    }
}
