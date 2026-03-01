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

impl RuntimeState {
    pub fn transition(self, to: RuntimeState) -> Result<RuntimeState, RuntimeTransitionError> {
        let valid = matches!(
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
        );

        if valid {
            Ok(to)
        } else {
            Err(RuntimeTransitionError { from: self, to })
        }
    }
}

