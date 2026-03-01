use crate::config::TierSet;

#[derive(Debug, Clone)]
pub struct RuntimeOptions {
    pub deterministic_seed: u128,
    pub initial_tiers: TierSet,
}

#[derive(Debug, Clone)]
pub struct RuntimeLifecycle {
    pub options: RuntimeOptions,
    booted: bool,
}

impl RuntimeLifecycle {
    pub fn new(options: RuntimeOptions) -> Self {
        Self {
            options,
            booted: false,
        }
    }

    pub fn boot(&mut self) {
        self.booted = true;
    }

    pub fn model_load(&self) -> Result<(), &'static str> {
        if !self.booted {
            return Err("runtime must be booted before model_load");
        }
        Ok(())
    }

    pub fn activate_tiers(&mut self, tiers: TierSet) {
        self.options.initial_tiers = tiers;
    }

    pub fn shutdown(&mut self) {
        self.booted = false;
    }
}

