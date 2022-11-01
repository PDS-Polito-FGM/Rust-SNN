use crate::snn::neuron::Neuron;

// * LIF submodule *

/**
    Object representing a Neuron in the LIF (Leaky Integrate-and-Fire) model
*/
pub struct LifNeuron {
    v_th: f64,
    v_rest: f64,
    v_reset: f64,
    tau: f64,
    v_mem_ts_prev: f64,
    ts_prev: u64,
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
            v_mem_ts_prev: 0f64,
            ts_prev: 0u64,
        }
    }
}

impl Neuron for LifNeuron {
    fn compute_v_mem(&mut self, t: u64, extra_weighted_sum: f64, intra_weighted_sum: f64) -> u8 {
        // * compute the neuron membrane potential with the LIF formula *
        let exponent = -(((t - self.ts_prev) as f64) / self.tau);
        let weighted_sum = extra_weighted_sum + intra_weighted_sum;
        let v_mem =
            self.v_rest + (self.v_mem_ts_prev - self.v_rest) * exponent.exp() + weighted_sum;

        // check if the neuron has received at least 1 spike from the previous layer
        // if so, *update ts_prev*
        if extra_weighted_sum > (0 as f64) {
            self.ts_prev = t;
        }

        return if v_mem > self.v_th {
            // reset membrane potential
            self.v_mem_ts_prev = self.v_reset;
            1 // * fire *
        } else {
            // save membrane potential for later
            self.v_mem_ts_prev = v_mem;
            0
        };
    }
}
