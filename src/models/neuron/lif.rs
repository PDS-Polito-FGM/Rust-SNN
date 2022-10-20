
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
    ts_prev: u64
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64, v_mem_ts_prev: f64, ts_prev: u64) -> Self {
        Self { v_th, v_rest, v_reset, tau, v_mem_ts_prev, ts_prev }
    }
}

impl Neuron for LifNeuron {
    fn compute_v_mem(&mut self,t: u64, weighted_sum: f64) -> u8 {
        let exponent = -( ((t - self.ts_prev) as f64 ) / self.tau);
        let v_mem = self.v_rest + (self.v_mem_ts_prev - self.v_rest)  * exponent.exp() + weighted_sum;
        self.v_mem_ts_prev = v_mem;
        return if v_mem > self.v_th {
            self.ts_prev = t;
            1
        } else {
            0
        }
    }
}

// TODO: implement LIF Neuron struct - implement Neuron Trait
