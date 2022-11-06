use crate::snn::neuron::Neuron;

// * LIF submodule *

/**
    Object representing a Neuron in the LIF (Leaky Integrate-and-Fire) model
*/
#[derive(Debug)]
pub struct LifNeuron {
    v_th:    f64,       /* threshold potential  */
    v_rest:  f64,       /* resting potential    */
    v_reset: f64,       /* reset potential      */
    tau:     f64,
    v_mem:   f64,       /* membrane potential   */
    ts:      u64,       /* last instant in which receiving at least one spike */
}

impl LifNeuron {
    pub fn new(v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> Self {
        Self {
            v_th,
            v_rest,
            v_reset,
            tau,
            v_mem: v_rest,      // TODO Note: we have to decide how to initialize this value - Francesco
                                // TODO Note: v_rest could be a good option - Mario
            ts: 0u64,
        }
    }
}

impl Neuron for LifNeuron {
    fn compute_v_mem(&mut self, t: u64, extra_weighted_sum: f64, intra_weighted_sum: f64) -> u8 {
        // update neuron membrane potential *subtracting* the intra_weighted_sum from the previous t_s
        self.v_mem += intra_weighted_sum;   /* negative contribute */

        // * compute the neuron membrane potential with the LIF formula *

        let exponent = -(((t - self.ts) as f64) / self.tau);
        self.v_mem = self.v_rest + (self.v_mem - self.v_rest) * exponent.exp() + extra_weighted_sum;

        // * update ts - last instant in which at least one spike (1) is received *
        self.ts = t;

        // TODO Note: if the neuron fires, we have to decrement all the other
        //            intra weights of the other neurons in the same layer - Francesco

        return if self.v_mem > self.v_th {
            // reset membrane potential
            self.v_mem = self.v_reset;
            1   // * fire *
        } else {
            0   // not fire
        };
    }
}

impl Clone for LifNeuron {
    fn clone(&self) -> Self {
        Self {
            v_th:    self.v_th,
            v_rest:  self.v_rest,
            v_reset: self.v_reset,
            tau:     self.tau,
            v_mem:   self.v_mem,
            ts:      self.ts
        }
    }
}

unsafe impl Send for LifNeuron {}
