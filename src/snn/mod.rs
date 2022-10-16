use crate::snn::layer::Layer;

// * submodules *
pub mod builder;
    mod layer;     // private
pub mod neuron;

// * SNN module *

/**
    Object representing the Spiking Neural Network itself
 */
pub struct SNN {    // test
    pub s: bool,
    pub l: Layer
}

// TODO: implement SNN struct
impl SNN {      // test
    pub fn new(s: bool, l: f64) -> Self {
        let la = Layer{l};
        Self {
            s,
            l: la
        }
    }
}
