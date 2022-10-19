use std::sync::mpsc::channel;
use crate::snn::layer::{Layer, SpikeEvent};
use crate::snn::neuron::Neuron;

// * submodules *
pub mod builder;
    mod layer;     // private
pub mod neuron;

// * SNN module *

/**
    Object representing the Spiking Neural Network itself
 */
pub struct SNN<N: Neuron> {    // test
    pub s: bool,
    pub l: Layer<N>
}

// TODO: implement SNN struct
impl<N: Neuron> SNN<N> {      // test
    pub fn new(s: bool, l: f64) -> Self {
        let (tx,rc) = channel::<SpikeEvent>();
        let la: Layer<N> = Layer::new(vec![],vec![], vec![], rc, tx);
        Self {
            s,
            l: la
        }
    }
}
