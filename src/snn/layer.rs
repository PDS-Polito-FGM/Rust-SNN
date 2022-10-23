// * private Layer submodule *
use crate::snn::neuron::Neuron;
use std::sync::mpsc::{Receiver, Sender};
use crate::snn::SpikeEvent;


/* Object representing a Layer of the Spiking Neural Network */
pub struct Layer<N: Neuron> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    rc: Receiver<SpikeEvent>,
    tx: Sender<SpikeEvent>,
}

impl<N: Neuron> Layer<N> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>,
        rc: Receiver<SpikeEvent>,
        tx: Sender<SpikeEvent>,
    ) -> Self {
        Self {
            neurons,
            weights,
            intra_weights,
            rc,
            tx,
        }
    }
}
