// * private Layer submodule *
use crate::snn::neuron::Neuron;
use std::sync::mpsc::{Receiver, Sender};
use crate::snn::SpikeEvent;


/* Object representing a Layer of the Spiking Neural Network */
pub struct Layer<N: Neuron+'static> {
    _neurons: Vec<N>,
    _weights: Vec<Vec<f64>>,
    _intra_weights: Vec<Vec<f64>>
    // TODO Note: I removed the tx and rc from here as well for the same reasons of the SNN - Mario
}


impl<N: Neuron+'static> Layer<N> {
    pub fn new(
        _neurons: Vec<N>,
        _weights: Vec<Vec<f64>>,
        _intra_weights: Vec<Vec<f64>>
    ) -> Self {
        Self {
            _neurons,
            _weights,
            _intra_weights
        }
    }

    /** It processes the input SpikeEvent(s) from the previous layer, according to the model
        of the Neurons in the network
        - layer_input_rc: is a (sync) channel receiver from the previous layer
        - layer_output_tx: is a (sync) channel sender to the next Network Layer
            (or to the SNN itself, if this is the output layer) */
    pub fn process(&self, _layer_input_rc: Receiver<SpikeEvent> , _layer_output_tx: Sender<SpikeEvent>) {
        todo!()

        // TODO: wait on the channel rc for new input

        // TODO: process input (loop on the neurons, etc.);
        //       recall to ignore the right matrix elements in the intra_weights
        //       (the ones corresponding to reflexive links)

        // TODO: send output to the next layer
    }
}

unsafe impl<N: Neuron> Sync for Layer<N> {}
