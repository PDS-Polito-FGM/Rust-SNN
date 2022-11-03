use std::borrow::BorrowMut;
// * private Layer submodule *
use crate::snn::neuron::Neuron;
use std::sync::mpsc::{Receiver, Sender};
use crate::snn::SpikeEvent;


/* Object representing a Layer of the Spiking Neural Network */
#[derive(Debug)]
pub struct Layer<N: Neuron + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_output_spikes: Vec<u8>
}


impl<N: Neuron + Send + 'static> Layer<N> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>,
        prev_output_spikes: Vec<u8>
    ) -> Self {
        Self {
            neurons,
            weights,
            intra_weights,
            prev_output_spikes
        }
    }

    /** It processes the input SpikeEvent(s) from the previous layer, according to the model
        of the Neurons in the network
        - layer_input_rc: is a channel receiver from the previous layer
        - layer_output_tx: is a channel sender to the next Network Layer
            (or to the SNN itself, if this is the output layer) */
    pub fn process(&mut self, layer_input_rc: Receiver<SpikeEvent>, layer_output_tx: Sender<SpikeEvent>) {

        while let Ok(input_spike_event) = layer_input_rc.recv() {

            let instant = input_spike_event.ts;
            let mut output_spikes = Vec::<u8>::with_capacity(self.neurons.len());

            // * computing for each neuron the intra and extra weighted sums, retrieving also their output spikes *
            for index in 0..self.neurons.len() {
                let neuron = self.neurons[index].borrow_mut();
                let mut intra_weighted_sum = 0f64;
                let mut extra_weighted_sum = 0f64;

                let extra_weights = self.weights[index].iter().zip(input_spike_event.spikes.iter())
                    .map(|(weight,spike)| (weight.clone(),spike.clone())).collect::<Vec<(f64,u8)>>();

                for (weight,spike) in extra_weights {
                    if spike != 0 {
                        extra_weighted_sum += weight;
                    }
                }

                let mut intra_weights = self.intra_weights[index].iter().zip(self.prev_output_spikes.iter())
                    .map(|(weight,spike)| (weight.clone(),spike.clone())).collect::<Vec<(f64,u8)>>();

                // remove the reflexive link
                intra_weights.remove(index);

                for (weight,spike) in intra_weights {
                    if spike != 0 {
                        intra_weighted_sum += weight;
                    }
                }

                let neuron_spike = neuron.compute_v_mem(instant,extra_weighted_sum,intra_weighted_sum);
                output_spikes.push(neuron_spike);
            }

            let output_spike_event = SpikeEvent::new(instant,output_spikes.clone());

            layer_output_tx.send(output_spike_event)
                .expect(&format!("Unexpected error sending input spike event t={}", instant));

            self.prev_output_spikes = output_spikes;
        }

        // we don't need to drop the sender, because it will be dropped automatically when the layer goes out of scope

    }
}

unsafe impl<N: Neuron> Sync for Layer<N> {}