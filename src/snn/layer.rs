// * private Layer submodule *
use crate::snn::neuron::Neuron;
use std::sync::mpsc::{Receiver, Sender};
use crate::snn::SpikeEvent;

/* Object representing a Layer of the Spiking Neural Network */
#[derive(Debug)]
pub struct Layer<N: Neuron + Clone + Send + 'static> {
    neurons: Vec<N>,
    weights: Vec<Vec<f64>>,
    intra_weights: Vec<Vec<f64>>,
    prev_output_spikes: Vec<u8>
}

impl<N: Neuron + Clone + Send + 'static> Layer<N> {
    pub fn new(
        neurons: Vec<N>,
        weights: Vec<Vec<f64>>,
        intra_weights: Vec<Vec<f64>>
    ) -> Self {
        let num_neurons = neurons.len();
        Self {
            neurons,
            weights,
            intra_weights,
            prev_output_spikes: vec![0; num_neurons]
        }
    }

    //Getters of the layer
    pub fn get_neurons_number(&self) -> usize {
        self.neurons.len()
    }

    pub fn get_weights(&self) -> Vec<Vec<f64>> {
        self.weights.clone()
    }

    pub fn get_intra_weights(&self) -> Vec<Vec<f64>> {
        self.intra_weights.clone()
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
            let mut at_least_one_spike = false;

            // * for each neuron compute the intra and the extra weighted sums,
            //   then retrieve the output spike *
            for index in 0..self.neurons.len() {
                let neuron = &mut self.neurons[index];
                let mut intra_weighted_sum = 0f64;
                let mut extra_weighted_sum = 0f64;

                // compute extra weighted sum
                let extra_weights_pairs =
                    self.weights[index].iter().zip(input_spike_event.spikes.iter());

                for (weight, spike) in extra_weights_pairs {
                    if *spike != 0 {
                        extra_weighted_sum += *weight;
                    }
                }

                // compute intra weighted sum
                let intra_weights_pairs =
                    self.intra_weights[index].iter().zip(self.prev_output_spikes.iter());

                for (i, (weight, spike)) in intra_weights_pairs.enumerate() {
                    // ignore the reflexive link
                    if i != index && *spike != 0 {
                        intra_weighted_sum += *weight;
                    }
                }

                // * compute neuron membrane potential and determine if it fires or not *
                let neuron_spike = neuron.compute_v_mem(instant, extra_weighted_sum, intra_weighted_sum);
                output_spikes.push(neuron_spike);

                if !at_least_one_spike && neuron_spike > 0u8 {
                    at_least_one_spike = true;
                }
            }

            // save output spikes for later
            self.prev_output_spikes = output_spikes.clone();

            // * check if at least one neuron fired - if not, not send any spike *
            if !at_least_one_spike {
                continue;
            }
            // at least one neuron fired -> send output spikes to the next layer

            let output_spike_event = SpikeEvent::new(instant, output_spikes);

            layer_output_tx.send(output_spike_event)
                .expect(&format!("Unexpected error sending input spike event t={}", instant));
        }

        // * we don't need to drop the sender, because it will be
        // automatically dropped when the layer goes out of scope *
    }
}

unsafe impl<N: Neuron + Clone> Sync for Layer<N> {}

impl<N: Neuron + Clone + Send + 'static> Clone for Layer<N> {
    fn clone(&self) -> Self {
        Self {
            neurons: self.neurons.clone(),
            weights: self.weights.clone(),
            intra_weights: self.intra_weights.clone(),
            prev_output_spikes: self.prev_output_spikes.clone()
        }
    }
}