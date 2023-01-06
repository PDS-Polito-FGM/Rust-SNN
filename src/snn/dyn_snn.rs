use std::slice::IterMut;
use std::sync::{Arc, Mutex};
use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::snn::processor::Processor;
use crate::SpikeEvent;

/* * Dynamic Spiking Neural Network structure * */

/**
    Object representing the (dynamic) Spiking Neural Network itself
    - N: is a generic type representing the Neuron implementation

    There aren't generic const variables such as NET_INPUT_DIM, NET_OUTPUT_DIM, etc. because all the
    computations and checks are done at *runtime*.
*/
#[derive(Debug, Clone)]
pub struct DynSNN<N: Neuron + Clone + 'static> {
    layers: Vec<Arc<Mutex<Layer<N>>>>
}

impl<N: Neuron + Clone> DynSNN<N> {
    pub fn new(layers: Vec<Arc<Mutex<Layer<N>>>>) -> Self {
        Self { layers }
    }

    /* Getters */
    pub fn get_layers_number(&self) -> usize {
        self.layers.len()
    }

    fn get_input_layer_dimension(&self) -> usize {
        let first_layer = self.layers[0].lock().unwrap();
        let input_layer_dimension = first_layer.get_weights().first().unwrap().len();

        input_layer_dimension
    }

    fn get_output_layer_dimension(&self) -> usize {
        let last_layer = self.layers.last().unwrap().lock().unwrap();
        let output_dimension = last_layer.get_neurons_number();

        output_dimension
    }

    pub fn get_layers(&self) -> Vec<Layer<N>> {
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /**
        Actually process input spikes by means of the Spiking Neural Network and produce corresponding output spikes.
        'spikes' contains an array for each input layer's neuron, and each array has the same
        number of spikes, equal to the duration of the input
        (spikes is a matrix, one row for each input neuron, and one column for each time instant)
        This method check user input at *run-time*
    */
    pub fn process(&mut self, spikes: &Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // * check and compute the spikes duration *
        let spikes_duration = self.compute_spikes_duration(spikes);

        let input_layer_dimension = self.get_input_layer_dimension();
        let output_layer_dimension = self.get_output_layer_dimension();

        // * encode spikes into SpikeEvent(s) *
        let input_spike_events =
            DynSNN::<N>::encode_spikes(input_layer_dimension, spikes, spikes_duration);

        // * process input *
        let processor = Processor{};
        let output_spike_events = processor.process_events(self,input_spike_events);

        // * decode output into array shape *
        let decoded_output =  DynSNN::<N>::decode_spikes(output_layer_dimension,
                                                         output_spike_events, spikes_duration);

        decoded_output
    }

    /**
        This function checks if each vector passed in 'spikes' has the same number of spikes.
        If yes, it returns the duration, otherwise it triggers an error
     */
    fn compute_spikes_duration(&self, spikes: &Vec<Vec<u8>>) -> usize {
        // compute length of the first Vec (0 if it does not exist)
        let spikes_duration = spikes.get(0)
                                            .unwrap_or(&Vec::new())
                                            .len();

        for neuron_spikes in spikes {
            if neuron_spikes.len() != spikes_duration {
                panic!("The number of spikes duration must be equal for each neuron");
            }
        }
        spikes_duration
    }

    /**
        This function encodes the received input spikes in a Vec of **SpikeEvent** to process them.
     */
    fn encode_spikes(input_layer_dimension: usize, spikes: &Vec<Vec<u8>>, spikes_duration: usize) -> Vec<SpikeEvent> {
        let mut spike_events = Vec::<SpikeEvent>::new();

        if spikes.len() != input_layer_dimension {
            panic!("The number of input spikes is not coherent with the input layer dimension: \
                    'spikes' must have a Vec for each neuron");
        }

        for t in 0..spikes_duration {
            let mut t_spikes = Vec::<u8>::new();

            /* retrieve the input spikes for each neuron */
            for in_neuron_index in 0..spikes.len(){
                /* check for 0 or 1 only */
                if spikes[in_neuron_index][t] != 0 && spikes[in_neuron_index][t] != 1 {
                    panic!("Error: input spike must be 0 or 1 at for N={} at t={}", in_neuron_index, t);
                }
                t_spikes.push(spikes[in_neuron_index][t]);
            }

            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }

        spike_events
    }

    /**
        This function decodes a Vec of SpikeEvents and returns an output spikes matrix of 0/1
     */
    fn decode_spikes(output_layer_dimension: usize, spikes: Vec<SpikeEvent>, spikes_duration: usize) -> Vec<Vec<u8>> {
        let mut raw_spikes  = vec![vec![0; spikes_duration]; output_layer_dimension];

        for spike_event in spikes {
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
                raw_spikes[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }

        raw_spikes
    }
}

impl<'a, N: Neuron + Clone + 'static> IntoIterator for &'a mut DynSNN<N> {
    type Item = &'a mut Arc<Mutex<Layer<N>>>;
    type IntoIter = IterMut<'a, Arc<Mutex<Layer<N>>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
