use std::slice::IterMut;
use std::sync::{Arc, Mutex};
use crate::snn::layer::Layer;
use crate::snn::neuron::Neuron;
use crate::snn::processor::Processor;
use crate::snn::SpikeEvent;

/* * Spiking Neural Network structure * */

/**
    Object representing the Spiking Neural Network itself
    - N: is the generic type representing the Neuron implementation
    - NET_INPUT_DIM: is the input dimension of the network, i.e. the size of the input layer
    - NET_OUTPUT_DIM: is the output dimension of the network, i.e. the size of the output layer
    Having a generic constant type such as NET_INPUT_DIM allows to check at compile time
    the size of the input provided by the user
 */
#[derive(Debug)]
pub struct SNN<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize> {
    layers: Vec<Arc<Mutex<Layer<N>>>>,
}

impl<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize>
SNN<N, NET_INPUT_DIM, NET_OUTPUT_DIM> {
    pub fn new(layers: Vec<Arc<Mutex<Layer<N>>>>) -> Self {
        Self {
            layers
        }
    }

    /* Getters for the SNN object */
    pub fn get_layers_number(&self) -> usize {
        self.layers.len()
    }

    pub fn get_layers(&self) -> Vec<Layer<N>> {
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /**
        Actually process input spikes by means of the Spiking Neural Network and produce corresponding output spikes
        - 'spikes' contains a binary array for each input layer's neuron, and each array has the same
        number of spikes, equal to the duration of the input (spikes is a matrix of 0/1,
        one row for each input neuron, and one column for each time instant).
        This method is able to check user input at compile-time.
        Ex:
            snn.process(&[[0,1,1], [1,0,1]])  /* input layer with 2 neurons, each receiving 3 spikes */
     */
    pub fn process<const SPIKES_DURATION: usize>(&mut self, spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
                                                 -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {
        /* encode spikes into SpikeEvent(s) */
        let input_spike_events = SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::encode_spikes(spikes);

        /* process input and produce SNN output spikes */
        let processor = Processor {};
        let output_spike_events = processor.process_events(self, input_spike_events);

        /* decode output into array shape */
        let decoded_output: [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] =
            SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::decode_spikes(output_spike_events);

        decoded_output
    }

     /**
        (same as process(), but it checks input spikes sizes at *run-time*:
        spikes must have a number of Vec(s) equal to NET_INPUT_DIM, and all
        these Vec(s) must have the same length), otherwise panic!()
     */
    pub fn process_dyn(&mut self, spikes: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        /* check num of spikes vec(s) */
        if spikes.len() != NET_INPUT_DIM {
            panic!("Error: dimensions mismatch - each input layer's neuron must have its own spikes vec");
        }

        /* encode input spikes in spike events */

        let mut spikes_events = Vec::<SpikeEvent>::new();
        let mut spikes_duration: Option<usize> = None;

        for (n, neuron_spikes) in spikes.into_iter().enumerate() {
            let temp_len = neuron_spikes.len();

            /* check spikes durations - they must have all the same size */
            match spikes_duration {
                None => spikes_duration = Some(temp_len),
                Some(duration) => if temp_len != duration {
                    panic!("Error: different size spikes vec(s) found \
                            - spikes must have the same duration for each input layer's neuron")
                }
            }

            if n == 0 {    /* the first cycle... */
                /* ...create all the spike events */
                (0..spikes_duration.unwrap()).for_each(|t| {
                    let spike_event = SpikeEvent::new(t as u64, Vec::<u8>::new());
                    spikes_events.push(spike_event);
                });
            }

            /* copy each spike in the spike_events vec */
            for t in 0..spikes_duration.unwrap() {
                let temp_spike = neuron_spikes[t];
                if temp_spike != 0 && temp_spike != 1 {
                    panic!("Error: input spike must be 0 or 1 for neuron {} in t={}", n, t);
                }

                spikes_events[t].spikes.push(temp_spike);
            }
        }

        /* run SNN */
         let processor = Processor {};
         let output_spike_events = processor.process_events(self, spikes_events);

        /* decode output spikes events */

        /* create and initialize output object */
        let mut output_spikes: Vec<Vec<u8>> = Vec::new();

        for _ in &output_spike_events.get(0)
            .unwrap_or(&SpikeEvent::new(0, Vec::<u8>::new()))
            .spikes {
            /* create as many internal Vec<u8> as the length of the first output spike_event (num of output neurons) */
            output_spikes.push(vec![0u8; spikes_duration.unwrap()]);
        }

        /* copy processed spikes in the output spikes vec */
        for spike_event in output_spike_events {
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
                output_spikes[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }

        output_spikes
    }

    /* private functions */

    /**
        This function encodes the input spikes matrix (of 0/1) into a Vec of SpikeEvents
    */
    fn encode_spikes<const SPIKES_DURATION: usize>(spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
        -> Vec<SpikeEvent> {

        let mut spike_events = Vec::<SpikeEvent>::new();

        for t in 0..SPIKES_DURATION {
            let mut t_spikes = Vec::<u8>::new();

            /* retrieve the input spikes for each neuron */
            for in_neuron_index in 0..NET_INPUT_DIM {
                if spikes[in_neuron_index][t] != 0 && spikes[in_neuron_index][t] != 1 {
                    panic!("Error: input spike must be 0 or 1 ");
                }
                t_spikes.push(spikes[in_neuron_index][t]);
            }

            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }

        spike_events
    }

    /**
        This function decodes a Vec of SpikeEvents and returns an output spikes matrix of 1/0,
        one row for each output neuron
     */
    fn decode_spikes<const SPIKES_DURATION: usize>(spikes: Vec<SpikeEvent>)
        -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {

        let mut raw_spikes = [[0u8; SPIKES_DURATION]; NET_OUTPUT_DIM];

        for spike_event in spikes {
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
                raw_spikes[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }

        raw_spikes
    }
}

impl<'a, N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize>
IntoIterator for &'a mut SNN<N, NET_INPUT_DIM, NET_OUTPUT_DIM> {
    type Item = &'a mut Arc<Mutex<Layer<N>>>;
    type IntoIter = IterMut<'a, Arc<Mutex<Layer<N>>>>;

    fn into_iter(self) -> Self::IntoIter {
        self.layers.iter_mut()
    }
}
