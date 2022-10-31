use std::sync::mpsc::channel;
use std::thread;
use crate::snn::layer::Layer;
use crate::snn::neuron::Neuron;

pub mod builders;
    mod layer; // private
pub mod neuron;

// * SNN module *

/**
    Object representing the Spiking Neural Network itself
    - N: is the type representing the Neuron
    - NET_INPUT_DIM: is the input dimension of the network, i.e. the size of the input layer
    - NET_OUTPUT_DIM: is the output dimension of the network, i.e. the size of the output layer
    Having a generic cons type such as NET_INPUT_DIM allows to check at compile time
    the size of the input provided by the user
*/
pub struct SNN<N: Neuron+'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize> {
    layers: Vec<Layer<N>>
    // TODO Note: I removed tx and rc because it is better to create them on the fly before processing
    //            the input, as well as all the others tx(s) and rc(s) for the layers. In this way, they will be
    //            dropped as soon as they are not needed anymore (after processing the input), instead of keeping
    //            them as fixed struct fields even when the computation is done
    //            Furthermore, it simplifies the layer.process() method, because in this way the tx(s) are dropped
    //            as soon as the input is processed, and this causes the correspondent rc(s) to stop waiting in
    //            the next layer, leading that layer's thread to return - Mario
}

impl<N: Neuron+'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize>
    SNN<N, NET_INPUT_DIM, NET_OUTPUT_DIM> {
    // test
    pub fn new(layers: Vec<Layer<N>>) -> Self {
        Self { layers }
    }

    // spikes contains an array for each input layer's neuron, and each array has the same
    // number of spikes, equal to the duration of the input
    // (spikes is a matrix, one row for each input neuron, and one column for each time instant)
    // * this method is able to check user input at compile-time *
    // TODO Note: I don't know if these are the best input and output for this method, let's think about that - Mario
    pub fn process<const SPIKES_DURATION: usize> (&'static self, spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
        -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {
        // * encode spikes into SpikeEvent(s) *
        let input_spike_events = SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::encode_spikes(spikes);

        // * process input *
        let output_spike_events = self.process_events(input_spike_events);

        // * decode output into array shape *
        let decoded_output: [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] =
            SNN::<N, NET_INPUT_DIM, NET_OUTPUT_DIM>::decode_spikes(output_spike_events);

        decoded_output
    }

    pub fn process_events(&'static self, spikes: Vec<SpikeEvent>) -> Vec<SpikeEvent> {
        // create input TX and output RC for each layers and spawn layers threads
        let (net_input_tx, mut layer_rc) = channel::<SpikeEvent>();

        for layer in &self.layers {
            let (layer_tx, next_layer_rc) = channel::<SpikeEvent>();

            let _ = thread::spawn(move || {
                layer.process(layer_rc, layer_tx);
            });

            layer_rc = next_layer_rc; // update external rc, to pass it to the next layer
        }

        let net_output_rc = layer_rc;

        // * fire input SpikeEvents into *net_input* tx *
        for spike_event in spikes {
            let instant = spike_event.ts;
            net_input_tx.send(spike_event)
                        .expect(&format!("Unexpected error sending input spike event t={}", instant));
        }
        drop(net_input_tx); // * drop input tx, to make all the threads terminate *

        // * get output SpikeEvents from *net_output* rc *
        let mut output_events = Vec::<SpikeEvent>::new();

        while let Ok(spike_event) = net_output_rc.recv() {
            output_events.push(spike_event);
        }

        output_events
    }

    // (same as process(), but it checks input spikes sizes at *run-time*:
    // spikes must have a number of Vec(s) equal to NET_INPUT_DIM, and all
    // these Vec(s) must have the same length), otherwise panic!()
    pub fn process_dyn(&'static self, spikes: Vec<Vec<u8>>) -> Vec<Vec<u8>> {
        // check num of spikes vec(s)
        if spikes.len() != NET_INPUT_DIM {
            panic!("Error: dimensions mismatch - each input layer's neuron must have its own spikes vec");
        }

        // * encode input spikes in spike events *

        let mut spikes_events = Vec::<SpikeEvent>::new();
        let mut spikes_duration: Option<usize> = None;

        for neuron_spikes in spikes {
            let temp_len = neuron_spikes.len();

            // check spikes durarions - they must have all the same size
            match spikes_duration {
                None => spikes_duration = Some(temp_len),
                Some(duration) => if temp_len != duration {
                    panic!("Error: different size spikes vec(s) found \
                            - spikes must have the same duration for each input layer's neuron")
                }
            }

            if spikes_events.len() == 0 {    // the first cycle...
                // ...create the spike events
                (0..spikes_duration.unwrap()).for_each(|t| {
                    let spike_event = SpikeEvent::new(t as u64, Vec::<u8>::new());
                    spikes_events.push(spike_event);
                });
            }

            // copy each spike in the spike_events vec
            for t in 0..spikes_duration.unwrap() {
                spikes_events[t].spikes.push(neuron_spikes[t]);
            }
        }

        let output_spike_events = self.process_events(spikes_events);

        // * decode output spikes in spike events *

        // create and initialize output object
        let mut output_spikes: Vec<Vec<u8>> = Vec::new();

        for _ in &output_spike_events.get(0)
                                        .unwrap_or(&SpikeEvent::new(0, Vec::<u8>::new()))
                                        .spikes {
            // create as many internal Vec<u8> as the time duration of the first spike_event
            output_spikes.push(Vec::<u8>::new());
        }

        let mut out_neuron_index;
        // copy processed spikes in the output spikes vec
        for spike_event in output_spike_events {
            out_neuron_index = 0;
            for spike in spike_event.spikes {
                output_spikes[out_neuron_index].push(spike);
                out_neuron_index += 1;
            }
        }

        output_spikes
    }

    // * private functions *
    fn encode_spikes<const SPIKES_DURATION: usize>(spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
        -> Vec<SpikeEvent> {
        let mut spike_events = Vec::<SpikeEvent>::new();

        for t in 0..SPIKES_DURATION {
            let mut t_spikes = Vec::<u8>::new();

            // retrieve the input spikes for each neuron
            for in_neuron_index in 0..NET_INPUT_DIM {
                t_spikes.push(spikes[in_neuron_index][t]);
            }

            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }

        spike_events
    }

    fn decode_spikes<const SPIKES_DURATION: usize>(spikes: Vec<SpikeEvent>)
        -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM] {
        let mut result = [[0u8; SPIKES_DURATION]; NET_OUTPUT_DIM];
        let mut out_neuron_index;

        for spike_event in spikes {
            out_neuron_index = 0;

            for spike in spike_event.spikes {
                result[out_neuron_index][spike_event.ts as usize] = spike;
                out_neuron_index += 1;
            }
        }

        result
    }
}

/* Object representing the output spikes generated by a single layer */
pub struct SpikeEvent {
    ts: u64,
    spikes: Vec<u8>,
}

impl SpikeEvent {
    pub fn new(ts: u64, spikes: Vec<u8>) -> Self {
        Self { ts, spikes }
    }
}
