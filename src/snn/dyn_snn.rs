use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread;
use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::SpikeEvent;

/* * Dynamic Spiking Neural Network structure * */

/**
    Object representing the Spiking Neural Network itself
    - N: is a generic type representing the Neuron

    There aren't generic const variables such as NET_INPUT_DIM, NET_OUTPUT_DIM, etc. because all the
    computations and checks are done at runtime.
*/
#[derive(Debug, Clone)]
pub struct DynSNN <N: Neuron + Clone + 'static>{
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

    pub fn get_layers(&self) -> Vec<Layer<N>> {
        self.layers.iter().map(|layer| layer.lock().unwrap().clone()).collect()
    }

    /**
        Spikes contains an array for each input layer's neuron, and each array has the same
        number of spikes, equal to the duration of the input
        (spikes is a matrix, one row for each input neuron, and one column for each time instant)
        This method check  user input at run-time
    */
    pub fn process(&mut self, spikes: &Vec<Vec<u8>>)
                                                 -> Vec<Vec<u8>> {
        // * check and compute the spikes duration *
        let spikes_duration = self.compute_spikes_duration(spikes);
        // * encode spikes into SpikeEvent(s) *
        let input_spike_events = self.encode_spikes(spikes, spikes_duration);
        // * process input *
        let output_spike_events = self.process_events(input_spike_events);
        // * decode output into array shape *
        let decoded_output =  self.decode_spikes(output_spike_events, spikes_duration);
        decoded_output
    }
    /**
        This function checks if each vector passed in the spikes one contains the same spikes duration.
        If yes, it returns the duration, otherwise it returns an error
     */
    fn compute_spikes_duration(&self, spikes:&Vec<Vec<u8>>) -> usize {
        let spike_duration = spikes[0].len();
        for spike in spikes{
            if spike.len() != spike_duration {
                panic!("The number of spikes duration must be equal");
            }
        }
        spike_duration
    }

    /**
        Spikes is a Vec of spike events that will be processed through the layers of the network.
        - This method creates a new thread for each layer.
        Each thread will process the input spike events received from the previous layer through a shared channel
        and will send the computed output spike events to the next layer by using another shared channel.
     */
    fn process_events(&mut self, spikes: Vec<SpikeEvent>) -> Vec<SpikeEvent> {
        /* create the threads' pool */
        let mut threads = Vec::new();
        /* IT'S BETTER TO SHARE THIS FUNCTION WITH THE OTHER SNN */
        /* create input TX and output RC for each layers and spawn layers threads */
        let (net_input_tx, mut layer_rc) = channel::<SpikeEvent>();

        for layer in &mut self.layers {
            let (layer_tx, next_layer_rc) = channel::<SpikeEvent>();

            let layer = layer.clone();

            let thread = thread::spawn(move || {
                layer.lock().unwrap().process(layer_rc, layer_tx);
            });

            threads.push(thread);   /* push the new thread into threads' pool */
            layer_rc = next_layer_rc;    /* update external rc, to pass it to the next layer */
        }

        let net_output_rc = layer_rc;

        /* fire input SpikeEvents into *net_input* tx */
        for spike_event in spikes {
            /* check if there is at least 1 spike, otherwise skip to the next instant */
            if spike_event.spikes.iter().all(|spike| *spike == 0u8) {
                continue;
            }

            /* (process only *effective* spike events) */
            let instant = spike_event.ts;
            net_input_tx.send(spike_event)
                .expect(&format!("Unexpected error sending input spike event t={}", instant));
        }
        drop(net_input_tx); /* drop input tx, to make all the threads terminate */

        /* get output SpikeEvents from *net_output* rc */
        let mut output_events = Vec::<SpikeEvent>::new();

        while let Ok(spike_event) = net_output_rc.recv() {
            output_events.push(spike_event);
        }

        /* waiting for threads to terminate */
        for thread in threads {
            thread.join().unwrap();
        }

        output_events
    }

    /**
        (same as process(), but it checks input spikes sizes at *run-time*:
        spikes must have a number of Vec(s) equal to NET_INPUT_DIM, and all
        these Vec(s) must have the same length), otherwise panic!()
     */
    fn _process_dyn(&mut self, spikes: Vec<Vec<u8>>) -> Vec<Vec<u8>> {

        /* check num of spikes vec(s) */
        if spikes.len() != self.layers[0].lock().unwrap().get_neurons_number() {
            panic!("Error: number of input spikes must be equal to the number of input neurons");
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

            if spikes_events.len() == 0 {    /* the first cycle... */
                /* ...create the spike events */
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
        let output_spike_events = self.process_events(spikes_events);

        /* decode output spikes in spike events */

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

    /**
        This function encodes the received input in a Vec of **SpikeEvent** to process it.
     */
    fn encode_spikes(&self, spikes: &Vec<Vec<u8>>,spikes_duration:usize) -> Vec<SpikeEvent> {
        let mut spike_events = Vec::<SpikeEvent>::new();

        if spikes.len()!= self.layers[0].lock().unwrap().get_weights().first().unwrap().len() {
            panic!("The number of input spikes is not coherent with the input layer dimension");
        }
        for t in 0..spikes_duration {
            let mut t_spikes = Vec::<u8>::new();

            /* retrieve the input spikes for each neuron */
            for in_neuron_index in 0..spikes.len(){
                if spikes[in_neuron_index][t] != 0 && spikes[in_neuron_index][t] != 1 {
                    panic!("Error: input spike must be 0 or 1");
                }
                t_spikes.push(spikes[in_neuron_index][t]);
            }

            let t_spike_event = SpikeEvent::new(t as u64, t_spikes);
            spike_events.push(t_spike_event);
        }

        spike_events
    }

    /**
        This function decodes and returns an output spikes matrix from a Vec of SpikeEvents
     */
    fn decode_spikes(&self ,spikes: Vec<SpikeEvent>, spikes_duration:usize) -> Vec<Vec<u8>> {

        let last_layer = self.layers.last().unwrap().lock().unwrap();

        let output_dimension = last_layer.get_neurons_number();
        let mut result  = vec![vec![0; spikes_duration];  output_dimension];

        for spike_event in spikes {
            for (out_neuron_index, spike) in spike_event.spikes.into_iter().enumerate() {
                result[out_neuron_index][spike_event.ts as usize] = spike;
            }
        }
        result
    }
}




