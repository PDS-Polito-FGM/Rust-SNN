use std::sync::{Arc, Mutex};
use std::sync::mpsc::channel;
use std::thread;
use std::thread::JoinHandle;
use crate::neuron::Neuron;
use crate::snn::layer::Layer;
use crate::SpikeEvent;

#[derive(Debug)]
pub struct Processor { }

impl Processor {
    /**
        Spikes is a Vec of spike events that will be processed through the layers of the network.
        - This method creates a new thread for each layer.
        Each thread will process the input spike events received from the previous layer through a shared channel
        and will send the computed output spike events to the next layer by using another shared channel.
     */
    pub fn process_events<'a, N: Neuron + Clone + Send + 'static, S: IntoIterator<Item=&'a mut Arc<Mutex<Layer<N>>>>>
    (&self, snn: S, spikes: Vec<SpikeEvent>) -> Vec<SpikeEvent> {
        /* create the threads' pool */
        let mut threads = Vec::<JoinHandle<()>>::new();

        /* create channel to feed the (first layer of the) network */
        let (net_input_tx, mut layer_rc) = channel::<SpikeEvent>();

        /* create input TX and output RC for each layer and spawn layers' threads */
        for layer_ref in snn {
            /* create channel to feed the next layer */
            let (layer_tx, next_layer_rc) = channel::<SpikeEvent>();

            let layer_ref = layer_ref.clone();

            let thread = thread::spawn(move || {
                /* retrieve layer */
                let mut layer = layer_ref.lock().unwrap();
                /* execute layer task */
                layer.process(layer_rc, layer_tx);
            });

            threads.push(thread);   /* push the new thread into threads' pool */
            layer_rc = next_layer_rc;    /* update external rc, to pass it to the next layer */
        }

        let net_output_rc = layer_rc;

        /* fire input SpikeEvents into *net_input_tx* */
        for spike_event in spikes {
            /* * check if there is at least 1 spike, otherwise skip to the next instant * */
            if spike_event.spikes.iter().all(|spike| *spike == 0u8) {
                continue;   /* (process only *effective* spike events) */
            }

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
}