mod demo_internals;
use pds_snn::builders::{DynSnnBuilder, SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;
use crate::demo_internals::demo_internals::{print_instants, print_layer, print_spikes};


fn main() {
    println!("* Verbose demo for the Spiking Neural Network Rust library *");

    /* demo of DynSnnBuilder and DynSNN */
    verbose_demo_dynamic_snn();

    println!();
    /* demo of SnnBuilder and SNN */
    verbose_demo_static_snn();
}

fn verbose_demo_dynamic_snn() {
    println!("\n• *Dynamic* building demo");

    let mut builder = DynSnnBuilder::<LifNeuron>::new(5);
    println!("Created Builder for *dynamic* SNN with 5 input neurons");

    /* add first (hidden) layer */
    builder = builder.add_layer(vec![
        /* 4 LIF Neurons */
        LifNeuron::new(0.1, 0.10, 0.23, 0.45, 1.0),
        LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0),
        LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0),
        LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)
    ], vec![
        vec![0.9 , 0.42, 0.1, 0.31, 0.3 ],      /* 1st neuron incoming extra-weights from input layer */
        vec![0.2 , 0.56, 0.1, 0.9 , 0.76],      /* 2nd neuron incoming extra-weights from input layer */
        vec![0.2 , 0.23, 0.3, 0.95, 0.5 ],      /* 3rd neuron incoming extra-weights from input layer */
        vec![0.23, 0.1 , 0.2, 0.4 , 0.8 ]       /* 4th neuron incoming extra-weights from input layer */
    ], vec![
        vec![ 0.0 , -0.34, -0.12, -0.23],       /* 1st neuron incoming intra-weights */
        vec![-0.23,  0.0 , -0.56, -0.23],       /* 2nd neuron incoming intra-weights */
        vec![-0.05, -0.01,  0.0 , -0.23],       /* 3rd neuron incoming intra-weights */
        vec![-0.23, -0.23, -0.23,  0.0 ]        /* 4th neuron incoming intra-weights */
    ]);

    println!("\nAdded 1st hidden layer with 4 LifNeurons:");
    print_layer(&builder.get_params().neurons[0]);

    /* add 2nd hidden layer */
    builder = builder.add_layer(vec![
        /* 2 LIF Neurons */
        LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),
        LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)
    ], vec![
        vec![0.1, 0.3, 0.4, 0.2],   /* 1st neuron incoming extra-weights from 1st layer */
        vec![0.7, 0.3, 0.1, 0.3],   /* 2nd neuron incoming extra-weights from 1st layer */
    ], vec![
        vec![ 0.0 , -0.62],         /* 1st neuron incoming intra-weights */
        vec![-0.12,  0.0 ],         /* 2nd neuron incoming intra-weights */
    ]);

    println!("\nAdded 2nd hidden layer with 2 LifNeurons:");
    print_layer(&builder.get_params().neurons[1]);

    println!("\nBuilding the network...");
    let mut snn = builder.build();
    println!("Done!");


    println!("\n• Dynamic processing");

    /* creating input spikes */
    let input_spikes = vec![
        vec![1, 0, 1, 1, 0, 0, 1, 0, 0, 1],     /* 1st neuron input train of spikes */
        vec![0, 0, 1, 1, 1, 0, 1, 1, 0, 1],     /* 2nd neuron input train of spikes */
        vec![0, 1, 0, 1, 0, 0, 1, 0, 0, 0],     /* 3rd neuron input train of spikes */
        vec![0, 1, 0, 1, 1, 0, 1, 0, 0, 0],     /* 4th neuron input train of spikes */
        vec![1, 1, 1, 0, 0, 0, 1, 0, 0, 1]      /* 5th neuron input train of spikes */
    ];

    println!("Considering input spikes over 10 time instants:");
    print_instants(input_spikes[0].len());
    print_spikes(&input_spikes, "input");

    /* processing input spikes */
    println!("\nFeeding DynSNN with the provided input spikes...");
    let output_spikes = snn.process(&input_spikes);
    println!("Done!");

    println!("\nOutput spikes produced by the Spiking Neural Network:");
    print_instants(output_spikes[0].len());
    print_spikes(&output_spikes, "output");
}

fn verbose_demo_static_snn() {
    println!("\n• *Static* building demo");

    let builder = SnnBuilder::<LifNeuron>::new();
    println!("Created Builder for *static* SNN");

    /* add first (hidden) layer */
    let builder = builder.add_layer()
        .weights([
            [0.9 , 0.42, 0.1, 0.31, 0.3 ],      /* 1st neuron incoming extra-weights from input layer */
            [0.2 , 0.56, 0.1, 0.9 , 0.76],      /* 2nd neuron incoming extra-weights from input layer */
            [0.2 , 0.23, 0.3, 0.95, 0.5 ],      /* 3rd neuron incoming extra-weights from input layer */
            [0.23, 0.1 , 0.2, 0.4 , 0.8 ]       /* 4th neuron incoming extra-weights from input layer */
        ]).neurons([
            /* 4 LIF Neurons */
            LifNeuron::new(0.1, 0.10, 0.23, 0.45, 1.0),
            LifNeuron::new(0.3, 0.12, 0.54, 0.23, 1.0),
            LifNeuron::new(0.2, 0.23, 0.23, 0.65, 1.0),
            LifNeuron::new(0.4, 0.34, 0.12, 0.45, 1.0)
        ]).intra_weights([
            [ 0.0 , -0.34, -0.12, -0.23],       /* 1st neuron incoming intra-weights */
            [-0.23,  0.0 , -0.56, -0.23],       /* 2nd neuron incoming intra-weights */
            [-0.05, -0.01,  0.0 , -0.23],       /* 3rd neuron incoming intra-weights */
            [-0.23, -0.23, -0.23,  0.0 ]        /* 4th neuron incoming intra-weights */
        ]);

    println!("\nAdded 1st hidden layer with 4 LifNeurons:");
    print_layer(&builder.get_params().neurons[0]);

    /* add 2nd hidden layer */
    let builder = builder.add_layer()
        .weights([
            [0.1, 0.3, 0.4, 0.2],   /* 1st neuron incoming extra-weights from 1st layer */
            [0.7, 0.3, 0.1, 0.3]    /* 2nd neuron incoming extra-weights from 1st layer */
        ]).neurons([
            /* 2 LIF Neurons */
            LifNeuron::new(0.17, 0.12, 0.78, 0.67, 1.0),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84, 1.0)
        ]).intra_weights([
            [ 0.0 , -0.62],         /* 1st neuron incoming intra-weights */
            [-0.12,  0.0 ],         /* 2nd neuron incoming intra-weights */
        ]);

    println!("\nAdded 2nd hidden layer with 2 LifNeurons:");
    print_layer(&builder.get_params().neurons[1]);

    println!("\nBuilding the network...");
    let mut snn = builder.build();
    println!("Done!");


    println!("\n• Static processing");

    /* creating input spikes */
    let input_spikes = [
        [1, 0, 1, 1, 0, 0, 1, 0, 0, 1],     /* 1st neuron input train of spikes */
        [0, 0, 1, 1, 1, 0, 1, 1, 0, 1],     /* 2nd neuron input train of spikes */
        [0, 1, 0, 1, 0, 0, 1, 0, 0, 0],     /* 3rd neuron input train of spikes */
        [0, 1, 0, 1, 1, 0, 1, 0, 0, 0],     /* 4th neuron input train of spikes */
        [1, 1, 1, 0, 0, 0, 1, 0, 0, 1]      /* 5th neuron input train of spikes */
    ];

    println!("Considering input spikes over 10 time instants:");
    print_instants(input_spikes[0].len());
    print_spikes(&input_spikes, "input");

    /* processing input spikes */
    println!("\nFeeding SNN with the provided input spikes...");
    let output_spikes = snn.process(&input_spikes);
    println!("Done!");

    println!("\nOutput spikes produced by the Spiking Neural Network:");
    print_instants(output_spikes[0].len());
    print_spikes(&output_spikes, "output");
}
