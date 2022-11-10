use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

//This function verifies the correct parameters of the LIF neuron
fn verify_neuron(lif_neuron: &LifNeuron, v_th: f64, v_rest: f64, v_reset: f64, tau: f64) -> bool {

    if lif_neuron.get_v_th() != v_th {
        return false;
    }
    if lif_neuron.get_v_rest() != v_rest {
        return false;
    }
    if lif_neuron.get_v_reset() != v_reset {
        return false;
    }
    if  lif_neuron.get_tau() != tau {
        return false;
    }
    if lif_neuron.get_v_mem() != v_rest {
        return false;
    }
    if lif_neuron.get_ts() != 0u64 {
        return false;
    }

    true
}

//Function that prints the output spikes obtained from the SNN processing
fn print_output(test_name: &str, output_spikes: Vec<Vec<u8>>) -> () {
    println!("\nOUTPUT SPIKES for {}:\n",test_name);
    print!("t   ");

    for (n, spikes) in output_spikes.into_iter().enumerate() {
        if n == 0 {
            (0..spikes.len()).for_each(|t| print!("{} ", t));
            println!();
        }

        print!("N{}  ", n);

        for spike in spikes {
            print!("{} ", spike);
        }
        println!();
    }
    println!();
}

//Test related to the SNN fluent builder

#[test]
fn test_add_one_layer() {
    #[rustfmt::skip]

    let snn = SnnBuilder::<LifNeuron>::new()
        .add_layer::<0>()
        .weights([])
        .neurons([])
        .intra_weights([])
        .build();

    assert_eq!(snn.get_layers_number(),1);
}

#[test]
fn test_add_more_than_one_layer() {
    #[rustfmt::skip]

    let snn = SnnBuilder::<LifNeuron>::new()
        .add_layer::<0>().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .add_layer().weights([]).neurons([]).intra_weights([])
        .build();

    assert_eq!(snn.get_layers_number(),4);
}

#[test]
fn test_add_weights_to_layers() {
    #[rustfmt::skip]

    let snn_params = SnnBuilder::<LifNeuron>::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.4, 0.5, 0.6]
        ]).neurons([
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)
        ]).intra_weights([
            [0.0, -0.2],
            [-0.9, 0.0]
        ]).add_layer()
        .weights([
            [0.2, 0.3]
        ]).neurons([
            LifNeuron::new(0.45, 0.7, 0.1, 0.6)
        ]).intra_weights([
            [0.0]
        ])
        .get_params();

    let weights_layer1 = snn_params.extra_weights.get(0);
    let weights_layer2 = snn_params.extra_weights.get(1);
    let weights_layer3 = snn_params.extra_weights.get(2);

    assert_eq!(weights_layer1.is_some(), true);
    assert_eq!(weights_layer2.is_some(), true);
    assert_eq!(weights_layer3.is_none(), true);

    assert_eq!(weights_layer1.unwrap(), &[[0.1, 0.2, 0.3],[0.4, 0.5, 0.6]]);
    assert_eq!(weights_layer2.unwrap(), &[[0.2, 0.3]]);

}

#[test]
fn test_layer_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.5, 0.1, 0.6, 0.3]
        ]).neurons([
            LifNeuron::new(0.12, 0.1, 0.03, 0.98)
        ]).intra_weights([
            [0.0]
        ]).get_params();

    let layer_neurons1 = snn_params.neurons.get(0);
    let layer_neurons2 = snn_params.neurons.get(1);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 1);
    assert_eq!(layer_neurons2.is_none(), true);

    let neuron = layer_neurons1.unwrap().get(0);

    assert_eq!(neuron.is_some(), true);
    assert_eq!(verify_neuron(neuron.unwrap(), 0.12, 0.1, 0.03, 0.98), true);

}

#[test]
fn test_layer_with_more_than_one_neuron() {
    #[rustfmt::skip]

        let snn_params = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.5, 0.1, 0.6, 0.3],
            [0.2, 0.3, 0.1, 0.9, 0.76],
            [0.1, 0.2, 0.3, 0.4, 0.5]
        ]).neurons([
            LifNeuron::new(0.127, 0.12, 0.78, 0.67),
            LifNeuron::new(0.12, 0.22, 0.31, 0.47),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84)
    ]).intra_weights([
        [0.0, -0.34, -0.12],
        [-0.23, 0.0, -0.56],
        [-0.05, -0.01, 0.0]
    ]).get_params();

    let layer_neurons1 = snn_params.neurons.get(0);
    let layer_neurons2 = snn_params.neurons.get(1);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 3);
    assert_eq!(layer_neurons2.is_none(), true);

    let neuron1 = layer_neurons1.unwrap().get(0);

    assert_eq!(neuron1.is_some(), true);
    assert_eq!(verify_neuron(neuron1.unwrap(), 0.127, 0.12, 0.78, 0.67), true);

    let neuron2 = layer_neurons1.unwrap().get(1);

    assert_eq!(neuron2.is_some(), true);
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.22, 0.31, 0.47), true);

    let neuron3 = layer_neurons1.unwrap().get(2);

    assert_eq!(neuron3.is_some(), true);
    assert_eq!(verify_neuron(neuron3.unwrap(), 0.25, 0.36, 0.71, 0.84), true);

}

#[test]
fn test_intra_layer_weights_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.5, 0.1, 0.6, 0.3]
        ]).neurons([
            LifNeuron::new(0.12, 0.1, 0.03, 0.98)
        ]).intra_weights([
            [0.0]
        ]).get_params();

    let layer_intra_weights1 = snn_params.intra_weights.get(0);
    let layer_intra_weights2 = snn_params.intra_weights.get(1);

    assert_eq!(layer_intra_weights1.is_some(), true);
    assert_eq!(layer_intra_weights1.unwrap().len(), 1);
    assert_eq!(layer_intra_weights2.is_none(), true);

    let intra_weights = layer_intra_weights1.unwrap().get(0);

    assert_eq!(intra_weights.is_some(), true);
    assert_eq!(intra_weights.unwrap(), &[0.0]);

}

#[test]
fn test_intra_layer_weights_with_more_than_one_neuron() {
    #[rustfmt::skip]

    let snn_params = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.5, 0.1, 0.6, 0.3],
            [0.2, 0.3, 0.1, 0.9, 0.76],
            [0.1, 0.2, 0.3, 0.4, 0.5]
        ]).neurons([
            LifNeuron::new(0.127, 0.12, 0.78, 0.67),
            LifNeuron::new(0.12, 0.22, 0.31, 0.47),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84)
    ]).intra_weights([
        [0.0, -0.34, -0.12],
        [-0.23, 0.0, -0.56],
        [-0.05, -0.01, 0.0]
    ]).get_params();

    let layer_intra_weights1 = snn_params.intra_weights.get(0);
    let layer_intra_weights2 = snn_params.intra_weights.get(1);

    assert_eq!(layer_intra_weights1.is_some(), true);
    assert_eq!(layer_intra_weights1.unwrap().len(), 3);
    assert_eq!(layer_intra_weights2.is_none(), true);

    let weights1 = layer_intra_weights1.unwrap().get(0);

    assert_eq!(weights1.is_some(), true);
    assert_eq!(weights1.unwrap(), &[0.0, -0.34, -0.12]);

    let weights2 = layer_intra_weights1.unwrap().get(1);

    assert_eq!(weights2.is_some(), true);
    assert_eq!(weights2.unwrap(), &[-0.23, 0.0, -0.56]);

    let weights3 = layer_intra_weights1.unwrap().get(2);

    assert_eq!(weights3.is_some(), true);
    assert_eq!(weights3.unwrap(), &[-0.05, -0.01, 0.0]);

}

//Test related to the SNN process

#[test]
fn test_process_snn_with_only_one_layer() {
    #[rustfmt::skip]

    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
        ]).intra_weights([
            [0.0, -0.1, -0.15],
            [-0.05, 0.0, -0.1],
            [-0.15, -0.1, 0.0]
        ]).build();

    let output_spikes = snn.process(&[[1,0,1],[0,0,1]]);
    let output_expected = [[0,0,0],[1,0,1],[1,0,1]];

    print_output("test_process_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());

    assert_eq!(output_spikes, output_expected);
}

#[test]
#[should_panic]
fn test_snn_with_negative_weights() {
    #[rustfmt::skip]

    let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [-0.2, 0.5]
        ]).neurons([
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)
        ]).intra_weights([
            [0.0]
        ]).build();

}

#[test]
#[should_panic]
fn test_snn_with_weights_greater_than_one() {
    #[rustfmt::skip]

        let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.2, 1.5]
        ]).neurons([
            LifNeuron::new(0.45, 0.7, 0.1, 0.6)
        ]).intra_weights([
            [0.0]
        ]).build();

}

#[test]
#[should_panic]
fn test_snn_with_positive_intra_weights() {
    #[rustfmt::skip]

        let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.2, 0.5],
            [0.3, 0.4]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0)
    ]).intra_weights([
        [0.0, 0.5],
        [-0.05, 0.0]
    ]).build();

}

#[test]
#[should_panic]
fn test_snn_with_intra_weights_greater_than_one() {
    #[rustfmt::skip]

        let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.2, 0.5],
            [0.3, 0.4]
        ]).neurons([
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)
        ]).intra_weights([
            [0.0, -1.5],
            [-0.05, 0.0]
        ]).build();

}