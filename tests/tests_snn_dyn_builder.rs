use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

//Tests related to the SNN dyn builder

//TODO: add tests when all the controls of the dyn builder will be implemented - Francesco

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

#[test]
#[ignore]
fn test_add_one_layer() {
    #[rustfmt::skip]

    let snn = DynSnnBuilder::<LifNeuron>::new()
            .add_layer(vec![], vec![], vec![])
            .build();

    assert_eq!(snn.get_layers_number(),1);
}

#[test]
#[ignore]
fn test_add_more_than_one_layer() {
    #[rustfmt::skip]

    let snn = DynSnnBuilder::<LifNeuron>::new()
            .add_layer(vec![], vec![], vec![])
            .add_layer(vec![], vec![], vec![])
            .add_layer(vec![], vec![], vec![])
            .add_layer(vec![], vec![], vec![])
            .build();

    assert_eq!(snn.get_layers_number(),4);
}

#[test]
#[ignore]
fn test_add_weights_to_layers() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)], vec![
            vec![0.1, 0.2, 0.3],
            vec![0.4, 0.5, 0.6]], vec![
            vec![0.0, -0.2],
            vec![-0.9, 0.0]
        ])
        .add_layer(vec![
            LifNeuron::new(0.45, 0.7, 0.1, 0.6)], vec![
            vec![0.2, 0.3]], vec![
            vec![0.0]
        ]).get_params();

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
#[ignore]
fn test_layer_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.12, 0.8, 0.03, 0.64)], vec![
            vec![0.3, 0.5, 0.1, 0.6, 0.3]], vec![
            vec![0.0]
        ]).get_params();

    let layer_neurons1 = snn_params.neurons.get(0);
    let layer_neurons2 = snn_params.neurons.get(1);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 1);
    assert_eq!(layer_neurons2.is_none(), true);

    let neuron = layer_neurons1.unwrap().get(0);

    assert_eq!(neuron.is_some(), true);
    assert_eq!(verify_neuron(neuron.unwrap(), 0.12, 0.8, 0.03, 0.64), true);

}

#[test]
#[ignore]
fn test_neurons_with_same_parameters() {
    #[rustfmt::skip]

let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.12, 0.8, 0.03, 0.64),
            LifNeuron::new(0.12, 0.8, 0.03, 0.64)], vec![
            vec![0.3, 0.5, 0.1, 0.6, 0.3],
            vec![0.2, 0.3, 0.1, 0.4, 0.2]], vec![
            vec![0.0, -0.3],
            vec![-0.2, 0.0]
        ]).get_params();

    let layer_neurons1 = snn_params.neurons.get(0);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 2);

    let neuron1 = layer_neurons1.unwrap().get(0);
    let neuron2 = layer_neurons1.unwrap().get(1);

    assert_eq!(neuron1.is_some(), true);
    assert_eq!(neuron2.is_some(), true);

    assert_eq!(verify_neuron(neuron1.unwrap(), 0.12, 0.8, 0.03, 0.64), true);
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.8, 0.03, 0.64), true);

}

#[test]
#[ignore]
fn test_layer_with_more_than_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.127, 0.46, 0.78, 0.67),
            LifNeuron::new(0.12, 0.22, 0.31, 0.47),
            LifNeuron::new(0.25, 0.36, 0.5, 0.84)], vec![
            vec![0.3, 0.5, 0.1, 0.6, 0.3],
            vec![0.2, 0.3, 0.1, 0.9, 0.76],
            vec![0.1, 0.2, 0.3, 0.4, 0.5]], vec![
            vec![0.0, -0.34, -0.12],
            vec![-0.23, 0.0, -0.56],
            vec![-0.05, -0.01, 0.0]
        ]).get_params();

    let layer_neurons1 = snn_params.neurons.get(0);
    let layer_neurons2 = snn_params.neurons.get(1);

    assert_eq!(layer_neurons1.is_some(), true);
    assert_eq!(layer_neurons1.unwrap().len(), 3);
    assert_eq!(layer_neurons2.is_none(), true);

    let neuron1 = layer_neurons1.unwrap().get(0);

    assert_eq!(neuron1.is_some(), true);
    assert_eq!(verify_neuron(neuron1.unwrap(), 0.127, 0.46, 0.78, 0.67), true);

    let neuron2 = layer_neurons1.unwrap().get(1);

    assert_eq!(neuron2.is_some(), true);
    assert_eq!(verify_neuron(neuron2.unwrap(), 0.12, 0.22, 0.31, 0.47), true);

    let neuron3 = layer_neurons1.unwrap().get(2);

    assert_eq!(neuron3.is_some(), true);
    assert_eq!(verify_neuron(neuron3.unwrap(), 0.25, 0.36, 0.5, 0.84), true);

}

#[test]
#[ignore]
fn test_intra_layer_weights_with_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.12, 0.1, 0.03, 0.98)], vec![
            vec![0.3, 0.5, 0.1, 0.6, 0.3]], vec![
            vec![0.0]
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
#[ignore]
fn test_intra_layer_weights_with_more_than_one_neuron() {
    #[rustfmt::skip]

    let snn_params = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.127, 0.12, 0.78, 0.67),
            LifNeuron::new(0.12, 0.22, 0.31, 0.47),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84)], vec![
            vec![0.3, 0.5, 0.1, 0.6, 0.3],
            vec![0.2, 0.3, 0.1, 0.9, 0.76],
            vec![0.1, 0.2, 0.3, 0.4, 0.5]], vec![
            vec![0.0, -0.34, -0.12],
            vec![-0.23, 0.0, -0.56],
            vec![-0.05, -0.01, 0.0]
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

#[test]
#[ignore]
fn test_complete_snn() {
    #[rustfmt::skip]

    let snn = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.1, 0.1, 0.23, 0.45),
            LifNeuron::new(0.3, 0.12, 0.54, 0.23),
            LifNeuron::new(0.2, 0.23, 0.23, 0.65),
            LifNeuron::new(0.4, 0.34, 0.12, 0.45)], vec![
            vec![0.9, 0.42, 0.1, 0.31, 0.3],
            vec![0.2, 0.56, 0.1, 0.9, 0.76],
            vec![0.2, 0.23, 0.3, 0.95, 0.5],
            vec![0.23, 0.1, 0.2, 0.4, 0.8]], vec![
            vec![0.0, -0.34, -0.12, -0.23],
            vec![-0.23, 0.0, -0.56, -0.23],
            vec![-0.05, -0.01, 0.0, -0.23],
            vec![-0.23, -0.23, -0.23, 0.0]
        ]).add_layer(vec![
            LifNeuron::new(0.17, 0.12, 0.78, 0.67),
            LifNeuron::new(0.25, 0.36, 0.71, 0.84)], vec![
            vec![0.1, 0.3, 0.4, 0.2],
            vec![0.7, 0.3, 0.1, 0.3]
        ], vec![
            vec![0.0, -0.62],
            vec![-0.12, 0.0]
        ]).build();

    let snn_layers = snn.get_layers();

    assert_eq!(snn_layers.len(),2);

    let layer1 = snn_layers.get(0);
    let layer2 = snn_layers.get(1);
    let layer3 = snn_layers.get(2);

    assert_eq!(layer1.is_some(),true);
    assert_eq!(layer2.is_some(),true);
    assert_eq!(layer3.is_none(),true);

    let neurons_layer1 = layer1.unwrap().get_neurons();
    let weights_layer1 = layer1.unwrap().get_weights();
    let intra_weights_layer1 = layer1.unwrap().get_intra_weights();

    assert_eq!(neurons_layer1.len(),4);
    assert_eq!(weights_layer1.len(),4);
    assert_eq!(intra_weights_layer1.len(),4);

    assert_eq!(verify_neuron(&neurons_layer1[0],0.1,0.1,0.23,0.45),true);
    assert_eq!(verify_neuron(&neurons_layer1[1],0.3,0.12,0.54,0.23),true);
    assert_eq!(verify_neuron(&neurons_layer1[2],0.2,0.23,0.23,0.65),true);
    assert_eq!(verify_neuron(&neurons_layer1[3],0.4,0.34,0.12,0.45),true);

    assert_eq!(weights_layer1, &[
        [0.9, 0.42, 0.1, 0.31, 0.3],
        [0.2, 0.56, 0.1, 0.9, 0.76],
        [0.2, 0.23, 0.3, 0.95, 0.5],
        [0.23, 0.1, 0.2, 0.4, 0.8]
    ]);

    assert_eq!(intra_weights_layer1, &[
        [0.0, -0.34, -0.12, -0.23],
        [-0.23, 0.0, -0.56, -0.23],
        [-0.05, -0.01, 0.0, -0.23],
        [-0.23, -0.23, -0.23, 0.0]
    ]);

    let neurons_layer2 = layer2.unwrap().get_neurons();
    let weights_layer2 = layer2.unwrap().get_weights();
    let intra_weights_layer2 = layer2.unwrap().get_intra_weights();

    assert_eq!(neurons_layer2.len(),2);
    assert_eq!(weights_layer2.len(),2);
    assert_eq!(intra_weights_layer2.len(),2);

    assert_eq!(verify_neuron(&neurons_layer2[0],0.17,0.12,0.78,0.67),true);
    assert_eq!(verify_neuron(&neurons_layer2[1],0.25,0.36,0.71,0.84),true);

    assert_eq!(weights_layer2, &[
        [0.1, 0.3, 0.4, 0.2],
        [0.7, 0.3, 0.1, 0.3]
    ]);

    assert_eq!(intra_weights_layer2, &[
        [0.0, -0.62],
        [-0.12, 0.0]
    ]);

}

#[test]
#[should_panic]
#[ignore]
fn test_snn_with_negative_weights() {
    #[rustfmt::skip]

    let _snn = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)
        ], vec![
            vec![-0.2, 0.5]
        ], vec![
            vec![0.0]
        ]).build();

}

#[test]
#[should_panic]
#[ignore]
fn test_snn_with_weights_greater_than_one() {
    #[rustfmt::skip]

    let _snn = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.45, 0.7, 0.1, 0.6)
        ], vec![
            vec![0.2, 1.5]
        ], vec![
            vec![0.0]
        ]).build();

}

#[test]
#[should_panic]
#[ignore]
fn test_snn_with_positive_intra_weights() {
    #[rustfmt::skip]

    let _snn = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0)
        ], vec![
            vec![0.2, 0.5],
            vec![0.3, 0.4]
        ], vec![
            vec![0.0, 0.5],
            vec![-0.05, 0.0]
        ]).build();

}

#[test]
#[should_panic]
#[ignore]
fn test_snn_with_intra_weights_greater_than_one() {
    #[rustfmt::skip]

    let _snn = DynSnnBuilder::<LifNeuron>::new()
        .add_layer(vec![
            LifNeuron::new(0.1, 0.05, 0.1, 1.0),
            LifNeuron::new(0.3, 0.23, 0.1, 0.89)
        ], vec![
            vec![0.2, 0.5],
            vec![0.3, 0.4]
        ], vec![
            vec![0.0, -1.5],
            vec![-0.05, 0.0]
        ]).build();

}