use pds_snn::builders::{DynSnnBuilder, SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;

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

//Tests related to the SNN process function

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
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,0,1],[0,0,1]]);
    let output_expected:[[u8; 3]; 3] = [[0,0,0],[1,0,1],[1,0,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_dyn_snn_with_only_one_layer() {
    #[rustfmt::skip]

        let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)], vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6]], vec![
            vec![0.0, -0.1, -0.15],
            vec![-0.05, 0.0, -0.1],
            vec![-0.15, -0.1, 0.0]
        ])
        .build();

    let output_spikes = snn.process(&vec![vec![1,0,1],vec![0,0,1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![0,0,0],vec![1,0,1],vec![1,0,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_dyn_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_only_one_layer_and_different_dt() {
    #[rustfmt::skip]

    let dt = 0.1;

    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
    ]).intra_weights([
        [0.0, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,0,1],[0,0,1]]);
    let output_expected:[[u8; 3]; 3] = [[0,0,0],[1,0,1],[1,0,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_dyn_snn_with_only_one_layer_and_different_dt() {
    #[rustfmt::skip]

    let dt = 0.1;

    let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, dt)], vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4],
            vec![0.5, 0.6]], vec![
            vec![0.0, -0.1, -0.15],
            vec![-0.05, 0.0, -0.1],
            vec![-0.15, -0.1, 0.0]
        ])
        .build();

    let output_spikes = snn.process(&vec![vec![1,0,1],vec![0,0,1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![0,0,0],vec![1,0,1],vec![1,0,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_dyn_snn_with_only_one_layer", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_more_than_one_layer_and_same_neurons() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.6]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]).add_layer()
        .weights([
            [0.3, 0.2, 0.1]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([[0.0]])
        .add_layer()
        .weights([
            [0.3],
            [0.2],
            [0.5],
            [0.3]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
    ]).intra_weights([
        [0.0, -0.1, -0.2, -0.3],
        [-0.1, 0.0, -0.4, -0.2],
        [-0.6, -0.2, 0.0, -0.9],
        [-0.5, -0.3, -0.8, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,0,1],[0,0,1]]);
    let output_expected:[[u8; 3]; 4] = [[1,0,0],[0,0,0],[1,0,0],[1,0,0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_more_than_one_layer_and_same_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_only_one_layer_and_different_neurons() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.1, 0.2],
            [0.5, 0.1, 0.7, 0.25]
        ]).neurons([
        LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),
        LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),
        LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.6, -0.3],
        [-0.5, 0.0, -0.15],
        [-0.4, -0.05, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,1,0],[0,1,0],[0,1,1],[0,0,1]]);
    let output_expected: [[u8; 3]; 3]  = [[0,1,0],[0,1,0],[1,1,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_only_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_more_than_one_layer_and_different_neurons() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.3, 0.4]
        ]).neurons([
        LifNeuron::new(0.5, 0.1, 0.2, 0.7, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.4],
        [-0.1, 0.0]
    ]).add_layer()
        .weights([
            [0.7, 0.2],
            [0.3, 0.8],
            [0.5, 0.6],
            [0.3, 0.2]
        ]).neurons([
        LifNeuron::new(0.2, 0.1, 0.15, 0.1, 1.0),
        LifNeuron::new(0.3, 0.2, 0.05, 0.3, 1.0),
        LifNeuron::new(0.4, 0.15, 0.1, 0.8, 1.0),
        LifNeuron::new(0.05, 0.35, 0.01, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.2, -0.4, -0.9],
        [-0.1, 0.0, -0.3, -0.2],
        [-0.6, -0.2, 0.0, -0.9],
        [-0.5, -0.3, -0.8, 0.0]
    ])
        .add_layer()
        .weights([
            [0.3, 0.3, 0.2, 0.7]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
    ]).intra_weights([
        [0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,0,1,0],[0,0,1,1]]);
    let output_expected:[[u8; 4]; 1] = [[1,0,1,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_more_than_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_dyn_snn_with_more_than_one_layer_and_different_neurons() {
    #[rustfmt::skip]

        let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.5, 0.1, 0.2, 0.7, 1.0),
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)], vec![
            vec![0.1, 0.2],
            vec![0.3, 0.4]], vec![
            vec![0.0, -0.4],
            vec![-0.1, 0.0]
        ])
        .add_layer(vec![
            LifNeuron::new(0.2, 0.1, 0.15, 0.1, 1.0),
            LifNeuron::new(0.3, 0.2, 0.05, 0.3, 1.0),
            LifNeuron::new(0.4, 0.15, 0.1, 0.8, 1.0),
            LifNeuron::new(0.05, 0.35, 0.01, 1.0, 1.0)], vec![
            vec![0.7, 0.2],
            vec![0.3, 0.8],
            vec![0.5, 0.6],
            vec![0.3, 0.2]], vec![
            vec![0.0, -0.2, -0.4, -0.9],
            vec![-0.1, 0.0, -0.3, -0.2],
            vec![-0.6, -0.2, 0.0, -0.9],
            vec![-0.5, -0.3, -0.8, 0.0]])
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)], vec![
            vec![0.3, 0.3, 0.2, 0.7]], vec![
            vec![0.0]])
        .build();

    let output_spikes = snn.process(&vec![vec![1,0,1,0],vec![0,0,1,1]]);
    let output_expected: Vec<Vec<u8>> = vec![vec![1,0,1,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_dyn_snn_with_more_than_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_different_neurons_and_more_than_five_input_spikes() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.3, 0.21, 0.36, 0.47],
            [0.6, 0.45, 0.34, 0.21],
            [0.1, 0.62, 0.72, 0.82],
            [0.12, 0.23, 0.6, 0.8]
        ]).neurons([
        LifNeuron::new(0.67, 0.01, 0.1, 0.8, 1.0),
        LifNeuron::new(0.4, 0.02, 0.3, 0.9, 1.0),
        LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
        LifNeuron::new(0.9, 0.05, 0.7, 0.5, 1.0),
    ]).intra_weights([
        [0.0, -0.6, -0.3, -0.2],
        [-0.5, 0.0, -0.15, -0.4],
        [-0.4, -0.05, 0.0, -0.2],
        [-0.1, -0.25, -0.15, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,1,0,1,0,1,1,1,1],[0,1,0,0,1,0,0,0,0],[0,1,1,0,1,0,0,0,0],[0,0,1,0,1,0,0,0,0]]);
    let output_expected: [[u8; 9]; 4] = [[0,0,0,0,1,0,0,0,0],[1,1,1,0,1,0,1,1,1],[0,1,1,0,1,0,0,0,0],[0,0,1,0,1,0,0,0,0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_different_neurons_and_more_than_five_input_spikes", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_all_zeros_as_input() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.1, 0.4, 0.3],
            [0.5, 0.6, 0.7]
        ]).neurons([
        LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),
        LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),
        LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.6, -0.3],
        [-0.5, 0.0, -0.15],
        [-0.4, -0.05, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[0,0,0,0],[0,0,0,0],[0,0,0,0]]);
    let output_expected: [[u8; 4]; 3] = [[0,0,0,0],[0,0,0,0],[0,0,0,0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_all_zeros_as_input", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_zero_inputs() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.1, 0.4, 0.3],
            [0.5, 0.6, 0.7]
        ]).neurons([
        LifNeuron::new(0.31, 0.01, 0.1, 0.8, 1.0),
        LifNeuron::new(0.32, 0.02, 0.3, 0.9, 1.0),
        LifNeuron::new(0.33, 0.03, 0.2, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.6, -0.3],
        [-0.5, 0.0, -0.15],
        [-0.4, -0.05, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[],[],[]]);
    let output_expected: [[u8; 0]; 3] = [[],[],[]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_zero_inputs", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_only_one_input() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.3, 0.4],
            [0.5, 0.25]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.1, -0.15],
        [-0.05, 0.0, -0.1],
        [-0.15, -0.1, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[0],[1]]);
    let output_expected: [[u8; 1]; 3] = [[0],[1],[0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_only_one_input", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_all_zeros_as_extra_weights() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 0.0]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, -0.5, -0.6],
        [-0.3, 0.0, -0.2],
        [-0.7, -0.3, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,1,0],[0,1,0],[0,1,1],[0,0,1]]);
    let output_expected: [[u8; 3]; 3] = [[0,0,0],[0,0,0],[0,0,0]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_all_zeros_as_extra_weights", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
fn test_process_snn_with_all_zeros_as_intra_weights() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.4, 0.2, 0.4, 0.6],
            [0.7, 0.3, 0.5, 0.8],
            [0.1, 0.2, 0.7, 0.9]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
        LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
    ]).intra_weights([
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0]
    ]).build();

    let output_spikes = snn.process(&[[1,1,0],[0,1,0],[0,1,1],[0,0,1]]);
    let output_expected: [[u8; 3]; 3] = [[1,1,1],[1,1,1],[0,1,1]];

    assert_eq!(output_spikes, output_expected);

    print_output("test_process_snn_with_all_zeros_as_intra_weights", output_spikes.iter().map(|x| x.to_vec()).collect());
}

#[test]
#[should_panic]
fn test_input_spikes_greater_than_one() {
    #[rustfmt::skip]

        let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.12, 0.5],
            [0.53, 0.43]
        ]).neurons([
        LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),
        LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0)
    ]).intra_weights([
        [0.0, -0.3],
        [-0.4, 0.0]
    ]).build();

    let _output_spikes = snn.process(&[[0,50],[0,1]]);
}

#[test]
#[should_panic]
fn test_dyn_snn_input_spikes_greater_than_one() {
    #[rustfmt::skip]

        let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),
            LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0)], vec![
            vec![0.12, 0.5],
            vec![0.53, 0.43]], vec![
            vec![0.0, -0.3],
            vec![-0.4, 0.0]
        ]).build();

    let _output_spikes = snn.process(&vec![vec![0,50],vec![0,1]]);
}

#[test]
#[should_panic]
fn test_dyn_snn_input_spikes_greater_than_input_dimension() {
    #[rustfmt::skip]

        /*
            This test case has to panic because the input spikes entered
            are different than the input dimension
        */

        let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),
            LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0)], vec![
            vec![0.12, 0.5],
            vec![0.53, 0.43]], vec![
            vec![0.0, -0.3],
            vec![-0.4, 0.0]
        ]).build();

    let _output_spikes = snn.process(&vec![vec![0,0],vec![0,1],vec![1,1]]);
}

#[test]
#[should_panic]
fn test_dyn_snn_input_spikes_lower_than_input_dimension() {
    #[rustfmt::skip]

        /*
            This test case has to panic because the input spikes entered
            are different than the input dimension
        */

        let mut snn = DynSnnBuilder::new(2)
        .add_layer(vec![
            LifNeuron::new(0.3, 0.05, 0.84, 1.0, 1.0),
            LifNeuron::new(0.3, 0.87, 0.12, 0.89, 1.0)], vec![
            vec![0.12, 0.5],
            vec![0.53, 0.43]], vec![
            vec![0.0, -0.3],
            vec![-0.4, 0.0]
        ]).build();

    let _output_spikes = snn.process(&vec![vec![1,0]]);
}
