use pds_snn::builders::SnnBuilder;
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
fn test_process_snn_with_only_one_layer_and_different_neurons() {
    #[rustfmt::skip]

    let mut snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3, 0.4],
            [0.1, 0.4, 0.3, 0.2],
            [0.5, 0.6, 0.7, 0.8]
        ]).neurons([
            LifNeuron::new(0.31, 0.01, 0.1, 0.8),
            LifNeuron::new(0.32, 0.02, 0.3, 0.9),
            LifNeuron::new(0.33, 0.03, 0.2, 1.0),
        ]).intra_weights([
            [0.0, -0.6, -0.3],
            [-0.5, 0.0, -0.15],
            [-0.4, -0.05, 0.0]
        ]).build();

    let output_spikes = snn.process(&[[1,1,0],[0,1,0],[0,1,1],[0,0,1]]);
    //let output_expected = [[0,0,0],[1,0,1],[1,0,1]];

    print_output("test_process_snn_with_only_one_layer_and_different_neurons", output_spikes.iter().map(|x| x.to_vec()).collect());

    //assert_eq!(output_spikes, output_expected);
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
            LifNeuron::new(0.3, 0.05, 0.84, 1.0),
            LifNeuron::new(0.3, 0.87, 0.12, 0.89)
        ]).intra_weights([
            [0.0, -0.3],
            [-0.4, 0.0]
        ]).build();

    let _output_spikes = snn.process(&[[0,50],[0,1]]);

}