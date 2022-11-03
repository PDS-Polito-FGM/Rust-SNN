use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

#[test]
fn fluent_builder_test1() {
    println!("This is the first test useful to check the Fluent Builder pattern");

    let snn = SnnBuilder::new()
        .add_layer()
            .weights([
                [0.1, 0.1, 0.3],
                [0.2, 0.1, 0.3]
            ])
            .neurons([
                LifNeuron::new(0.3, 0.3, 0.25, 0.8),
                LifNeuron::new(0.05, 0.2, 0.2, 0.3),
            ])
            .intra_weights([
                [ 0.0, -0.1],
                [-0.1, -0.0]
            ])
        .add_layer()
            .weights([
                [0.1, 0.1],
                [0.2, 0.3],
                [0.5, 0.1]
            ])
            .neurons([
                LifNeuron::new(0.1, 0.1, 0.4, 0.4),
                LifNeuron::new(0.02, 0.3, 0.7, 0.67),
                LifNeuron::new(0.01, 0.2, 0.9, 0.456)
            ])
            .intra_weights([
                [ 0.0, -0.1, -0.3],
                [-0.2,  0.0, -0.5],
                [-0.1, -0.9,  0.0]
            ])
        .build();

    // printing the SNN network just built
    println!("This is the SNN network built: \n{:?}\n", snn);

    let output_spikes = snn.process(&[[0,1,1],[1,1,1],[0,0,1]]);

    println!("\nOUTPUT SPIKES\n");

    for spikes in output_spikes.to_vec() {
        for spike in spikes.to_vec() {
            print!("{} ",spike);
        }
        println!();
    }

    let output_expected:[[u8;3];3] = [[0,0,1],[1,1,1],[1,0,0]];

    assert_eq!(output_spikes, output_expected);
}

#[test]
fn fluent_builder_test2() {
    println!("This is the second test useful to check the Fluent Builder pattern");

    let snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.2, 0.9, 0.4]
        ])
        .neurons([
            LifNeuron::new(0.2, 0.01, 0.4, 0.8),
            LifNeuron::new(0.1, 0.2, 0.2, 0.2)
        ])
        .intra_weights([
            [ 0.0, -0.1],
            [-0.4,  0.0]
        ])
        .add_layer()
        .weights([
            [0.1, 0.2],
            [0.2, 0.9],
            [0.5, 0.1],
            [0.9, 0.8]
        ])
        .neurons([
            LifNeuron::new(0.1, 0.4, 0.23, 0.7),
            LifNeuron::new(0.2, 0.3, 0.43, 0.6),
            LifNeuron::new(0.3, 0.2, 0.54, 0.5),
            LifNeuron::new(0.4, 0.1, 0.1, 0.3)
        ])
        .intra_weights([
            [ 0.0, -0.1, -0.4, -0.8],
            [-0.4,  0.0, -0.9, -0.3],
            [-0.2, -0.3,  0.0, -0.2],
            [-0.3, -0.5, -0.6,  0.0]
        ])
        .build();

    // printing the SNN network just built
    println!("This is the SNN network built: \n{:?}\n", snn);

    let output_spikes = snn.process(&[[0,1,0],[1,0,1],[0,1,1]]);

    println!("\nOUTPUT SPIKES\n");

    for spikes in output_spikes.to_vec() {
        for spike in spikes.to_vec() {
            print!("{} ",spike);
        }
        println!();
    }

    let output_expected:[[u8;3];4] = [[1,0,0],[1,1,1],[0,0,0],[1,1,1]];

    assert_eq!(output_spikes,output_expected);
}


