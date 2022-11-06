use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

#[test]
fn fluent_builder_test1() {
    println!("This is the first test useful to check the Fluent Builder pattern");

    let mut snn = SnnBuilder::new()
        .add_layer()
            .weights([
                [0.5, 0.4, 0.3],
                [0.2, 0.2, 0.6]
            ])
            .neurons([
                LifNeuron::new(0.5, 0.1, 0.25, 1.0),
                LifNeuron::new(0.5, 0.2, 0.25, 1.0),
            ])
            .intra_weights([
                [ 0.0, -0.2],
                [-0.3, -0.0]
            ])
        .add_layer()
            .weights([
                [0.1, 0.1],
                [0.2, 0.3],
                [0.5, 0.1]
            ])
            .neurons([
                LifNeuron::new(0.5, 0.1, 0.2, 1.0),
                LifNeuron::new(0.5, 0.1, 0.2, 1.0),
                LifNeuron::new(0.5, 0.1, 0.2, 1.0)
            ])
            .intra_weights([
                [ 0.0, -0.1, -0.1],
                [-0.1,  0.0, -0.1],
                [-0.1, -0.1,  0.0]
            ])
        .build();

    // printing the SNN network just built
    println!("This is the SNN network built: \n{:?}\n", snn);

    let input_spikes = [
        [0,1,1,0],  /* 1st neuron spikes */
        [1,1,1,0],  /* 2nd neuron spikes */
        [0,0,1,1]   /* 3rd neuron spikes */
    ];

    let output_spikes = snn.process(&input_spikes);

    println!("OUTPUT SPIKES");
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

    let output_expected:[[u8;4];3] = [[0,0,0,0], [0,1,1,0], [0,1,1,0]];

    assert_eq!(output_spikes, output_expected);
}

#[test]
fn fluent_builder_test2() {
    println!("This is the second test useful to check the Fluent Builder pattern");

    let mut snn = SnnBuilder::new()
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

    println!("OUTPUT SPIKES");

    for spikes in output_spikes.to_vec() {
        for spike in spikes.to_vec() {
            print!("{} ",spike);
        }
        println!();
    }
    println!();

    let output_expected:[[u8;3];4] = [[1,1,1],[1,1,1],[1,1,1],[1,1,1]];

    assert_eq!(output_spikes,output_expected);
}

#[test]
#[should_panic]
fn panic_builder_test1() {
    println!("This test should panic!");

    let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [-2.6, 0.2, 0.3],   /* Negative weight */
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
}

#[test]
#[should_panic]
fn panic_builder_test2() {
    println!("This test should panic!");

    let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.2, 3.2, 0.4]     /* weight > 1 */
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
}

#[test]
#[should_panic]
fn panic_builder_test3() {
    println!("This test should panic!");

    let _snn = SnnBuilder::new()
        .add_layer()
        .weights([
            [0.1, 0.2, 0.3],
            [0.2, 0.6, 0.4]
        ])
        .neurons([
            LifNeuron::new(0.2, 0.01, 0.4, 0.8),
            LifNeuron::new(0.1, 0.2, 0.2, 0.2)
        ])
        .intra_weights([
            [ -4.6, -0.1],   /* intra weight < -1 */
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
}