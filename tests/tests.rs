use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

#[test]
fn fake_test1() {
    println!("This is a fake test to check the Fluent Builder pattern");

    let snn = Box::new(SnnBuilder::new()
        .add_layer()
            .weights([
                [0.1, 0.1, 0.3],
                [0.2, 0.1, 0.3]
            ])
            .neurons([
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
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
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0)
            ])
            .intra_weights([
                [ 0.0, -0.1, -0.3],
                [-0.2,  0.0, -0.5],
                [-0.1, -0.9,  0.0]
            ])
        .build());

    // printing the SNN network just built
    println!("This is the SNN network built: \n{:?}\n", snn);

    let static_snn= Box::leak(snn);

    /*
       By including the whole SNN network into the Box smart pointer, we can create a static network
       that will be never deallocated, so we are granting the right lifetime of the network among the
       threads
    */

    let _output_spikes = static_snn.process(&[[0,1,1],[1,1,1],[0,0,1]]);

    let _snn2 = SnnBuilder::new()
        .add_layer()
            .weights([
                [0.1, 0.2, 0.3],
                [0.2, 0.9, 0.4]
            ])
            .neurons([
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0)
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
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0)
            ])
            .intra_weights([
                [ 0.0, -0.1, -0.4, -0.8],
                [-0.4,  0.0, -0.9, -0.3],
                [-0.2, -0.3,  0.0, -0.2],
                [-0.3, -0.5, -0.6,  0.0]
            ])
        .build();

    assert_eq!(true, true);
}


