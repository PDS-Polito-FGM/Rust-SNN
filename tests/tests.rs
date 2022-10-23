use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;
use pds_snn::SNN;

#[test]
fn fake_test1() {
    println!("This is a fake test to check the Fluent Builder pattern");

    let builder = SnnBuilder::new();

    let snn = builder
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
        .build();

    let snn2 = SnnBuilder::new()
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


    assert_eq!(snn.s, true);
    assert_eq!(snn2.s, true);
}

#[test]
fn fake_test2() {
    println!("This is a fake test");

    // let l: LifNeuron = LifNeuron::new(1.0, 2.0, 3.0, 4.0);

    assert_eq!(1, 1);
}

#[test]
#[ignore]
fn fake_test3() {
    println!("This is a fake test");

    let snn: SNN<LifNeuron> = SNN::<LifNeuron>::new(true);

    assert_eq!(snn.s, true);
}
