use pds_snn::models::neuron::lif::LifNeuron;
use pds_snn::snn::builder::Builder;
use pds_snn::snn::SNN;

#[test]
fn fake_test1() {
    println!("This is a fake test");

    /*
    let mut b = Builder::new();
    let snn = b
        .with_input_dimensions(3)
        .add_layer()
        .weights(vec![vec![0.1, 0.1, 0.3], vec![0.2, 0.1, 0.3]])
        .neurons(vec![
            LifNeuron::new(0.0, 0.0, 0.0, 0.0),
            LifNeuron::new(0.0, 0.0, 0.0, 0.0),
        ])
        .intraweights(vec![vec![-0.1], vec![-0.2]])
        .build();
    .add_layer()
    .weights(vec![
        vec![0.1, 0.1, 0.3],
        vec![0.2, 0.1, 0.3]
    ])
    .neurons(vec![
        LifNeuron::new(0.0, 0.0, 0.0, 0.0),
        LifNeuron::new(0.0, 0.0, 0.0, 0.0)
    ])
    .intraweights(vec![
        vec![-0.1],
        vec![-0.2]
    ]);*/

    /*
    let snn_builder = Builder::new();
    let snn = snn_builder.with_input_dimensions(3)
        .add_layer()
            .weights(vec![
                vec![0.1, 0.1, 0.3],
                vec![0.2, 0.1, 0.3]
            ])
            .neurons(vec![
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0)
            ])
            .intraweights(vec![
                vec![-0.1],
                vec![-0.2]
            ])
        .add_layer()
            .weights(vec![
                vec![0.1, 0.1, 0.3],
                vec![0.2, 0.1, 0.3]
            ])
            .neurons(vec![
                LifNeuron::new(0.0, 0.0, 0.0, 0.0),
                LifNeuron::new(0.0, 0.0, 0.0, 0.0)
            ])
            .intraweights(vec![
                vec![-0.1],
                vec![-0.2]
            ])
        .build();*/

    assert_eq!(true, true);
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
