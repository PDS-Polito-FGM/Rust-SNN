use pds_snn::snn::builder::Builder;
use pds_snn::models::neuron::lif::LifNeuron;
use pds_snn::snn::SNN;

#[test]
fn fake_test1() {
    println!("This is a fake test");

    let b = Builder{
        a: 10
    };

    assert_eq!(b.a, 10);
}

#[test]
fn fake_test2() {
    println!("This is a fake test");

    let l = LifNeuron {
        l: 5.3
    };

    assert_eq!(l.l, 5.3);
}

#[test]
fn fake_test3() {
    println!("This is a fake test");

    let snn = SNN::new(true, 5.3);

    assert_eq!(snn.s, true);
    assert_eq!(snn.l.l, 5.3);
}
