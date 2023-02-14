
pub mod demo_internals {
    use std::fmt::Debug;

    /* internal functions */

    pub fn print_instants(n: usize) {
        print!(" t\t ");
        (0..n).into_iter().for_each(|t| print!("{}  ", t));
        println!();
    }

    pub fn print_spikes<'a, S: IntoIterator<Item=K>, K: IntoIterator<Item=&'a u8> + Debug>(spikes: S, role: &str) {
        spikes.into_iter()
            .zip(vec!["1st", "2nd", "3rd", "4th", "5th"].into_iter())
            .for_each(|(train_of_spikes, pos)|
                println!("\t{:?} \t\t {} *{}* neuron train of spikes", train_of_spikes, pos, role));
    }

    #[allow(dead_code)]
    pub fn print_layer<L: IntoIterator<Item=N>, N: Debug>(neurons: L) {
        neurons.into_iter().for_each(|neuron| println!("- {:?}", neuron));
        println!("Added corresponding extra-layer and intra-layer weights");
    }
}

#[allow(dead_code)]
fn main() {}

