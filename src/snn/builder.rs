// * builder submodule *

use crate::snn::neuron::Neuron;
use crate::snn::SNN;

pub trait FirstLayerBuilder<N: Neuron, const INPUT_DIMENSION: usize> {
    fn add_layer(&mut self)
        -> &mut dyn WeightsBuilder<N, INPUT_DIMENSION>;
}

pub trait WeightsBuilder<N: Neuron, const INPUT_DIMENSION: usize> {
    fn weights<const NUM_NEURONS: usize>(&mut self, weights: [[f64; INPUT_DIMENSION]; NUM_NEURONS])
        -> &mut dyn NeuronsBuilder<N, NUM_NEURONS>;
}

pub trait NeuronsBuilder<N: Neuron, const NUM_NEURONS: usize> {
    fn neurons(&mut self, neurons: [N; NUM_NEURONS]) -> &mut dyn IntraweightsBuilder<N, NUM_NEURONS>;
}

pub trait IntraweightsBuilder<N: Neuron, const NUM_NEURONS: usize> {
    fn intraweights(&mut self, intra_weights: [[f64; NUM_NEURONS-1]; NUM_NEURONS])
        -> &mut dyn SnnBuilder<N, NUM_NEURONS>;
}

pub trait SnnBuilder<N: Neuron, const OUTPUT_DIMENSION: usize> {
    fn build(&mut self) -> SNN<N>;
    fn add_layer(&mut self) -> &mut dyn WeightsBuilder<N, OUTPUT_DIMENSION>;
}

/**
    Object for the configuration and creation of the Spiking Neural Network.
    It allows to configure the network step-by-step, adding one layer at a time,
    setting the input dimension, and then specifying the weights and the neurons for each layer.
    - It follows the (fluent) Builder design pattern.
*/

pub struct Builder<N: Neuron> {
    input_dimensions: usize,
    neurons: Vec<Vec<N>>,
    extra_weights: Vec<Vec<Vec<f64>>>,
    intra_weights: Vec<Vec<Vec<f64>>>,
}

impl<N: Neuron> Builder<N> {
    pub fn new() -> Self {
        Self {
            input_dimensions: 0,
            neurons: vec![],
            extra_weights: vec![],
            intra_weights: vec![],
        }
    }

    pub fn with_input_dimensions <const INPUT_DIMENSION: usize> (&mut self)
            -> &mut dyn FirstLayerBuilder<N, INPUT_DIMENSION> {
        self.input_dimensions = INPUT_DIMENSION;
        return self;
    }
}

impl<N: Neuron, const INPUT_DIMENSION: usize> FirstLayerBuilder<N, INPUT_DIMENSION> for Builder<N> {
    fn add_layer(&mut self) -> &mut dyn WeightsBuilder<N, INPUT_DIMENSION> {
        self
    }
}

impl<N: Neuron, const INPUT_DIMENSION: usize> WeightsBuilder<N, INPUT_DIMENSION> for Builder<N> {
    fn weights<const NUM_NEURONS: usize>(&mut self, weights: [[f64; INPUT_DIMENSION]; NUM_NEURONS])
        -> &mut dyn NeuronsBuilder<N, NUM_NEURONS> {
        self.extra_weights.push(Vec::from(&weights));
        self
    }
}

impl<N: Neuron> NeuronsBuilder<N> for Builder<N> {
    fn neurons(&mut self, neurons: Vec<N>) -> &mut dyn IntraweightsBuilder<N> {
        self.neurons.push(neurons);
        self
    }
}

impl<N: Neuron> IntraweightsBuilder<N> for Builder<N> {
    fn intraweights(&mut self, intra_weights: Vec<Vec<f64>>) -> &mut dyn SnnBuilder<N> {
        self.intra_weights.push(intra_weights);
        self
    }
}

impl<N: Neuron> SnnBuilder<N> for Builder<N> {
    fn build(&mut self) -> SNN<N> {
        SNN::<N>::new(true)
    }
}
