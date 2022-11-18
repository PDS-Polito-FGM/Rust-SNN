use crate::neuron::Neuron;
use crate::snn::dyn_snn::DynSNN;
use crate::snn::layer::Layer;

#[derive(Clone)]
pub struct DynSnnParams<N: Neuron> {
    pub input_dimensions: usize,            /* dimension of the network input layer */
    pub neurons: Vec<Vec<N>>,               /* neurons per each layer */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* (positive) weights between layers */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* (negative) weights inside the same layer */
    pub num_layers: usize,
}
#[derive(Clone)]
pub struct DynSnnBuilder<N: Neuron> {
    params: DynSnnParams<N>
}

impl<N: Neuron + Clone> DynSnnBuilder<N> {
    pub fn new() -> Self {
        Self {
            params: DynSnnParams {
                input_dimensions: 0,
                neurons: vec![],
                extra_weights: vec![],
                intra_weights: vec![],
                num_layers: 0
            }
        }
    }

    pub fn get_params(&self) -> DynSnnParams<N> {
        self.params.clone()
    }

    pub fn add_layer(mut self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self {
        if self.params.num_layers == 0 {
            self.params.input_dimensions = neurons.len();
        }
        let mut params = self.params;
        params.neurons.push(neurons);
        params.extra_weights.push(extra_weights);
        params.intra_weights.push(intra_weights);
        params.num_layers += 1;
        Self { params }
    }

    pub fn build(self) -> DynSNN<N> {
        if  self.params.neurons.len() != self.params.extra_weights.len() &&
            self.params.neurons.len() != self.params.intra_weights.len() {
            // it must not happen
            panic!("Error: the number of neurons layers does not correspond to the number of weights layers")
        }
        let mut layers: Vec<Layer<N>> = Vec::new();
        let mut neurons_iter = self.params.neurons.into_iter();
        let mut extra_weights_iter = self.params.extra_weights.into_iter();
        let mut intra_weights_iter = self.params.intra_weights.into_iter();


        // * retrieve the Neurons, the extra weights and the intra weights for each layer *
        while let Some(layer_neurons) = neurons_iter.next() {
            let layer_extra_weights = extra_weights_iter.next().unwrap();
            let layer_intra_weights = intra_weights_iter.next().unwrap();

            // create and save the new layer
            let new_layer = Layer::new(layer_neurons, layer_extra_weights, layer_intra_weights);
            layers.push(new_layer);
        }


        DynSNN::new(layers)
    }

}