# Spiking Neural Network library
- [Description](#description)
- [Group Members](#group-members)
- [Dependencies](#dependencies)
- [File Structure](#file-structure)
- [Organization](#organization)
- [Main Structures](#main-structures)
- [Main Methods](#main-methods)
- [Usage Examples](#usage-examples)

## Description
This is a `Rust library` aiming to model a `Spiking Neural Network`. It is carried out for the `group project` related to the "Programmazione di Sistema" course of the Politecnico di Torino, a.y. 2021-2022.

The library provide support for the implementation of `Spiking Neural Network` models to be executed over spikes datasets.
It does *not* support the training phase of the network, but only the execution one.

## Group members
- Francesco Rosati
- Giuseppe Lazzara
- Mario Mastrandrea

## Dependencies
- `Rust` (version 1.56.1)
- `Cargo` (version 1.56.0)

No other particular dependencies are required.

## File Structure
The code is structured as follows:
- `src/` contains the source code of the library
  + `models/` contains the specific models' implementations (here only `Lif Neuron`)
  + `snn/`    contains the SNN generic implementation
    + `builders` contains the builder objects for the SNN
- `tests/` contains the tests of the library

## Organization
The library is organized as follows:

- ### Builder
  The `Builder` module allows you to actually create the structure of
  the network with the corresponding layers, neurons per each layer,
  the corresponding weights between them and between neurons of the same layer.
  The library provides two `Builder` implementations:
  - #### SnnBuilder 

    The `SnnBuilder` allows to *statically* create a `Spiking Neural Network` taking for each layer static vectors of neurons,
    weights and intra-layer weights. The library can check the correctness of the network structure at *compile-time*, but this implies that all
    the structures of the network are allocated on the **Stack** (**Not fitting with large networks**).

  - #### DynSnnBuilder 
    The `DynSnnBuilder` allows to *dynamically* create a `Spiking Neural Network` taking for each layer dynamic vectors of neurons,
    weights and intra-layer weights. The library cannot check the correctness of the network structure until the *execution time*, but this implies that all
    the structures of the network are allocated on the **Heap** (**Fitting with large networks**).

- ### Network
  The `Network` module allows you to actually execute the network on a given input.
  The library provides two `Network` implementations:
  - #### Snn (Spiking Neural Network)

    The `Snn` is created by the `SnnBuilder` and allows to execute the network on a given input through the `process()` method.
    As the `SnnBuilder`, the `Snn` receives the input as a static vector of spikes and produces as output a static vector of spikes too.
    The correctness of the input can be checked at *compile time*.

  - #### DynSnn (Dynamic Spiking Neural Network)

    The `DynSnn` is created by the `DynSnnBuilder` and allows to execute the network on a given input through the `process()` method.
    As the `DynSnnBuilder`, the `DynSnn` receives the input as a dynamic vector of spikes and produces as output a dynamic vector of spikes too.
    The correctness of the input can be checked only at *run time*.

## Main structures
The library provides the following main structures:

- `LifNeuron` represents a neuron for the `Leaky Integrate and Fire` model, it can be used to build a `Layer` of neurons. 

```rust
pub struct LifNeuron {
    /* const fields */
    v_th:    f64,       /* threshold potential */
    v_rest:  f64,       /* resting potential */
    v_reset: f64,       /* reset potential */
    tau:     f64, 
    dt:      f64,       /* time interval between two consecutive instants */
    /* mutable fields */
    v_mem:   f64,       /* membrane potential */
    ts:      u64,       /* last instant in which receiving at least one spike */
}
```
For more information about the `Leaky Integrate and Fire` model, see [here](https://www.nature.com/articles/s41598-017-07418-y).

- `Layer` represents a layer of neurons, it can be used to build a `SNN` or `DynSNN`  of layers.
```rust
pub struct Layer<N: Neuron + Clone + Send + 'static> {
    neurons: Vec<N>,                /* neurons of the layer */
    weights: Vec<Vec<f64>>,         /* weights between the neurons of this layer and the previous one */
    intra_weights: Vec<Vec<f64>>,   /* weights between the neurons of this layer */
    prev_output_spikes: Vec<u8>     /* output spikes of the previous instant */
}
```

- `SpikeEvent` represents an event of a neurons layer firing at a certain instant of time.
  It wraps the spikes flowing through the network
```rust
pub struct SpikeEvent {
    ts: u64,            /* discrete time instant */
    spikes: Vec<u8>,    /* vector of spikes in that instant (a 1/0 for each input neuron)  */
}
```

- `SNN` represents a `Spiking Neural Network` composed by a vector of `Layer`s.
```rust
pub struct SNN<N: Neuron + Clone + Send + 'static, const NET_INPUT_DIM: usize, const NET_OUTPUT_DIM: usize> {
    layers: Vec<Arc<Mutex<Layer<N>>>>
}
```

- `DynSNN` represents a `Dynamic Spiking Neural Network` composed by a vector of `Layer`s.
```rust
pub struct DynSNN <N: Neuron + Clone + 'static>{
    layers: Vec<Arc<Mutex<Layer<N>>>>
}
```

- `Processor` is the object in charge of managing the layers' threads and processing the input spikes events
```rust
pub struct Processor { }
```

- `SnnBuilder` represents the builder for a `SNN`
```rust
pub struct SnnBuilder<N: Neuron + Clone + Send + 'static> {
    params: SnnParams<N>
}

pub struct SnnParams<N: Neuron + Clone + Send + 'static> {
    pub neurons: Vec<Vec<N>>,               /* neurons per each layer */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* (positive) weights between layers */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* (negative) weights inside the same layer */
}
```

-  `DynSnnBuilder` represents the builder for a `DynSNN`
```rust
pub struct DynSnnBuilder<N: Neuron> {
    params: DynSnnParams<N>
}

pub struct DynSnnParams<N: Neuron> {
    pub input_dimensions: usize,            /* dimension of the network input layer */
    pub neurons: Vec<Vec<N>>,               /* neurons per each layer */
    pub extra_weights: Vec<Vec<Vec<f64>>>,  /* (positive) weights between layers */
    pub intra_weights: Vec<Vec<Vec<f64>>>,  /* (negative) weights inside the same layer */
    pub num_layers: usize,                  /* number of layers */
}
```

## Main methods
The library provides the following main methods:
 - ### Builder Methods
   - #### `SnnBuilder` methods:
   
     - **new()** method: 
     
        ```rust
          pub fn new() -> Self
         ```         
       
         creates a new `SnnBuilder` 
     
     - **add_layer()** method:
     
       ```rust
          pub fn add_layer(self) -> WeightsBuilder<N, OUTPUT_DIM, NET_INPUT_DIM> 
       ``` 
         adds a new (empty) layer to the `SnnBuilder`
     
     -  **weights()** method:
     
        ```rust
          pub fn weights<const NUM_NEURONS: usize>(mut self, weights: [[f64; INPUT_DIM]; NUM_NEURONS])
                                           -> NeuronsBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
          ```
          adds weights to the current layer 
     
       - **neurons()** method:
       
         ```rust
             pub fn neurons(mut self, neurons: [N; NUM_NEURONS]) -> IntraWeightsBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
         ```
          adds neurons to the current layer 
     - **intra_weights()** method
        ```rust
         pub fn intra_weights(mut self, intra_weights: [[f64; NUM_NEURONS]; NUM_NEURONS])
                    -> LayerBuilder<N, NUM_NEURONS, NET_INPUT_DIM>
         ```
       
         adds intra-weights to the current layer

     - **build()** method:
     
        ```rust
         pub fn build(self) -> SNN<N, { NET_INPUT_DIM }, { OUTPUT_DIM }>
         ```
       
         builds the `SNN` from the information collected so far by the `SnnBuilder`

 
- #### `DynSnnBuilder` methods:
   - **new()** method:
   
     ```rust
     pub fn new(input_dimension: usize) -> Self 
        ```
     
     creates a new `DynSnnBuilder`
   - **add_layer()** method:
   
     ```rust
     pub fn add_layer(self, neurons: Vec<N>, extra_weights: Vec<Vec<f64>>, intra_weights: Vec<Vec<f64>>) -> Self
     ```

     adds a new `layer` to the SNN with the given `neurons`, `weights` and `intra_weights` passed as parameters

   - **build()** method:
   
     ```rust
     pub fn build(self) -> DynSNN<N>
     ```
     
     builds the `DynSNN` from the information collected so far by the `DynSnnBuilder`

 - ### Network Methods
   - #### `Snn` method:
      - process() method:
      
        ```rust
         pub fn process<const SPIKES_DURATION: usize>(&mut self, spikes: &[[u8; SPIKES_DURATION]; NET_INPUT_DIM])
                                                 -> [[u8; SPIKES_DURATION]; NET_OUTPUT_DIM]
        ```
        
        processes the input spikes passed as parameter and returns the output spikes of the network
      - 
   - #### `DynSnn` method:
        - process() method:
        
            ```rust
             pub fn process(&mut self, spikes: &Vec<Vec<u8>>)
                                                 -> Vec<Vec<u8>> 
            ```
          
            processes the input spikes passed as parameter and returns the output spikes of the network
   


## Usage examples
The following example shows how to *statically* create a `Spiking Neural Network` with 2 input neurons and  
a single layer of 3 `LifNeuron`s using the `SnnBuilder`, and how to execute it on a given input of 3 instants per neuron.

```rust
use pds_snn::builders::SnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;


 let mut snn = SnnBuilder::new()
        .add_layer()    /* first layer (input dimension automatically inferred) */
            .weights([
                [0.1, 0.2],     /* weigths from input layer to the 1st neuron */
                [0.3, 0.4],     /* weigths from input layer to the 2nd neuron */
                [0.5, 0.6]      /* weigths from input layer to the 3rd neuron */
            ]).neurons([
                /* 3 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.1, -0.15],     /* weigths from the same layer to the 1st neuron */
                [-0.05, 0.0, -0.1],     /* weigths from the same layer to the 2nd neuron */
                [-0.15, -0.1, 0.0]      /* weigths from the same layer to the 3rd neuron */
        ])
        .add_layer()    /* second layer */
            .weights([
                [0.11, 0.29, 0.3],      /* weigths from previous layer to the 1st neuron */
                [0.33, 0.41, 0.57]      /* weigths from previous layer to the 2nd neuron */
            ]).neurons([
                /* 2 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ]).intra_weights([
                [0.0, -0.25],       /* weigths from the same layer to the 1st neuron */
                [-0.10, 0.0]        /* weigths from the same layer to the 2nd neuron */
            ]).build();     /* create the network */

        /* process input spikes */
        let output_spikes = snn.process(&[
            [1,0,1],    /* 1st neuron input */
            [0,0,1]     /* 2ns neuron input */
        ]);    
```
The following example shows how to *dynamically* create a `Spiking Neural Network` with 2 input neurons
and a single layer of 3 `LifNeuron`s using the `DynSnnBuilder`, and how to execute it on a given input of 3 instants per neuron.

```rust
use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

    let mut snn = DynSnnBuilder::new(2)     /* input dimension of 2 */
        .add_layer(     /* first layer*/
            vec![   /* 3 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ], 
            vec![   /* weights */
                vec![0.1, 0.2],     /* weigths from input layer to the 1st neuron */
                vec![0.3, 0.4],     /* weigths from input layer to the 2nd neuron */
                vec![0.5, 0.6]      /* weigths from input layer to the 3rd neuron */
            ], 
            vec![   /* intra-weights */
                vec![0.0, -0.1, -0.15],     /* weigths from the same layer to the 1st neuron */
                vec![-0.05, 0.0, -0.1],     /* weigths from the same layer to the 2nd neuron */
                vec![-0.15, -0.1, 0.0]      /* weigths from the same layer to the 3rd neuron */
            ]
        ).add_layer(    /* second layer */
            vec![   /* 2 LIF neurons */
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0),
                LifNeuron::new(0.3, 0.05, 0.1, 1.0, 1.0)
            ],
            vec![   /* weights */
                vec![0.11, 0.29, 0.3],      /* weigths from previous layer to the 1st neuron */
                vec![0.33, 0.41, 0.57]      /* weigths from previous layer to the 2nd neuron */
            ],
            vec![   /* intra-weights */
                vec![0.0, -0.25],       /* weigths from the same layer to the 1st neuron */
                vec![-0.10, 0.0]        /* weigths from the same layer to the 2nd neuron */
            ]
        ).build();      /* create the network */

        /* process input spikes */
        let output_spikes = snn.process(&vec![
            vec![1,0,1],    /* 1st neuron input */
            vec![0,0,1]     /* 2nd neuron input */
        ]);
```
