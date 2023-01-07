use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use colored::Colorize;
use pds_snn::builders::DynSnnBuilder;
use pds_snn::models::neuron::lif::LifNeuron;

fn main() {
    let n_neurons: usize = 400;
    let n_inputs: usize = 784;
    let n_instants: usize = 3500;

    /* read input spikes from file (coming from python script) */
    let input_spikes: Vec<Vec<u8>> = read_input_spikes(n_instants, n_inputs);

    /* read neurons' and weights' parameters from config files */
    let neurons: Vec<LifNeuron> = build_neurons(n_neurons);
    let extra_weights: Vec<Vec<f64>> = read_extra_weights(n_neurons, n_inputs);
    let intra_weights: Vec<Vec<f64>> = build_intra_weights(n_neurons);

    println!("{}","Building the SNN network...".yellow());

    let building_start = Instant::now();

    /*  Build dynamic SNN network  */
    let mut snn = DynSnnBuilder::new(n_inputs)
        .add_layer(neurons, extra_weights, intra_weights)
        .build();

    let building_end = building_start.elapsed();

    println!("{}","Done!".green());
    println!("{}",format!("\nTime elapsed in building the network: {}.{:03} seconds\n", building_end.as_secs(), building_end.subsec_millis()).blue());
    println!("{}","Computing the output spikes...".yellow());

    let computing_start = Instant::now();

    /* calling the dynSNN process function to process input spikes */
    let output_spikes = snn.process(&input_spikes);

    let computing_end = computing_start.elapsed();

    println!("{}","Done!".green());
    println!("{}", format!("\nTime elapsed in computing the output spikes: {}.{:03} seconds\n", computing_end.as_secs(), computing_end.subsec_millis()).blue());

    /* write output spikes to file (to pass them to python script) */
    write_to_output_file(output_spikes, n_neurons, n_instants);
}

/* * USEFUL FUNCTIONS * */

/**
    This function builds the neurons of the network.
*/
fn build_neurons(n_neurons: usize) -> Vec<LifNeuron> {
    let thresholds: Vec<f64> = read_thresholds(n_neurons);

    let v_rest: f64 = -65.0;
    let v_reset: f64 = -60.0;
    let tau: f64 = 100.0;
    let dt: f64 = 0.1;

    let mut neurons: Vec<LifNeuron> = Vec::with_capacity(n_neurons);

    println!("{}","Building the neurons...".yellow());

    for i in 0..n_neurons {
        let neuron = LifNeuron::new(thresholds[i], v_rest, v_reset, tau, dt);
        neurons.push(neuron);
    }

    println!("{}", "Done!".green());
    neurons
}

/**
    This function builds a 2D Vec of intra weights
*/
fn build_intra_weights(n_neurons: usize) -> Vec<Vec<f64>> {
    let value: f64 = -15.0;

    let mut intra_weights: Vec<Vec<f64>> = vec![vec![0f64; n_neurons]; n_neurons];

    println!("{}","Building the intra weights...".yellow());

    for i in 0..n_neurons {
        for j in 0..n_neurons {
            if i == j {
                intra_weights[i][j] = 0.0;
            } else {
                intra_weights[i][j] = value;
            }
        }
    }

    println!("{}", "Done!".green());

    intra_weights
}

/**
    This function reads the input spikes from the input file and returns a 2D Vec of u8.
*/
fn read_input_spikes(n_instants: usize, n_inputs: usize) -> Vec<Vec<u8>> {
    let path_input = "./inputSpikes.txt";
    let input = File::open(path_input).expect("Something went wrong opening the file inputSpikes.txt!");
    let buffered = BufReader::new(input);

    let mut input_spikes: Vec<Vec<u8>> = vec![vec![0; n_instants]; n_inputs];

    println!("{}","Reading input spikes from file inputSpikes.txt...".yellow());

    let mut i = 0;

    for line in buffered.lines() {
        let mut j = 0;
        let chars = convert_line_into_u8(line.unwrap());
        chars.into_iter().for_each(| ch | {
            input_spikes[j][i] = ch;
            j += 1;
        });
        i += 1;
    }

    println!("{}","Done!".green());

    input_spikes
}

/**
    This function reads the weights file and returns a 2D Vec of weights
*/
fn read_extra_weights(n_neurons: usize, n_inputs: usize) -> Vec<Vec<f64>> {
    let path_weights_file = "./networkParameters/weightsOut.txt";
    let input = File::open(path_weights_file).expect("Something went wrong opening the file weightsOut.txt!");
    let buffered = BufReader::new(input);

    let mut extra_weights: Vec<Vec<f64>> = vec![vec![0f64; n_inputs]; n_neurons];

    println!("{}","Reading weights from file weightsOut.txt...".yellow());

    let mut i = 0;

    for line in buffered.lines() {
        let split: Vec<String> = line.unwrap().as_str().split(" ").map(|el| el.to_string()).collect::<Vec<String>>();

        for j in 0..n_inputs {
            extra_weights[i][j] = split[j].parse::<f64>().expect("Cannot parse string into f64!");
        }

        i += 1;
    }

    println!("{}", "Done!".green());

    extra_weights
}

/**
    This function reads the threshold file and returns a Vec of thresholds
*/
fn read_thresholds(n_neurons: usize) -> Vec<f64> {
    let path_threshold_file = "./networkParameters/thresholdsOut.txt";
    let input = File::open(path_threshold_file).expect("Something went wrong opening the file thresholdsOut.txt!");
    let buffered = BufReader::new(input);

    let mut thresholds: Vec<f64> = vec![0f64; n_neurons];

    println!("{}","Reading thresholds from file thresholdsOut.txt...".yellow());

    let mut i = 0;

    for line in buffered.lines() {

        thresholds[i] = line.unwrap().parse::<f64>().expect("Cannot parse String into f64!");

        i += 1;
    }

    println!("{}", "Done!".green());

    thresholds
}

/**
    This function writes the output spikes to the output file.
*/
fn write_to_output_file(output_spikes: Vec<Vec<u8>>, n_neurons: usize, n_instants: usize) -> () {
    let path_output = "./outputCounters.txt";
    let mut output_file = File::create(path_output).expect("Something went wrong opening the file outputCounters.txt!");

    let mut neurons_sum: Vec<u32> = vec![0; n_neurons];

    println!("{}","Computing sum of the spikes for each neuron...".yellow());

    for i in 0..n_neurons {
        for j in 0..n_instants {
            neurons_sum[i] += output_spikes[i][j] as u32;
        }
    }

    println!("{}","Done!".green());
    println!("{}","Writing data into file outputCounters.txt...".yellow());

    for i in 0..n_neurons {
        output_file.write_all(format!("{}\n", neurons_sum[i]).as_bytes()).expect("Something went wrong writing into the file outputCounters.txt!");
    }

    println!("{}","Done!".green());
}

/**
    This function converts a line of the input file into a Vec of u8.
*/
fn convert_line_into_u8(line: String) -> Vec<u8> {
       line.chars()
            .map(|ch| (ch.to_digit(10).unwrap()) as u8)
            .collect::<Vec<u8>>()
}