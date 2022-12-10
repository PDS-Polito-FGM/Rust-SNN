use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use colored::Colorize;
use pds_snn::builders::{SnnBuilder, DynSnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;

fn main() {

    const N_NEURONS: usize = 400;
    const N_INPUTS: usize = 784;
    const N_INSTANTS: usize = 3500;

    let _input_spikes: [[u8; N_INSTANTS]; N_INPUTS] = read_input_spikes();

    let neurons: [LifNeuron; N_NEURONS] = build_neurons();

    let extra_weights: [[f64; N_NEURONS]; N_NEURONS] = read_extra_weights();

    let intra_weights: [[f64; N_NEURONS]; N_NEURONS] = build_intra_weights();

    println!("{}","Building the SNN network...".yellow());

    let building_start = Instant::now();

    /* SNN network */
    /*
    let _snn = SnnBuilder::new()
        .add_layer()
        .weights(extra_weights)
        .neurons(neurons)
        .intra_weights(intra_weights)
        .build();
     */

    /*  Dynamic SNN network  */
    let _snn = DynSnnBuilder::new(N_INPUTS)
        .add_layer(neurons.to_vec(),
                   extra_weights.map(|el| el.to_vec()).to_vec(),
                   intra_weights.map(|el| el.to_vec()).to_vec())
        .build();

    let building_end = building_start.elapsed();

    println!("{}","Done!".green());

    println!("{}",format!("\nTime elapsed in building the network: {}.{:03} seconds\n", building_end.as_secs(), building_end.subsec_millis()).blue());

    println!("{}","Computing the output spikes...".yellow());

    let computing_start = Instant::now();

    //TODO: Call the process function to compute the real output spikes - Francesco

    let computing_end = computing_start.elapsed();

    println!("{}","Done!".green());

    println!("{}", format!("\nTime elapsed in computing the output spikes: {}.{:03} seconds\n", computing_end.as_secs(), computing_end.subsec_millis()).blue());

    let _output_spikes: [[u8; N_INSTANTS]; N_NEURONS] = [[0; N_INSTANTS]; N_NEURONS];
    write_to_output_file(_output_spikes);

}

/* * USEFUL FUNCTIONS * */

/**
    This function builds the neurons of the network.
*/
fn build_neurons<const N_NEURONS: usize>() -> [LifNeuron; N_NEURONS] {

    let thresholds: [f64; N_NEURONS] = read_thresholds();

    let v_rest: f64 = -65.0;
    let v_reset: f64 = -60.0;
    let tau: f64 = 100.0;

    let mut neurons: Vec<LifNeuron> = Vec::with_capacity(N_NEURONS);

    for i in 0..N_NEURONS {

        let neuron = LifNeuron::new(thresholds[i], v_rest, v_reset, tau);

        neurons.push(neuron);
    }

    neurons.try_into().unwrap()
}

/**
    This function builds a 2D array of intra weights
*/
fn build_intra_weights<const N_NEURONS: usize>() -> [[f64; N_NEURONS]; N_NEURONS] {

    let value: f64 = -15.0;

    let mut intra_weights: [[f64; N_NEURONS]; N_NEURONS] = [[0f64; N_NEURONS]; N_NEURONS];

    println!("{}","Computing intra weights...".yellow());

    for i in 0..N_NEURONS {
        for j in 0..N_NEURONS {
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
    This function reads the input spikes from the input file and returns a 2D array of u8.
*/
fn read_input_spikes<const N_INSTANTS: usize, const N_INPUTS: usize>() -> [[u8; N_INSTANTS]; N_INPUTS] {
    let path_input = "inputSpikes.txt";
    let input = File::open(path_input).expect("Something went wrong opening the file inputSpikes.txt!");
    let buffered = BufReader::new(input);

    let mut input_spikes: [[u8; N_INSTANTS]; N_INPUTS] = [[0; N_INSTANTS]; N_INPUTS];

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
    This function reads the weights file and returns a 2D array of weights
*/
fn read_extra_weights<const N_NEURONS: usize, const N_INPUTS: usize>() -> [[f64; N_INPUTS]; N_NEURONS] {
    let path_weights_file = "./networkParameters/weightsOut.txt";
    let input = File::open(path_weights_file).expect("Something went wrong opening the file weightsOut.txt!");
    let buffered = BufReader::new(input);

    let mut extra_weights: [[f64; N_INPUTS]; N_NEURONS] = [[0f64; N_INPUTS]; N_NEURONS];

    println!("{}","Reading weights from file weightsOut.txt...".yellow());

    let mut i = 0;

    for line in buffered.lines() {
        let split: Vec<String> = line.unwrap().as_str().split(" ").map(|el| el.to_string()).collect::<Vec<String>>();

        for j in 0..N_INPUTS {
            extra_weights[i][j] = split[j].parse::<f64>().expect("Cannot parse string into f64!");
        }

        i += 1;
    }

    println!("{}", "Done!".green());

    extra_weights
}

/**
    This function reads the threshold file and returns an array of thresholds
*/
fn read_thresholds<const N_NEURONS: usize>() -> [f64; N_NEURONS] {
    let path_threshold_file = "./networkParameters/thresholdsOut.txt";
    let input = File::open(path_threshold_file).expect("Something went wrong opening the file thresholdsOut.txt!");
    let buffered = BufReader::new(input);

    let mut thresholds: [f64; N_NEURONS] = [0f64; N_NEURONS];

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
fn write_to_output_file<const N_NEURONS: usize, const N_INSTANTS: usize>(output_spikes: [[u8; N_INSTANTS]; N_NEURONS]) -> () {
    let path_output = "outputCounters.txt";
    let mut output_file = File::create(path_output).expect("Something went wrong opening the file outputCounters.txt!");

    let mut neurons_sum: [u32; N_NEURONS] = [0; N_NEURONS];

    println!("{}","Computing sum of the spikes for each neuron...".yellow());

    for i in 0..N_NEURONS {
        for j in 0..N_INSTANTS {
            neurons_sum[i] += output_spikes[i][j] as u32;
        }
    }

    println!("{}","Done!".green());

    println!("{}","Writing data into file outputCounters.txt...".yellow());

    for i in 0..N_NEURONS {
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