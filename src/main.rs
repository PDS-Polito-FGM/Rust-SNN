use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use std::time::Instant;
use colored::Colorize;
use pds_snn::builders::{SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;

fn main() {

    const N_NEURONS: usize = 400;
    const N_INPUTS: usize = 784;
    const N_INSTANTS: usize = 3500;

    let _input_spikes: [[u8; N_INSTANTS]; N_INPUTS] = read_input_spikes();

    println!("{}","Building the SNN network...".yellow());

    let building_start = Instant::now();

    //TODO: Build the SNN network - Francesco

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