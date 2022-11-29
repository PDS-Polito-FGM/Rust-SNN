use std::fs::File;
use std::io::{BufRead, BufReader, Write};
use pds_snn::builders::{SnnBuilder};
use pds_snn::models::neuron::lif::LifNeuron;

fn main() {

    const N_NEURONS: usize = 400;
    const N_INPUTS: usize = 784;
    const N_INSTANTS: usize = 3500;

    let _input_spikes: [[u8; N_INSTANTS]; N_INPUTS] = read_input_spikes();

    println!("Building the SNN network...");

    //TODO: Build the SNN network - Francesco

    println!("Done!");

    println!("Computing the output spikes...");

    //TODO: Call the process function to compute the real output spikes - Francesco

    println!("Done!");

    let _output_spikes: [[u8; N_INSTANTS]; N_NEURONS] = [[0; N_INSTANTS]; N_NEURONS];
    write_to_output_file(_output_spikes);

}

/*
    This function reads the input spikes from the input file and returns a 2D array of u8.
*/
fn read_input_spikes<const N_INSTANTS: usize, const N_INPUTS: usize>() -> [[u8; N_INSTANTS]; N_INPUTS] {
    let path_input = "inputSpikes.txt";
    let input = File::open(path_input).expect("Something went wrong opening the file inputSpikes.txt!");
    let buffered = BufReader::new(input);

    let mut input_spikes: [[u8; N_INSTANTS]; N_INPUTS] = [[0; N_INSTANTS]; N_INPUTS];

    println!("Reading input spikes from file inputSpikes.txt...");

    let mut i = 0;

    for line in buffered.lines() {
        let mut j = 0;
        let chars = line.unwrap()
                                .chars()
                                .map(| ch| ch.to_digit(10).unwrap())
                                .map(|ch| ch as u8)
                                .collect::<Vec<u8>>();
        chars.into_iter().for_each(| ch | {
            input_spikes[j][i] = ch;
            j += 1;
        });
        i += 1;
    }

    println!("Done!");

    input_spikes
}

/*
    This function writes the output spikes to the output file.
*/
fn write_to_output_file<const N_NEURONS: usize, const N_INSTANTS: usize>(output_spikes: [[u8; N_INSTANTS]; N_NEURONS]) -> () {
    let path_output = "outputCounters.txt";
    let mut output_file = File::create(path_output).expect("Something went wrong opening the file outputCounters.txt!");

    let mut output_to_file: [[u8; N_NEURONS]; N_INSTANTS] = [[0; N_NEURONS]; N_INSTANTS];

    for i in 0..N_INSTANTS {
        for j in 0..N_NEURONS {
            output_to_file[i][j] = output_spikes[j][i];
        }
    }

    println!("Writing output spikes to file outputCounters.txt...");

    for i in 0..N_INSTANTS {
        let mut output_line = String::new();
        for j in 0..N_NEURONS {
            output_line.push_str(&output_to_file[i][j].to_string());
        }
        output_file.write(output_line.as_bytes()).expect("Something went wrong writing to the file outputCounters.txt!");
    }

    println!("Done!");

}