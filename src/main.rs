use std::fs::File;
use std::io::{BufRead, BufReader};

fn main() {

    //This main is just to learn how to open and manage a file
    //TODO: integrate the script and try to create the snn

    let path = "./src/test.txt";

    let input = File::open(path).expect("Something went wrong opening the file test.txt!");
    let buffered = BufReader::new(input);

    for line in buffered.lines() {
        println!("{}",line.unwrap());
    }

}