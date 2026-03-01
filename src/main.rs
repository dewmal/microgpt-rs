use native_tls::TlsConnector;
use std::{
    io::{BufRead, BufReader, Read, Write},
    net::TcpStream,
};

fn load_data() {
    let url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt";
    let file_name = "data.txt";
    let without_scheme = url.replace("https://", "");

    let (host, path) = match without_scheme.find("/") {
        Some(i) => (&without_scheme[..i], &without_scheme[i..]),
        None => (without_scheme.as_str(), "/"),
    };
    let tcp = TcpStream::connect(format!("{host}:443")).expect("Could not Connect to the server");

    let connector = TlsConnector::new().expect("Failed to create TLS connector");
    let mut stream = connector.connect(host, tcp).expect("TLS handshake failed");

    let request = format!("GET {path} HTTP/1.1\r\nHost: {host}\r\nConnection: close\r\n\r\n");

    stream
        .write_all(request.as_bytes())
        .expect("Failed to send");

    let mut reader = BufReader::new(stream);

    let mut line = String::new();

    loop {
        line.clear();
        reader.read_line(&mut line).expect("Failed to read");
        if line == "\r\n" || line.is_empty() {
            break;
        }
    }
    let mut body = String::new();
    reader
        .read_to_string(&mut body)
        .expect("Failed to read body");

    save_to_file(file_name, &mut body);
}

fn save_to_file(file_name: &str, contents: &str) {
    std::fs::write(file_name, contents).expect("Could not save the file");
    println!("File saved as: {file_name}")
}

fn main() {
    load_data();
}
