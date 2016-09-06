extern crate ccv;
extern crate lepton;

use ccv::*;
use ccv::swt::TSwt;
use ccv::transform::TTransform;

use lepton::*;
use lepton::rotate::*;
use lepton::skew::*;

use std::env::args;
use std::f32::consts::PI;

fn main() {
    let mut args = args();
    let _ = args.next().unwrap(); // Ignore executable name.
    let source = args.next().expect("Expected source file name");
    let dest   = args.next().expect("Expected destination file name");

    // FIXME: Capture camera -> image (gstreamer? OpenCV?)

    // Detect text using SWT (implemented by CCV).
    let mut matrix = Matrix::read(source, OpenAs::ToGray).expect("Could not read image");
    let mut words = matrix.detect_words(Default::default());
    let len = words.len();
    println!("Found {} chunks of text", len);

    // Extract each block of text as a chunk.
    let chunks = words.drain(..).map(|rect| {
        matrix.slice(rect)
    });

    // Rewrite chunks to temporary files (FIXME: we can probably keep them in memory)
    for (chunk, i) in chunks.zip(0..) {
        chunk.write(format!("{}-{}.chunk.png", dest, i), FileFormat::PNG)
            .expect("Could not write temporary image containing chunk of text.");
    }

    // Read back with Lepton
    let pixes = (0..len).map(|i| {
        Pix::read(format!("{}-{}.chunk.png", dest, i))
            .expect("Could not read temporary image containing chunk of text.")
    });

    // Find skew and correct it.
    let deskewed = pixes.map(|mut pix| {
        let skew = pix.find_skew().expect("Could not find skew.");
        pix.rotate(skew.degrees * PI / 180., Rotation::AREA_MAP, Background::WHITE, None)
            .expect("Could not rotate image.")
    });

    // Write back to temporary files (FIXME: we can probably keep them in memory)
    for (mut pix, i) in deskewed.zip(0..) {
        pix.write(format!("{}-{}.deskewed.png", dest, i), Format::PNG)
            .expect("Could not write temporary deskewed image.");
    }

    // FIXME: Eliminate details (Edge detection - any lib)
    // FIXME: OCR (Tesseract)
}