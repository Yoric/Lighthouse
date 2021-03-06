extern crate ccv;
extern crate image;
extern crate imageproc;
extern crate itertools;
extern crate rand;
extern crate vec_map;

#[macro_use]
extern crate log;
extern crate env_logger;

use ccv::{ Matrix, OpenAs, FileFormat };
use ccv::swt::*;
//use ccv::edges::*;
//use image::*;
//use imageproc::edges::*;
//use imageproc::gradients::*;

//use image::*;
//use imageproc::map::*;

// mod clean;
mod line;
mod swt;
mod util;

use std::default::Default;
use std::env::args;
//use std::cmp::min;

fn main() {
    env_logger::init().unwrap();

    let mut args = args();
    let _ = args.next().unwrap(); // Ignore executable name.
    let source = args.next().expect("Expected source file name.");
    //let dest   = args.next().expect("Expected destination file name.");

/*
    let params = swt::SwtParams::default();

    // Lighthouse.

    let image = image::open(source.clone())
        .expect("Could not load image.")
        .to_luma();
    let image = canny(&image, 0.1, 0.3);
    image.save("/tmp/output-lighthouse-canny.png").unwrap();


    // Compute gradients.
    let x_grad = vertical_sobel(&image);
    let x_grad = util::colorize(&x_grad);
    x_grad.save("/tmp/output-lighthouse-dx.png").unwrap();
    let y_grad = horizontal_sobel(&image);
    let y_grad = util::colorize(&y_grad);
    y_grad.save("/tmp/output-lighthouse-dy.png").unwrap();
/*
    util::colorize(&x_grad)
        .save("/tmp/output-lighthouse-dx.png")
        .expect("Could not save colorized-x");
    util::colorize(&y_grad)
        .save("/tmp/output-lighthouse-dy.png")
        .expect("Could not save colorized-y");
*/
    // CCV.

    let mut matrix = Matrix::read(source.clone(), OpenAs::ToGray)
        .expect("Could not read image (ccv)");
    matrix = matrix.canny(3, params.canny_low as f64, params.canny_high as f64);
    matrix.write("/tmp/output-buf-canny.png", FileFormat::PNG);
    util::colorize(&image::open("/tmp/output-buf-canny.png")
        .expect("Could not load ccv image")
        .to_luma()
    )
        .save("/tmp/output-ref-canny.png")
        .expect("Coult not save back colorized ccv image");

    matrix.sobel(3, 0)
        .write("/tmp/output-buf-dx.png", FileFormat::PNG);
    util::colorize(&image::open("/tmp/output-buf-dx.png")
        .expect("Could not load ccv image")
        .to_luma()
    )
        .save("/tmp/output-ref-dx.png")
        .expect("Coult not save back colorized ccv image");

    matrix.sobel(0, 3)
        .write("/tmp/output-buf-dy.png", FileFormat::PNG);
    util::colorize(&image::open("/tmp/output-buf-dy.png")
        .expect("Could not load ccv image")
        .to_luma()
    )
        .save("/tmp/output-ref-dy.png")
        .expect("Coult not save back colorized ccv image");

/*
    // Our algorithm.
    let params = swt::SwtParams::default();
    let image = image::open(source.clone())
        .expect("Could not load image.");

/*
    println!("swt: detect words starting");
    swt::detect_words(&image, &swt::SwtParams::default());
    println!("swt: detect words complete");
*/
    util::colorize(&swt::swt(&image.to_luma(), &params, swt::SwtDirection::BrightToDark))
        .save("/tmp/output-lighthouse-swt.png")
        .unwrap();


    let mut matrix = Matrix::read(source.clone(), OpenAs::ToGray).expect("Could not read image (ccv)");
    matrix.swt(SwtParams::default())
        .write("/tmp/output-ccv-swt.png", FileFormat::PNG)
        .expect("Coult not save ccv image");

    util::colorize(&image::open("/tmp/output-ccv-swt.png")
        .expect("Could not load ccv image")
        .to_luma()
    )
        .save("/tmp/output-ref-swt.png")
        .expect("Coult not save back colorized ccv image");
*/
*/
    // Our algorithm.
    let image = image::open(source.clone())
        .expect("Could not load image.");

    swt::detect_words(&image, &swt::SwtParams::default());

    // Reference algorithm.
    let mut matrix = Matrix::read(source.clone(), OpenAs::ToGray).expect("Could not read image (ccv)");
    matrix = matrix.swt(ccv::swt::SwtParams::default());
    matrix.write("/tmp/output-ccv-swt.png", FileFormat::PNG)
        .expect("Coult not save ccv image");

    util::colorize(&image::open("/tmp/output-ccv-swt.png")
        .expect("Could not load ccv image")
        .to_luma()
    )
        .save("/tmp/output-ref-swt.png")
        .expect("Coult not save back colorized ccv image");


/*
    let swt = swt::swt(&image, &Default::default(), swt::SwtDirection::DarkToBright);

    util::colorize(&swt).save(format!("{}-swt-transform-color-1.png", dest))
        .expect("Could not save output file");

    let swt = swt::swt(&image, &Default::default(), swt::SwtDirection::BrightToDark);

    util::colorize(&swt).save(format!("{}-swt-transform-color-2.png", dest))
        .expect("Could not save output file");
*/
/*
    println!("Fast detection of text");
    // Detect text using SWT (implemented by CCV).
    let words =
    {
        // Detect text using SWT (implemented by CCV).
        let mut matrix = Matrix::read(source.clone(), OpenAs::ToGray).expect("Could not read image (ccv)");
        matrix.detect_words(Default::default())
    };

    println!("Loading image.");
    let mut image = image::open(source)
        .expect("Could not load image.");

    let cleanup = clean::CleanupParams {
        min_width: 8,
        min_height: 10,
        max_width: 100,
        max_height: 100,
        canny_low: 0.2,
        canny_high: 0.3,
        split_channels: true,
        prefix: "".to_owned(),
        suffix: "".to_owned(),
    };
    for (word, i) in words.iter().zip(0..) {
        println!("Cleaning up chunk {}.", i);
        let image = image.sub_image(word.x as u32, word.y as u32, word.width as u32, word.height as u32)
            .to_image();
        image.save(format!("{}-before-{}-{}x{}.png", dest, i, word.x, word.y))
            .expect("Could not save output file");

/*
        let mut gray = image.convert();
        let otsu = otsu_level(&gray);
        threshold_mut(&mut gray, otsu);
        image.save(format!("{}-contrasted-{}-{}x{}.png", dest, i, word.x, word.y))
            .expect("Could not save output file");
*/
        let cleaned = clean::clean_text(&mut DynamicImage::ImageRgba8(image), &clean::CleanupParams {
            prefix: dest.clone(),
            suffix: format!("{}-{}x{}.png", i, word.x, word.y),
            ..cleanup.clone()
        });

        cleaned.save(format!("{}-cleaned-{}-{}x{}.png", dest, i, word.x, word.y))
            .expect("Could not save output file");
    }
    */
}

/*

extern crate ccv;
extern crate lepton;

use ccv::*;
use ccv::swt::TSwt;

use lepton::*;
use lepton::clip::*;
use lepton::contrast::*;
use lepton::edge::*;
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

    println!("Detecting text");
    // Detect text using SWT (implemented by CCV).
    let mut words =
    {
        // Detect text using SWT (implemented by CCV).
        let mut matrix = Matrix::read(source.clone(), OpenAs::ToGray).expect("Could not read image (ccv)");
        matrix.detect_words(Default::default())
    };

    println!("Detecting text");
    // Extract chunks.
    let mut pix = Pix::read(source.clone().replace("/", "//")) // FIXME: For some reason, leptonica requires // instead of /.)
        .expect("Could not read image (lepton)");
    pix.write("//tmp//sample-back.png", Format::PNG).unwrap();
    pix.clip(Rect {
        x: 0,
        y: 0,
        w: 10000,
        h: 10000
    }).unwrap().write("//tmp//sample-crop.png", Format::PNG).unwrap();

    let mut contrasted = pix
        .convert(Conversion::EightBits)
        .expect("Could not convert to 8bits")
        .background_norm_flex(3, 3, 3, 3, 0)
        .expect("Could not improve contrast");


    println!("Extracting {} chunks", words.len());
    for (word, i) in words.drain(..).zip(0..) {
        let mut chunk = contrasted.clip(Rect {
            x: word.x,
            y: word.y,
            w: word.width,
            h: word.height
        }).expect("Could not clip");
        let path = format!("{}-{}.chunk.png", dest, i).replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
        println!("Writing chunk to {}", path);
        chunk.write(path, Format::PNG)
            .expect("Could not write temporary image containing chunk of text.");

        let skew = pix.convert(Conversion::OneBit { threshold: 130, factor: None })
            .expect("Could not convert to 1bit")
            .find_skew().expect("Could not find skew.");

        let mut deskewed = chunk
            .rotate(skew.degrees * PI / 180., Rotation::AREA_MAP, Background::BLACK, None)
            .expect("Could not rotate");

            let path = format!("{}-{}.deskewed.png", dest, i).replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
            println!("Writing chunk to {}", path);
            deskewed.write(path, Format::PNG)
                .expect("Could not write temporary image containing chunk of text.");

/*
        let mut edge = chunk.convert(Conversion::EightBits)
            .expect("Could not convert to 8bits")
            .sobel(EdgeFilter::Both)
            .expect("Could not run edge detection");
        let mut deskewed = edge
            .rotate(skew.degrees * PI / 180., Rotation::AREA_MAP, Background::BLACK, None)
            .expect("Could not rotate");

        let path = format!("{}-{}.deskewed.png", dest, i).replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
        println!("Writing chunk to {}", path);
        deskewed.write(path, Format::PNG)
            .expect("Could not write temporary image containing chunk of text.");
*/

    }

/*

    // Rewrite chunks to temporary files (FIXME: we can probably keep them in memory)
    for (chunk, i) in chunks.zip(0..) {
        let path = format!("{}-{}.chunk.png", dest, i);
        println!("Writing chunk to {}", path);
        chunk.write(path, FileFormat::PNG)
            .expect("Could not write temporary image containing chunk of text.");
    }

    // Read back with Lepton
    for i in 0..len {
        let path = format!("{}-{}.chunk.png", dest, i)
            .replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
        println!("Reading {}", path);
        let mut pix = Pix::read(path.clone())
            .expect("Could not read temporary image containing chunk of text.");
        pix.write(format!("//tmp//DEBUG-{}.png", i), Format::PNG)
            .expect("Could not write debug file");

        //let mut bw = pix.clone().convert(Conversion::OneBit { threshold: 130, factor: None })
        //    .expect("Could not convert to 1bit");
        //let skew = bw.find_skew().expect("Could not find skew.");
        //pix.rotate(0., Rotation::AREA_MAP, Background::WHITE, None)

        let dest = format!("{}-{}.deskewed.png", dest, i)
            .replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
        println!("Writing {}", dest);
        pix.write(dest, Format::PNG)
            .expect("Could not write temporary deskewed image.");

/*
        let mut pix = Pix::read(path)
            .expect("Could not read temporary image containing chunk of text.")
            .rotate(skew.degrees * PI / 180., Rotation::AREA_MAP, Background::WHITE, None)
            .expect("Could not rotate image.");
        let dest = format!("{}-{}.deskewed.png", dest, i)
            .replace("/", "//"); // FIXME: For some reason, leptonica requires // instead of /.
        println!("Writing {}", dest);
        pix.write(dest, Format::PNG)
            .expect("Could not write temporary deskewed image.");
*/
    }*/

    // FIXME: Eliminate details (Edge detection - any lib)
    // FIXME: OCR (Tesseract)
}

*/