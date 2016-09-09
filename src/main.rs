extern crate image;
extern crate imageproc;
extern crate itertools;
extern crate rand;

#[macro_use]
extern crate log;
extern crate env_logger;

mod clean;
mod util;

use std::env::args;

use image::*;
use image::imageops::colorops::*;
use imageproc::contrast::*;
use imageproc::drawing::*;
use imageproc::map::*;
use imageproc::edges::canny;
use imageproc::regionlabelling::*;
use imageproc::rect::*;

fn main() {
    env_logger::init().unwrap();

    let mut args = args();
    let _ = args.next().unwrap(); // Ignore executable name.
    let source = args.next().expect("Expected source file name.");
    let dest   = args.next().expect("Expected destination file name.");

    println!("Loading image.");
    let mut image = image::open(source)
        .expect("Could not load image.")
        .to_luma();

    let cleaned = clean::clean_text(&mut image, &clean::CleanupParams {
        min_width: 10,
        min_height: 15,
        max_width: 100,
        max_height: 100,
        canny_low: 0.2,
        canny_high: 0.3,
        split_channels: false,
    });

    cleaned.save(dest)
        .expect("Could not save output file");
/*




    println!("Let's output a new intermediate version to see what the components look like.");

    colored.save(format!("{}-components-cleanup.png", dest))
        .expect("Could not write image.");

    let mut rectangles = image.clone();
    for bound in bounds_per_color.values() {
        let rect = Rect::at(bound.x0 as i32, bound.y0 as i32).of_size(bound.x1 - bound.x0, bound.y1 - bound.y0);
        draw_hollow_rect_mut(&mut rectangles, rect, Rgb::from_channels(255, 255, 0, 0));
    }
    rectangles.save(format!("{}-rectangles.png", dest))
        .expect("Could not write image.");


    rebuilt.save(format!("{}-final.png", dest))
        .expect("Could not write image.");

/*
    let mut stack : Vec<Luma<u32>> = vec![]; // We're assuming that connected components worked, so no intersections.
    const ZERO : Luma<u32> = Luma { data: [0] };
    let mut latest = ZERO;
    let mut fill = ZERO;
    let mut pixels = Vec::new();
//    let mut depth = HashMap::new();
    for (x, _, pixel) in components.enumerate_pixels() {
        if x == 0 {
            // Beginning of line, reset.
            stack.clear();
            latest = ZERO;
            fill = ZERO;
        }
        if pixel.data[0] == 0 {
            // Not a component boundary, nothing to do.
            // Keep filling.
            latest = ZERO;
        } else if pixel.data == latest.data {
            // No change in color, nothing to do.
        } else {
            // This is a boundary change.
            latest.data = pixel.data;
            if stack.is_empty() || stack[stack.len() - 1].data != pixel.data {
                // We're entering.
                stack.push(*pixel);
                fill = latest;
            } else {
                // We're leaving.
                stack.pop();
                if stack.is_empty() {
                    fill = ZERO;
                } else {
                    fill = stack[stack.len() - 1]
                }
            }
        }
        pixels.extend_from_slice(colors.get(&fill.data[0]).unwrap())
    }
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), pixels)
        .expect("Could not create filled image.");

    colored.save(format!("{}-filled.png", dest))
        .expect("Could not write image.");
*/
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