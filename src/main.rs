extern crate image;
extern crate imageproc;
extern crate itertools;
extern crate rand;

#[macro_use]
extern crate log;
extern crate env_logger;

use std::collections::HashMap;
use std::collections::hash_map::Entry::*;
use std::env::args;

use image::*;
use imageproc::map::*;
use imageproc::edges::canny;
use imageproc::regionlabelling::*;

use itertools::*;

use rand::*;

fn main() {
    env_logger::init().unwrap();

    let mut args = args();
    let _ = args.next().unwrap(); // Ignore executable name.
    let source = args.next().expect("Expected source file name.");
    let dest   = args.next().expect("Expected destination file name.");

    println!("Loading image.");
    let image = image::open(source)
        .expect("Could not load image.")
        .to_rgb();

    let low = 0.2;
    let high = 0.3;
    let chan_edges : Vec<_> = [red_channel(&image), green_channel(&image), blue_channel(&image)].iter().map(|chan| {
        println!("Computing channel edge.");
        canny(chan, low, high)
    }).collect();

    println!("Combining channel edges.");
    let edges : Vec<_> = Zip::new((chan_edges[0].iter(), chan_edges[1].iter(), chan_edges[2].iter())).map(|(r, g, b)| {
        r | g | b
    }).collect();

    println!("Converting to image.");
    let edges : GrayImage = ImageBuffer::from_raw(image.width(), image.height(), edges)
        .expect("Could not create edges image");

    println!("Computing components.");
    let components = connected_components(&edges, Connectivity::Eight);

/*
    let colored : Vec<_> = components.pixels().map(|pix| {
        let full = pix.data[0];
        let foo : [u8; 4] = unsafe { std::mem::transmute(full) };
        Rgb::from_channels(foo[3], foo[2], foo[1], foo[0])
    }).collect();
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), colored)
        .expect("Could not create colored image");
*/

    println!("Let's output an intermediate version to see what the components look like.");
    let mut random = XorShiftRng::new_unseeded();
    let mut colors = HashMap::new();
    let mut buffer = Vec::new();
    let _ = colors.insert(0, [0, 0, 0]);
    for pixel in components.pixels() {
        match colors.entry(pixel.data[0]) {
            Occupied(color) => {
                let bytes: &[u8; 3] = color.get();
                buffer.extend_from_slice(bytes)
            },
            Vacant(slot) => {
                let mut bytes = [0, 0, 0];
                random.fill_bytes(&mut bytes);
                slot.insert(bytes);
                buffer.extend_from_slice(&bytes)
            }
        }
    }
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), buffer)
        .expect("Could not create colored image.");

    colored.save(format!("{}-components.png", dest))
        .expect("Could not write image.");


    println!("Let's compute edge boxes.");
    #[derive(Debug)]
    struct Bounds {
        x0: u32,
        y0: u32,
        x1: u32,
        y1: u32,
        color: u32,
    }
    let mut bounds_per_color = HashMap::new(); // FIXME: Could be a VecMap.
    for (x, y, pixel) in components.enumerate_pixels() {
        use std::collections::hash_map::Entry::*;
        match bounds_per_color.entry(pixel.data[0]) {
            Occupied(mut slot) => {
                // Extend bounds.
                let bounds : &mut Bounds = slot.get_mut();
                if x < bounds.x0 {
                    bounds.x0 = x;
                } else if x > bounds.x1 {
                    bounds.x1 = x;
                }
                if y < bounds.y0 {
                    bounds.y0 = y;
                } else if y > bounds.y1 {
                    bounds.y1 = y;
                }
            },
            Vacant(slot) => {
                slot.insert(Bounds {
                    x0: x,
                    y0: y,
                    x1: x,
                    y1: y,
                    color: pixel.data[0],
                });
            }
        }
    }
    println!("Cleanup: We have {} boundaries", bounds_per_color.len());

    // Remove Bounds that are too small/too large/too elongated.
    const MIN_WIDTH : u32 = 15;
    const MIN_HEIGHT : u32 = 15;
    let mut remove = vec![];
    {
        for (key, bound) in &bounds_per_color {
            println!("Cleanup: Examining {:?}", bound);
            let width = bound.x1 - bound.x0;
            let height = bound.y1 - bound.y0;
            if width < MIN_WIDTH || width > image.width() / 5 {
                println!("Cleanup: width too small/too large");
                remove.push(key.clone());
                continue
            }
            if height < MIN_HEIGHT || height > image.height() / 5 {
                println!("Cleanup: height too small/too large");
                remove.push(key.clone());
                continue
            }
            let ratio = (width as f32) / (height as f32);
            if ratio < 0.1 || ratio > 10. {
                println!("Cleanup: boundary too thin");
                remove.push(key.clone());
                continue
            }
        }
    }
    println!("Cleanup: removing {} boundaries", remove.len());
    for key in remove.drain(..) {
        println!("Cleanup: removing color {}", key);
        bounds_per_color.remove(&key).unwrap();
    }

    // Determine if a Bound is strictly inside another one.
    let mut sort : Vec<_> = bounds_per_color.values().collect();
    sort.sort_by(|a, b| {
        use std::cmp::Ordering::*;
        if a.x0 != b.x0 {
            return a.x0.cmp(&b.x0)
        }
        if a.y0 != b.y0 {
            return a.y0.cmp(&b.y0)
        }
        if a.x1 != b.x1 {
            return b.x1.cmp(&a.x1)
        }
        if a.y1 != b.y1 {
            return b.y1.cmp(&a.y1)
        }
        return Equal
    });
    let mut children : HashMap<u32/*color*/, u32> = HashMap::new();
    for (bound0, i) in sort.iter().zip(0..) {
        println!("Examining bound {}: {:?}", i, bound0);
        if i + 1 >= sort.len() {
            println!("Bound: last one, it can't be a parent");
            continue;
        }
        for bound1 in sort[i+1..].iter() {
            println!("Bound: Comparing with {:?}", bound1);
            // Invariant: bound0.x0 <= bound1.x0
            if bound1.x0 >= bound0.x1 {
                // The rectangles are disjoint, and all further rectangles are disjoint.
                println!("Bound: Rectangles are disjoint");
                break;
            }
            if bound1.x1 > bound0.x1 {
                // Possible intersection but no inclusion.
                println!("Bound: Rectangles is too far on the right");
                continue;
            }
            if bound0.y0 <= bound1.y0 && bound1.y1 <= bound0.y1 {
                // bound0 has one more child
                println!("Bound: Found child");
                match children.entry(bound0.color) {
                    Occupied(mut slot) => {
                        *slot.get_mut() = slot.get() + 1
                    },
                    Vacant(slot) => {
                        slot.insert(1);
                    }
                }
            } else {
                println!("Bound: Not a child");
            }
        }
    }

    println!("Let's output a new intermediate version to see what the components look like.");
    let mut buffer = Vec::new();
    for pixel in components.pixels() {
        if let Some(total) = children.get(&pixel.data[0]) {
            println!("{}: {}", pixel.data[0], total);
            if *total >= 3 {
                buffer.extend_from_slice(&[0, 0, 0]); // Get rid of this component.
                continue;
            }
        } else if let None = bounds_per_color.get(&pixel.data[0]) {
            buffer.extend_from_slice(&[0, 0, 0]);
            continue;
        }
        let color = colors.get(&pixel.data[0])
            .expect("Cannot find color");
        buffer.extend_from_slice(color);
    }
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), buffer)
        .expect("Could not create image with component cleanup.");

    colored.save(format!("{}-components-cleanup.png", dest))
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