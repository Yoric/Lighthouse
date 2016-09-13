use std::collections::{ HashMap, HashSet };
use std::collections::hash_map::Entry::*;
use std::cmp::max;

use image::*;
use image::imageops::colorops::*;
use imageproc::contrast::*;
use imageproc::drawing::*;
use imageproc::edges::canny;
use imageproc::map::*;
use imageproc::regionlabelling::*;
use imageproc::rect::*;

use itertools::*;

use rand::*;

use util;

//use rand::*;

#[derive(Clone)]
pub struct CleanupParams {
    pub min_width: u32,
    pub max_width: u32,
    pub min_height: u32,
    pub max_height: u32,
    pub canny_low: f32,
    pub canny_high: f32,
    pub split_channels: bool,
    pub prefix: String,
    pub suffix: String,
}

#[derive(Debug)]
struct Bounds {
    x0: u32,
    y0: u32,
    x1: u32,
    y1: u32,
    color: [u32;1],
}
impl Bounds {
    fn to_rect(&self) -> Rect {
        Rect::at(self.x0 as i32, self.y0 as i32)
            .of_size(
                max(self.x1 as i32 - self.x0 as i32, 0) as u32 + 1,
                max(self.y1 as i32 - self.y0 as i32, 0) as u32 + 1,
            )
    }
}
/*
pub enum ImageToClean {
    Gray(GrayImage),
    Rgb(RgbImage)
}
*/

pub fn clean_text(image: &/*FIXME: mut, Really?*/DynamicImage, params: &CleanupParams) -> GrayImage {
    const WHITE : [u8; 3] = [255, 255, 255];
    const BLACK : [u8; 3] = [0, 0, 0];

    let mut gray = image.to_luma();
    let edges =
        if params.split_channels {
            let image = image.to_rgb();
            let chan_edges : Vec<_> = [red_channel(&image), green_channel(&image), blue_channel(&image)].iter().map(|chan| {
                println!("Computing channel edge.");
                canny(chan, params.canny_low, params.canny_high)
            }).collect();

            println!("Combining channel edges.");
            let edges : Vec<_> = Zip::new((chan_edges[0].iter(), chan_edges[1].iter(), chan_edges[2].iter())).map(|(r, g, b)| {
                r | g | b
            }).collect();
            let edges : GrayImage = ImageBuffer::from_raw(image.width(), image.height(), edges)
                .expect("Could not create edges image");
            edges
        } else {
            canny(&gray, params.canny_low, params.canny_high)
        };

    let components = connected_components(&edges, Connectivity::Eight);

    println!("Let's compute edge boxes.");
    let mut bounds_per_color = HashMap::new(); // FIXME: Could be a VecMap.
    for (x, y, pixel) in components.enumerate_pixels() {
        use std::collections::hash_map::Entry::*;
        match bounds_per_color.entry(pixel.data[0]) {
            Occupied(mut slot) => {
                // Extend bounds as necessary.
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
                    color: pixel.data,
                });
            }
        }
    }
    println!("Cleanup: We have {} boundaries", bounds_per_color.len());
    let mut all_rectangles = image.to_rgb();
    for bound in bounds_per_color.values() {
        assert!(bound.x1 >= bound.x0);
        if bound.x1 <= bound.x0 {
            continue;
        }
        assert!(bound.y1 >= bound.y0);
        if bound.y1 <= bound.y0 {
            continue;
        }
        let rect = bound.to_rect();
        draw_hollow_rect_mut(&mut all_rectangles, rect, Rgb::from_channels(255, 0, 0, 0));
    }
    all_rectangles.save(format!("{}-rectangles-before-{}", params.prefix, params.suffix))
        .expect("Could not write image.");

    // Remove Bounds that are too small/too large/too elongated.
    let mut remove = vec![];
    {
        for (key, bound) in &bounds_per_color {
            println!("Cleanup: Examining {:?}", bound);
            let width = bound.x1 - bound.x0;
            let height = bound.y1 - bound.y0;
            if width < params.min_width || width > params.max_width {
                println!("Cleanup: width {} too small/too large", width);
                remove.push(key.clone());
                continue
            }
            if height < params.min_height || height > params.max_height {
                println!("Cleanup: height {} too small/too large", height);
                remove.push(key.clone());
                continue
            }
            let ratio = (width as f32) / (height as f32);
            if ratio < 0.03 || ratio > 30. {
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
    let sorted = {
        let mut to_sort : Vec<_> = bounds_per_color.values().collect();
        to_sort.sort_by(|a, b| {
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
        to_sort
    };
    let mut children_by_color : HashMap<[u32;1]/*color*/, Vec<usize> /*children*/> = HashMap::new();
    for (left_rect, i) in sorted.iter().zip(0..) {
        println!("Examining bound {}: {:?}", i, left_rect);
        for j in i+1 .. sorted.len() {
            let right_rect = sorted[j];
            // Invariant: left_rect.x0 <= right_rect.x0
            if right_rect.x0 >= left_rect.x1 {
                // The rectangles are disjoint, and all further rectangles are disjoint.
                break;
            }
            if right_rect.x1 > left_rect.x1 {
                // Possible intersection but no inclusion.
                continue;
            }
            if left_rect.y0 <= right_rect.y0 && right_rect.y1 <= left_rect.y1 {
                // left_rect has one more child
                match children_by_color.entry(left_rect.color) {
                    Occupied(mut slot) => {
                        slot.get_mut().push(j);
                        println!("Bounds: rect {} {:?} now has {} children", i, left_rect, slot.get().len())
                    },
                    Vacant(slot) => {
                        slot.insert(vec![j]);
                        println!("Bounds: rect {} {:?} now has 1 child", i, left_rect);
                    }
                }
            }
        }
    }

    // Now cleanup rectangles that have too many children/that are useless children themselves.
    let mut remove = HashSet::new();
    {
        for (color, children) in &children_by_color {
            if children.len() > 3 {
                remove.insert(color.clone());
            } else {
                for child in children {
                    remove.insert(sorted[*child].color);
                }
            }
        }
    }
    for color in remove.drain() {
        children_by_color.remove(&color);
    }

    println!("Let's output an intermediate version to see what the components look like.");
    let mut random = XorShiftRng::new_unseeded();
    let mut colors = HashMap::new();
    let mut used = HashSet::new();
    let mut buffer = Vec::new();
    let _ = colors.insert(0, BLACK);
    used.insert(BLACK);
    for pixel in components.pixels() {
        match colors.entry(pixel.data[0]) {
            Occupied(color) => {
                let bytes: &[u8; 3] = color.get();
                buffer.extend_from_slice(bytes)
            },
            Vacant(slot) => {
                let mut bytes = BLACK;
                loop {
                    random.fill_bytes(&mut bytes);
                    if used.insert(bytes) {
                        break;
                    }
                }
                slot.insert(bytes);
                buffer.extend_from_slice(&bytes)
            }
        }
    }
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), buffer)
        .expect("Could not create colored image.");

    colored.save(format!("{}-colored-{}", params.prefix, params.suffix))
            .expect("Could not write image.");


    let mut buffer = Vec::new();
    for pixel in components.pixels() {
        if let None = bounds_per_color.get(&pixel.data[0]) {
            buffer.extend_from_slice(&BLACK);
            continue;
        } else {
            buffer.extend_from_slice(&WHITE);
        }
//        let color = colors.get(&pixel.data[0])
//            .expect("Cannot find color");
//        buffer.extend_from_slice(color);
    }
    let colored : RgbImage = ImageBuffer::from_vec(image.width(), image.height(), buffer)
        .expect("Could not create image with component cleanup.");


    println!("Now, let's use the information to build a filter and threshold the image");
    let mut rebuilt : GrayImage = ImageBuffer::from_pixel(image.width(), image.height(), Luma { data: [255] });
    let mut thresholds = vec![];
    for (_, bound) in &bounds_per_color {
        // Determine background threshold.
        let mut background = Vec::with_capacity(12);
        {
            let mut collect = |x: i32, y: i32| {
                if x < 0 || y < 0 {
                    return;
                }
                let x = x as u32;
                let y = y as u32;
                if x >= image.width() || y >= image.height() {
                    return;
                }
                // We're in the image. Otherwise, don't bother with the pixel.
                if colored.get_pixel(x, y).data == WHITE {
                    // We're in the background. Otherwise, don't bother with the pixel.
                    background.push(image.get_pixel(x, y).data[0]);
                }
            };
            let x0 = bound.x0 as i32;
            let y0 = bound.y0 as i32;
            let x1 = bound.x1 as i32;
            let y1 = bound.y1 as i32;
            for (x, y) in [
                (x0 - 1, y0 - 1),
                (x0 - 1, y0),
                (x0,     y0 - 1),

                (x1 + 1, y0 - 1),
                (x1 + 1, y0),
                (x1,     y0 - 1),

                (x0 - 1, y1 + 1),
                (x0 - 1, y1),
                (x0,     y1 + 1),

                (x1 + 1, y1 + 1),
                (x1 + 1, y1),
                (x1,     y1 + 1),
            ].iter().cloned() {
                collect(x, y);
            }
        }
        background.sort();

        // Compute median
        let bg = match util::median(&background) {
            None => 0,
            Some(bg) => bg
        };


        // Determine foreground threshold
        let mut foreground_total : u64 = 0;
        let mut foreground_pixels = 0;
        for x in bound.x0 .. bound.x1 + 1 {
            for y in bound.y0 .. bound.y1 + 1 {
                if colored.get_pixel(x, y).data == WHITE {
                    // Part of the background, we don't care.
                } else {
                    foreground_total += image.get_pixel(x, y).data[0] as u64;
                    foreground_pixels += 1;
                }
            }
        }
        let fg =
            if foreground_pixels == 0 {
                0
            } else {
                (foreground_total / foreground_pixels) as u8
            };
        thresholds.push((bg, fg));

        let sub = gray.sub_image(bound.x0, bound.y0, bound.x1 - bound.x0 + 1, bound.y1 - bound.y0 + 1);
        let mut sub = threshold(&sub, fg);


        if bg < fg {
//            draw_filled_rect(&rebuilt, bound.to_rect(), Luma { data: [0] });
            invert(&mut sub)
        }
        rebuilt.copy_from(&sub, bound.x0, bound.y0);
    }

    let mut rectangles = rebuilt.convert();
    for bound in bounds_per_color.values() {
        let rect = Rect::at(bound.x0 as i32, bound.y0 as i32).of_size(bound.x1 - bound.x0, bound.y1 - bound.y0);
        draw_hollow_rect_mut(&mut rectangles, rect, Rgb::from_channels(255, 0, 0, 0));
    }
    rectangles.save(format!("{}-rectangles-after-{}", params.prefix, params.suffix))
        .expect("Could not write image.");


    rebuilt
}
