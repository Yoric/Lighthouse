use image::*;

use rand::*;

use std::collections::{ HashMap, HashSet };
use std::collections::hash_map::Entry::*;
use std::hash::Hash;
use std::default::Default;

pub fn median(numbers: &[u8]) -> Option<u8> {
    match numbers.len() {
        0 => None,
        1 => Some(numbers[0]),
        len => {
            if len % 2 == 1 {
                Some(numbers[len / 2])
            } else {
                let a = numbers[len / 2 - 1] as u32;
                let b = numbers[len / 2] as u32;
                Some(((a + b) / 2) as u8)
            }
        }
    }
}

pub fn colorize<T>(image: &ImageBuffer<Luma<T>, Vec<T>>) -> RgbImage where T: Primitive + Eq + Hash + Default + 'static {
    const BLACK: [u8; 3] = [0, 0, 0];
    let mut random = XorShiftRng::new_unseeded();
    let mut colors : HashMap<T, [u8;3]> = HashMap::new();
    let mut used = HashSet::new();
    let mut buffer = Vec::new();

    // By convention, black => black. This makes backgrounds easier to read.
    let _ = colors.insert(T::default(), BLACK);
    used.insert(BLACK);

    for pixel in image.pixels() {
        match colors.entry(pixel.data[0]) {
            Occupied(color) => {
                let bytes: &[u8; 3] = color.get();
                buffer.extend_from_slice(bytes)
            },
            Vacant(slot) => {
                let mut bytes = BLACK;
                loop {
                    // Find a color that hasn't been used yet.
                    random.fill_bytes(&mut bytes);
                    let sum : i64 = bytes.iter().map(|x| *x as i64).sum();
                    if sum < 50 {
                        // Assume that this will look black, discard color.
                        continue;
                    }
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
    colored
}