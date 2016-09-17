use image::*;
use imageproc::rect::Rect;

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

pub struct Queue<T> where T: Copy {
    data: Vec<T>,
    consumed: usize,
}

impl<T> Queue<T> where T: Copy {
    pub fn new() -> Self {
        Queue {
            data: vec![],
            consumed: 0,
        }
    }
    pub fn push(&mut self, value: T) {
        self.data.push(value)
    }
    pub fn pop(&mut self) -> Option<T> {
        if self.consumed < self.data.len() {
            let result = self.data[self.consumed];
            self.consumed += 1;
            Some(result)
        } else {
            None
        }
    }
    pub fn clear(&mut self) {
        self.data.clear();
        self.consumed = 0;
    }
    pub fn is_empty(&self) -> bool {
        self.data.len() == self.consumed
    }
    pub fn total_pushed(&self) -> usize {
        self.data.len()
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Point {
    pub x: u32,
    pub y: u32,
}

pub struct Contour {
    pub points: Vec<Point>,
    pub top_left: Point,
    pub bottom_right: Point,
    id: u32,
    sum_x: u32,
    sum_y: u32,
    sum_x_y: u64,
    sum_x_x: u64,
    sum_y_y: u64,
}
impl Contour {
    pub fn new(point: Point, id: u32) -> Self {
        Contour {
           top_left: point,
           bottom_right: point,
           points: vec![point],
           id: id,
           sum_x: 0,
           sum_y: 0,
           sum_x_x: 0,
           sum_x_y: 0,
           sum_y_y: 0,
       }
    }
    pub fn push(&mut self, point: Point) {
        if point.x < self.top_left.x {
            self.top_left.x = point.x;
        } else if point.x > self.bottom_right.x {
            self.bottom_right.x = point.x;
        }
        if point.y < self.top_left.y {
            self.top_left.y = point.y;
        } else if point.y > self.bottom_right.y {
            self.bottom_right.y = point.y;
        }
        debug_assert!(self.top_left.x <= self.bottom_right.x);
        debug_assert!(self.top_left.y <= self.bottom_right.y);
        self.sum_x += point.x;
        self.sum_y += point.y;
        self.sum_x_x += point.x as u64 * point.x as u64;
        self.sum_x_y += point.x as u64 * point.y as u64;
        self.sum_y_y += point.y as u64 * point.y as u64;
        self.points.push(point);
    }
    pub fn height(&self) -> u32 {
        // Note: Operation is on u32, so this will assert that the result > 0
        self.bottom_right.x - self.top_left.x + 1
    }
    pub fn width(&self) -> u32 {
        // Note: Operation is on u32, so this will assert that the result > 0
        self.bottom_right.y - self.top_left.y + 1
    }
    pub fn ratio(&self) -> f32 {
        self.height() as f32 / self.width() as f32
    }
    pub fn sq_ratio(&self) -> f32 {
        let xc = self.sum_x as f32 / self.size() as f32;
        let yc = self.sum_y as f32 / self.size() as f32;
        let af = self.sum_x_x as f32 / self.size() as f32 - xc * xc;
        let cf = self.sum_y_y as f32 / self.size() as f32 - yc * yc;
        let bf = self.sum_x_y as f32 / self.size() as f32 - xc * yc;
        let delta = f32::sqrt (bf * bf + (af - cf) * (af - cf));
        let ratio = f32::sqrt ((af + cf + delta) / (af + cf - delta));
        ratio
    }
    pub fn size(&self) -> u32 {
        self.points.len() as u32
    }
    pub fn center(&self) -> Point {
        let x = (self.top_left.x + self.bottom_right.x) / 2;
        let y = (self.top_left.y + self.bottom_right.y) / 2;
        Point {
            x: x,
            y: y
        }
    }
    pub fn bound(&self) -> Rect {
        Rect::at(self.top_left.x as i32, self.top_left.y as i32).of_size(self.width(), self.height())
    }
    pub fn id(&self) -> u32 {
        self.id
    }
}