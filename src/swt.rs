use image::*;
use imageproc::definitions::*;
use imageproc::edges::canny;
use imageproc::gradients::*;

use std::cmp::max;

use util::*;

#[derive(Clone, Copy)]
pub enum SwtDirection {
    DarkToBright = 1,
    BrightToDark = -1,
}
impl SwtDirection {
    pub fn reverse(&self) -> SwtDirection {
        use self::SwtDirection::*;
        match *self {
            DarkToBright => BrightToDark,
            BrightToDark => DarkToBright
        }
    }
}
pub struct SwtParams {
    /// Intervals for scale invariant option.
    pub interval: u32,

    /// Minimal neighbors to make a detection valid, this is for scale-invariant version.
    pub min_neighbors: u32,

    /// Enable scale invariant swt (to scale to different sizes and then combine the results)
    pub scale_invariant: bool,

    pub direction: SwtDirection,

    /// Overlapping more than 0.1 of the bigger one (0), and 0.9 of the smaller one (1)
    pub same_word_thresh: [f32; 2],

    pub canny_low: f32,
    pub canny_high: f32,

    pub letter_min_height: u32,
    pub letter_max_height: u32,
    pub letter_min_area: u32,
    pub letter_occlude_thresh: u32,
    pub max_diameter: u32,

    /// The maximum aspect ratio for a letter.
    pub letter_max_aspect_ratio: f32,

    /// The inner-class standard derivation when grouping letters.
    pub std_ratio: f32,

    /// The allowable thickness variance when grouping letters.
    pub thickness_ratio: f32,

    /// The allowable height variance when grouping letters.
    pub height_ratio: f32,
    pub intensity_thresh: u32,
    pub distance_ratio: f32,
    pub intersect_ratio: f32,
    pub elongate_ratio: f32,
    pub letter_thresh: u32,

    pub breakdown: bool,
    pub breakdown_ratio: f32,
}

/// Complete an outline (as devised by calling sobel), by adding the missing pixels at the
/// intersection of two edges.
pub fn close_outline(image: &DynamicImage, threshold: u8) -> GrayImage {
    let gray = image.to_luma();
    let mut result = GrayImage::from_pixel(gray.width(), gray.height(), Luma::black());

    if gray.width() == 0 || gray.height() == 0 {
        return result;
    }

    for i in 0 .. gray.width() {
        for j in 0 .. gray.height() {
            let is_white = |i, j| {
                gray.get_pixel(i, j).data[0] >= threshold
            };

            if result.get_pixel(i, j).data[0] == 0 && is_white(i, j) {
                *result.get_pixel_mut(i, j) = Luma::white()
            }

            if i + 1 >= gray.width() || j + 1 >= gray.height() {
                continue;
            }
            if is_white(i, j) && is_white(i + 1, j + 1) {
                *result.get_pixel_mut(i + 1, j) = Luma::white();
                *result.get_pixel_mut(i, j + 1) = Luma::white();
            }
            if is_white(i + 1, j) && is_white(i, j + 1) {
                *result.get_pixel_mut(i, j) = Luma::white();
                *result.get_pixel_mut(i + 1, j + 1) = Luma::white();
            }
        }
    }
    gray
}

#[derive(Clone)]
struct Ray {
    x: i32,
    slope_x: i32,
    adx: i32,
    y: i32,
    slope_y: i32,
    ady: i32,
    err: i32,
}

struct RayDirection {
    xx: i32,
    xy: i32,
    yx: i32,
    yy: i32
}

struct Stroke {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    width: u8,
}
impl Stroke {
    fn iter(&self) -> StrokeIterator {
        StrokeIterator {
            ray: Some(Ray::from_stroke(&self)),
            x1: self.x1,
            y1: self.y1
        }
    }
}


struct StrokeIterator {
    ray: Option<Ray>,
    x1: i32,
    y1: i32
}
impl Iterator for StrokeIterator {
    type Item = (u32, u32);
    fn next(&mut self) -> Option<Self::Item> {
        let done;
        let result;
        match self.ray {
            None => {
                done = true;
                result = None
            }
            Some(ref mut ray) => {
                result = Some((ray.x as u32, ray.y as u32));
                if ray.x == self.x1 && ray.y == self.y1 {
                    done = true;
                } else {
                    ray.increment();
                    done = false;
                }
            }
        }
        if done {
            self.ray = None
        }
        result
    }
}

impl Ray {
    fn from_stroke(stroke: &Stroke) -> Ray {
        let adx = i32::abs(stroke.x1 - stroke.x0);
        let ady = i32::abs(stroke.y1 - stroke.y0);
        let slope_x = if stroke.x1 > stroke.x0 { 1 } else { -1 };
        let slope_y = if stroke.y1 > stroke.y0 { 1 } else { -1 };

        Ray {
            x: stroke.x0,
            y: stroke.y0,
            slope_x: slope_x,
            slope_y: slope_y,
            adx: adx,
            ady: ady,
            err: adx - ady
        }
    }
}

#[derive(Clone)]
struct Line(Ray);

struct LineIterator(Ray);

struct Point {
    x: u32,
    y: u32,
}

impl Iterator for LineIterator {
    type Item = Point;
    fn next(&mut self) -> Option<Self::Item> {
        let ref mut ray = self.0;
        let result = Some(Point {
            x: ray.x as u32,
            y: ray.y as u32
        });
        ray.increment();
        result
    }
}

impl Line {
    fn from_grad(x: u32, y: u32, x_grad: &ImageBuffer<Luma<i16>, Vec<i16>>, y_grad: &ImageBuffer<Luma<i16>, Vec<i16>>, direction: &RayDirection, params: &SwtParams) -> Line {
        let x_grad_pixel = x_grad.get_pixel(x, y).data[0] as i32;
        let y_grad_pixel = y_grad.get_pixel(x, y).data[0] as i32;
        let rx_grad = x_grad_pixel * direction.xx + y_grad_pixel * direction.xy;
        let ry_grad = x_grad_pixel * direction.yx + y_grad_pixel * direction.yy;

        let adx = i32::abs(rx_grad);
        let ady = i32::abs(ry_grad);
        return Line(Ray {
            x: x as i32,
            slope_x: if rx_grad <= 0 { params.direction } else { params.direction.reverse() } as i32,
            adx: adx,

            y: y as i32,
            slope_y: if ry_grad <= 0 { params.direction } else { params.direction.reverse() } as i32,
            ady: ady,

            err: adx - ady
        });
    }
    fn iter(&self) -> LineIterator {
        LineIterator(self.0.clone())
    }
}

impl Ray {
    fn increment(&mut self) {
        let e2 = 2 * self.err;
        if e2 > -self.ady {
            self.err -= self.ady;
            self.y = self.y + self.slope_y
        }
        if e2 < self.adx {
            self.err += self.adx;
            self.x = self.x + self.slope_x;
        }
    }
}


#[allow(dead_code)]
pub fn swt(image: &DynamicImage, params: &SwtParams) -> GrayImage {
    const BW_THRESHOLD : u8 = 1; /* Any number other than 0 (black) and 255 (white) works.*/
    let gray = image.to_luma();

    // Each pixel contains either 0 (if it is not part of an outline) or the length of the smallest
    // segment found so far that connects two edges of an outline. Once we have finished computing
    // all (reasonable) segments, this will contain an approximation of the width of each outline,
    // i.e. the stroke width.
    let mut stroke_widths = GrayImage::new(image.width(), image.height());
    let mut strokes = vec![];

    // Compute all outlines on the image...
    let edges = canny(&gray, params.canny_low, params.canny_high);
    // ... and improve the chances that they are closed.
    let outlines = close_outline(&DynamicImage::ImageLuma8(edges), BW_THRESHOLD);

    // Compute gradients.
    let x_grad = horizontal_sobel(&gray);
    let y_grad = vertical_sobel(&gray);

    for (x, y, pixel) in outlines.enumerate_pixels() {
        if pixel.data[0] < BW_THRESHOLD {
            // This pixel is not part of the outline, no need to throw rays.
            continue;
        }

        // This pixel is part of the outline, so we suspect that it's the border of a shape,
        // possibly a letter.
        //
        // Cast a few rays to find an opposite border. Each call to `ray_emit` corresponds to
        // casting a ray in a different direction. Note that we only cast rays towards the
        // right, as we are scanning the image from the left.
        let mut ray_emit = |direction| {
            let line = Line::from_grad(x, y, &x_grad, &y_grad, &direction, params);

            // `Some((kx, ky)) once we have found an opposite border.
            let mut opposite_border = None;

            // For performance reasons, limit how far we are willing to search for an
            // opposite border.
            for (ray, _) in line.iter().zip(0 .. params.max_diameter) {
                if ray.x < 1 || ray.x >= gray.width() - 1 {
                    // Leaving the image, no border found.
                    break;
                }
                if ray.y < 1 || ray.y >= gray.height() - 1 {
                    // Leaving the image, no border found.
                    break;
                }
                // Note that we are not certain that we will encounter an edge. Despite calling
                // `close_outline`, we may have lost/missed pixels that should be part of the
                // opposite border.
                if i32::abs(y as i32 - ray.y as i32) >= 2
                || i32::abs(x as i32 - ray.x as i32) >= 2 {
                    'search_neighbour: for &(dx, dy) in &[
                        (-1, 0),
                        (0,  0),
                        (1,  0),
                        (0, -1),
                        (0,  1)
                    ] {
                        let kx = ray.x as i32 + dx;
                        let ky = ray.y as i32 + dy;
                        if outlines.get_pixel(kx as u32, ky as u32 - y).data[0] > BW_THRESHOLD {
                            // FIXME: Why is ky >= y?
                            opposite_border = Some((ray, kx, ky));
                            break 'search_neighbour;
                        }
                    }
                }
            }

            // Make sure that the opposite angle is in d_p ± ~pi/6. Otherwise, we assume that
            // the shape is too exotic and this is not a letter.
            let check_angle = |kx, ky| {
                for &(dx, dy) in &[
                    (-1,  0),
                    (0,   0),
                    (1,   0),
                    (-1, -1),
                    (0,  -1),
                    (1,  -1),
                    (-1,  1),
                    (0,   1),
                    (1,   1)
                ] {
                    let x1 = (kx as i32 + dx) as u32;
                    let y1 = (ky as i32 - y as i32 + dy) as u32;
                    let tn =
                        (y_grad.get_pixel(x, y).data[0] * x_grad.get_pixel(x1,  y1).data[0]) as i32 // FIXME: Or am I confusing x and y?
                       - (x_grad.get_pixel(x, y).data[0] * y_grad.get_pixel(x1, y1).data[0]) as i32;
                    let td =
                        (x_grad.get_pixel(x, y).data[0] * x_grad.get_pixel(x1,  y1).data[0]) as i32 // FIXME: Or am I confusing x and y?
                       - (y_grad.get_pixel(x, y).data[0] * y_grad.get_pixel(x1, y1).data[0]) as i32;

                    // Compute a reasonable apprxomation of `|| tn/td || < pi/6`.
                    if tn * 7 < -td * 4 && tn * 7 > td * 4 {
                        return true
                    }
                }
                false
            };
            if let Some((ray, kx, ky)) = opposite_border {
                if check_angle(kx, ky) {
                    // We have a hit. Now, fill the line with the Stroke Width.
                    let square = |z: i32| z as f32 * z as f32;
                    let width = f32::sqrt (square(ray.x as i32 - x as i32) + square(ray.y as i32 - y as i32) + 0.5 /*extend the line to be of width 1*/) as u8;
                    let stroke = Stroke {
                        x0: x as i32,
                        y0: y as i32,
                        x1: ray.x as i32,
                        y1: ray.y as i32,
                        width: width
                    };
                    for (x1, y1) in stroke.iter() {
                        let pixel = stroke_widths.get_pixel_mut(x1 - x, y1 - y);
                        if pixel.data[0] == 0 || pixel.data[0] > width {
                            // We have found a shorter width. Update.
                            pixel.data[0] = width;
                        }
                    }

                    // Finally, record the stroke.
                    strokes.push(stroke);
                }
            }
        };
        ray_emit(RayDirection {
            xx: 1,
            xy: 0,
            yx: 0,
            yy: 1
        });
        ray_emit(RayDirection {
            xx: 1,
            xy: -1,
            yx: 1,
            yy: 1
        });
        ray_emit(RayDirection {
            xx: 1,
            xy: 1,
            yx: -1,
            yy: 1
        });
    }

    // The Stroke Width computed so far works nicely for simple forms, but forks, intersections,
    // angles, ... make it meaningless. For instance, consider letter the corner of letter "L".
    // Each pixel in the border is matched by an opposing border, which can be very far. However,
    // this is a fluke of being in the edge. The true width of the stroke is actually the same
    // for the entire stroke.
    //
    //  second pass replaces the computed width with the median width across the entire stroke.
    // We proceed from shortest to longest. // FIXME: Does this guarantee anything?
    strokes.sort_by(|a, b| a.width.cmp(&b.width));
    let mut buf = Vec::with_capacity(max(image.width(), image.height()) as usize);
    for stroke in strokes.drain(..) {
        for (x, y) in stroke.iter() {
            buf.push(stroke_widths.get_pixel(x, y).data[0]);
        }
        if let Some(nw) = median(&buf) {
            if nw != stroke.width {
                // Repaint stroke.
                assert!(nw < stroke.width, "I believe that we can only decrease here, right?");
                for (x, y) in stroke.iter() {
                    stroke_widths.get_pixel_mut(x, y).data[0] = nw;
                }
            }
        }
    }
    stroke_widths
}