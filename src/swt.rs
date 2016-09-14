use image::*;
use image::imageops::resize;

use imageproc::definitions::*;
use imageproc::edges::canny;
use imageproc::gradients::*;
use imageproc::rect::Rect;

use std::cmp::{ max, min };

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

    /// Overlapping more than 0.1 of the bigger one (0), and 0.9 of the smaller one (1)
    pub same_word_thresh: [f32; 2],

    pub canny_low: f32,
    pub canny_high: f32,

    /// The minimal height of a letter. This measures the distance between the topmost
    /// pixel and the bottommost pixel. Smaller shapes are discarded as noise.
    pub letter_min_height: u32,

    /// The maximal height of a letter. This measures the distance between the leftmost
    /// pixel and the rightmost pixel. Smaller shapes are discarded as background.
    pub letter_max_height: u32,

    /// The minimal number of pixels in a letter. Shapes containing fewer pixels are discarded
    /// as noise. This measure the total number of "black" pixels in the shape.
    pub letter_min_pixels: u32,
    pub letter_occlude_thresh: Option<u32>,

    /// Maximal width of a strike, in iterations (~pixels). Used to give up when attempting to
    /// find the opposite edge of a shape.
    pub max_width: u32,

    /// The maximum aspect ratio for a letter.
    pub letter_max_aspect_ratio: f32,

    /// The inner-class standard derivation when grouping letters.
    pub std_ratio: f32,

    /// Limit on how quickly the stroke width is expected to change in a letter.
    /// If the stroke width measure for two consecutive pixels is not comprised
    /// between 1/smoothness_ratio and smoothness_ratio, then these pixels cannot
    /// be part of the same letter candidate.
    pub smoothness_ratio: f32,

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

impl Default for SwtParams {
    fn default() -> SwtParams {
        SwtParams {
            interval: 1,
            same_word_thresh: [0.1, 0.8 ],
            min_neighbors: 1,
            scale_invariant: false,
            canny_low: 0.48,
            canny_high: 0.8,
            letter_min_height: 8,
            letter_max_height: 300,
            letter_min_pixels: 38,
            letter_occlude_thresh: Some(3),
            max_width: 70,
            letter_max_aspect_ratio: 8.,
            std_ratio: 0.83,
            smoothness_ratio: 3.0,
            thickness_ratio: 1.5,
            height_ratio: 1.7,
            intensity_thresh: 31,
            distance_ratio: 2.9,
            intersect_ratio: 1.3,
            elongate_ratio: 1.9,
            letter_thresh: 3,
            breakdown: true,
            breakdown_ratio: 1.0,
        }
    }
}

/// Complete an outline (as devised by calling sobel), by adding the missing pixels at the
/// intersection of two edges.
pub fn close_outline(image: &GrayImage, threshold: u8) -> GrayImage {
    let mut result = GrayImage::from_pixel(image.width(), image.height(), Luma::black());

    if image.width() == 0 || image.height() == 0 {
        return result;
    }

    for i in 0 .. image.width() {
        for j in 0 .. image.height() {
            let is_white = |i, j| {
                image.get_pixel(i, j).data[0] >= threshold
            };

            if result.get_pixel(i, j).data[0] == 0 && is_white(i, j) {
                *result.get_pixel_mut(i, j) = Luma::white()
            }

            if i + 1 >= image.width() || j + 1 >= image.height() {
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
    result
}

#[derive(Clone, Debug)]
struct Ray {
    x: i32,
    slope_x: i32,
    adx: i32,
    y: i32,
    slope_y: i32,
    ady: i32,
    err: i32,
}

#[derive(Debug)]
struct RayDirection {
    xx: i32,
    xy: i32,
    yx: i32,
    yy: i32
}

#[derive(Debug)]
struct Stroke {
    x0: i32,
    y0: i32,
    x1: i32,
    y1: i32,
    width: u8,
}
impl Stroke {
    fn iter(&self) -> StrokeIterator {
        let adx = i32::abs(self.x1 - self.x0);
        let ady = i32::abs(self.y1 - self.y0);
        let slope_x = if self.x1 > self.x0 { 1 } else { -1 };
        let slope_y = if self.y1 > self.y0 { 1 } else { -1 };

        let ray = Ray {
            x: self.x0,
            y: self.y0,
            slope_x: slope_x,
            slope_y: slope_y,
            adx: adx,
            ady: ady,
            err: adx - ady
        };

        StrokeIterator {
            ray: Some(ray),
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

#[derive(Clone, Debug)]
struct Line(Ray);

struct LineIterator(Ray);


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
    fn from_grad(x: u32, y: u32,
            x_grad: &ImageBuffer<Luma<i16>, Vec<i16>>,
            y_grad: &ImageBuffer<Luma<i16>, Vec<i16>>,
            vector: &RayDirection,
            direction: SwtDirection) -> Line {
        let x_grad_pixel = x_grad.get_pixel(x, y).data[0] as i32;
        let y_grad_pixel = y_grad.get_pixel(x, y).data[0] as i32;
        println!("Line::from_grad {}/{}", x_grad_pixel, y_grad_pixel);
        let rx_grad = x_grad_pixel * vector.xx + y_grad_pixel * vector.xy;
        let ry_grad = x_grad_pixel * vector.yx + y_grad_pixel * vector.yy;

        let adx = i32::abs(rx_grad);
        let ady = i32::abs(ry_grad);
        Line(Ray {
            x: x as i32,
            slope_x: if rx_grad <= 0 { direction } else { direction.reverse() } as i32,
            adx: adx,

            y: y as i32,
            slope_y: if ry_grad <= 0 { direction } else { direction.reverse() } as i32,
            ady: ady,

            err: adx - ady
        })
    }
    fn iter(&self) -> LineIterator {
        LineIterator(self.0.clone())
    }
}

impl Ray {
    fn increment(&mut self) {
        let e2 = 2 * self.err;
//        println!("Ray::increment {} >?> {}: {}", e2, -self.ady, e2 > -self.ady);
        if e2 > -self.ady {
//            println!("Ray::increment: moving along x");
            self.err -= self.ady;
            self.x += self.slope_x
        }
//        println!("Ray::increment {} <?< {}: {}", e2, self.adx, e2 < self.adx);
        if e2 < self.adx {
//            println!("Ray::increment: moving along y");
            self.err += self.adx;
            self.y += self.slope_y;
        }
    }
}

pub fn swt(image: &GrayImage, params: &SwtParams, direction: SwtDirection) -> GrayImage {
    const BW_THRESHOLD : u8 = 1; /* Any number other than 0 (black) and 255 (white) works.*/

    // Each pixel contains either 0 (if it is not part of an outline) or the length of the smallest
    // segment found so far that connects two edges of an outline. Once we have finished computing
    // all (reasonable) segments, this will contain an approximation of the width of each outline,
    // i.e. the stroke width.
    let mut stroke_widths = GrayImage::new(image.width(), image.height());
    let mut strokes = vec![];

    // Compute all outlines on the image...
    let edges = canny(image, params.canny_low, params.canny_high);
    // ... and improve the chances that they are closed.
    let outlines = close_outline(&edges, BW_THRESHOLD);

    // Compute gradients.
    let x_grad = horizontal_sobel(image);
    let y_grad = vertical_sobel(image);

    colorize(&x_grad)
        .save("/tmp/output-colorized-x.png")
        .expect("Could not save colorized-x");
    colorize(&y_grad)
        .save("/tmp/output-colorized-y.png")
        .expect("Could not save colorized-y");

    for (x, y, pixel) in outlines.enumerate_pixels() {
        if pixel.data[0] < BW_THRESHOLD {
            // This pixel is not part of the outline, no need to throw rays.
            continue;
        }
        println!("swt: {}x{}", x, y);

        // This pixel is part of the outline, so we suspect that it's the border of a shape,
        // possibly a letter.
        //
        // Cast a few rays to find an opposite border. Each call to `ray_emit` corresponds to
        // casting a ray in a different direction. Note that we only cast rays towards the
        // right, as we are scanning the image from the left.
        let mut ray_emit = |vector| {
            let line = Line::from_grad(x, y, &x_grad, &y_grad, &vector, direction);
            println!("swt line: {:?} => {:?}", vector, line);

            // `Some((kx, ky))` once we have found an opposite border.
            let mut opposite_border = None;

            // For performance reasons, limit how far we are willing to search for an
            // opposite border.
            'search_opposite: for (ray, _) in line.iter().skip(1).zip(0 .. params.max_width) {
                if ray.x < 1 || ray.x >= image.width() - 1 {
                    // Leaving the image, no border found.
                    break;
                }
                if ray.y < 1 || ray.y >= image.height() - 1 {
                    // Leaving the image, no border found.
                    break;
                }
                if i32::abs(y as i32 - ray.y as i32) < 3 && i32::abs(x as i32 - ray.x as i32) < 3 {
                    // We are looking at another pixel that belongs to the same border, ignore it.
                    println!("swt: ignoring {}x{}", ray.x, ray.y);
                    continue;
                }
                // Note that we are not certain that we will encounter an edge. Despite calling
                // `close_outline`, we may have lost/missed pixels that should be part of the
                // opposite border.
                for &(dx, dy) in &[
                    (-1, 0),
                    (0,  0),
                    (1,  0),
                    (0, -1),
                    (0,  1)
                ] {
                    let kx = ray.x as i32 + dx;
                    let ky = ray.y as i32 + dy;
                    if outlines.get_pixel(kx as u32, ky as u32).data[0] > BW_THRESHOLD {
                        opposite_border = Some((ray, kx, ky));
                        println!("connects to {}x{}", kx, ky);
                        break 'search_opposite;
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
                    let y1 = (ky as i32 + dy) as u32;
//                    println!("swt: checking angle {}x{}", x1, y1);
                    let x_grad_pixel = x_grad.get_pixel(x, y).data[0] as i64;
                    let y_grad_pixel = y_grad.get_pixel(x, y).data[0] as i64;
                    let x1_grad_pixel = x_grad.get_pixel(x1, y1).data[0] as i64;
                    let y1_grad_pixel = y_grad.get_pixel(x1, y1).data[0] as i64;
                    let tn = i64::abs(y_grad_pixel * x1_grad_pixel - x_grad_pixel * y1_grad_pixel);
                    let td = i64::abs(x_grad_pixel * x1_grad_pixel - y_grad_pixel * y1_grad_pixel);
                    println!("swt: checking angle {}x{} => {}x{}-{}x{}", x1, y1, x_grad_pixel, y_grad_pixel, x1_grad_pixel, y1_grad_pixel);
                    println!("swt: tangents {}, {} => {}", tn, td, tn as f64 / td as f64);
                    // Compute a reasonable apprxomation of `|| tn/td || < pi/6`.
                    if tn * 7 < td * 4 {
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
//                    println!("swt: filling {} pixels", width);
                    let stroke = Stroke {
                        x0: x as i32,
                        y0: y as i32,
                        x1: ray.x as i32,
                        y1: ray.y as i32,
                        width: width
                    };
                    println!("swt: stroke {:?}", stroke);
                    for (x1, y1) in stroke.iter() {
                        println!("swt: striking {}x{}", x1, y1);
                        let pixel = stroke_widths.get_pixel_mut(x1, y1);
                        if pixel.data[0] == 0 || pixel.data[0] > width {
                            // We have found a shorter width. Update.
                            pixel.data[0] = width;
                        }
                    }
//                    println!("swt: done filling");

                    // Finally, record the stroke.
                    strokes.push(stroke);
                } else {
                    println!("swt: wrong angle, though");
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



/// Group pixels by components.
///
/// Two neighbouring pixels are considered part of the same component if:
/// 1. They both have a non-zero, finite Stroke Width.
/// 2. Their Stroke Width are close enough, as defined by `params.smoothness_ratio`.
///
/// Components that are too small are discarded as noise.
#[allow(dead_code)]
fn get_connected_components(swt: &Swt, params: &SwtParams) -> Vec<Contour> {
    const WHITE: Luma<u8> = Luma { data: [255] };
    const BLACK: Luma<u8> = Luma { data: [0] };

    assert!(params.smoothness_ratio > 0.);
    // Invariant: `ratio_to_average >= 1`.
    let ratio_to_average = if params.smoothness_ratio < 1. { 1. / params.smoothness_ratio } else { params.smoothness_ratio };
    let letter_min_pixels = max(params.letter_min_pixels, 2);

    // BLACK for points that have not been explored yet, WHITE otherwise.
    let mut explored : GrayImage = ImageBuffer::new(swt.0.width(), swt.0.height());
    let mut contours : Vec<Contour> = vec![];

    // Points that have been identified as being part of the current component but have not been
    // explored yet. This vector is reused but emptied each time we start with a new `(x, y)`
    let mut pending = Queue::new();


    for (x, y, pixel) in swt.0.enumerate_pixels() {
        // Explore any component going through `(x, y)`, unless it has already been explored.
        if pixel.data[0] == 0 {
            // Not part of a component.
            continue;
        }
        if *explored.get_pixel(x, y) != BLACK {
            // Component has already been labelled.
            continue;
        }
        // Start labelling component.
        let point = Point {
            x: x,
            y: y
        };
        let mut component = Contour::new(point);

        // Start taking orders.
        pending.clear();
        pending.push(point);
        let mut total_stroke = 0;
        while !pending.is_empty() {
            // Advance by one.
            let closed = pending.pop().unwrap();

            // If we revisit this pixel, mark that the component has been handled already.
            *explored.get_pixel_mut(closed.x, closed.y) = WHITE;
            component.push(closed);

            // Prepare neighbouring pixels.
            for &(dx, dy) in &[
                (-1,  0),
                (1,   0),
                (-1, -1),
                (0,  -1),
                (1,  -1),
                (-1,  1),
                (0,   1),
                (1,   1)
            ] {
                let nx = x as i32 + dx;
                let ny = y as i32 + dy;
                if !(nx >= 0 && ny >= 0) {
                    // This pixel is out of the picture.
                    continue;
                }
                let nx = nx as u32;
                let ny = ny as u32;
                if !(nx < swt.0.width() && ny < swt.0.height()) {
                    // This pixel is out of the picture.
                    continue;
                }
                let pixel = *swt.0.get_pixel(nx, ny);
                let current = pixel.data[0] as u64;
                if pixel == BLACK {
                    // The pixel is not part of any contour.
                    continue;
                }
                if *explored.get_pixel(nx, ny) != BLACK {
                    // The component has been visited already.
                    continue;
                }
                total_stroke += current;

                // let average = total_stroke / pending.total_pushed()
                // we must have average <= pixel.data[0] <= average * ratio
                // or average * ratio <= pixel.data[0] <= average

                let t = current * pending.total_pushed() as u64;

                if (total_stroke <= t && t <= (total_stroke as f32 * ratio_to_average) as u64)
                || (total_stroke <= (t as f32 * ratio_to_average) as u64 && t <= total_stroke) {
                    // Smooth variation of Stroke Width: this pixel is part of the same component.
                    pending.push(Point {
                        x: nx,
                        y: ny
                    });
                    total_stroke += swt.0.get_pixel(nx, ny).data[0] as u64;
                }
            }
        }
        // Contour is complete.
        if component.height() < params.letter_min_height
        || component.height() > params.letter_max_height
        || component.size() < letter_min_pixels {
            // This component is weird, most likely not a letter.
            continue;
        }
        contours.push(component);
    }
    contours
}

pub struct Letter {
    center: Point,
    thickness: u8,
    intensity: u32,
    std: f32,
    mean: f32,
    contour: Contour,
    id: usize,
}


struct Swt(GrayImage);

/// Extract from a swt good candidates for being letters.
#[allow(dead_code)]
fn get_connected_letters(image: &GrayImage, swt: &Swt, params: &SwtParams) -> Vec<Letter> {
    let mut contours = get_connected_components(swt, params);
    let (aspect_ratio_min, aspect_ratio_max) =
        if params.letter_max_aspect_ratio < 1. {
            (params.letter_max_aspect_ratio, 1. / params.letter_max_aspect_ratio)
        } else {
            (1. / params.letter_max_aspect_ratio,  params.letter_max_aspect_ratio)
        };

    let mut strokes = vec![]; // Out of the loop to save some memory management.

    // FIXME: Merge contours that appear inside other contours.
    let mut id = 0;
    let letters : Vec<_> = contours.drain(..).filter_map(|contour| {
        strokes.clear();

        let ratio = contour.ratio();
        if ratio < aspect_ratio_min || ratio > aspect_ratio_max {
            // Bad ratio, discard contour.
            return None;
        }
        let ratio = contour.sq_ratio();
        if ratio < aspect_ratio_min || ratio > aspect_ratio_max {
            // Bad ratio, discard contour.
            return None;
        }

        // Compute mean stroke along the contour.
        let mut total_stroke = 0;
        for point in &contour.points {
            let stroke = swt.0.get_pixel(point.x, point.y).data[0];
            strokes.push(stroke);
            total_stroke += stroke as u64;
        }
        let mean = total_stroke as f32 / contour.size() as f32;

        // Compute variance along the contour.
        let mut total_sq_delta = 0.0;
        for point in &contour.points {
            let delta = mean as f32 - swt.0.get_pixel(point.x, point.y).data[0] as f32;
            total_sq_delta += delta * delta;
        }
        let variance = total_sq_delta / contour.size() as f32;

        if variance > mean * params.std_ratio {
            return None;
        }

        // Ok, this looks like a letter so far.
        id += 1;
        let letter = Letter {
            id: id,
            center: contour.center(),
            std: variance,
            mean: mean,
            contour: contour,
            thickness: median(&strokes).unwrap(),
            intensity: 0, // FIXME: Not computed yet?
        };

        Some(letter)
    }).collect();

    // The pixmap contains the id of the connected component.
    let mut pixmap : ImageBuffer<Luma<usize>, _> = ImageBuffer::new(image.width(), image.height());
    for letter in &letters {
        // Mark all the pixels that belong to this letter.
        for point in &letter.contour.points {
            pixmap.get_pixel_mut(point.x, point.y).data[0] = id;
        }
    }


    // Get rid of letters whose bounding box intersects too many other letters.
    let mut letters : Vec<Letter> =
        if let Some(letter_occlude_thresh) = params.letter_occlude_thresh {
            // `true` if there is an intersection between the bounding box of the current letter and
            // an actual pixel of another letter.
            let mut others : Vec<bool> = Vec::with_capacity(id);
            let mut filtered = vec![];
            'per_letter: for letter in letters {
                others.clear();
                let mut intersections = 0;
                for x in letter.contour.top_left.x .. min(letter.contour.bottom_right.x + 1, image.width()) {
                    for y in letter.contour.top_left.y .. min(letter.contour.bottom_right.y + 1, image.height()) {
                        let pix_id = pixmap.get_pixel(x, y).data[0];
                        if pix_id > 0 && pix_id != letter.id && !others[id] {
                            // Look, there's an intersection with another component.
                            others[pix_id] = true;
                            intersections += 1;
                            if intersections > letter_occlude_thresh {
                                // Get rid of letter.
                                continue 'per_letter
                            }
                        }
                    }
                }
                filtered.push(letter)
            }
            filtered
        } else {
            letters
        };

    for letter in &mut letters {
        let mut total_intensity = 0 as u32;
        for point in &letter.contour.points {
            total_intensity += image.get_pixel(point.x, point.y).data[0] as u32
        }
        letter.intensity = total_intensity / letter.contour.size()
    }

    // FIXME: Outputting intermediate image.
    println!("Outputting intermediate image");
    {
        use std::collections::HashSet;
        use imageproc::map;

        let mut ids = HashSet::new();
        for letter in &letters {
            ids.insert(letter.id);
        }
        let foo = map::map_pixels(&pixmap, |_, _, pixel| {
            if ids.contains(&pixel.data[0]) {
                pixel
            } else {
                Luma {
                    data: [0]
                }
            }
        });
        let colorized = colorize(&foo);
        colorized.save("/tmp/output-connected-letters.png")
            .expect("Could not write file");
    }

    letters
}

#[allow(dead_code)]
pub fn detect_words(image: &DynamicImage, params: &SwtParams) {
    // FIXME: Implement downsizing.
//    let mut words = vec![];
    let gray = image.to_luma();
    let swt = Swt(swt(&gray, params, SwtDirection::DarkToBright));
    let letters_1 = get_connected_letters(&gray, &swt, params);
}