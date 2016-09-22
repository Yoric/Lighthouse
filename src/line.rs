use util::Point;

pub struct Segment {
    pub start: Point,
    pub stop: Point,
}

impl Segment {
    pub fn iter(&self) -> SegmentIterator {
        SegmentIterator {
            ray: Some(RayIterator::new(self.start, self.stop.x - self.start.x, self.stop.y - self.start.y)),
            stop: self.stop
        }
    }
}

pub struct SegmentIterator {
    ray: Option<RayIterator>,
    stop: Point,
}

impl Iterator for SegmentIterator {
    type Item = Point;
    fn next(&mut self) -> Option<Self::Item> {
        let mut done = false;
        let result;
        if let Some(ref mut iter) = self.ray {
            result = iter.next().expect("A RayIterator should always return a point.");
            debug_assert!(iter.sx as i32 * result.x <= iter.sx as i32 * self.stop.x);
            debug_assert!(iter.sy as i32 * result.y <= iter.sy as i32 * self.stop.y);
            if result == self.stop {
                // FIXME: This won't necessarily hit.
                done = true;
            }
        } else {
            return None;
        }
        if done {
            self.ray = None;
        }
        Some(result)
    }
}

pub struct RayIterator {
    /// The current position.
    pos: Point,
    adx: i32,
    ady: i32,
    err: i32,
    sx:  i32, // FIXME: sx is always -1 or 1, consider packing.
    sy:  i32, // FIXME: sy is always -1 or 1, consider packing.
}

impl RayIterator {
    pub fn new(start: Point, dx: i32, dy: i32) -> Self {
        assert!(dx != 0 || dy != 0);
        let sx = if dx > 0 { 1 } else { -1 };
        let sy = if dy > 0 { 1 } else { -1 };
        let adx = i32::abs(dx);
        let ady = i32::abs(dy);
        let err = adx - ady;
        RayIterator {
            pos: start,
            adx: adx,
            ady: ady,
            sx: sx,
            sy: sy,
            err: err
        }
    }
}

impl Iterator for RayIterator {
    type Item = Point;
    fn next(&mut self) -> Option<Self::Item> {
        let result = self.pos.clone();

        let e2 = 2 * self.err;
        if e2 > -self.ady {
            self.err -= self.ady;
            self.pos.x += self.sx as i32;
        }
        if e2 < self.adx {
            self.err += self.adx;
            self.pos.y += self.sy as i32;
        }
        Some(result)
    }
}