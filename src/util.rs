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

