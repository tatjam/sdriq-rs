use std::io::{BufReader, BufWriter, Read, Write};
use thiserror::Error;

use num_complex::Complex;

#[derive(Error, Debug)]
pub enum ReadError {
    #[error("Header is incomplete")]
    HeaderIncomplete(),
    #[error("header CRC {header:?} does not match computed CRC {expected:?}")]
    HeaderInvalidCRC { header: u32, expected: u32 },
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
}
pub enum WriteError {}

pub struct Header {
    /// Sample rate in samples per second
    samp_rate: u32,
    /// Center frequency in Hz
    center_freq: u64,
    /// Unix timestamp (ms) of the first sample in the file
    start_timestamp: u64,
    /// Sample size, either 16 or 24 bits
    samp_size: u32,
}

impl Header {
    pub fn new<R: Read>(reader: &mut R) -> Result<Self, ReadError> {
        let mut header_bytes: [u8; 32] = [0; 32];
        let num_read = reader.read(&mut header_bytes)?;
        if num_read < 32 {
            return Err(ReadError::HeaderIncomplete());
        }

        let samp_rate = u32::from_le_bytes(
            header_bytes[0..4]
                .try_into()
                .expect("We have 32 bytes, this is gauranteed to succeed"),
        );

        let center_freq = u64::from_le_bytes(
            header_bytes[4..12]
                .try_into()
                .expect("We have 32 bytes, this is guaranteed to succeed"),
        );

        let start_timestamp = u64::from_le_bytes(
            header_bytes[12..20]
                .try_into()
                .expect("We have 32 bytes, this is guaranteed to succeed"),
        );

        let samp_size = u32::from_le_bytes(
            header_bytes[20..24]
                .try_into()
                .expect("We have 32 bytes, this is guaranteed to succeed"),
        );

        // 4 bytes of padding skipped

        let crc32_computed = crc32fast::hash(&header_bytes[0..28]);

        let crc32 = u32::from_le_bytes(
            header_bytes[28..32]
                .try_into()
                .expect("We have 32 bytes, this is guaranteed to succeed"),
        );

        if crc32_computed != crc32 {
            return Err(ReadError::HeaderInvalidCRC {
                header: crc32_computed,
                expected: crc32,
            });
        }

        Ok(Header {
            samp_rate,
            center_freq,
            start_timestamp,
            samp_size,
        })
    }
}

/// A Source wraps around a Read to be able to fetch samples from an sdriq file. Samples are obtained by
/// copying into an array of Complex values, optionally performing type conversion (at a slight
/// performance cost, of course!).
pub struct Source<R: Read> {
    header: Header,
    reader: BufReader<R>,
}

impl<R: Read> Source<R> {
    /// Creates a sample reader, expecting to find a valid header on the first few bytes of the Read argument.
    /// Uses a sane-default for the internal buffer size, which should perform acceptably on most use cases.
    fn new(source: R) -> Result<Self, ReadError> {
        // Default buffer size holds a bit over 1sec of 1Msps 16 bit samples, rounded to nearest power of 2
        Self::with_capacity(2097152, source)
    }

    /// Creates a sample reader, expecting to find a valid header on the first few bytes of the Read argument.
    fn with_capacity(bytes: usize, source: R) -> Result<Self, ReadError> {
        let mut reader = BufReader::with_capacity(bytes, source);
        let header = Header::new(&mut reader)?;
        Ok(Self { header, reader })
    }

    /// Returns samples, as long as they are directly readable to Complex<T> type without any
    /// conversion being performed. This is fast, as it's a simple memory copy.
    /// Valid types for T are i32 (for 24 bit samples) or i16 (for 16 bit samples)
    fn get_samples<T>(target: &mut [Complex<T>]) -> Result<usize, ReadError> {
        todo!("Implement");
    }

    /// Read samples, with possible conversion if type Complex<T> does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// If the conversion would result in truncation (for example, 24 bits -> i16), an error is generated.
    fn get_samples_conv<T>(target: &mut [Complex<T>]) -> Result<usize, ReadError> {
        todo!("Implement");
    }

    /// Read samples, with possible conversion if type Complex<T> does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// Allows silent truncating conversion if the target type is too small
    fn get_samples_trunc<T>(target: &mut [Complex<T>]) -> Result<usize, ReadError> {
        todo!("Implement");
    }

    /// Read samples, normalizing them to [-1, 1] range, so T must be a floating point type (f32 or f64).
    /// The original range is deduced from the bit-size of the samples, so
    /// - 16 bit samples map [-32768, 32767] -> [-1, 1]
    /// - 24 bit samples map [-16777216, 16777215] -> [-1, 1]
    fn get_samples_norm<T>(target: &mut [Complex<T>]) -> Result<usize, ReadError> {
        todo!("Implement");
    }
}

/// A Source wraps around a Write to be able to write samples to an sdriq file. Samples are written by
/// copying from an array of Complex values, optionally performing type conversion (at a slight
/// performance cost, of course!)
struct Sink<W: Write> {
    writer: BufWriter<W>,
    header: Header,
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;

    #[test]
    fn read_16bit_file() {
        let file = File::open("test_files/test_1.sdriq").unwrap();
        let source = Source::new(file);
    }

    #[test]
    fn read_24bit_file() {}
}
