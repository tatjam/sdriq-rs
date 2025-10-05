//! # sdriq - Read and write .sdriq files
//! **The library is in active development, the API may change!**
//!
//! ## Example
//!
//! ```no_run
//! use sdriq::{Source, Complex};
//! // You can also use num_complex::Complex;
//! use std::fs::File;
//!
//! let file = File::open("file.sdriq").unwrap();
//! let mut source = Source::new(file).unwrap();
//! let mut samples = vec![Complex::new(0.0, 0.0); 1000000];
//! let num_samples = source.get_samples_norm(&mut samples).unwrap();
//!
//! let average = samples[0..num_samples]
//!     .iter()
//!     .fold(Complex::new(0.0, 0.0), |acc, &e| acc + e)
//!     / (num_samples as f32);
//!
//! println!("Average of first {} samples = {}", num_samples, average);
//!
//! ```
//!
//! ## Development state
//!
//! **Not yet implemented**:
//! - Benchmarking and performance tuning
//!
//! **Not yet tested**:
//! - 16 bit files (requires recompiling sdrangel)
//!

use std::{
    any::TypeId,
    io::{BufRead, BufReader, BufWriter, Read, Write},
    ptr::slice_from_raw_parts,
};
use thiserror::Error;

pub use num_complex::Complex;

#[derive(Error, Debug)]
pub enum ReadError {
    #[error("Header is incomplete")]
    HeaderIncomplete(),
    #[error("Header CRC {header:?} does not match computed CRC {expected:?}")]
    HeaderInvalidCRC { header: u32, expected: u32 },
    #[error("Header sample size is invalid value {samp_size:?}")]
    HeaderSampleSizeInvalid { samp_size: u32 },
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
    #[error("Sample size stored in the file is not compatible with target type")]
    InvalidSampleSize(),
    #[error("Reading sample would truncate")]
    TruncationError(),
}

#[derive(Error, Debug)]
pub enum WriteError {
    #[error("I/O error")]
    IOError(#[from] std::io::Error),
    #[error("Header sample size is invalid value {samp_size:?}")]
    HeaderSampleSizeInvalid { samp_size: u32 },
    #[error("Sample size stored in the file is not compatible with source type")]
    InvalidSampleSize(),
    #[error("Writing sample would truncate")]
    TruncationError(),
}

/// Header content of the .sdriq file
#[derive(Copy, Clone, PartialEq, Debug)]
pub struct Header {
    /// Sample rate in samples per second
    pub samp_rate: u32,
    /// Center frequency in Hz
    pub center_freq: u64,
    /// Unix timestamp (ms) of the first sample in the file
    pub start_timestamp: u64,
    /// Sample size, either 16 (stored as 16 bits) or 24 bits (stored as 32 bits)
    pub samp_size: u32,
}

impl Header {
    /// Writes the header to the writer, returning the number of bytes written.
    pub fn write_to<W: Write>(&self, writer: &mut W) -> Result<usize, WriteError> {
        if self.samp_size != 16 && self.samp_size != 24 {
            return Err(WriteError::HeaderSampleSizeInvalid {
                samp_size: self.samp_size,
            });
        }

        let mut header_buf: [u8; 32] = [0; 32];
        header_buf[0..4].copy_from_slice(self.samp_rate.to_le_bytes().as_slice());
        header_buf[4..12].copy_from_slice(self.center_freq.to_le_bytes().as_slice());
        header_buf[12..20].copy_from_slice(self.start_timestamp.to_le_bytes().as_slice());
        header_buf[20..24].copy_from_slice(self.samp_size.to_le_bytes().as_slice());
        // 4 bytes of padding, left as 0
        let crc32_computed = crc32fast::hash(&header_buf[0..28]);
        header_buf[28..32].copy_from_slice(crc32_computed.to_le_bytes().as_slice());

        writer.write_all(&header_buf)?;
        Ok(header_buf.len())
    }

    /// Read a header as stored in a binary, from the given reader.
    pub fn read_from<R: Read>(reader: &mut R) -> Result<Self, ReadError> {
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

        if samp_size != 16 && samp_size != 24 {
            return Err(ReadError::HeaderSampleSizeInvalid { samp_size });
        }

        // 4 bytes of padding / zeroes skipped

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

    /// Returns the number of bits that each sample takes up in the binary file,
    /// not neccesarily equal to `samp_size`.
    pub fn get_stored_bits_per_sample(&self) -> usize {
        match self.samp_size {
            16 => 16,
            24 => 32,
            _ => unreachable!(),
        }
    }
}

#[doc(hidden)]
#[inline]
fn is_24bit(value: i32) -> bool {
    (value << 8) >> 8 == value
}

/// If you want to read samples to a type with conversion, this trait must be implemented
/// for said type.
pub trait SampleConvert: Copy + Sized {
    /// Read a sample, if possible without truncation, from a 16-bit value
    fn from_i16(value: i16) -> Option<Self>;
    /// Read a sample, if possible without truncation, from a 24-bit value (stored as 32 bit)
    fn from_i24(value: i32) -> Option<Self>;
    /// Read a sample, clamping if needed, from a 16-bit value
    fn from_i16_clamp(value: i16) -> Self;
    /// Read a sample, clamping if needed, from a 24-bit value (stored as 32 bit)
    fn from_i24_clamp(value: i32) -> Self;
    /// Write a 16-bit sample, if possible without truncation, from the value
    fn to_i16(self) -> Option<i16>;
    /// Write a 24-bit sample, if possible without truncation, from the value
    fn to_i24(self) -> Option<i32>;
    /// Write a 16-bit sample, clamping if needed, from the value
    fn to_i16_clamp(self) -> i16;
    /// Write a 24-bit sample, clamping if needed, from the value
    fn to_i24_clamp(self) -> i32;
}

impl SampleConvert for i16 {
    fn from_i16(value: i16) -> Option<Self> {
        Some(value)
    }

    fn from_i24(value: i32) -> Option<Self> {
        i16::try_from(value).ok()
    }

    fn from_i16_clamp(value: i16) -> Self {
        value as Self
    }

    fn from_i24_clamp(value: i32) -> Self {
        value.clamp(i16::MIN as i32, i16::MAX as i32) as Self
    }

    fn to_i16(self) -> Option<i16> {
        Some(self)
    }

    fn to_i24(self) -> Option<i32> {
        Some(self as i32)
    }

    fn to_i16_clamp(self) -> i16 {
        self
    }

    fn to_i24_clamp(self) -> i32 {
        self as i32
    }
}

impl SampleConvert for i32 {
    fn from_i16(value: i16) -> Option<Self> {
        Some(value as i32)
    }

    fn from_i24(value: i32) -> Option<Self> {
        Some(value)
    }

    fn from_i16_clamp(value: i16) -> Self {
        value as Self
    }

    fn from_i24_clamp(value: i32) -> Self {
        value as Self
    }

    fn to_i16(self) -> Option<i16> {
        i16::try_from(self).ok()
    }

    fn to_i24(self) -> Option<i32> {
        if !is_24bit(self) { None } else { Some(self) }
    }

    fn to_i16_clamp(self) -> i16 {
        (self).clamp(i16::MIN as i32, i16::MAX as i32) as i16
    }

    fn to_i24_clamp(self) -> i32 {
        (self).clamp(I24_MIN, I24_MAX)
    }
}

/// Note that decimal truncation is done if converting to i16 or i24, instead of rounding
impl SampleConvert for f32 {
    fn from_i16(value: i16) -> Option<Self> {
        Some(value as f32)
    }

    fn from_i24(value: i32) -> Option<Self> {
        // 24 bits can be fully represented as f32, but 32 bit cannot!
        debug_assert!(is_24bit(value), "value was not 24 bit");
        Some(value as f32)
    }

    fn from_i16_clamp(value: i16) -> Self {
        value as Self
    }

    fn from_i24_clamp(value: i32) -> Self {
        value as Self
    }

    fn to_i16(self) -> Option<i16> {
        debug_assert!(!self.is_nan());
        if self.is_infinite() {
            return None;
        }
        if self < i16::MIN as f32 || self > i16::MAX as f32 {
            return None;
        }

        Some(self as i16)
    }

    fn to_i24(self) -> Option<i32> {
        debug_assert!(!self.is_nan());
        if self.is_infinite() {
            return None;
        }
        if self < I24_MIN as f32 || self > I24_MAX as f32 {
            return None;
        }

        Some(self as i32)
    }

    fn to_i16_clamp(self) -> i16 {
        debug_assert!(!self.is_nan());
        if self < i16::MIN as f32 {
            i16::MIN
        } else if self > i16::MAX as f32 {
            i16::MAX
        } else {
            self as i16
        }
    }

    fn to_i24_clamp(self) -> i32 {
        debug_assert!(!self.is_nan());
        if self < I24_MIN as f32 {
            I24_MIN
        } else if self > I24_MAX as f32 {
            I24_MAX
        } else {
            self as i32
        }
    }
}

impl SampleConvert for f64 {
    fn from_i16(value: i16) -> Option<Self> {
        Some(value as f64)
    }

    fn from_i24(value: i32) -> Option<Self> {
        Some(value as f64)
    }

    fn from_i16_clamp(value: i16) -> Self {
        value as Self
    }

    fn from_i24_clamp(value: i32) -> Self {
        value as Self
    }

    fn to_i16(self) -> Option<i16> {
        debug_assert!(!self.is_nan());
        if self.is_infinite() {
            return None;
        }
        if self < i16::MIN as f64 || self > i16::MAX as f64 {
            return None;
        }

        Some(self as i16)
    }

    fn to_i24(self) -> Option<i32> {
        debug_assert!(!self.is_nan());
        if self.is_infinite() {
            return None;
        }
        if self < I24_MIN as f64 || self > I24_MAX as f64 {
            return None;
        }

        Some(self as i32)
    }

    fn to_i16_clamp(self) -> i16 {
        debug_assert!(!self.is_nan());
        if self < i16::MIN as f64 {
            i16::MIN
        } else if self > i16::MAX as f64 {
            i16::MAX
        } else {
            self as i16
        }
    }

    fn to_i24_clamp(self) -> i32 {
        debug_assert!(!self.is_nan());
        if self < I24_MIN as f64 {
            I24_MIN
        } else if self > I24_MAX as f64 {
            I24_MAX
        } else {
            self as i32
        }
    }
}

const I24_MIN: i32 = -8388608;
const I24_MAX: i32 = 8388607;
const F64_SCALE_I16: f64 = 32768.0;
const F64_SCALE_I24: f64 = 8388608.0;
const F32_SCALE_I16: f32 = 32768.0;
const F32_SCALE_I24: f32 = 8388608.0;
const I16_SCALE_F32: f32 = 1.0 / F32_SCALE_I16;
const I24_SCALE_F32: f32 = 1.0 / F32_SCALE_I24;
const I16_SCALE_F64: f64 = 1.0 / F64_SCALE_I16;
const I24_SCALE_F64: f64 = 1.0 / F64_SCALE_I24;

/// This trait must be implemented for your target type if you want to be able to perform "normalized" sample reads.
///
/// The implementations of this trait included in this crate use the following logic:
/// - For 16 bit numbers, maps [-32768, 32767] <-> [-1, 0.9999694824]
/// - For 24 bit numbers, maps [-8388608, 8388607] <-> [-1, 0.9999998808]
///
/// This guarantees 0 is mapped to 0. The error introduced is insignificant for almost any use, but be aware that
/// the returned values will always be contained in the interval [-1, 1).
pub trait SampleNormalizedConvert: Copy + Sized {
    fn from_i16_norm(value: i16) -> Self;
    fn from_i24_norm(value: i32) -> Self;
    fn to_i16_denorm(&self) -> i16;
    fn to_i24_denorm(&self) -> i32;
}

impl SampleNormalizedConvert for f32 {
    fn from_i16_norm(value: i16) -> Self {
        (value as f32) * I16_SCALE_F32
    }

    fn from_i24_norm(value: i32) -> Self {
        // This would result in values outside [-1, 1)
        debug_assert!(is_24bit(value), "value was not 24 bit");
        (value as f32) * I24_SCALE_F32
    }

    fn to_i16_denorm(&self) -> i16 {
        let denorm = *self * F32_SCALE_I16;
        debug_assert!(i16::MIN as f32 <= denorm && denorm <= i16::MAX as f32);
        denorm as i16
    }

    fn to_i24_denorm(&self) -> i32 {
        let denorm = *self * F32_SCALE_I24;
        debug_assert!(I24_MIN as f32 <= denorm && denorm <= I24_MAX as f32);
        denorm as i32
    }
}

impl SampleNormalizedConvert for f64 {
    fn from_i16_norm(value: i16) -> Self {
        (value as f64) * I16_SCALE_F64
    }

    fn from_i24_norm(value: i32) -> Self {
        // This would result in values outside [-1, 1)
        debug_assert!(is_24bit(value), "value was not 24 bit");
        (value as f64) * I24_SCALE_F64
    }

    fn to_i16_denorm(&self) -> i16 {
        let denorm = *self * F64_SCALE_I16;
        debug_assert!(i16::MIN as f64 <= denorm && denorm <= i16::MAX as f64);
        denorm as i16
    }

    fn to_i24_denorm(&self) -> i32 {
        let denorm = *self * F64_SCALE_I24;
        debug_assert!(I24_MIN as f64 <= denorm && denorm <= I24_MAX as f64);
        denorm as i32
    }
}

/// A Source wraps around a Read to be able to fetch samples from an sdriq file.
pub struct Source<R: Read> {
    header: Header,
    reader: BufReader<R>,
}

impl<R: Read> Source<R> {
    /// Creates a sample reader, expecting to find a valid header on the first few bytes of the Read argument.
    /// Uses a sane-default for the internal buffer size, which should perform acceptably on most use cases.
    pub fn new(source: R) -> Result<Self, ReadError> {
        // Default buffer size holds a bit over 1sec of 1Msps 16 bit samples, rounded to nearest power of 2
        Self::with_capacity(2097152, source)
    }

    /// Creates a sample reader, expecting to find a valid header on the first few bytes of the Read argument.
    /// `bytes` represents the internal buffer size, which should be chosen to reduce I/O overhead.
    pub fn with_capacity(bytes: usize, source: R) -> Result<Self, ReadError> {
        let mut reader = BufReader::with_capacity(bytes, source);
        let header = Header::read_from(&mut reader)?;
        Ok(Self { header, reader })
    }

    /// Returns the header applicable to this `Source`
    pub fn get_header(&self) -> Header {
        self.header
    }

    #[doc(hidden)]
    // Reads samples by memcpy as long as sizeof(T) is correct. Not intented to be used directly.
    fn get_samples_memcpy<T>(&mut self, target: &mut [Complex<T>]) -> Result<usize, ReadError> {
        let header_bits = self.header.get_stored_bits_per_sample();
        let incoming_bits = std::mem::size_of::<T>() * 8;
        if header_bits != incoming_bits {
            return Err(ReadError::InvalidSampleSize());
        }

        let mut num_written: usize = 0;
        let bytes_per_sample = std::mem::size_of::<T>() * 2;

        while num_written < target.len() {
            let num_remain = target.len() - num_written;
            let buffer = self.reader.fill_buf()?;
            if buffer.is_empty() {
                // EOF
                break;
            }

            let num_in_buf = buffer.len() / bytes_per_sample;

            if num_in_buf == 0 {
                // Partial EOF
                // TODO: This could be reported as an error, as the file is expected to contain non-clampated samples
                break;
            }

            let num_copy = num_remain.min(num_in_buf);
            let target_ptr = target[num_written..].as_mut_ptr() as *mut u8;

            unsafe {
                // SAFETY:
                // - target_ptr contains atleast num_remain samples
                // - buffer contains num_in_buf samples
                // - num_copy = min(num_in_buf, num_remain), so no out of bounds access can take place
                std::ptr::copy_nonoverlapping(
                    buffer.as_ptr(),
                    target_ptr,
                    num_copy * bytes_per_sample,
                );
            }

            self.reader.consume(num_copy * bytes_per_sample);
            num_written += num_copy;
        }

        Ok(num_written)
    }

    /// Reads samples, as long as they are directly copyable to `Complex<T>` type without any
    /// conversion being performed. This is fast, as it's a simple, flat memory copy.
    ///
    /// **NOTE**: On big endian systems, the direct copy is not done, and instead "slow" loading is performed
    ///
    /// Returns the number of complex samples read.
    pub fn get_samples_direct<T: 'static>(
        &mut self,
        tgt: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        let header_bits = self.header.get_stored_bits_per_sample();
        match header_bits {
            32 => {
                if TypeId::of::<T>() != TypeId::of::<i32>() {
                    return Err(ReadError::InvalidSampleSize());
                }
            }
            16 => {
                if TypeId::of::<T>() != TypeId::of::<i16>() {
                    return Err(ReadError::InvalidSampleSize());
                }
            }
            _ => unreachable!(),
        }

        #[cfg(not(target_endian = "little"))]
        {
            // Fall-back to slow reading
            return self.get_samples::<T>(tgt);
        }

        #[cfg(target_endian = "little")]
        {
            self.get_samples_memcpy::<T>(tgt)
        }
    }

    /// Read samples, with possible conversion if type `Complex<T>` does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// If the conversion would result in truncation (for example, 24 bits -> i16), an error is generated.
    ///
    /// No truncation errors are ever generated is T is a floating point number, as 24 bit integers can fully
    /// fit into f32.
    /// Note that the resulting floating point numbers are NOT normalized!
    ///
    /// Note that this function will choose the most optimized possible call for the type `T`, so if you have
    /// `T = i32` or `T = i16`, it will automatically call [Source::get_samples_direct] on little endian systems.
    ///
    /// Returns the number of complex samples read.
    pub fn get_samples<T: SampleConvert + 'static>(
        &mut self,
        target: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        // Try optimized functions if possible
        #[cfg(target_endian = "little")]
        {
            let header_bits = self.header.get_stored_bits_per_sample();
            if TypeId::of::<T>() == TypeId::of::<i32>() && header_bits == 32 {
                return self.get_samples_direct(target);
            }
            if TypeId::of::<T>() == TypeId::of::<i16>() && header_bits == 16 {
                return self.get_samples_direct(target);
            }
        }

        // Fallback "slow" method
        let mut num_read = 0;
        while num_read < target.len() {
            let maybe_smp = match self.header.samp_size {
                16 => self.read_next_sample_16bit::<T>()?,
                24 => self.read_next_sample_24bit::<T>()?,
                _ => unreachable!(),
            };

            target[num_read] = match maybe_smp {
                Some(v) => v,
                None => break,
            };

            num_read += 1
        }

        Ok(num_read)
    }

    #[inline]
    #[doc(hidden)]
    fn read_raw_16bit(&mut self) -> Result<Option<(i16, i16)>, ReadError> {
        let mut buffer_i: [u8; 2] = [0; 2];
        let mut buffer_q: [u8; 2] = [0; 2];
        let mut num_read = self.reader.read(&mut buffer_i)?;
        num_read += self.reader.read(&mut buffer_q)?;

        if num_read == 0 {
            return Ok(None);
        }
        if num_read < 4 {
            // TODO: This could be reported as an error, as the file is expected to contain full samples
            return Ok(None);
        }

        let i = i16::from_le_bytes(buffer_i);
        let q = i16::from_le_bytes(buffer_q);

        Ok(Some((i, q)))
    }

    #[inline]
    #[doc(hidden)]
    fn read_raw_24bit(&mut self) -> Result<Option<(i32, i32)>, ReadError> {
        let mut buffer_i: [u8; 4] = [0; 4];
        let mut buffer_q: [u8; 4] = [0; 4];
        let mut num_read = self.reader.read(&mut buffer_i)?;
        num_read += self.reader.read(&mut buffer_q)?;

        if num_read == 0 {
            return Ok(None);
        }
        if num_read < 8 {
            // TODO: This could be reported as an error, as the file is expected to contain full samples
            return Ok(None);
        }

        let i = i32::from_le_bytes(buffer_i);
        let q = i32::from_le_bytes(buffer_q);

        Ok(Some((i, q)))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_16bit<T: SampleConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_16bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i16(i).ok_or(ReadError::TruncationError())?,
            im: T::from_i16(q).ok_or(ReadError::TruncationError())?,
        }))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_24bit<T: SampleConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_24bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i24(i).ok_or(ReadError::TruncationError())?,
            im: T::from_i24(q).ok_or(ReadError::TruncationError())?,
        }))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_16bit_clamp<T: SampleConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_16bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i16_clamp(i),
            im: T::from_i16_clamp(q),
        }))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_24bit_clamp<T: SampleConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_24bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i24_clamp(i),
            im: T::from_i24_clamp(q),
        }))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_16bit_norm<T: SampleNormalizedConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_16bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i16_norm(i),
            im: T::from_i16_norm(q),
        }))
    }

    #[inline]
    #[doc(hidden)]
    fn read_next_sample_24bit_norm<T: SampleNormalizedConvert>(
        &mut self,
    ) -> Result<Option<Complex<T>>, ReadError> {
        let (i, q) = match self.read_raw_24bit()? {
            Some(v) => v,
            None => return Ok(None),
        };

        Ok(Some(Complex::<T> {
            re: T::from_i24_norm(i),
            im: T::from_i24_norm(q),
        }))
    }

    /// Read samples, with possible conversion if type `Complex<T>` does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// If T is too small (i16) and the incoming number if bigger, it will be clamped.
    ///
    /// Note that this function will choose the most optimized possible call for the type "T", as explained
    /// in [Source::get_samples].
    ///
    /// Returns the number of samples read.
    pub fn get_samples_clamp<T: SampleConvert + 'static>(
        &mut self,
        target: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        // Try optimized functions if possible
        #[cfg(target_endian = "little")]
        {
            let header_bits = self.header.get_stored_bits_per_sample();
            if TypeId::of::<T>() == TypeId::of::<i32>() && header_bits == 32 {
                return self.get_samples_direct(target);
            }
            if TypeId::of::<T>() == TypeId::of::<i16>() && header_bits == 16 {
                return self.get_samples_direct(target);
            }
        }

        // Fallback "slow" method
        let mut num_read = 0;
        while num_read < target.len() {
            let maybe_smp = match self.header.samp_size {
                16 => self.read_next_sample_16bit_clamp::<T>()?,
                24 => self.read_next_sample_24bit_clamp::<T>()?,
                _ => unreachable!(),
            };

            target[num_read] = match maybe_smp {
                Some(v) => v,
                None => break,
            };

            num_read += 1
        }

        Ok(num_read)
    }

    /// Read samples, normalizing them. See [SampleNormalizedConvert].
    pub fn get_samples_norm<T: SampleNormalizedConvert>(
        &mut self,
        target: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        let mut num_read = 0;
        while num_read < target.len() {
            let maybe_smp = match self.header.samp_size {
                16 => self.read_next_sample_16bit_norm::<T>()?,
                24 => self.read_next_sample_24bit_norm::<T>()?,
                _ => unreachable!(),
            };

            target[num_read] = match maybe_smp {
                Some(v) => v,
                None => break,
            };

            num_read += 1
        }

        Ok(num_read)
    }
}

/// A Sink wraps around a Write to be able to write samples to an sdriq file.
/// Because the number of bytes written does not directly map to the number of samples written,
/// in order to prevent "half-samples" being written, write_all is used at all times, and all
/// operations either write the full passed buffer, or fail.
///
/// It's important to use [Sink::flush] instead of relying on Drop flushing the buffer, as otherwise
/// any I/O errors will be silently ignored.
pub struct Sink<W: Write> {
    writer: BufWriter<W>,
    header: Header,
}

impl<W: Write> Sink<W> {
    /// Creates a sample writer from the given writer and header.
    /// Uses a sane-default for the internal buffer size, which should perform acceptably on most use cases.
    pub fn new(writer: W, header: Header) -> Result<Self, WriteError> {
        // Default buffer size holds a bit over 1sec of 1Msps 16 bit samples, rounded to nearest power of 2
        Self::with_capacity(2097152, writer, header)
    }

    /// Creates a sample writer from the given writer and header.
    /// `bytes` represents the internal buffer size, which should be chosen to reduce I/O overhead.
    pub fn with_capacity(bytes: usize, mut writer: W, header: Header) -> Result<Self, WriteError> {
        // Write the header, non-buffered
        header.write_to(&mut writer)?;
        let writer_buf = BufWriter::with_capacity(bytes, writer);
        Ok(Self {
            header,
            writer: writer_buf,
        })
    }

    /// Flushes the internal buffer. If you rely on `Drop` calling flush, any I/O errors will be silently
    /// ignored, so it's important to use this function! See [std::io::BufWriter] documentation.
    pub fn flush(&mut self) -> Result<(), std::io::Error> {
        self.writer.flush()
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_16bit<T: SampleConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i16().ok_or(WriteError::TruncationError())?;
        let q = sample.im.to_i16().ok_or(WriteError::TruncationError())?;
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_24bit<T: SampleConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i24().ok_or(WriteError::TruncationError())?;
        let q = sample.im.to_i24().ok_or(WriteError::TruncationError())?;
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_16bit_clamp<T: SampleConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i16_clamp();
        let q = sample.im.to_i16_clamp();
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_24bit_clamp<T: SampleConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i24_clamp();
        let q = sample.im.to_i24_clamp();
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_16bit_denorm<T: SampleNormalizedConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i16_denorm();
        let q = sample.im.to_i16_denorm();
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    #[inline]
    #[doc(hidden)]
    fn write_next_sample_24bit_denorm<T: SampleNormalizedConvert>(
        &mut self,
        sample: Complex<T>,
    ) -> Result<(), WriteError> {
        let i = sample.re.to_i24_denorm();
        let q = sample.im.to_i24_denorm();
        self.writer.write_all(i.to_le_bytes().as_slice())?;
        self.writer.write_all(q.to_le_bytes().as_slice())?;

        Ok(())
    }

    /// Read samples, with possible conversion if type `Complex<T>` does not match the type
    /// stored in the sdriq file. This could have slight overhead.
    /// If the conversion would result in clamping (for example, 24 bits -> i16), an error is generated,
    /// but rounding errors for floating point numbers are not reported.
    ///
    /// **WARNING**: Floating point numbers are stored as is, without "denormalizing". You likely want [Sink::write_samples_norm].
    ///
    /// Note that this function will choose the most optimized possible call for the type `T`, so if you have
    /// `T = i32` or `T = i16`, it will automatically call [Sink::write_samples_direct] on little endian systems.
    ///
    /// Returns the number of complex samples written.
    pub fn write_all_samples<T: SampleConvert>(
        &mut self,
        samples: &[Complex<T>],
    ) -> Result<(), WriteError> {
        for &samp in samples {
            match self.header.samp_size {
                16 => self.write_next_sample_16bit(samp)?,
                24 => self.write_next_sample_24bit(samp)?,
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Writes samples, as long as they are directly copyable to the binary format without any
    /// conversion being performed. This is fast, as it's a simple, flat memory copy.
    ///
    /// **NOTE**: On big endian systems, the direct copy is not done, and instead "slow" loading is performed
    ///
    /// **NOTE**: No range-checking is done for 32-bit -> 24-bit storage, if any out of bound values are used,
    /// they will be stored in the binary file as is, and may cause issues on users of the generated file. Note that
    /// this is possible because 24-bit values are stored in the .sdriq file with 32-bit alignment.
    ///
    /// Returns the number of complex samples written.
    pub fn write_all_samples_direct<T: SampleConvert + 'static>(
        &mut self,
        samples: &[Complex<T>],
    ) -> Result<(), WriteError> {
        let header_bits = self.header.get_stored_bits_per_sample();
        match header_bits {
            32 => {
                if TypeId::of::<T>() != TypeId::of::<i32>() {
                    return Err(WriteError::InvalidSampleSize());
                }
            }
            16 => {
                if TypeId::of::<T>() != TypeId::of::<i16>() {
                    return Err(WriteError::InvalidSampleSize());
                }
            }
            _ => unreachable!(),
        }

        #[cfg(not(target_endian = "little"))]
        {
            // Fall-back to slow writing
            return self.write_all_samples::<T>(samples);
        }

        #[cfg(target_endian = "little")]
        {
            self.write_all_samples_memcpy::<T>(samples)
        }
    }

    #[doc(hidden)]
    pub fn write_all_samples_memcpy<T: SampleConvert>(
        &mut self,
        samples: &[Complex<T>],
    ) -> Result<(), WriteError> {
        let header_bits = self.header.get_stored_bits_per_sample();
        let incoming_bits = std::mem::size_of::<T>() * 8;
        if header_bits != incoming_bits {
            return Err(WriteError::InvalidSampleSize());
        }

        let num_bytes = std::mem::size_of_val(samples);
        let raw_slice = slice_from_raw_parts(samples.as_ptr() as *const u8, num_bytes);
        unsafe {
            // SAFETY:
            // - If we have Complex<i16>, then it's directly binary-compatible
            // - If we have Complex<i32>, it's directly binary-compatible, with the caveat that
            //   no size checking is done to make sure the i32 are actually i24.
            self.writer.write_all(&*raw_slice)?;
        };

        Ok(())
    }

    /// Read samples, denormalizing them. See [SampleNormalizedConvert].
    pub fn write_all_samples_denorm<T: SampleNormalizedConvert>(
        &mut self,
        samples: &[Complex<T>],
    ) -> Result<(), WriteError> {
        for &samp in samples {
            match self.header.samp_size {
                16 => self.write_next_sample_16bit_denorm(samp)?,
                24 => self.write_next_sample_24bit_denorm(samp)?,
                _ => unreachable!(),
            }
        }

        Ok(())
    }

    /// Read samples, with possible conversion if type `Complex<T>` does not match the binary type
    /// in the sdriq file. This could have slight overhead.
    /// If the binary type is too small to hold the values, they are clamped.
    ///
    /// Note that this function will choose the most optimized possible call for the type "T", as explained
    /// in [Sink::get_samples].
    ///
    /// Returns the number of samples written.
    pub fn write_all_samples_clamp<T: SampleConvert>(
        &mut self,
        samples: &[Complex<T>],
    ) -> Result<(), WriteError> {
        for &samp in samples {
            match self.header.samp_size {
                16 => self.write_next_sample_16bit_clamp(samp)?,
                24 => self.write_next_sample_24bit_clamp(samp)?,
                _ => unreachable!(),
            }
        }

        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use std::fs::File;

    use super::*;
    use num::Signed;

    #[test]
    fn test_conversion_16bit() {
        // All values of i16 are fully representable by i16, i32, f32 and f64
        for i in i16::MIN..=i16::MAX {
            assert_eq!(i, i16::from_i16(i).unwrap());
            assert_eq!(i as i32, i32::from_i16(i).unwrap());
            assert_eq!(i as f32, f32::from_i16(i).unwrap());
            assert_eq!(i as f64, f64::from_i16(i).unwrap());
            assert_eq!(i as f32, f32::from_i16(i).unwrap());
            assert_eq!(i as f64, f64::from_i16(i).unwrap());
        }
    }

    #[test]
    fn test_conversion_24bit() {
        // All 16 bit values are fully representable by i16, i32, f32 and f64
        for i in i16::MIN..=i16::MAX {
            let i = i as i32;
            assert_eq!(i as i16, i16::from_i24(i).unwrap());
            assert_eq!(i, i32::from_i24(i).unwrap());
            assert_eq!(i as f32, f32::from_i24(i).unwrap());
            assert_eq!(i as f64, f64::from_i24(i).unwrap());
            assert_eq!(i as f32, f32::from_i24(i).unwrap());
            assert_eq!(i as f64, f64::from_i24(i).unwrap());
        }

        // Not all 24 bit values are representable by i16, but they are correctly clipped
        for i in I24_MIN..=I24_MAX {
            let i_clamp = i16::from_i24_clamp(i);
            if i_clamp as i32 != i {
                assert!(i16::from_i24(i).is_none());
                if i < 0 {
                    assert!(i_clamp == i16::MIN);
                } else {
                    assert!(i_clamp == i16::MAX);
                }
            }
        }

        // All 24 bit values are fully representable by i32, f32 and f64
        for i in I24_MIN..=I24_MAX {
            assert_eq!(i, i32::from_i24(i).unwrap());
            assert_eq!(i as f32, f32::from_i24(i).unwrap());
            assert_eq!(i as f64, f64::from_i24(i).unwrap());
            assert_eq!(i as f32, f32::from_i24(i).unwrap());
            assert_eq!(i as f64, f64::from_i24(i).unwrap());
        }
    }

    #[test]
    fn is_24bit_checks() {
        assert!(is_24bit(0));
        assert!(is_24bit(1));
        assert!(is_24bit(-1));
        assert!(is_24bit(I24_MIN));
        assert!(is_24bit(I24_MAX));
        assert!(!is_24bit(I24_MIN - 1));
        assert!(!is_24bit(I24_MAX + 1));
        assert!(!is_24bit(I24_MIN - 1000));
        assert!(!is_24bit(I24_MAX + 1000));
        assert!(!is_24bit(i32::MIN));
        assert!(!is_24bit(i32::MAX));
    }

    #[test]
    fn testsignal_constant_header() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let source = Source::new(file).unwrap();
        let header = source.get_header();
        assert_eq!(header.samp_rate, 768000);
        assert_eq!(header.center_freq, 123456000);
        assert_eq!(header.start_timestamp, 1759607614307);
        assert_eq!(header.samp_size, 24);
    }

    #[test]
    fn testsignal_constant_get_samples() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<i32>; 8] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 8);

        for v in samples {
            assert_eq!(v, Complex::<i32> { re: 65536, im: 0 });
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    fn testsignal_constant_partial_read() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<i32>; 4] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 4);

        for v in samples {
            assert_eq!(v, Complex::<i32> { re: 65536, im: 0 });
        }

        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 4);

        for v in samples {
            assert_eq!(v, Complex::<i32> { re: 65536, im: 0 });
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    fn testsignal_constant_excessive_read() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<i32>; 10] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 8);

        for &v in samples[0..num].iter() {
            assert_eq!(v, Complex::<i32> { re: 65536, im: 0 });
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    fn testsignal_constant_f32() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let mut samples: [Complex<f32>; 10] = Default::default();

        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 8);

        let expected = Complex::<f32> {
            re: 65536.0,
            im: 0.0,
        };

        for &v in samples[0..num].iter() {
            assert_eq!(v, expected);
        }
    }

    #[test]
    fn testsignal_constant_f32_norm() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let mut samples: [Complex<f32>; 10] = Default::default();

        let num = source.get_samples_norm(&mut samples).unwrap();
        assert_eq!(num, 8);

        let expected = Complex::<f32> {
            re: 65536.0 / 8388608.0,
            im: 0.0,
        };

        for &v in samples[0..num].iter() {
            assert_eq!(v, expected);
        }

        // EOF
        let num = source.get_samples_norm(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    #[should_panic]
    fn testsignal_constant_i16_fail() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let mut samples: [Complex<i16>; 10] = Default::default();

        // This will explode because 65536 > i16::MAX
        source.get_samples(&mut samples).unwrap();
    }

    #[test]
    fn testsignal_constant_clamp() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let mut samples: [Complex<i16>; 10] = Default::default();

        let num = source.get_samples_clamp(&mut samples).unwrap();
        assert_eq!(num, 8);

        let expected = Complex::<i16> {
            re: i16::MAX,
            im: 0,
        };

        for &v in samples[0..num].iter() {
            assert_eq!(v, expected);
        }

        // EOF
        let num = source.get_samples_clamp(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[derive(Default, Copy, Clone, PartialEq, Eq, Debug)]
    struct CustomI32 {
        v: i32,
    }

    impl SampleConvert for CustomI32 {
        fn from_i16(value: i16) -> Option<Self> {
            Some(Self { v: value as i32 })
        }

        fn from_i24(value: i32) -> Option<Self> {
            Some(Self { v: value })
        }

        fn from_i16_clamp(value: i16) -> Self {
            Self { v: value as i32 }
        }

        fn from_i24_clamp(value: i32) -> Self {
            Self { v: value }
        }

        fn to_i16(self) -> Option<i16> {
            i32::to_i16(self.v)
        }

        fn to_i24(self) -> Option<i32> {
            i32::to_i24(self.v)
        }

        fn to_i16_clamp(self) -> i16 {
            i32::to_i16_clamp(self.v)
        }

        fn to_i24_clamp(self) -> i32 {
            i32::to_i24_clamp(self.v)
        }
    }

    #[test]
    fn testsignal_constant_custom_type_get_samples() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<CustomI32>; 10] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 8);

        let expected = Complex::<CustomI32> {
            re: CustomI32 { v: 65536 },
            im: CustomI32 { v: 0 },
        };

        for &v in samples[0..num].iter() {
            assert_eq!(v, expected);
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    fn testsignal_constant_custom_type_partial_read() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<CustomI32>; 4] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 4);

        let expected = Complex::<CustomI32> {
            re: CustomI32 { v: 65536 },
            im: CustomI32 { v: 0 },
        };

        for v in samples {
            assert_eq!(v, expected);
        }

        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 4);

        for v in samples {
            assert_eq!(v, expected);
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    #[test]
    fn testsignal_constant_custom_type_excessive_read() {
        let file = File::open("test_files/constant.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();

        let mut samples: [Complex<CustomI32>; 10] = Default::default();
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 8);

        let expected = Complex::<CustomI32> {
            re: CustomI32 { v: 65536 },
            im: CustomI32 { v: 0 },
        };

        for &v in samples[0..num].iter() {
            assert_eq!(v, expected);
        }

        // Try to read past EOF, it should write no samples
        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 0);
    }

    fn check_pos_and_neg<T: Signed>(samples: &[Complex<T>]) {
        let mut has_positive_re = false;
        let mut has_negative_re = false;
        let mut has_positive_im = false;
        let mut has_negative_im = false;
        for v in samples {
            if v.re.is_positive() {
                has_positive_re = true;
            }
            if v.re.is_negative() {
                has_negative_re = true;
            }
            if v.im.is_positive() {
                has_positive_im = true;
            }
            if v.im.is_negative() {
                has_negative_im = true;
            }
        }

        assert!(has_positive_re);
        assert!(has_negative_re);
        assert!(has_positive_im);
        assert!(has_negative_im);
    }

    #[test]
    fn testsignal_offset_tone() {
        let file = File::open("test_files/offset_tone.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let zero = Complex::<i32> { re: 0, im: 0 };
        let mut samples: [Complex<i32>; 60] = [zero; 60];

        let num = source.get_samples(&mut samples).unwrap();
        assert_eq!(num, 60);

        // We expect positive and negative in both real and imaginary
        // as the signal is positive frequency
        check_pos_and_neg(&samples);
    }

    #[test]
    fn testsignal_offset_tone_norm() {
        let file = File::open("test_files/offset_tone.sdriq").unwrap();
        let mut source = Source::new(file).unwrap();
        let zero = Complex::<f32> { re: 0.0, im: 0.0 };
        let mut samples: [Complex<f32>; 60] = [zero; 60];

        let num = source.get_samples_norm(&mut samples).unwrap();
        assert_eq!(num, 60);

        // We expect positive and negative in both real and imaginary
        // as the signal is positive frequency
        check_pos_and_neg(&samples);

        for &sample in samples.iter() {
            assert!(sample.norm() < 1.0);
        }
    }

    #[test]
    #[should_panic]
    fn header_not_verify_crc() {
        let file = File::open("test_files/malformed.sdriq").unwrap();
        let _ = Source::new(file).unwrap();
    }

    #[test]
    fn header_write_and_read() {
        let header = Header {
            samp_rate: 12345,
            center_freq: 67890,
            start_timestamp: 12345,
            samp_size: 24,
        };

        let mut buffer: Vec<u8> = Vec::new();
        header.write_to(&mut buffer).unwrap();

        assert_eq!(buffer.len(), 32);

        let mut as_slice = buffer.as_slice();
        let read_header = Header::read_from(&mut as_slice).unwrap();

        assert_eq!(read_header, header);
    }
}
