use std::{
    any::TypeId,
    io::{BufRead, BufReader, BufWriter, Read, Write},
};
use thiserror::Error;

use num_complex::Complex;

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
pub enum WriteError {}

#[derive(Copy, Clone, PartialEq)]
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

    /// Returns the number of
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

pub trait SampleConvert: Copy + Sized {
    fn from_i16(value: i16) -> Option<Self>;
    fn from_i24(value: i32) -> Option<Self>;
    fn from_i16_clamp(value: i16) -> Self;
    fn from_i24_clamp(value: i32) -> Self;
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
}

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
}

const I16_SCALE_F32: f32 = 1.0 / 32768.0;
const I32_SCALE_F32: f32 = 1.0 / 8388608.0;
const I16_SCALE_F64: f64 = 1.0 / 32768.0;
const I32_SCALE_F64: f64 = 1.0 / 8388608.0;

/// For 16 bit numbers, maps [-32768, 32767] -> [-1, 0.9999694824]
/// For 24 bit numbers, maps [-8388608, 8388607] -> [-1, 0.9999998808]
/// This guarantees 0 is mapped to 0. The error introduced is insignificant for almost any use, but be aware that
/// the returned values will always be contained in the interval [-1, 1).
pub trait SampleNormalizedConvert: Copy + Sized {
    fn from_i16_norm(value: i16) -> Self;
    fn from_i24_norm(value: i32) -> Self;
}

impl SampleNormalizedConvert for f32 {
    fn from_i16_norm(value: i16) -> Self {
        (value as f32) * I16_SCALE_F32
    }

    fn from_i24_norm(value: i32) -> Self {
        // This would result in values outside [-1, 1)
        debug_assert!(is_24bit(value), "value was not 24 bit");
        (value as f32) * I32_SCALE_F32
    }
}

impl SampleNormalizedConvert for f64 {
    fn from_i16_norm(value: i16) -> Self {
        (value as f64) * I16_SCALE_F64
    }

    fn from_i24_norm(value: i32) -> Self {
        // This would result in values outside [-1, 1)
        debug_assert!(is_24bit(value), "value was not 24 bit");
        (value as f64) * I32_SCALE_F64
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
    pub fn new(source: R) -> Result<Self, ReadError> {
        // Default buffer size holds a bit over 1sec of 1Msps 16 bit samples, rounded to nearest power of 2
        Self::with_capacity(2097152, source)
    }

    /// Creates a sample reader, expecting to find a valid header on the first few bytes of the Read argument.
    pub fn with_capacity(bytes: usize, source: R) -> Result<Self, ReadError> {
        let mut reader = BufReader::with_capacity(bytes, source);
        let header = Header::new(&mut reader)?;
        Ok(Self { header, reader })
    }

    /// Returns the header applicable to this Source
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

    /// Reads samples, as long as they are directly copyable to Complex<T> type without any
    /// conversion being performed. This is fast, as it's a simple, flat memory copy.
    ///
    /// WARNING: On big endian systems, the driect copy is not done, and instead "slow" loading is performed
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

    /// Read samples, with possible conversion if type Complex<T> does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// If the conversion would result in truncation (for example, 24 bits -> i16), an error is generated.
    ///
    /// No truncation errors are ever generated is T is a floating point number, as 24 bit integers can fully
    /// fit into f32.
    /// Note that the resulting floating point numbers are NOT normalized!
    ///
    /// Note that this function will choose the most optimized possible call for the type "T"
    ///
    /// Returns the number of samples read.
    pub fn get_samples<T: SampleConvert + 'static>(
        &mut self,
        target: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        // Try optimized functions if possible
        let header_bits = self.header.get_stored_bits_per_sample();
        if TypeId::of::<T>() == TypeId::of::<i32>() && header_bits == 32 {
            return self.get_samples_direct(target);
        }
        if TypeId::of::<T>() == TypeId::of::<i16>() && header_bits == 16 {
            return self.get_samples_direct(target);
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

    /// Read samples, with possible conversion if type Complex<T> does not match the incoming type
    /// from the sdriq file. This could have slight overhead.
    /// If T is too small (i16) and the incoming number if bigger, it will be clamped.
    ///
    /// Note that this function will choose the most optimized possible call for the type "T"
    ///
    /// Returns the number of samples read.
    pub fn get_samples_clamp<T: SampleConvert + 'static>(
        &mut self,
        target: &mut [Complex<T>],
    ) -> Result<usize, ReadError> {
        // Try optimized functions if possible
        let header_bits = self.header.get_stored_bits_per_sample();
        if TypeId::of::<T>() == TypeId::of::<i32>() && header_bits == 32 {
            return self.get_samples_direct(target);
        }
        if TypeId::of::<T>() == TypeId::of::<i16>() && header_bits == 16 {
            return self.get_samples_direct(target);
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

    /// Read samples, normalizing them to [-1, 1] range, so T must be a floating point type (f32 or f64).
    /// The original range is deduced from the bit-size of the samples, so
    /// - 16 bit samples map [-32768, 32767] -> [-1, 1]
    /// - 24 bit samples map [-16777216, 16777215] -> [-1, 1]
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

/// A Sink wraps around a Write to be able to write samples to an sdriq file. Samples are written by
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
    use num::Signed;
    use num_complex::ComplexFloat;

    const I24_MIN: i32 = -8388608;
    const I24_MAX: i32 = 8388607;

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
            Self { v: value as i32 }
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
}
