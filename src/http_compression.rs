use std::io::{Error, ErrorKind, Result};

use async_compression::codecs::{BrotliDecoder, DecodeV2, DeflateDecoder, GzipDecoder};
use async_compression::core::util::{PartialBuffer, WriteBuffer};
use bytes::Bytes;

#[derive(Copy, Clone, Debug, PartialEq, Eq)]
pub enum Algorithm {
    Brotli,
    Gzip,
    Deflate,
}

const MAX_DECOMPRESSED_BYTES: usize = 100 * 1024 * 1024;
const SCRATCH_BUF_SIZE: usize = 64 * 1024;

pub async fn decompress(bytes: Bytes, algorithm: Algorithm) -> Result<Bytes> {
    tokio::task::spawn_blocking(move || match algorithm {
        Algorithm::Brotli => {
            let mut decoder = BrotliDecoder::new();
            decode_sync(&mut decoder, &bytes).map(Bytes::from)
        }
        Algorithm::Gzip => {
            let mut decoder = GzipDecoder::new();
            decode_sync(&mut decoder, &bytes).map(Bytes::from)
        }
        Algorithm::Deflate => {
            let mut decoder = DeflateDecoder::new();
            decode_sync(&mut decoder, &bytes).map(Bytes::from)
        }
    })
    .await
    .map_err(|err| Error::other(format!("Decompression task join failed: {}", err)))?
}

fn decode_sync(decoder: &mut dyn DecodeV2, input_bytes: &[u8]) -> Result<Vec<u8>> {
    let mut input = PartialBuffer::new(input_bytes);
    let initial_cap = input_bytes.len().saturating_mul(2).min(16 * 1024 * 1024);
    let mut out = Vec::with_capacity(initial_cap);
    let mut scratch = vec![0u8; SCRATCH_BUF_SIZE];

    loop {
        let mut wb = WriteBuffer::new_initialized(&mut scratch);
        let done = decoder.decode(&mut input, &mut wb)?;
        let written = wb.written();
        if out.len().saturating_add(written.len()) > MAX_DECOMPRESSED_BYTES {
            return Err(Error::new(
                ErrorKind::InvalidData,
                "Decompressed response exceeds 100 MiB safety limit",
            ));
        }
        out.extend_from_slice(written);

        if done {
            loop {
                let mut wb = WriteBuffer::new_initialized(&mut scratch);
                let finished = decoder.finish(&mut wb)?;
                let written = wb.written();
                if out.len().saturating_add(written.len()) > MAX_DECOMPRESSED_BYTES {
                    return Err(Error::new(
                        ErrorKind::InvalidData,
                        "Decompressed response exceeds 100 MiB safety limit",
                    ));
                }
                out.extend_from_slice(written);

                if finished {
                    return Ok(out);
                }
                if written.is_empty() {
                    return Err(Error::new(
                        ErrorKind::UnexpectedEof,
                        "Compressed stream did not finish cleanly",
                    ));
                }
            }
        }

        if input.unwritten().is_empty() && written.is_empty() {
            return Err(Error::new(
                ErrorKind::UnexpectedEof,
                "Truncated compressed stream",
            ));
        }
    }
}
