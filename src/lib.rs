use pyo3::prelude::*;
use numpy::{PyArray3, IntoPyArray, ndarray::Array3};
use std::path::Path;

// ═══════════════════════════════════════════════
// INISIALISASI FFmpeg (sekali saja saat dimuat)
// ═══════════════════════════════════════════════

fn init_ffmpeg() -> PyResult<()> {
    ffmpeg_next::init().map_err(|e| {
        pyo3::exceptions::PyRuntimeError::new_err(format!("Inisialisasi FFmpeg gagal: {}", e))
    })?;
    Ok(())
}

// ═══════════════════════════════════════════════
// BAGIAN 1: INFORMASI MEDIA
// ═══════════════════════════════════════════════

#[pyclass]
struct MediaInfo {
    #[pyo3(get)] path: String,
    #[pyo3(get)] duration: f64,
    #[pyo3(get)] width: u32,
    #[pyo3(get)] height: u32,
    #[pyo3(get)] fps: f64,
    #[pyo3(get)] codec: String,
    #[pyo3(get)] codec_id: String,
    #[pyo3(get)] bitrate: i64,
}

#[pymethods]
impl MediaInfo {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        init_ffmpeg()?;

        let path = Path::new(file_path);
        let input = ffmpeg_next::format::input(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Buka file: {}", e)))?;

        let stream = input.streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tidak ada aliran video"))?;

        let codec_id = stream.parameters().id().name().to_string();
        let codec = stream.parameters().id().description().unwrap_or("Tidak diketahui").to_string();

        let fps = stream.avg_frame_rate();
        let fps = if fps.denominator() > 0 {
            fps.numerator() as f64 / fps.denominator() as f64
        } else { 0.0 };

        let duration = stream.duration() as f64 * f64::from(stream.time_base());
        let bitrate = input.bit_rate() as i64;

        let decoder = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;

        let (width, height) = if let Some(vdec) = decoder.decoder().video() {
            (vdec.width(), vdec.height())
        } else { (0, 0) };

        Ok(MediaInfo {
            path: file_path.to_string(),
            duration,
            width,
            height,
            fps,
            codec,
            codec_id,
            bitrate,
        })
    }

    fn __repr__(&self) -> String {
        format!(
            "MediaInfo(dur={:.2}s, {}x{} @{:.2}fps, {}, {} bitrate={}bps)",
            self.duration, self.width, self.height, self.fps, self.codec_id, self.codec, self.bitrate
        )
    }
}

// ═══════════════════════════════════════════════
// BAGIAN 2: DEKODER BINGKAI VIDEO
// ═══════════════════════════════════════════════

#[pyclass]
struct VideoDecoder {
    input_ctx: ffmpeg_next::format::context::Input,
    stream_idx: usize,
    decoder: ffmpeg_next::codec::context::decoder::Video,
    time_base: f64,
    #[pyo3(get)] duration: f64,
    #[pyo3(get)] width: u32,
    #[pyo3(get)] height: u32,
    #[pyo3(get)] codec: String,
}

#[pymethods]
impl VideoDecoder {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        init_ffmpeg()?;

        let path = Path::new(file_path);
        let mut input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Buka file: {}", e)))?;

        let (stream_idx, stream) = input_ctx.streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tidak ada aliran video"))?
            .into();

        let tb = stream.time_base();
        let time_base = if tb.denominator() > 0 {
            tb.numerator() as f64 / tb.denominator() as f64
        } else { 0.0 };

        let duration = stream.duration() as f64 * time_base;
        let codec = stream.parameters().id().name().to_string();

        let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;

        let mut decoder = ctx.decoder().video()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Buat dekoder: {}", e)))?;

        // 🔑 Otomatis pilih dekoder VP9 jika perlu
        if codec == "vp9" {
            if let Ok(vp9_decoder) = ffmpeg_next::codec::decoder::find(ffmpeg_next::codec::Id::VP9) {
                decoder.set_codec(vp9_decoder);
            }
        }

        let (width, height) = (decoder.width(), decoder.height());

        Ok(VideoDecoder {
            input_ctx,
            stream_idx,
            decoder,
            time_base,
            duration,
            width,
            height,
            codec,
        })
    }

    /// Lompat ke detik tertentu → kembalikan bingkai RGB sebagai np.array [H, W, 3]
    fn seek_frame(&mut self, second: f64) -> PyResult<Py<PyArray3<u8>>> {
        let target_ts = (second / self.time_base).round() as i64;

        self.input_ctx.seek(target_ts..target_ts + 10)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lompat gagal: {}", e)))?;

        let mut pkt = ffmpeg_next::packet::Packet::empty();
        let mut frame_count = 0;
        let max_try = 50;

        loop {
            if frame_count > max_try {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Bingkai tidak ditemukan"));
            }

            match self.input_ctx.read(&mut pkt) {
                Ok(_) => {
                    if pkt.stream_index() != self.stream_idx {
                        frame_count += 1;
                        continue;
                    }

                    pkt.decode(&mut self.decoder)
                        .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Dekode gagal: {}", e)))?;

                    if let Ok(frame) = self.decoder.frame() {
                        // Konversi ke RGB24
                        let mut converter = ffmpeg_next::software::scaling::Context::get(
                            ffmpeg_next::util::format::Pixel::RGB24,
                            self.width,
                            self.height,
                            ffmpeg_next::util::format::Pixel::RGB24,
                        );

                        let mut rgb_frame = ffmpeg_next::frame::Video::new(
                            ffmpeg_next::util::format::Pixel::RGB24,
                            self.width,
                            self.height,
                        );

                        converter.run(&frame, &mut rgb_frame)
                            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konversi warna gagal: {}", e)))?;

                        let data = rgb_frame.data(0).to_vec();
                        let arr = Array3::from_shape_vec(
                            (self.height as usize, self.width as usize, 3),
                            data,
                        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Array gagal: {}", e)))?;

                        return Python::with_gil(|py| {
                            Ok(arr.into_pyarray_bound(py).into())
                        });
                    }
                }
                Err(ffmpeg_next::Error::Eof) => break,
                Err(e) => return Err(pyo3::exceptions::PyRuntimeError::new_err(format!("Baca gagal: {}", e))),
            }
            frame_count += 1;
        }

        Err(pyo3::exceptions::PyRuntimeError::new_err("Tidak dapat membaca bingkai"))
    }
}

// ═══════════════════════════════════════════════
// DAFTARKAN KE MODUL
// ═══════════════════════════════════════════════

#[pymodule]
fn media_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MediaInfo>()?;
    m.add_class::<VideoDecoder>()?;
    Ok(())
}