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

        let id = stream.parameters().id();
        let codec_id = id.name().to_string();
        // Id tidak punya .description() lagi di ffmpeg-next 7.x — harus cari
        // Codec-nya dulu lewat decoder::find() (Option, bukan Result).
        let codec = ffmpeg_next::codec::decoder::find(id)
            .map(|c| c.description().to_string())
            .unwrap_or_else(|| "Tidak diketahui".to_string());

        let fps = stream.avg_frame_rate();
        let fps = if fps.denominator() > 0 {
            fps.numerator() as f64 / fps.denominator() as f64
        } else { 0.0 };

        let duration = stream.duration() as f64 * f64::from(stream.time_base());
        let bitrate = input.bit_rate() as i64;

        let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;

        let (width, height) = match ctx.decoder().video() {
            Ok(vdec) => (vdec.width(), vdec.height()),
            Err(_) => (0, 0),
        };

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

// unsendable: SwsContext (dipakai lewat scaler cache) berisi raw pointer
// (*mut SwsContext) yang gak implement Send. Karena PyO3 pyclass secara
// default butuh Send, VideoDecoder harus ditandai unsendable — artinya
// instance-nya cuma boleh diakses dari thread Python tempat dia dibikin
// (biasanya main thread), gak bisa dipindah-pindah antar thread.
#[pyclass(unsendable)]
struct VideoDecoder {
    input_ctx: ffmpeg_next::format::context::Input,
    stream_idx: usize,
    decoder: ffmpeg_next::decoder::Video,
    // 🧠 Cache konteks scaling — dibuat sekali, dipakai ulang tiap seek_frame()
    scaler: Option<ffmpeg_next::software::scaling::Context>,
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
        let input_ctx = ffmpeg_next::format::input(&path)
            .map_err(|e| pyo3::exceptions::PyIOError::new_err(format!("Buka file: {}", e)))?;

        // streams().best() balikin Stream langsung (bukan tuple), index-nya
        // diambil lewat .index() setelahnya.
        let stream = input_ctx.streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| pyo3::exceptions::PyValueError::new_err("Tidak ada aliran video"))?;
        let stream_idx = stream.index();

        let tb = stream.time_base();
        let time_base = if tb.denominator() > 0 {
            tb.numerator() as f64 / tb.denominator() as f64
        } else { 0.0 };

        let duration = stream.duration() as f64 * time_base;
        let codec = stream.parameters().id().name().to_string();

        let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;

        // Catatan: dulu ada override manual buat pilih dekoder VP9 lewat
        // decoder.set_codec(), tapi method itu udah gak ada di API 7.x.
        // Gak masalah — ctx.decoder().video() otomatis pakai codec yang
        // bener berdasarkan codecpar dari stream, jadi override manual
        // ini emang udah gak perlu lagi.
        let decoder = ctx.decoder().video()
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Buat dekoder: {}", e)))?;

        let (width, height) = (decoder.width(), decoder.height());

        Ok(VideoDecoder {
            input_ctx,
            stream_idx,
            decoder,
            scaler: None,
            time_base,
            duration,
            width,
            height,
            codec,
        })
    }

    /// Lompat ke detik tertentu → kembalikan bingkai RGB sebagai np.array [H, W, 3]
    fn seek_frame(&mut self, py: Python<'_>, second: f64) -> PyResult<Py<PyArray3<u8>>> {
        let target_ts = (second / self.time_base).round() as i64;

        // Lompat ke belakang sedikit supaya nemu keyframe terdekat sebelum target,
        // lalu baca maju sampai lewati target_ts.
        self.input_ctx
            .seek(target_ts, ..target_ts)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Lompat gagal: {}", e)))?;

        // Buang state dekoder lama (penting setelah seek, biar gak nyampur GOP lama)
        self.decoder.flush();

        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut packet_count = 0;
        let max_try = 200;
        let mut got_frame = false;

        // Input gak punya method .read() langsung — cara baca paket di API
        // 7.x adalah lewat iterator .packets(), yang menghasilkan pasangan
        // (Stream, Packet). Index stream diambil dari situ, bukan dari
        // Packet.stream_index() (yang juga udah gak ada).
        // Ini adalah &mut borrow ke field input_ctx saja (bukan seluruh
        // self), jadi self.decoder masih bisa diakses lepas di dalam loop.
        'read_loop: for (stream, packet) in self.input_ctx.packets() {
            if packet_count > max_try {
                return Err(pyo3::exceptions::PyRuntimeError::new_err("Bingkai tidak ditemukan"));
            }

            if stream.index() != self.stream_idx {
                packet_count += 1;
                continue;
            }

            self.decoder.send_packet(&packet)
                .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Kirim paket gagal: {}", e)))?;

            // Satu paket bisa hasilkan >1 bingkai; ambil yang terakhir
            // yang tersedia sebelum lanjut baca paket berikutnya.
            while self.decoder.receive_frame(&mut decoded).is_ok() {
                got_frame = true;
                if decoded.timestamp().unwrap_or(0) >= target_ts {
                    break 'read_loop;
                }
            }

            packet_count += 1;
        }

        if !got_frame {
            return Err(pyo3::exceptions::PyRuntimeError::new_err("Tidak dapat membaca bingkai"));
        }

        // Bikin konteks scaling sekali saja lalu di-cache; dipakai ulang tiap panggilan.
        let width = self.width;
        let height = self.height;
        let src_format = self.decoder.format();
        let scaler = match &mut self.scaler {
            Some(s) => s,
            None => {
                let ctx = ffmpeg_next::software::scaling::Context::get(
                    src_format,
                    width,
                    height,
                    ffmpeg_next::util::format::Pixel::RGB24,
                    width,
                    height,
                    ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
                ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Buat konteks scaling: {}", e)))?;
                self.scaler.insert(ctx)
            }
        };

        let mut rgb_frame = ffmpeg_next::frame::Video::new(
            ffmpeg_next::util::format::Pixel::RGB24,
            width,
            height,
        );

        scaler.run(&decoded, &mut rgb_frame)
            .map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Konversi warna gagal: {}", e)))?;

        let data = rgb_frame.data(0).to_vec();
        let arr = Array3::from_shape_vec(
            (height as usize, width as usize, 3),
            data,
        ).map_err(|e| pyo3::exceptions::PyRuntimeError::new_err(format!("Array gagal: {}", e)))?;

        Ok(arr.into_pyarray(py).into())
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
