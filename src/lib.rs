use pyo3::prelude::*;
use pyo3::exceptions::{PyIOError, PyRuntimeError, PyValueError};
use numpy::{IntoPyArray, PyArray3, ndarray::Array3};
use std::collections::VecDeque;
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicI64, AtomicU64, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::{self, JoinHandle};
use std::time::{Duration, Instant};

use cpal::traits::{DeviceTrait, HostTrait, StreamTrait};

// ═══════════════════════════════════════════════
// INISIALISASI FFmpeg (sekali saja saat dimuat)
// ═══════════════════════════════════════════════

fn init_ffmpeg() -> PyResult<()> {
    ffmpeg_next::init().map_err(|e| {
        PyRuntimeError::new_err(format!("Inisialisasi FFmpeg gagal: {}", e))
    })?;
    Ok(())
}

// Input::seek() di ffmpeg-next dipanggil dengan stream_index = -1 di
// belakang layar (avformat_seek_file). Kalau stream_index = -1, FFmpeg
// nganggep timestamp yang dikasih itu dalam satuan AV_TIME_BASE
// (mikrodetik / 1_000_000), BUKAN time_base stream video. Ini beda sama
// yang dipakai di VideoDecoder lama (yang salah pakai time_base stream) —
// gak dibenerin di situ biar VideoDecoder lama tetep stabil buat thumbnail,
// tapi PlayerEngine di bawah pakai yang bener.
fn av_time_base_ts(seconds: f64) -> i64 {
    (seconds * 1_000_000.0) as i64
}

// ═══════════════════════════════════════════════
// BAGIAN 1: INFORMASI MEDIA (gak diubah)
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
            .map_err(|e| PyIOError::new_err(format!("Buka file: {}", e)))?;

        let stream = input.streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| PyValueError::new_err("Tidak ada aliran video"))?;

        let id = stream.parameters().id();
        let codec_id = id.name().to_string();
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
            .map_err(|e| PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;

        let (width, height) = match ctx.decoder().video() {
            Ok(vdec) => (vdec.width(), vdec.height()),
            Err(_) => (0, 0),
        };

        Ok(MediaInfo {
            path: file_path.to_string(),
            duration, width, height, fps, codec, codec_id, bitrate,
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
// BAGIAN 2: DEKODER BINGKAI TUNGGAL (dipertahankan apa adanya
// untuk kebutuhan seek-frame / thumbnail. Player beneran pakai
// PlayerEngine di bawah.)
// ═══════════════════════════════════════════════

#[pyclass(unsendable)]
struct VideoDecoder {
    input_ctx: ffmpeg_next::format::context::Input,
    stream_idx: usize,
    decoder: ffmpeg_next::decoder::Video,
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
            .map_err(|e| PyIOError::new_err(format!("Buka file: {}", e)))?;

        let stream = input_ctx.streams()
            .best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| PyValueError::new_err("Tidak ada aliran video"))?;
        let stream_idx = stream.index();

        let tb = stream.time_base();
        let time_base = if tb.denominator() > 0 {
            tb.numerator() as f64 / tb.denominator() as f64
        } else { 0.0 };

        let duration = stream.duration() as f64 * time_base;
        let codec = stream.parameters().id().name().to_string();

        let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters())
            .map_err(|e| PyRuntimeError::new_err(format!("Konteks dekoder: {}", e)))?;
        let decoder = ctx.decoder().video()
            .map_err(|e| PyRuntimeError::new_err(format!("Buat dekoder: {}", e)))?;

        let (width, height) = (decoder.width(), decoder.height());

        Ok(VideoDecoder {
            input_ctx, stream_idx, decoder, scaler: None, time_base, duration, width, height, codec,
        })
    }

    fn seek_frame(&mut self, py: Python<'_>, second: f64) -> PyResult<Py<PyArray3<u8>>> {
        let target_ts = (second / self.time_base).round() as i64;
        self.input_ctx.seek(target_ts, ..target_ts)
            .map_err(|e| PyRuntimeError::new_err(format!("Lompat gagal: {}", e)))?;
        self.decoder.flush();

        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut packet_count = 0;
        let max_try = 200;
        let mut got_frame = false;

        'read_loop: for (stream, packet) in self.input_ctx.packets() {
            if packet_count > max_try {
                return Err(PyRuntimeError::new_err("Bingkai tidak ditemukan"));
            }
            if stream.index() != self.stream_idx {
                packet_count += 1;
                continue;
            }
            self.decoder.send_packet(&packet)
                .map_err(|e| PyRuntimeError::new_err(format!("Kirim paket gagal: {}", e)))?;
            while self.decoder.receive_frame(&mut decoded).is_ok() {
                got_frame = true;
                if decoded.timestamp().unwrap_or(0) >= target_ts {
                    break 'read_loop;
                }
            }
            packet_count += 1;
        }

        if !got_frame {
            return Err(PyRuntimeError::new_err("Tidak dapat membaca bingkai"));
        }

        rgb_from_decoded(py, &decoded, &mut self.scaler, self.decoder.format(), self.width, self.height)
    }

    fn read_next_frame(&mut self, py: Python<'_>) -> PyResult<Py<PyArray3<u8>>> {
        let mut decoded = ffmpeg_next::frame::Video::empty();
        let mut got_frame = false;

        for (stream, packet) in self.input_ctx.packets() {
            if stream.index() != self.stream_idx { continue; }
            self.decoder.send_packet(&packet)
                .map_err(|e| PyRuntimeError::new_err(format!("Kirim paket: {}", e)))?;
            if self.decoder.receive_frame(&mut decoded).is_ok() {
                got_frame = true;
                break;
            }
        }

        if !got_frame {
            return Err(PyRuntimeError::new_err("Akhir video"));
        }

        rgb_from_decoded(py, &decoded, &mut self.scaler, self.decoder.format(), self.width, self.height)
    }
}

/// Helper bareng: konversi 1 frame YUV/dst hasil decode -> np.array RGB [H,W,3].
/// Dipisah dari VideoDecoder biar gak duplikat logic scaler+stride-strip.
fn rgb_from_decoded(
    py: Python<'_>,
    decoded: &ffmpeg_next::frame::Video,
    scaler_slot: &mut Option<ffmpeg_next::software::scaling::Context>,
    src_format: ffmpeg_next::util::format::Pixel,
    width: u32,
    height: u32,
) -> PyResult<Py<PyArray3<u8>>> {
    let scaler = match scaler_slot {
        Some(s) => s,
        None => {
            let ctx = ffmpeg_next::software::scaling::Context::get(
                src_format, width, height,
                ffmpeg_next::util::format::Pixel::RGB24, width, height,
                ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
            ).map_err(|e| PyRuntimeError::new_err(format!("Buat konteks scaling: {}", e)))?;
            scaler_slot.insert(ctx)
        }
    };

    let mut rgb_frame = ffmpeg_next::frame::Video::new(
        ffmpeg_next::util::format::Pixel::RGB24, width, height,
    );
    scaler.run(decoded, &mut rgb_frame)
        .map_err(|e| PyRuntimeError::new_err(format!("Konversi warna gagal: {}", e)))?;

    let stride = rgb_frame.stride(0);
    let row_width = (width as usize) * 3;
    let mut data = Vec::with_capacity((height as usize) * row_width);
    let raw_data = rgb_frame.data(0);
    for y in 0..(height as usize) {
        let start = y * stride;
        data.extend_from_slice(&raw_data[start..start + row_width]);
    }

    let arr = Array3::from_shape_vec((height as usize, width as usize, 3), data)
        .map_err(|e| PyRuntimeError::new_err(format!("Array gagal: {}", e)))?;
    Ok(arr.into_pyarray(py).into())
}

// ═══════════════════════════════════════════════
// BAGIAN 3: PLAYER ENGINE — decode+audio jalan di thread sendiri,
// audio dipakai sebagai master clock, video di-sync ke clock itu.
// ═══════════════════════════════════════════════

struct QueuedFrame {
    rgb: Vec<u8>,
    pts: f64, // detik
}

/// State yang dishare antara thread decode dan method Python.
/// Semua field pakai atomic/Mutex karena diakses dari 2 thread.
struct Shared {
    video_q: Mutex<VecDeque<QueuedFrame>>,
    audio_buf: Mutex<VecDeque<f32>>, // interleaved, sample rate & channel = device output
    // Jumlah "sample frame" (bukan float individual) yang udah beneran
    // dibunyikan lewat callback cpal -> ini jadi master clock kalau ada audio.
    audio_frames_played: AtomicU64,
    out_sample_rate: AtomicI64,
    out_channels: AtomicI64,
    has_audio: AtomicBool,

    seek_to: Mutex<Option<f64>>,
    playing: AtomicBool,
    stop: AtomicBool,
    eof: AtomicBool,

    // Basis clock: pts pemutaran dimulai dari sini + (waktu berjalan sejak play()).
    // Dipakai kalau video tanpa audio, atau sebelum audio callback mulai jalan.
    clock_base_pts: Mutex<f64>,
    clock_base_wall: Mutex<Option<Instant>>,
}

impl Shared {
    fn position(&self) -> f64 {
        if self.has_audio.load(Ordering::Relaxed) {
            let sr = self.out_sample_rate.load(Ordering::Relaxed).max(1) as f64;
            let base = *self.clock_base_pts.lock().unwrap();
            let frames = self.audio_frames_played.load(Ordering::Relaxed) as f64;
            base + frames / sr
        } else {
            let base = *self.clock_base_pts.lock().unwrap();
            match *self.clock_base_wall.lock().unwrap() {
                Some(t) if self.playing.load(Ordering::Relaxed) => {
                    base + t.elapsed().as_secs_f64()
                }
                _ => base,
            }
        }
    }

    fn reset_clock(&self, pts: f64) {
        *self.clock_base_pts.lock().unwrap() = pts;
        *self.clock_base_wall.lock().unwrap() = Some(Instant::now());
        self.audio_frames_played.store(0, Ordering::Relaxed);
    }
}

const MAX_VIDEO_FRAMES: usize = 90; // ~3 detik @30fps, batas memori readahead
const LATE_FRAME_DROP_SEC: f64 = 0.1; // frame yg telat >100ms dianggap basi

#[pyclass(unsendable)]
struct PlayerEngine {
    shared: Arc<Shared>,
    decode_thread: Option<JoinHandle<()>>,
    _audio_stream: Option<cpal::Stream>, // harus tetep hidup selama playback
    #[pyo3(get)] duration: f64,
    #[pyo3(get)] width: u32,
    #[pyo3(get)] height: u32,
    #[pyo3(get)] fps: f64,
    #[pyo3(get)] has_audio: bool,
}

#[pymethods]
impl PlayerEngine {
    #[new]
    fn new(file_path: &str) -> PyResult<Self> {
        init_ffmpeg()?;

        let path = Path::new(file_path);
        let probe = ffmpeg_next::format::input(&path)
            .map_err(|e| PyIOError::new_err(format!("Buka file: {}", e)))?;

        let vstream = probe.streams().best(ffmpeg_next::media::Type::Video)
            .ok_or_else(|| PyValueError::new_err("Tidak ada aliran video"))?;
        let vtb = vstream.time_base();
        let vtb = if vtb.denominator() > 0 { vtb.numerator() as f64 / vtb.denominator() as f64 } else { 0.0 };
        let duration = vstream.duration() as f64 * vtb;
        let fps = {
            let f = vstream.avg_frame_rate();
            if f.denominator() > 0 { f.numerator() as f64 / f.denominator() as f64 } else { 30.0 }
        };
        let vctx = ffmpeg_next::codec::context::Context::from_parameters(vstream.parameters())
            .map_err(|e| PyRuntimeError::new_err(format!("Konteks dekoder video: {}", e)))?;
        let vdec_probe = vctx.decoder().video()
            .map_err(|e| PyRuntimeError::new_err(format!("Buat dekoder video: {}", e)))?;
        let (width, height) = (vdec_probe.width(), vdec_probe.height());
        drop(vdec_probe);

        let has_audio = probe.streams().best(ffmpeg_next::media::Type::Audio).is_some();
        drop(probe);

        let shared = Arc::new(Shared {
            video_q: Mutex::new(VecDeque::new()),
            audio_buf: Mutex::new(VecDeque::new()),
            audio_frames_played: AtomicU64::new(0),
            out_sample_rate: AtomicI64::new(48000),
            out_channels: AtomicI64::new(2),
            has_audio: AtomicBool::new(has_audio),
            seek_to: Mutex::new(None),
            playing: AtomicBool::new(false),
            stop: AtomicBool::new(false),
            eof: AtomicBool::new(false),
            clock_base_pts: Mutex::new(0.0),
            clock_base_wall: Mutex::new(None),
        });

        // ── Setup cpal output stream (kalau ada audio) ──
        // Callback ini jalan di audio thread milik OS (real-time), jadi
        // JANGAN pernah block lama di sini. Mutex di sini masih ada risiko
        // kecil (priority inversion) — kalau kedengeran krek-krek di
        // playback, ganti audio_buf ke ring buffer lock-free (crate `ringbuf`).
        let mut audio_stream: Option<cpal::Stream> = None;
        if has_audio {
            let host = cpal::default_host();
            if let Some(device) = host.default_output_device() {
                if let Ok(cfg) = device.default_output_config() {
                    let sample_rate = cfg.sample_rate().0 as i64;
                    let channels = cfg.channels() as i64;
                    shared.out_sample_rate.store(sample_rate, Ordering::Relaxed);
                    shared.out_channels.store(channels, Ordering::Relaxed);

                    let stream_cfg: cpal::StreamConfig = cfg.clone().into();
                    let shared_cb = Arc::clone(&shared);

                    let build_result = match cfg.sample_format() {
                        cpal::SampleFormat::F32 => device.build_output_stream(
                            &stream_cfg,
                            move |data: &mut [f32], _| {
                                let mut buf = shared_cb.audio_buf.lock().unwrap();
                                let ch = shared_cb.out_channels.load(Ordering::Relaxed).max(1) as usize;
                                let mut i = 0;
                                while i < data.len() {
                                    data[i] = buf.pop_front().unwrap_or(0.0); // 0.0 = diam kalau buffer kosong (underrun)
                                    i += 1;
                                }
                                shared_cb.audio_frames_played.fetch_add((data.len() / ch) as u64, Ordering::Relaxed);
                            },
                            |err| eprintln!("[media_engine] audio stream error: {err}"),
                            None,
                        ),
                        // Device jarang minta selain f32 di WASAPI shared mode,
                        // tapi kalau ketemu kasusnya, tambahin cabang i16/u16 di sini.
                        _ => Err(cpal::BuildStreamError::StreamConfigNotSupported),
                    };

                    if let Ok(stream) = build_result {
                        let _ = stream.play();
                        audio_stream = Some(stream);
                    } else {
                        // Gagal setup audio device -> tetep jalan sebagai video-only,
                        // jangan gagalin seluruh player gara-gara audio device bermasalah.
                        shared.has_audio.store(false, Ordering::Relaxed);
                    }
                } else {
                    shared.has_audio.store(false, Ordering::Relaxed);
                }
            } else {
                shared.has_audio.store(false, Ordering::Relaxed);
            }
        }

        let final_has_audio = shared.has_audio.load(Ordering::Relaxed);

        // ── Spawn thread decode ──
        let thread_shared = Arc::clone(&shared);
        let path_owned = file_path.to_string();
        let decode_thread = thread::Builder::new()
            .name("media_engine-decode".into())
            .spawn(move || decode_loop(path_owned, thread_shared))
            .map_err(|e| PyRuntimeError::new_err(format!("Spawn thread decode gagal: {}", e)))?;

        Ok(PlayerEngine {
            shared,
            decode_thread: Some(decode_thread),
            _audio_stream: audio_stream,
            duration,
            width,
            height,
            fps,
            has_audio: final_has_audio,
        })
    }

    /// Mulai/lanjutkan playback.
    fn play(&mut self) {
        if !self.shared.playing.swap(true, Ordering::SeqCst) {
            *self.shared.clock_base_wall.lock().unwrap() = Some(Instant::now());
        }
    }

    /// Jeda playback (decode thread juga berhenti ngedecode paket baru).
    fn pause(&mut self) {
        // Simpen posisi sekarang sbg base baru biar gak lompat pas resume.
        let pos = self.shared.position();
        *self.shared.clock_base_pts.lock().unwrap() = pos;
        self.shared.playing.store(false, Ordering::SeqCst);
    }

    /// Lompat ke detik tertentu. Non-blocking — decode thread yang
    /// beneran ngerjain seek di background.
    fn seek(&mut self, second: f64) {
        self.shared.eof.store(false, Ordering::SeqCst);
        *self.shared.seek_to.lock().unwrap() = Some(second);
    }

    /// Posisi playback sekarang (detik), dihitung dari clock audio
    /// (atau wall-clock kalau video-only).
    fn position(&self) -> f64 {
        self.shared.position()
    }

    fn is_eof(&self) -> bool {
        self.shared.eof.load(Ordering::Relaxed)
            && self.shared.video_q.lock().unwrap().is_empty()
    }

    /// Panggil ini tiap tick UI (misal tiap ~8-16ms). Balikin None kalau
    /// belum waktunya nampilin frame baru — di situasi itu Python cukup
    /// nampilin frame terakhir yang ada, jangan blok nunggu.
    /// Return: Some((array_rgb, pts_detik)) atau None.
    fn get_frame(&mut self, py: Python<'_>) -> PyResult<Option<(Py<PyArray3<u8>>, f64)>> {
        let now = self.shared.position();
        let mut q = self.shared.video_q.lock().unwrap();

        // Buang frame yang udah basi (telat) biar gak numpuk delay -> ini
        // yang bikin video "ngejar" balik ke posisi seharusnya kalau sempet
        // ketinggalan, dibanding numpuk dan diputer kayak fast-forward.
        while let Some(front) = q.front() {
            if front.pts + LATE_FRAME_DROP_SEC < now {
                q.pop_front();
                continue;
            }
            break;
        }

        let ready = matches!(q.front(), Some(f) if f.pts <= now);
        if !ready {
            return Ok(None);
        }
        let frame = q.pop_front().unwrap();
        drop(q);

        let arr = Array3::from_shape_vec(
            (self.height as usize, self.width as usize, 3),
            frame.rgb,
        ).map_err(|e| PyRuntimeError::new_err(format!("Array gagal: {}", e)))?;

        Ok(Some((arr.into_pyarray(py).into(), frame.pts)))
    }

    /// Hentikan playback & thread decode. Panggil eksplisit sebelum
    /// object di-drop kalau mau nutup lebih cepat / ganti file.
    fn close(&mut self) {
        self.shared.stop.store(true, Ordering::SeqCst);
        if let Some(h) = self.decode_thread.take() {
            let _ = h.join();
        }
    }
}

impl Drop for PlayerEngine {
    fn drop(&mut self) {
        self.close();
    }
}

/// Loop utama thread decode: demux, decode video+audio, isi queue/buffer.
/// Jalan independen dari GIL Python — cuma method get_frame() yang perlu GIL,
/// dan itu cuma buat konversi Vec<u8> -> PyArray, bukan buat decode-nya.
fn decode_loop(file_path: String, shared: Arc<Shared>) {
    let path = Path::new(&file_path);
    let mut input_ctx = match ffmpeg_next::format::input(&path) {
        Ok(i) => i,
        Err(e) => {
            eprintln!("[media_engine] gagal buka file di decode thread: {e}");
            return;
        }
    };

    let video_idx = input_ctx.streams().best(ffmpeg_next::media::Type::Video).map(|s| s.index());
    let audio_idx = input_ctx.streams().best(ffmpeg_next::media::Type::Audio).map(|s| s.index());

    let mut vdecoder = video_idx.and_then(|_| {
        let stream = input_ctx.streams().best(ffmpeg_next::media::Type::Video)?;
        let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).ok()?;
        ctx.decoder().video().ok()
    });
    let vwidth = vdecoder.as_ref().map(|d| d.width()).unwrap_or(0);
    let vheight = vdecoder.as_ref().map(|d| d.height()).unwrap_or(0);
    let vtb = video_idx.and_then(|_| {
        let s = input_ctx.streams().best(ffmpeg_next::media::Type::Video)?;
        let tb = s.time_base();
        if tb.denominator() > 0 { Some(tb.numerator() as f64 / tb.denominator() as f64) } else { None }
    }).unwrap_or(0.0);

    let mut adecoder = if shared.has_audio.load(Ordering::Relaxed) {
        audio_idx.and_then(|_| {
            let stream = input_ctx.streams().best(ffmpeg_next::media::Type::Audio)?;
            let ctx = ffmpeg_next::codec::context::Context::from_parameters(stream.parameters()).ok()?;
            ctx.decoder().audio().ok()
        })
    } else {
        None
    };
    let atb = audio_idx.and_then(|_| {
        let s = input_ctx.streams().best(ffmpeg_next::media::Type::Audio)?;
        let tb = s.time_base();
        if tb.denominator() > 0 { Some(tb.numerator() as f64 / tb.denominator() as f64) } else { None }
    }).unwrap_or(0.0);

    let out_rate = shared.out_sample_rate.load(Ordering::Relaxed) as u32;
    let out_channels = shared.out_channels.load(Ordering::Relaxed) as u16;
    let out_layout = ffmpeg_next::util::channel_layout::ChannelLayout::default(out_channels as i32);

    let mut scaler: Option<ffmpeg_next::software::scaling::Context> = None;
    let mut resampler: Option<ffmpeg_next::software::resampling::Context> = None;

    let mut vframe = ffmpeg_next::frame::Video::empty();
    let mut aframe = ffmpeg_next::frame::Audio::empty();

    loop {
        if shared.stop.load(Ordering::Relaxed) {
            break;
        }

        if let Some(sec) = shared.seek_to.lock().unwrap().take() {
            let ts = av_time_base_ts(sec);
            if input_ctx.seek(ts, ..ts).is_ok() {
                if let Some(d) = vdecoder.as_mut() { d.flush(); }
                if let Some(d) = adecoder.as_mut() { d.flush(); }
                shared.video_q.lock().unwrap().clear();
                shared.audio_buf.lock().unwrap().clear();
                shared.reset_clock(sec);
                shared.eof.store(false, Ordering::Relaxed);
            }
        }

        if !shared.playing.load(Ordering::Relaxed) {
            thread::sleep(Duration::from_millis(10));
            continue;
        }

        // Backpressure: jangan decode lebih cepet dari yang dibutuhin buat nampilin.
        let q_len = shared.video_q.lock().unwrap().len();
        let a_len = shared.audio_buf.lock().unwrap().len();
        let a_cap = (out_rate as usize) * (out_channels as usize) * 2; // ~2 detik
        if q_len >= MAX_VIDEO_FRAMES || a_len >= a_cap {
            thread::sleep(Duration::from_millis(5));
            continue;
        }

        let next_packet = input_ctx.packets().next();
        let (stream, packet) = match next_packet {
            Some(p) => p,
            None => {
                // EOF: flush sisa frame yang masih ngendon di decoder.
                if let Some(d) = vdecoder.as_mut() { let _ = d.send_eof(); }
                if let Some(d) = adecoder.as_mut() { let _ = d.send_eof(); }
                shared.eof.store(true, Ordering::Relaxed);
                thread::sleep(Duration::from_millis(20));
                continue;
            }
        };

        if Some(stream.index()) == video_idx {
            if let Some(d) = vdecoder.as_mut() {
                if d.send_packet(&packet).is_ok() {
                    while d.receive_frame(&mut vframe).is_ok() {
                        let pts = vframe.timestamp().unwrap_or(0) as f64 * vtb;
                        if let Ok(rgb) = scale_to_rgb(&vframe, &mut scaler, d.format(), vwidth, vheight) {
                            shared.video_q.lock().unwrap().push_back(QueuedFrame { rgb, pts });
                        }
                    }
                }
            }
        } else if Some(stream.index()) == audio_idx {
            if let Some(d) = adecoder.as_mut() {
                if d.send_packet(&packet).is_ok() {
                    while d.receive_frame(&mut aframe).is_ok() {
                        if resampler.is_none() {
                            let src_layout = if aframe.channel_layout().bits() != 0 {
                                aframe.channel_layout()
                            } else {
                                ffmpeg_next::util::channel_layout::ChannelLayout::default(d.channels() as i32)
                            };
                            resampler = ffmpeg_next::software::resampling::Context::get(
                                aframe.format(), src_layout, aframe.rate(),
                                ffmpeg_next::util::format::Sample::F32(ffmpeg_next::util::format::sample::Type::Packed),
                                out_layout, out_rate,
                            ).ok();
                        }
                        if let Some(rs) = resampler.as_mut() {
                            let mut resampled = ffmpeg_next::frame::Audio::empty();
                            if rs.run(&aframe, &mut resampled).is_ok() {
                                let n_samples = resampled.samples() * out_channels as usize;
                                let raw = resampled.data(0);
                                if raw.len() >= n_samples * 4 {
                                    let floats: &[f32] = unsafe {
                                        std::slice::from_raw_parts(raw.as_ptr() as *const f32, n_samples)
                                    };
                                    shared.audio_buf.lock().unwrap().extend(floats.iter().copied());
                                }
                            }
                        }
                        let _ = aframe.timestamp().unwrap_or(0) as f64 * atb; // dicadangkan buat resync halus kalau perlu nanti
                    }
                }
            }
        }
    }
}

fn scale_to_rgb(
    decoded: &ffmpeg_next::frame::Video,
    scaler_slot: &mut Option<ffmpeg_next::software::scaling::Context>,
    src_format: ffmpeg_next::util::format::Pixel,
    width: u32,
    height: u32,
) -> Result<Vec<u8>, ffmpeg_next::Error> {
    let scaler = match scaler_slot {
        Some(s) => s,
        None => {
            let ctx = ffmpeg_next::software::scaling::Context::get(
                src_format, width, height,
                ffmpeg_next::util::format::Pixel::RGB24, width, height,
                ffmpeg_next::software::scaling::flag::Flags::BILINEAR,
            )?;
            scaler_slot.insert(ctx)
        }
    };

    let mut rgb_frame = ffmpeg_next::frame::Video::new(
        ffmpeg_next::util::format::Pixel::RGB24, width, height,
    );
    scaler.run(decoded, &mut rgb_frame)?;

    let stride = rgb_frame.stride(0);
    let row_width = (width as usize) * 3;
    let mut data = Vec::with_capacity((height as usize) * row_width);
    let raw_data = rgb_frame.data(0);
    for y in 0..(height as usize) {
        let start = y * stride;
        data.extend_from_slice(&raw_data[start..start + row_width]);
    }
    Ok(data)
}

// ═══════════════════════════════════════════════
// DAFTARKAN KE MODUL
// ═══════════════════════════════════════════════

#[pymodule]
fn media_engine(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<MediaInfo>()?;
    m.add_class::<VideoDecoder>()?;
    m.add_class::<PlayerEngine>()?;
    Ok(())
}
