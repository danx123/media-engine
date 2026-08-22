# media_engine

A native Rust media decoding & playback engine for the **Macan Angkasa Suite**, exposed to Python via [PyO3](https://pyo3.rs). Built on top of [`ffmpeg-next`](https://github.com/zmwangx/rust-ffmpeg) for demuxing/decoding and [`cpal`](https://github.com/RustAudio/cpal) for cross-platform audio output.

Ships as an `abi3` wheel — one compiled extension works across Python 3.10+.

## Why this exists

Macan Video Player originally relied on VLC/Qt multimedia backends for playback. `media_engine` replaces that with a purpose-built decode pipeline that:

- Exposes raw decoded frames as `numpy` arrays, so the suite's existing PySide6 UI code (overlays, filters, thumbbar previews, desktop lyrics-style widgets, etc.) can work directly on pixel data instead of being locked behind an opaque player widget.
- Keeps the binary footprint small — no bundling of a full VLC runtime or its plugin tree into the Nuitka build.
- Avoids being at the mercy of a third-party player's OS-specific backend quirks (e.g. Media Foundation codec availability differences across Windows versions).
- Gives full control over audio/video sync, buffering, and frame timing — all standard `AtomicXxx`/`Mutex`-based state, no hidden behavior.

The trade-off: correctness and edge cases (obscure containers, subtitle tracks, DRM, etc.) are on us to handle, not a battle-tested upstream project. This crate intentionally stays scoped to what the suite actually needs.

## Module layout

Everything lives in a single `lib.rs` (flat layout, consistent with the rest of the suite's Nuitka-friendly single-directory convention). Three `#[pyclass]` types are exported:

| Class | Purpose |
|---|---|
| `MediaInfo` | One-shot metadata probe (duration, resolution, fps, codec, bitrate). No decoding of frame data. |
| `VideoDecoder` | Frame-accurate single-frame extraction (`seek_frame`, `read_next_frame`). No audio, no internal threading — the caller drives it directly. Used for thumbnails / scrubbing previews / single-frame grabs. |
| `PlayerEngine` | Full playback engine: background decode thread, audio output via `cpal`, audio-clock-driven A/V sync, non-blocking frame polling. This is what backs actual video playback. |

`VideoDecoder` and `PlayerEngine` solve different problems and are kept separate on purpose — don't try to make one do the other's job.

## `PlayerEngine` — architecture

```
┌─────────────────────────────────────────────────────────────┐
│  Python (UI thread)                                          │
│                                                                │
│   PlayerEngine.play() / .pause() / .seek(t) / .set_volume(v)  │
│   PlayerEngine.get_frame()  ──── polled every UI tick ────┐   │
│   PlayerEngine.position() / .is_eof()                     │   │
└────────────────────────────┬───────────────────────────────┼──┘
                              │ Arc<Shared> (atomics + mutexes)   │
┌─────────────────────────────▼───────────────────────────────┼──┐
│  Decode thread (spawned in PlayerEngine::new)              │  │
│                                                              │  │
│   demux → decode video → scale to RGB24 ──► video_q        │  │
│   demux → decode audio → resample to f32 ──► audio_buf      │  │
│   handles seek requests, pause, backpressure, EOF           │  │
└──────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────▼──────────────────────────────┐
│  cpal audio callback (OS real-time audio thread)             │
│   pops samples from audio_buf, applies volume/mute,          │
│   increments audio_frames_played (→ master clock)            │
└────────────────────────────────────────────────────────────┘
```

Key design points:

- **Audio is the master clock.** `position()` derives playback time from the number of audio frames actually consumed by the output device, not from wall-clock time or decode speed. If the file has no audio track, it falls back to a wall-clock-based clock started at `play()`.
- **`get_frame()` is non-blocking and can legitimately return `None`.** That means "not time to show a new frame yet" — not an error. The caller should keep displaying the last frame and just poll again next tick.
- **Late frames are dropped, not queued.** If a decoded frame's PTS is more than ~100ms behind the current clock, it's discarded instead of being shown late. This prevents drift from turning into a visible backlog (fast-forward effect) after any hiccup.
- **Decoding never blocks the Python/UI thread.** The decode thread only touches the GIL indirectly, and only at the moment `get_frame()` converts a already-decoded `Vec<u8>` into a `PyArray3<u8>`.
- **Backpressure**: the decode thread throttles itself once the video queue or audio buffer holds a few seconds of readahead, so memory use stays bounded instead of decoding as fast as possible.
- **Seeking uses `AV_TIME_BASE` (microseconds)**, matching what `ffmpeg-next`'s `Input::seek()` expects internally (it calls `avformat_seek_file` with `stream_index = -1`). Note: `VideoDecoder::seek_frame` (the older, separate class) uses the video stream's own time base instead — this is a known discrepancy, kept as-is there for backward compatibility with existing thumbnail-extraction call sites. Don't copy that pattern into new code.

## Python API reference

### `MediaInfo(file_path: str)`

```python
info = MediaInfo("movie.mkv")
info.duration   # float, seconds
info.width      # int
info.height     # int
info.fps        # float
info.codec      # str, human-readable decoder description
info.codec_id   # str, e.g. "vp9", "h264"
info.bitrate    # int, bits/sec
```

### `VideoDecoder(file_path: str)`

```python
dec = VideoDecoder("movie.mkv")
frame = dec.seek_frame(12.5)     # np.ndarray [H, W, 3], uint8, RGB
frame = dec.read_next_frame()    # sequential decode, raises on EOF
dec.duration, dec.width, dec.height, dec.codec
```

No audio. No background thread — every call blocks the caller until a frame is ready. Fine for thumbnails/scrubbing; not meant for realtime playback.

### `PlayerEngine(file_path: str)`

```python
engine = PlayerEngine("movie.mkv")

engine.duration     # float, seconds
engine.width        # int
engine.height       # int
engine.fps          # float
engine.has_audio    # bool — false if the file has no audio track,
                     #        or if opening the output audio device failed
                     #        (engine still works as video-only in that case)

engine.play()
engine.pause()
engine.seek(seconds: float)          # non-blocking; decode thread handles it
engine.position() -> float           # current playback time, seconds
engine.is_eof() -> bool

engine.set_volume(v: float)          # 0.0–1.0, clamped
engine.get_volume() -> float
engine.set_muted(m: bool)
engine.is_muted() -> bool

engine.get_frame() -> tuple[np.ndarray, float] | None
    # np.ndarray: [H, W, 3] uint8 RGB, shares no memory with FFmpeg buffers
    # float: the frame's PTS in seconds
    # None: not time to show a new frame yet — keep the last one on screen

engine.close()                        # stops the decode thread & audio stream
                                       # explicitly; also called on drop
```

Typical UI loop (see `player_engine_test.py` for a full PySide6 example):

```python
engine = PlayerEngine(path)
engine.play()

def on_tick():                        # QTimer, ~10ms interval
    result = engine.get_frame()
    if result is not None:
        rgb_array, pts = result
        update_display(rgb_array)     # keep a reference alive until Qt is done with it
    if engine.is_eof():
        timer.stop()
```

**Important:** when converting `rgb_array` to a `QImage`, keep a Python-side reference to the array alive for as long as the `QImage`/`QPixmap` might still read from its buffer. `QImage` does not copy the buffer by default — if the numpy array gets garbage collected first, you'll get intermittent garbage pixels or crashes.

## Building

Windows wheels are cross-compiled from Ubuntu via `maturin` + `mingw-w64` — see `.github/workflows/build.yml`. Summary:

1. FFmpeg is compiled manually with `--target-os=mingw32` (not left to `ffmpeg-sys-next`'s own `build` feature, which sends the wrong `--target_os` flag during cross-compilation) and cached across runs.
2. `dav1d` is built separately via Meson/Ninja and linked in for AV1 support.
3. `cpal`'s Windows backend (WASAPI) needs no extra native libraries beyond what's already linked for COM (`ole32`, etc.) — it talks to the OS purely through `windows-sys`. If the linker complains about an undefined COM/audio symbol, that's the first place to look.
4. Local (non-Windows) builds let `ffmpeg-sys-next` compile FFmpeg from source itself via its `build` feature, since host == target there and the cross-compile bug doesn't apply.

To build locally for testing on the compile machine:

```bash
maturin develop --release
```


