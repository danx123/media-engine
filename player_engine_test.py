"""
player_engine_test.py
══════════════════════════════════════════════════════════
Testbed sederhana buat nyoba PlayerEngine (media_engine.pyd) abis build
sukses. Bukan buat production Macan Video Player — ini cuma harness buat
mastiin: video jalan smooth, audio nyala & sinkron, seek gak nge-hang,
sebelum logic-nya dipindah ke app beneran.

Cara pake:
    pip install PySide6 numpy
    # taro media_engine.pyd / .so satu folder sama file ini (atau di PYTHONPATH)
    python player_engine_test.py
══════════════════════════════════════════════════════════
"""

import sys
import os

from PySide6.QtCore import Qt, QTimer, QSettings, QThread, QObject, Signal, QPoint, Slot, QMetaObject
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox, QStyle, QStyleOptionSlider,
)

try:
    from media_engine import PlayerEngine, VideoDecoder
except ImportError as e:
    PlayerEngine = None
    VideoDecoder = None
    _IMPORT_ERROR = e


def format_time(seconds: float) -> str:
    """Detik float -> "MM:SS" (biar gampang dibaca di label posisi)."""
    if seconds < 0 or seconds != seconds:  # NaN guard
        seconds = 0.0
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


# ══════════════════════════════════════════════════════════
# Hover thumbnail: worker thread + slider custom + popup
# ══════════════════════════════════════════════════════════
#
# Pola threading-nya: ThumbnailRequester tinggal di MAIN thread, cuma
# nampung 1 Signal buat "ngirim" request nyebrang ke worker thread (Qt
# otomatis pakai queued connection krn beda thread affinity). ThumbnailWorker
# di-moveToThread() ke QThread terpisah dan bikin VideoDecoder-nya SENDIRI di
# situ -- terpisah total dari PlayerEngine yang lagi dipakai buat playback,
# jadi hover-preview gak akan pernah nyenggol/ganggu decode thread player.
#
# VideoDecoder itu unsendable di sisi Rust (PyO3), artinya cuma boleh
# diakses dari thread Python tempat dia dibikin. Karena kita bikin instance-nya
# di dalam slot yang jalan DI worker thread (bukan di __init__ main thread),
# syarat itu otomatis kepenuhin.

class ThumbnailRequester(QObject):
    """Proxy tipis yang tinggal di main thread. request_thumbnail cuma
    dipakai buat ngirim sinyal ke worker -- gak ada logic di sini."""
    request_thumbnail = Signal(int, float, QPoint)  # req_id, detik, posisi global cursor


class ThumbnailWorker(QObject):
    """Jalan di QThread sendiri. Buka VideoDecoder-nya sendiri (terpisah
    dari PlayerEngine yang lagi playback) dan proses satu permintaan
    seek_frame() per sinyal masuk."""
    thumbnail_ready = Signal(int, float, QPoint, object)  # req_id, detik, posisi, np.ndarray|None

    def __init__(self, path: str):
        super().__init__()
        self._path = path
        self._decoder = None

    def handle_request(self, req_id: int, second: float, global_pos: QPoint):
        try:
            if self._decoder is None:
                self._decoder = VideoDecoder(self._path)
            frame = self._decoder.seek_frame(second)
        except Exception as e:
            print(f"[ThumbnailWorker] gagal ambil frame @ {second:.2f}s: {e}")
            frame = None  # posisi susah didecode (mis. mepet EOF) -- gapapa, skip aja
        self.thumbnail_ready.emit(req_id, second, global_pos, frame)

    @Slot()
    def cleanup(self):
        """WAJIB dipanggil lewat QMetaObject.invokeMethod(..., Qt.BlockingQueuedConnection)
        dari main thread, BUKAN dipanggil langsung. VideoDecoder di sisi
        Rust itu "unsendable" -- objeknya cuma boleh disentuh (termasuk
        di-drop) dari thread yang sama tempat dia dibikin. self._decoder
        dibikin di handle_request() yang jalan DI THREAD WORKER INI (lewat
        queued connection), jadi drop-nya juga harus kejadian di sini, bukan
        di main thread pas main thread nge-null-in reference ke worker ini."""
        self._decoder = None


class HoverSeekSlider(QSlider):
    """QSlider biasa + sinyal hover yang ngasih tau posisi WAKTU (bukan cuma
    posisi piksel) yang lagi ditunjuk cursor, dipetakan lewat QStyle biar
    akurat ngitung margin groove/handle (bukan sekadar x/width linear)."""
    hovered = Signal(float, QPoint)  # fraksi 0.0-1.0, posisi global cursor
    hover_ended = Signal()

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.setMouseTracking(True)  # biar mouseMoveEvent nembak tanpa tombol ditekan

    def mouseMoveEvent(self, event):
        super().mouseMoveEvent(event)
        # PENTING: exception yang kelempar dari dalam mouseMoveEvent (method
        # virtual yang dipanggil dari sisi C++ Qt) BISA KETELAN DIAM-DIAM --
        # gak selalu muncul sebagai traceback normal di console, tergantung
        # excepthook yang aktif. Makanya dibungkus try/except eksplisit +
        # print, biar kalau ada apa2 minimal kelihatan, bukan cuma "hover
        # gak jalan" tanpa jejak sama sekali.
        try:
            frac = self._x_to_fraction(event.position().x())
        except Exception as e:
            print(f"[HoverSeekSlider] gagal hitung posisi hover: {e}")
            return
        if frac is not None:
            self.hovered.emit(frac, event.globalPosition().toPoint())

    def leaveEvent(self, event):
        super().leaveEvent(event)
        self.hover_ended.emit()

    def _x_to_fraction(self, x: float):
        opt = QStyleOptionSlider()
        self.initStyleOption(opt)
        style = self.style()
        # Pakai bentuk SCOPED (QStyle.ComplexControl.xxx / QStyle.SubControl.xxx),
        # bukan bentuk flat (QStyle.CC_Slider) -- di sebagian versi PySide6/Qt6,
        # enum QStyle yang jarang dipakai kayak gini cuma kebaca lewat bentuk
        # scoped-nya. Bentuk flat bisa lempar AttributeError yang (krn ini
        # dipanggil dari virtual method Qt) gak keliatan sebagai crash biasa.
        groove = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderGroove, self,
        )
        handle = style.subControlRect(
            QStyle.ComplexControl.CC_Slider, opt, QStyle.SubControl.SC_SliderHandle, self,
        )
        span = groove.width() - handle.width()
        if span <= 0:
            return None
        pos = min(max(int(x) - groove.x() - handle.width() // 2, 0), span)
        value = QStyle.sliderValueFromPosition(self.minimum(), self.maximum(), pos, span)
        value_range = max(self.maximum() - self.minimum(), 1)
        return (value - self.minimum()) / value_range


class ThumbnailPopup(QWidget):
    """Widget floating kecil, gak nyuri fokus, gak kena klik (transparan buat
    mouse) -- muncul di atas cursor pas hover di seekbar."""

    def __init__(self, parent=None):
        super().__init__(parent, Qt.ToolTip | Qt.FramelessWindowHint)
        self.setAttribute(Qt.WA_TransparentForMouseEvents)
        self.setStyleSheet(
            "background-color: #1c1c1c; border: 1px solid #555; border-radius: 4px;"
        )
        layout = QVBoxLayout(self)
        layout.setContentsMargins(4, 4, 4, 4)
        layout.setSpacing(2)
        self.image_label = QLabel()
        self.time_label = QLabel()
        self.time_label.setAlignment(Qt.AlignCenter)
        self.time_label.setStyleSheet("color: white; background: transparent; border: none;")
        layout.addWidget(self.image_label)
        layout.addWidget(self.time_label)

    def show_thumbnail(self, pixmap: QPixmap, time_text: str, global_pos: QPoint):
        self.image_label.setPixmap(pixmap)
        self.time_label.setText(time_text)
        self.adjustSize()

        x = global_pos.x() - self.width() // 2
        y = global_pos.y() - self.height() - 16
        screen = QApplication.screenAt(global_pos) or self.screen()
        if screen is not None:
            geo = screen.availableGeometry()
            x = max(geo.left(), min(x, geo.right() - self.width()))
            y = max(geo.top(), y)
        self.move(x, y)
        self.show()


class PlayerTestWindow(QMainWindow):
    # Interval polling get_frame(). Gak perlu presisi ke fps video — get_frame()
    # sendiri yang mutusin "belum waktunya" (return None) kalau dipanggil kepagian.
    # 10ms cukup buat video sampe ~90fps tanpa buang-buang CPU polling.
    TICK_MS = 10

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Macan Angkasa — PlayerEngine Testbed")
        self.resize(960, 620)

        self.settings = QSettings("MacanAngkasa", "PlayerEngineTest")

        self.engine = None          # instance PlayerEngine aktif
        self._current_frame_buf = None  # PENTING: nahan reference numpy array
                                          # selama QImage masih make bufernya,
                                          # kalau gak nanti dealokasi kepagian
                                          # -> QImage nampilin garbage/crash.
        self._is_seeking_by_user = False  # true selama slider lagi di-drag manual
        self._duration = 0.0

        # ── State buat hover-thumbnail di seekbar ──
        self.thumb_thread = QThread(self)
        self.thumb_worker = None            # dibikin ulang tiap ganti file, lihat _start_thumbnail_worker
        self.thumb_requester = ThumbnailRequester()
        self._thumb_frame_buf = None        # nahan reference array thumbnail, sama alasannya kayak _current_frame_buf
        self._is_hovering_seekbar = False
        self._hover_request_id = 0          # buat buang hasil basi kalau ada request lebih baru nyusul
        self._pending_hover_time = 0.0
        self._pending_hover_pos = QPoint()
        self._hover_debounce = QTimer(self)
        self._hover_debounce.setSingleShot(True)
        self._hover_debounce.setInterval(80)  # nunggu cursor "diem" dulu ~80ms sblm nembak request
        self._hover_debounce.timeout.connect(self._request_thumbnail)

        self._build_ui()

        self.tick_timer = QTimer(self)
        self.tick_timer.setInterval(self.TICK_MS)
        self.tick_timer.timeout.connect(self._on_tick)

        if PlayerEngine is None:
            QMessageBox.critical(
                self, "media_engine gak ketemu",
                f"Gagal import media_engine:\n{_IMPORT_ERROR}\n\n"
                "Pastiin file .pyd/.so hasil build maturin ada satu folder "
                "sama script ini, atau udah ke-install di venv."
            )

    # ────────────────────────────────────────────
    # UI
    # ────────────────────────────────────────────

    def _build_ui(self):
        central = QWidget(self)
        self.setCentralWidget(central)
        root = QVBoxLayout(central)

        # Area video — QLabel biasa, cukup buat testbed. Frame di-scale
        # otomatis ngikutin ukuran window.
        self.video_label = QLabel("Buka file video buat mulai...")
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setStyleSheet("background-color: #111; color: #888;")
        self.video_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.video_label.setMinimumHeight(400)
        root.addWidget(self.video_label, stretch=1)

        # Slider posisi + label waktu
        seek_row = QHBoxLayout()
        self.time_label = QLabel("00:00 / 00:00")
        self.position_slider = HoverSeekSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)  # dipetakan ke 0..duration
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        self.position_slider.hovered.connect(self._on_seekbar_hover)
        self.position_slider.hover_ended.connect(self._on_seekbar_hover_end)
        seek_row.addWidget(self.position_slider, stretch=1)
        seek_row.addWidget(self.time_label)
        root.addLayout(seek_row)

        self.thumbnail_popup = ThumbnailPopup(self)

        # Tombol transport
        btn_row = QHBoxLayout()
        self.open_btn = QPushButton("Buka File…")
        self.open_btn.clicked.connect(self._on_open_file)
        self.play_btn = QPushButton("▶ Play")
        self.play_btn.clicked.connect(self._on_play_pause)
        self.play_btn.setEnabled(False)

        self.stop_btn = QPushButton("⏹ Stop")
        self.stop_btn.clicked.connect(self._on_stop)
        self.stop_btn.setEnabled(False)

        self.next_frame_btn = QPushButton("⏭ Frame")
        self.next_frame_btn.setToolTip("Maju satu frame (aktif kalau lagi paused)")
        self.next_frame_btn.clicked.connect(self._on_next_frame)
        self.next_frame_btn.setEnabled(False)

        self.loop_checkbox = QCheckBox("Loop")
        self.loop_checkbox.toggled.connect(self._on_loop_toggled)
        self.loop_checkbox.setEnabled(False)

        self.mute_btn = QPushButton("🔊")
        self.mute_btn.setFixedWidth(36)
        self.mute_btn.clicked.connect(self._on_toggle_mute)
        self.mute_btn.setEnabled(False)

        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setFixedWidth(110)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(100)  # match default volume 1.0 di Rust
        self.volume_slider.valueChanged.connect(self._on_volume_changed)
        self.volume_slider.setEnabled(False)

        self.status_label = QLabel("")
        self.status_label.setStyleSheet("color: #888;")

        btn_row.addWidget(self.open_btn)
        btn_row.addWidget(self.play_btn)
        btn_row.addWidget(self.stop_btn)
        btn_row.addWidget(self.next_frame_btn)
        btn_row.addWidget(self.loop_checkbox)
        btn_row.addSpacing(12)
        btn_row.addWidget(self.mute_btn)
        btn_row.addWidget(self.volume_slider)
        btn_row.addStretch(1)
        btn_row.addWidget(self.status_label)
        root.addLayout(btn_row)

    # ────────────────────────────────────────────
    # Buka file & lifecycle engine
    # ────────────────────────────────────────────

    def _on_open_file(self):
        if PlayerEngine is None:
            return

        last_dir = self.settings.value("last_dir", "")
        path, _ = QFileDialog.getOpenFileName(
            self, "Pilih file media", last_dir,
            # [BARU] PlayerEngine udah dukung file audio murni (has_video
            # = false di sisi Rust) -- filter dialog cuma perlu dibukain,
            # gak ada perubahan lain yang dibutuhin buat bisa buka mp3/dll.
            "Semua media (*.mp4 *.mkv *.webm *.mov *.avi *.mp3 *.flac *.wav *.ogg *.m4a *.aac);;"
            "Video (*.mp4 *.mkv *.webm *.mov *.avi);;"
            "Audio (*.mp3 *.flac *.wav *.ogg *.m4a *.aac);;"
            "Semua file (*.*)",
        )
        if not path:
            return
        self.settings.setValue("last_dir", os.path.dirname(path))
        self._load_file(path)

    def _load_file(self, path: str):
        # Tutup engine lama dulu (kalau ada) sebelum bikin yang baru —
        # close() eksplisit matiin thread decode + audio stream sebelum
        # object-nya di-drop, jadi gak numpuk thread nganggur pas gonta-ganti file.
        self.tick_timer.stop()
        if self.engine is not None:
            self.engine.close()
            self.engine = None

        try:
            self.engine = PlayerEngine(path)
        except Exception as e:
            QMessageBox.critical(self, "Gagal buka file", str(e))
            self.status_label.setText("Gagal buka file.")
            return

        self._duration = self.engine.duration
        audio_note = "ada audio" if self.engine.has_audio else "TANPA audio"

        # [BARU] File audio murni (has_video=False): width/height/fps di
        # sisi Rust sengaja dikosongin (0), jadi jangan ditampilin di
        # status label -- bikin bingung ("0x0 @0.00fps"). Tampilan beda
        # buat kasus ini, fokus ke info audio-nya aja.
        if self.engine.has_video:
            self.status_label.setText(
                f"{os.path.basename(path)} — {self.engine.width}x{self.engine.height} "
                f"@{self.engine.fps:.2f}fps, {audio_note}"
            )
        else:
            self.status_label.setText(f"{os.path.basename(path)} — audio only")

        # [BARU] Gak ada frame video buat ditampilin di audio-only mode --
        # video_label dipake nunjukin placeholder ikon musik + nama file,
        # bukan dibiarin nampilin frame terakhir dari file sebelumnya (atau
        # kosong item, keliatan kayak nge-hang).
        if self.engine.has_video:
            self.video_label.setText("")
            self.video_label.setPixmap(QPixmap())
        else:
            self.video_label.setPixmap(QPixmap())
            self.video_label.setText(f"🎵  {os.path.basename(path)}\n(audio only, gak ada video)")

        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ Play")
        self.stop_btn.setEnabled(True)
        # [FIX] Frame-step gak masuk akal buat file tanpa video -- kalau
        # dibiarin aktif, step_frame() cuma diem-diem gak ngapa2in (gak
        # ada video_decode_loop yang beneran jalan buat file ini), keliatan
        # kayak tombolnya rusak drpd "emang gak berlaku di sini".
        self.next_frame_btn.setEnabled(self.engine.has_video)
        self.loop_checkbox.setEnabled(True)
        self.engine.set_loop(self.loop_checkbox.isChecked())

        # Sinkronin volume/mute engine baru ke posisi slider yang lagi aktif
        # di UI (bukan ke default engine), biar konsisten antar ganti file.
        self.mute_btn.setEnabled(self.engine.has_audio)
        self.volume_slider.setEnabled(self.engine.has_audio)
        self.engine.set_volume(self.volume_slider.value() / 100.0)
        self.engine.set_muted(self.mute_btn.text() == "🔇")

        self.engine.play()
        self.tick_timer.start()
        # [FIX] VideoDecoder (dipakai ThumbnailWorker) butuh stream video --
        # buat file audio-only bakal selalu gagal bikin instance-nya tiap
        # kali cursor hover di seekbar (exception ke-log berulang percuma).
        # Sekalian gak ada gunanya nampilin thumbnail buat file yang emang
        # gak ada framenya.
        if self.engine.has_video:
            self._start_thumbnail_worker(path)
        else:
            self._stop_thumbnail_worker()

    # ────────────────────────────────────────────
    # Transport controls
    # ────────────────────────────────────────────

    def _on_play_pause(self):
        if self.engine is None:
            return
        if self.engine.is_playing():
            self.engine.pause()
            self.play_btn.setText("▶ Play")
        else:
            self.engine.play()
            self.play_btn.setText("⏸ Pause")

    def _on_stop(self):
        if self.engine is None:
            return
        self.engine.stop()
        self.play_btn.setText("▶ Play")
        # Paksa satu tick manual biar frame pertama abis stop langsung
        # muncul di layar, gak nunggu play() ditekan dulu.
        self._on_tick()

    def _on_next_frame(self):
        if self.engine is None:
            return
        if self.engine.is_playing():
            return  # frame-step cuma masuk akal pas paused
        self.engine.step_frame()

    def _on_loop_toggled(self, checked: bool):
        if self.engine is not None:
            self.engine.set_loop(checked)

    def _on_toggle_mute(self):
        if self.engine is None:
            return
        now_muted = self.mute_btn.text() != "🔇"
        self.engine.set_muted(now_muted)
        self.mute_btn.setText("🔇" if now_muted else "🔊")

    def _on_volume_changed(self, value: int):
        if self.engine is not None:
            self.engine.set_volume(value / 100.0)

    def _on_slider_pressed(self):
        self._is_seeking_by_user = True

    def _on_slider_released(self):
        if self.engine is not None and self._duration > 0:
            frac = self.position_slider.value() / 1000.0
            self.engine.seek(frac * self._duration)
        self._is_seeking_by_user = False

    # ────────────────────────────────────────────
    # Hover-thumbnail di seekbar
    # ────────────────────────────────────────────

    def _start_thumbnail_worker(self, path: str):
        if VideoDecoder is None:
            return  # media_engine gagal ke-import, sudah dikeluhin pas startup
        self._stop_thumbnail_worker()

        self.thumb_worker = ThumbnailWorker(path)
        self.thumb_worker.moveToThread(self.thumb_thread)
        self.thumb_requester.request_thumbnail.connect(self.thumb_worker.handle_request)
        self.thumb_worker.thumbnail_ready.connect(self._on_thumbnail_ready)
        self.thumb_thread.start()

    def _stop_thumbnail_worker(self):
        if self.thumb_worker is not None and self.thumb_thread.isRunning():
            # BlockingQueuedConnection: manggil cleanup() dan NUNGGU sampe
            # beneran kelar dieksekusi DI THREAD WORKER, sebelum kita lanjut.
            # Ini yang nyegah crash "VideoDecoder is unsendable, but is
            # being dropped on another thread" -- tanpa ini, VideoDecoder-nya
            # baru ke-drop belakangan pas `self.thumb_worker = None` di
            # bawah, yang jalan di MAIN thread (thread yang salah).
            QMetaObject.invokeMethod(self.thumb_worker, "cleanup", Qt.BlockingQueuedConnection)
            # deleteLater() (bukan langsung None) supaya QObject-nya sendiri
            # juga dibersihin lewat event loop thread-nya sendiri, konsisten
            # sama alasan yang sama di atas.
            self.thumb_worker.deleteLater()
        if self.thumb_thread.isRunning():
            try:
                self.thumb_requester.request_thumbnail.disconnect()
            except TypeError:
                pass  # belum ada koneksi sama sekali (mis. file pertama kali dibuka)
            self.thumb_thread.quit()
            self.thumb_thread.wait(1000)
        self.thumb_worker = None

    def _on_seekbar_hover(self, frac: float, global_pos: QPoint):
        if self._duration <= 0 or VideoDecoder is None:
            return
        # [BARU] Audio-only: thumbnail worker sengaja gak dijalanin (lihat
        # _load_file), jadi hover di sini gak ada yang perlu diproses.
        if self.engine is not None and not self.engine.has_video:
            return
        self._is_hovering_seekbar = True
        self._pending_hover_time = frac * self._duration
        self._pending_hover_pos = global_pos
        self._hover_debounce.start()  # restart tiap gerak -- cuma nembak request pas cursor behenti sejenak

    def _on_seekbar_hover_end(self):
        self._is_hovering_seekbar = False
        self._hover_debounce.stop()
        self.thumbnail_popup.hide()

    def _request_thumbnail(self):
        if self.thumb_worker is None or not self._is_hovering_seekbar:
            return
        self._hover_request_id += 1
        self.thumb_requester.request_thumbnail.emit(
            self._hover_request_id, self._pending_hover_time, self._pending_hover_pos,
        )

    def _on_thumbnail_ready(self, req_id: int, requested_time: float, global_pos: QPoint, frame):
        # Buang hasil basi: ada request lebih baru yang udah nyusul sebelum
        # yang ini balik (misal cursor kepalang gerak lagi sebelum decode
        # kelar) -- tanpa cek ini, thumbnail bisa "ngetril" nampilin posisi
        # lama pas cursor udah pindah jauh.
        if req_id != self._hover_request_id or not self._is_hovering_seekbar:
            return
        if frame is None:
            return

        h, w, _ = frame.shape
        self._thumb_frame_buf = frame  # tahan reference selama QImage masih makenya
        image = QImage(frame.data, w, h, w * 3, QImage.Format_RGB888)
        thumb_w = 160
        thumb_h = max(1, int(thumb_w * h / w))
        pixmap = QPixmap.fromImage(image).scaled(
            thumb_w, thumb_h, Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.thumbnail_popup.show_thumbnail(pixmap, format_time(requested_time), global_pos)

    # ────────────────────────────────────────────
    # Loop utama — dipanggil tiap TICK_MS
    # ────────────────────────────────────────────

    def _on_tick(self):
        if self.engine is None:
            return

        result = self.engine.get_frame()
        if result is not None:
            rgb_array, _pts = result
            self._current_frame_buf = rgb_array  # tahan reference, lihat catatan di atas
            self._show_frame(rgb_array)

        # Update slider & label waktu, kecuali lagi di-drag manual sama user
        # (biar gak "ketarik balik" tiap tick pas user drag).
        if not self._is_seeking_by_user and self._duration > 0:
            pos = self.engine.position()
            self.position_slider.blockSignals(True)
            self.position_slider.setValue(int((pos / self._duration) * 1000))
            self.position_slider.blockSignals(False)
            self.time_label.setText(f"{format_time(pos)} / {format_time(self._duration)}")

        if self.engine.is_eof():
            self.tick_timer.stop()
            self.play_btn.setText("▶ Play")
            self.status_label.setText(self.status_label.text() + "  [Selesai]")
        # Catatan: kalau Loop dicentang, is_eof() gak akan pernah true --
        # engine otomatis seek balik ke 0 sendiri di dalem decode thread,
        # jadi gak perlu ditangani manual di sisi Python sama sekali.

    def _show_frame(self, rgb_array):
        h, w, _ = rgb_array.shape
        # rgb_array harus C-contiguous & tetep hidup selama QImage dipake
        # (makanya self._current_frame_buf di-set sebelum manggil ini).
        image = QImage(rgb_array.data, w, h, w * 3, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(image).scaled(
            self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation,
        )
        self.video_label.setPixmap(pixmap)

    # ────────────────────────────────────────────
    # Cleanup
    # ────────────────────────────────────────────

    def closeEvent(self, event):
        self.tick_timer.stop()
        self._stop_thumbnail_worker()
        if self.engine is not None:
            self.engine.close()
            self.engine = None
        super().closeEvent(event)


def main():
    app = QApplication(sys.argv)
    window = PlayerTestWindow()
    window.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
