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

from PySide6.QtCore import Qt, QTimer, QSettings
from PySide6.QtGui import QImage, QPixmap
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QSlider, QFileDialog, QMessageBox, QSizePolicy,
    QCheckBox,
)

try:
    from media_engine import PlayerEngine
except ImportError as e:
    PlayerEngine = None
    _IMPORT_ERROR = e


def format_time(seconds: float) -> str:
    """Detik float -> "MM:SS" (biar gampang dibaca di label posisi)."""
    if seconds < 0 or seconds != seconds:  # NaN guard
        seconds = 0.0
    m, s = divmod(int(seconds), 60)
    return f"{m:02d}:{s:02d}"


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
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 1000)  # dipetakan ke 0..duration
        self.position_slider.sliderPressed.connect(self._on_slider_pressed)
        self.position_slider.sliderReleased.connect(self._on_slider_released)
        seek_row.addWidget(self.position_slider, stretch=1)
        seek_row.addWidget(self.time_label)
        root.addLayout(seek_row)

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
            self, "Pilih file video", last_dir,
            "Video (*.mp4 *.mkv *.webm *.mov *.avi);;Semua file (*.*)",
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
        self.status_label.setText(
            f"{os.path.basename(path)} — {self.engine.width}x{self.engine.height} "
            f"@{self.engine.fps:.2f}fps, {audio_note}"
        )
        self.video_label.setText("")
        self.play_btn.setEnabled(True)
        self.play_btn.setText("▶ Play")
        self.stop_btn.setEnabled(True)
        self.next_frame_btn.setEnabled(True)
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
