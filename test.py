from __future__ import annotations

import ctypes
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import mss
import numpy as np
import psutil

import win32api
import win32con
import win32gui
import win32process

from PySide6.QtCore import Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QFont, QPainter, QPen
from PySide6.QtWidgets import QApplication, QWidget

from test_detection import ItemDetector, ItemMatchType
from image_recognition.dinov2_embedding import ImageRetriever, DinoV2, EmbeddingStore


# ============================================================
# CONFIGURATION
# ============================================================

# Exact process name shown in Task Manager.
TARGET_EXE = "discord.exe"


# ------------------------------------------------------------
# Detection
# ------------------------------------------------------------

# Put something like:
#
# TEMPLATE_PATH = Path("new_icon.png")
#
# If None, the program draws a test box so you can first verify
# that capturing + overlay work correctly.
TEMPLATE_PATH: Path | None = None

MATCH_THRESHOLD = 0.85
NMS_THRESHOLD = 0.30


# ------------------------------------------------------------
# Runtime
# ------------------------------------------------------------

CAPTURE_FPS = 30

# A few pixels of tolerance help with borderless-fullscreen
# windows and coordinate rounding.
FULLSCREEN_TOLERANCE = 4


# Press F8 to terminate this program.
EXIT_KEY = win32con.VK_F8


# ============================================================
# DATA TYPES
# ============================================================


@dataclass(frozen=True)
class MonitorInfo:
    left: int
    top: int
    right: int
    bottom: int
    device: str

    @property
    def width(self) -> int:
        return self.right - self.left

    @property
    def height(self) -> int:
        return self.bottom - self.top


@dataclass(frozen=True)
class CaptureState:
    hwnd: int
    monitor: MonitorInfo


@dataclass
class Detection:
    x1: int
    y1: int
    x2: int
    y2: int

    label: str
    amount: int


# ============================================================
# WINDOWS HELPERS
# ============================================================


def is_key_down(vk_code: int) -> bool:
    """
    Check whether a virtual key is physically down now.

    The high-order bit indicates the current state.
    """
    return bool(
        win32api.GetAsyncKeyState(vk_code)
        & 0x8000
    )


def unsafe_keyboard_state() -> bool:
    """
    Inputs that should immediately prevent capture.
    """

    windows_key = (
        is_key_down(win32con.VK_LWIN)
        or is_key_down(win32con.VK_RWIN)
    )

    alt_tab = (
        is_key_down(win32con.VK_MENU)
        and is_key_down(win32con.VK_TAB)
    )

    ctrl_escape = (
        is_key_down(win32con.VK_CONTROL)
        and is_key_down(win32con.VK_ESCAPE)
    )

    return (
        windows_key
        or alt_tab
        or ctrl_escape
    )


def get_process_name(hwnd: int) -> str | None:
    """
    Get executable name belonging to a HWND.
    """

    if not hwnd:
        return None

    try:
        _, pid = win32process.GetWindowThreadProcessId(
            hwnd
        )

        process = psutil.Process(pid)

        return process.name()

    except (
        psutil.NoSuchProcess,
        psutil.AccessDenied,
        OSError,
    ):
        return None


def get_monitor_info(hwnd: int) -> MonitorInfo | None:
    """
    Get the monitor containing the target window.
    """

    try:
        monitor_handle = win32api.MonitorFromWindow(
            hwnd,
            win32con.MONITOR_DEFAULTTONEAREST,
        )

        info = win32api.GetMonitorInfo(
            monitor_handle
        )

        left, top, right, bottom = info["Monitor"]

        return MonitorInfo(
            left=left,
            top=top,
            right=right,
            bottom=bottom,
            device=info.get(
                "Device",
                "",
            ),
        )

    except Exception:
        return None


def get_client_screen_rect(
    hwnd: int,
) -> tuple[int, int, int, int] | None:
    """
    Convert the application's client area into
    absolute desktop coordinates.
    """

    try:
        left, top, right, bottom = (
            win32gui.GetClientRect(hwnd)
        )

        screen_left, screen_top = (
            win32gui.ClientToScreen(
                hwnd,
                (left, top),
            )
        )

        screen_right, screen_bottom = (
            win32gui.ClientToScreen(
                hwnd,
                (right, bottom),
            )
        )

        return (
            screen_left,
            screen_top,
            screen_right,
            screen_bottom,
        )

    except win32gui.error:
        return None


def is_fullscreen(
    hwnd: int,
    monitor: MonitorInfo,
) -> bool:
    """
    Require the application's CLIENT AREA to cover
    the entire monitor.
    """

    client_rect = get_client_screen_rect(hwnd)

    if client_rect is None:
        return False

    left, top, right, bottom = client_rect

    tolerance = FULLSCREEN_TOLERANCE

    return (
        abs(left - monitor.left) <= tolerance
        and abs(top - monitor.top) <= tolerance
        and abs(right - monitor.right) <= tolerance
        and abs(bottom - monitor.bottom) <= tolerance
    )


def get_capture_state() -> CaptureState | None:
    """
    Central security gate.

    Capture is permitted ONLY if all checks pass.
    """

    # ------------------------------------------
    # 1. Sensitive keyboard shortcuts
    # ------------------------------------------

    if unsafe_keyboard_state():
        return None

    # ------------------------------------------
    # 2. Get foreground window
    # ------------------------------------------

    hwnd = win32gui.GetForegroundWindow()

    if not hwnd:
        return None

    # ------------------------------------------
    # 3. Must be visible
    # ------------------------------------------

    if not win32gui.IsWindowVisible(hwnd):
        return None

    # ------------------------------------------
    # 4. Must not be minimized
    # ------------------------------------------

    if win32gui.IsIconic(hwnd):
        return None

    # ------------------------------------------
    # 5. Must belong to requested executable
    # ------------------------------------------

    process_name = get_process_name(hwnd)

    if process_name is None:
        return None

    if process_name.lower() != TARGET_EXE.lower():
        return None

    # ------------------------------------------
    # 6. Find monitor
    # ------------------------------------------

    monitor = get_monitor_info(hwnd)

    if monitor is None:
        return None

    # ------------------------------------------
    # 7. Must be fullscreen
    # ------------------------------------------

    if not is_fullscreen(
        hwnd,
        monitor,
    ):
        return None

    return CaptureState(
        hwnd=hwnd,
        monitor=monitor,
    )


def same_capture_state(
    before: CaptureState,
    after: CaptureState | None,
) -> bool:

    if after is None:
        return False

    return (
        before.hwnd == after.hwnd
        and before.monitor == after.monitor
    )


# ============================================================
# CAPTURE WORKER
# ============================================================


class CaptureWorker(QThread):

    detections_ready = Signal(
        object,
        object,
    )

    capture_inactive = Signal()

    fatal_error = Signal(str)

    exit_requested = Signal()

    def __init__(
        self,
        detector: ItemDetector,
    ):
        super().__init__()

        self.detector = detector
        self._running = True

        self._exit_key_was_down = False

    def stop(self):
        self._running = False
        self.wait()

    def _check_exit_key(self) -> bool:

        down = is_key_down(EXIT_KEY)

        pressed = (
            down
            and not self._exit_key_was_down
        )

        self._exit_key_was_down = down

        return pressed

    def run(self):

        frame_duration = (
            1.0 / CAPTURE_FPS
        )

        capture_was_active = False

        try:

            # MSS must live inside this worker thread.
            with mss.mss() as screenshotter:

                while self._running:

                    loop_start = time.perf_counter()

                    # --------------------------------
                    # Global exit hotkey
                    # --------------------------------

                    if self._check_exit_key():

                        self.exit_requested.emit()

                        return

                    # --------------------------------
                    # SECURITY CHECK BEFORE CAPTURE
                    # --------------------------------

                    state_before = get_capture_state()

                    if state_before is None:

                        if capture_was_active:
                            self.capture_inactive.emit()

                        capture_was_active = False

                        self._sleep_until_next_frame(
                            loop_start,
                            frame_duration,
                        )

                        continue

                    monitor = state_before.monitor

                    capture_region = {
                        "left": monitor.left,
                        "top": monitor.top,
                        "width": monitor.width,
                        "height": monitor.height,
                    }

                    # =================================
                    # ACTUAL SCREEN CAPTURE
                    # =================================

                    screenshot = screenshotter.grab(
                        capture_region
                    )

                    # --------------------------------
                    # SECURITY CHECK AFTER CAPTURE
                    #
                    # If foreground changed during
                    # grab(), discard the frame.
                    # --------------------------------

                    state_after = get_capture_state()

                    if not same_capture_state(
                        state_before,
                        state_after,
                    ):

                        if capture_was_active:
                            self.capture_inactive.emit()

                        capture_was_active = False

                        del screenshot

                        self._sleep_until_next_frame(
                            loop_start,
                            frame_duration,
                        )

                        continue

                    # --------------------------------
                    # BGRA -> BGR
                    # --------------------------------

                    frame = np.asarray(
                        screenshot
                    )

                    frame_bgr = np.ascontiguousarray(
                        frame[:, :, :3]
                    )

                    # --------------------------------
                    # OpenCV detection
                    # --------------------------------

                    detections = self.detector.detect(
                        frame_bgr
                    )

                    # --------------------------------
                    # Send result to overlay
                    # --------------------------------

                    self.detections_ready.emit(
                        detections,
                        monitor,
                    )

                    capture_was_active = True

                    self._sleep_until_next_frame(
                        loop_start,
                        frame_duration,
                    )

        except Exception as exception:

            self.fatal_error.emit(
                repr(exception)
            )

    @staticmethod
    def _sleep_until_next_frame(
        loop_start: float,
        desired_duration: float,
    ):

        elapsed = (
            time.perf_counter()
            - loop_start
        )

        remaining = (
            desired_duration
            - elapsed
        )

        if remaining > 0:
            time.sleep(remaining)


# ============================================================
# TRANSPARENT OVERLAY
# ============================================================


class OverlayWindow(QWidget):

    WDA_EXCLUDEFROMCAPTURE = 0x00000011

    def __init__(self):

        super().__init__()

        self.boxes = []

        # ------------------------------------------------
        # Window configuration
        # ------------------------------------------------

        self.setWindowFlags(
            Qt.WindowType.FramelessWindowHint
            | Qt.WindowType.WindowStaysOnTopHint
            | Qt.WindowType.Tool
            | Qt.WindowType.WindowTransparentForInput
            | Qt.WindowType.WindowDoesNotAcceptFocus
        )

        # Transparent background
        self.setAttribute(
            Qt.WidgetAttribute.WA_TranslucentBackground,
            True,
        )

        # Mouse events pass through
        self.setAttribute(
            Qt.WidgetAttribute.WA_TransparentForMouseEvents,
            True,
        )

        # Do not steal foreground focus
        self.setAttribute(
            Qt.WidgetAttribute.WA_ShowWithoutActivating,
            True,
        )

        self.setFocusPolicy(
            Qt.FocusPolicy.NoFocus
        )

    # ========================================================
    # PUBLIC UPDATE
    # ========================================================

    @Slot(object, object)
    def set_detections(
        self,
        detections: list[Detection],
        monitor: MonitorInfo,
    ):

        screen = self._find_qt_screen(
            monitor.device
        )

        if screen is None:

            screen = (
                QApplication.primaryScreen()
            )

        if screen is None:
            return

        qt_geometry = screen.geometry()

        # ------------------------------------------------
        # Detection coordinates come from MSS in physical
        # capture pixels.
        #
        # Qt may use device-independent pixels under
        # Windows display scaling.
        #
        # Scale them to the Qt overlay size.
        # ------------------------------------------------

        scale_x = (
            qt_geometry.width()
            / monitor.width
        )

        scale_y = (
            qt_geometry.height()
            / monitor.height
        )

        scaled = []

        for detection in detections:
            if detection == []:
                continue
            
            scaled.append(
                Detection(
                    x1=round(
                        detection.x1
                        * scale_x
                    ),
                    y1=round(
                        detection.y1
                        * scale_y
                    ),
                    x2=round(
                        detection.x2
                        * scale_x
                    ),
                    y2=round(
                        detection.y2
                        * scale_y
                    ),
                    label=detection.label,
                    amount=detection.amount,
                )
            )

        self.boxes = scaled

        # Move transparent overlay onto
        # the target monitor.
        self.setGeometry(
            qt_geometry
        )

        if not self.isVisible():
            self.show()

        self.update()

    @Slot()
    def hide_overlay(self):

        self.boxes = []

        self.update()

        if self.isVisible():
            self.hide()

    # ========================================================
    # PAINT
    # ========================================================

    def paintEvent(self, event):

        painter = QPainter(self)

        painter.setRenderHint(
            QPainter.RenderHint.Antialiasing
        )

        pen = QPen(
            QColor(0, 255, 0),
        )

        pen.setWidth(3)

        painter.setPen(pen)

        font = QFont()
        font.setPointSize(11)
        font.setBold(True)

        painter.setFont(font)

        for detection in self.boxes:
            width = (
                detection.x2
                - detection.x1
            )

            height = (
                detection.y2
                - detection.y1
            )

            painter.drawRect(
                detection.x1,
                detection.y1,
                width,
                height,
            )

            label = (
                f"{detection.label} "
                f"{detection.amount:.2f}"
            )

            text_y = max(
                detection.y1 - 8,
                20,
            )

            painter.drawText(
                detection.x1,
                text_y,
                label,
            )

    # ========================================================
    # EXCLUDE OVERLAY FROM CAPTURE
    # ========================================================

    def showEvent(self, event):

        super().showEvent(event)

        self._exclude_from_capture()

    def _exclude_from_capture(self):

        try:

            hwnd = int(
                self.winId()
            )

            user32 = ctypes.windll.user32

            success = (
                user32.SetWindowDisplayAffinity(
                    hwnd,
                    self.WDA_EXCLUDEFROMCAPTURE,
                )
            )

            if not success:

                print(
                    "[WARNING] "
                    "Could not exclude overlay "
                    "from Windows capture."
                )

        except Exception as exception:

            print(
                "[WARNING] "
                "SetWindowDisplayAffinity failed:",
                exception,
            )

    # ========================================================
    # SCREEN MATCHING
    # ========================================================

    @staticmethod
    def _normalize_display_name(
        value: str,
    ) -> str:

        return (
            value
            .upper()
            .replace("\\\\.\\", "")
            .strip()
        )

    def _find_qt_screen(
        self,
        monitor_device: str,
    ):

        wanted = (
            self._normalize_display_name(
                monitor_device
            )
        )

        for screen in QApplication.screens():

            screen_name = (
                self._normalize_display_name(
                    screen.name()
                )
            )

            if screen_name == wanted:
                return screen

        return None


# ============================================================
# MAIN
# ============================================================


def main():

    print(
        "========================================"
    )
    print("Real-time Overlay Detector")
    print(
        "========================================"
    )

    print(
        f"Target executable: {TARGET_EXE}"
    )

    print(
        f"Capture FPS: {CAPTURE_FPS}"
    )

    print(
        "Press F8 to terminate."
    )

    print()

    if TARGET_EXE == "your_application.exe":

        print(
            "[WARNING] Change TARGET_EXE "
            "before using the program."
        )

    encoder = DinoV2(
        model_name="facebook/dinov2-small", 
        preprocessor_version="1.0",
    )

    embedding_store = EmbeddingStore(
        cache_path="cache/dinov2_small_cache_test.pt",
        input_dir="inputs",
        encoder=encoder,
    )

    stats = embedding_store.sync(
        batch_size=32, 
        force_hash_check=False, 
        rebuild_on_mismatch=True,
    )

    retriever = ImageRetriever(
        encoder=encoder,
        embedding_store=embedding_store,
        device_mode="auto",
        query_batch_size=32,
        reserved_vram=2.0,
        safety_factor=1.2,
    )

    detector = ItemDetector(
        retriever=retriever,
    )

    app = QApplication(sys.argv)

    # Keep running even when the overlay
    # temporarily becomes hidden.
    app.setQuitOnLastWindowClosed(False)

    overlay = OverlayWindow()

    worker = CaptureWorker(
        detector=detector
    )

    worker.detections_ready.connect(
        overlay.set_detections
    )

    worker.capture_inactive.connect(
        overlay.hide_overlay
    )

    worker.exit_requested.connect(
        app.quit
    )

    def handle_error(message: str):

        print(
            "[ERROR]",
            message,
        )

        app.quit()

    worker.fatal_error.connect(
        handle_error
    )

    worker.start()

    exit_code = app.exec()

    worker.stop()

    sys.exit(exit_code)


if __name__ == "__main__":
    main()