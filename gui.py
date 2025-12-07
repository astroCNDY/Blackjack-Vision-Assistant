import sys
import io
import json
from typing import Optional

import cv2
from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWebEngineWidgets import QWebEngineView
from PyQt5.QtCore import (
    QUrl,
    QBuffer,
    QIODevice,
    QRect,
    QPoint,
    QSettings,
)

from PIL import Image

from blackjack_core import (
    ShoeState,
    recommend_from_image,
    recommend_from_cards,
    create_openai_client,
    format_total_string,
    RANKS,
    SUITS,
)


def qpixmap_to_pil(pixmap: QtGui.QPixmap) -> Image.Image:
    buffer = QBuffer()
    buffer.open(QIODevice.ReadWrite)
    pixmap.save(buffer, b"PNG")
    pil_img = Image.open(io.BytesIO(buffer.data()))
    return pil_img.convert("RGB")


# ===== Card Edit Dialog =====

class CardEditDialog(QtWidgets.QDialog):
    def __init__(self, parent=None, initial_card: Optional[str] = None):
        super().__init__(parent)
        self.setWindowTitle("Edit card")
        self.setModal(True)

        if initial_card:
            init_rank = initial_card[:-1]
            init_suit = initial_card[-1]
        else:
            init_rank = "A"
            init_suit = "♠"

        layout = QtWidgets.QVBoxLayout(self)

        rank_label = QtWidgets.QLabel("Rank:")
        self.rank_box = QtWidgets.QComboBox()
        self.rank_box.addItems(RANKS)
        if init_rank in RANKS:
            self.rank_box.setCurrentText(init_rank)

        suit_label = QtWidgets.QLabel("Suit:")
        self.suit_box = QtWidgets.QComboBox()
        self.suit_box.addItems(SUITS)
        if init_suit in SUITS:
            self.suit_box.setCurrentText(init_suit)

        layout.addWidget(rank_label)
        layout.addWidget(self.rank_box)
        layout.addSpacing(6)
        layout.addWidget(suit_label)
        layout.addWidget(self.suit_box)
        layout.addStretch(1)

        btns = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btns.accepted.connect(self.accept)
        btns.rejected.connect(self.reject)
        layout.addWidget(btns)

    def get_card(self) -> str:
        return self.rank_box.currentText() + self.suit_box.currentText()


# ===== Card Button (card-looking widget) =====

class CardButton(QtWidgets.QPushButton):
    cardEdited = QtCore.pyqtSignal(str)

    def __init__(self, card: str, parent=None):
        super().__init__(card, parent)
        self.card = card
        self.setCursor(QtCore.Qt.PointingHandCursor)
        self.setMinimumWidth(44)
        self.setMinimumHeight(64)
        self._update_style()
        self.clicked.connect(self._edit_card)

    def _update_style(self):
        suit = self.card[-1] if self.card else ""
        text_color = "#dc2626" if suit in ("♥", "♦") else "#111827"
        self.setStyleSheet(
            f"""
            QPushButton {{
                background-color: #f9fafb;
                border-radius: 8px;
                border: 1px solid #9ca3af;
                padding: 6px 10px;
                min-width: 44px;
                min-height: 64px;
                color: {text_color};
                font-weight: 700;
                font-size: 15px;
            }}
            QPushButton:hover {{
                background-color: #e5e7eb;
            }}
            """
        )

    def _edit_card(self):
        dlg = CardEditDialog(self, self.card)
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            new_card = dlg.get_card()
            self.card = new_card
            self.setText(new_card)
            self._update_style()
            self.cardEdited.emit(new_card)


# ===== Selection Overlay (for region selection) =====

class SelectionOverlay(QtWidgets.QWidget):
    regionSelected = QtCore.pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._active = False
        self._start = QPoint()
        self._end = QPoint()
        self.setMouseTracking(True)
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.hide()

    def start_selection(self):
        self._active = True
        self._start = QPoint()
        self._end = QPoint()
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, False)
        self.show()
        self.raise_()
        self.setCursor(QtCore.Qt.CrossCursor)

    def stop_selection(self):
        self._active = False
        self.setAttribute(QtCore.Qt.WA_TransparentForMouseEvents, True)
        self.unsetCursor()
        self.hide()
        self.update()

    def mousePressEvent(self, event: QtGui.QMouseEvent):
        if self._active and event.button() == QtCore.Qt.LeftButton:
            self._start = event.pos()
            self._end = event.pos()
            self.update()

    def mouseMoveEvent(self, event: QtGui.QMouseEvent):
        if self._active and (event.buttons() & QtCore.Qt.LeftButton):
            self._end = event.pos()
            self.update()

    def mouseReleaseEvent(self, event: QtGui.QMouseEvent):
        if self._active and event.button() == QtCore.Qt.LeftButton:
            self._end = event.pos()
            rect = QRect(self._start, self._end).normalized()
            self.stop_selection()
            if rect.width() > 10 and rect.height() > 10:
                self.regionSelected.emit(rect)

    def paintEvent(self, event: QtGui.QPaintEvent):
        if not self._active:
            return
        painter = QtGui.QPainter(self)
        painter.setRenderHint(QtGui.QPainter.Antialiasing)
        rect = QRect(self._start, self._end).normalized()
        if rect.isNull():
            return
        pen = QtGui.QPen(QtGui.QColor("#f97316"))
        pen.setWidth(2)
        pen.setStyle(QtCore.Qt.DashLine)
        painter.setPen(pen)
        brush = QtGui.QBrush(QtGui.QColor(249, 115, 22, 60))
        painter.setBrush(brush)
        painter.drawRect(rect)


# ===== Browser Frame =====

class BrowserFrame(QtWidgets.QWidget):
    regionSelected = QtCore.pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view = QWebEngineView(self)
        layout.addWidget(self.view)

        self.overlay = SelectionOverlay(self.view)
        self.overlay.regionSelected.connect(self.regionSelected)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.view.rect())

    def load_url(self, url: str):
        self.view.setUrl(QUrl(url))

    def start_region_selection(self):
        self.overlay.setGeometry(self.view.rect())
        self.overlay.start_selection()

    def grab_for_analysis(self, region: Optional[QRect]) -> QtGui.QPixmap:
        pix = self.view.grab()
        if pix.isNull():
            return pix

        dpr = pix.devicePixelRatio() or 1.0

        if region is None:
            return pix

        scaled_rect = QRect(
            int(region.x() * dpr),
            int(region.y() * dpr),
            int(region.width() * dpr),
            int(region.height() * dpr),
        )

        bounds = QRect(0, 0, pix.width(), pix.height())
        scaled_rect = scaled_rect & bounds

        if scaled_rect.width() <= 0 or scaled_rect.height() <= 0:
            return pix

        cropped = pix.copy(scaled_rect)
        cropped.setDevicePixelRatio(dpr)
        return cropped


# ===== Camera Frame =====

class CameraFrame(QtWidgets.QWidget):
    regionSelected = QtCore.pyqtSignal(QRect)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.camera_index = 0
        self.cap = None

        layout = QtWidgets.QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)

        self.view_label = QtWidgets.QLabel("Camera preview")
        self.view_label.setAlignment(QtCore.Qt.AlignCenter)
        self.view_label.setMinimumSize(640, 360)
        layout.addWidget(self.view_label)

        self.overlay = SelectionOverlay(self.view_label)
        self.overlay.regionSelected.connect(self.regionSelected)

        self.timer = QtCore.QTimer(self)
        self.timer.timeout.connect(self._update_frame)

    def set_camera_index(self, index: int):
        self.camera_index = index

    def start_camera(self):
        if self.cap is not None:
            self.stop_camera()

        self.cap = cv2.VideoCapture(self.camera_index)
        if not self.cap.isOpened():
            self.view_label.setText(f"Could not open camera index {self.camera_index}")
            self.cap = None
            return

        self.timer.start(33)  # ~30 FPS

    def stop_camera(self):
        self.timer.stop()
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def _update_frame(self):
        if self.cap is None:
            return
        ret, frame = self.cap.read()
        if not ret:
            return
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qimg = QtGui.QImage(
            frame_rgb.data, w, h, bytes_per_line, QtGui.QImage.Format_RGB888
        )
        pix = QtGui.QPixmap.fromImage(qimg)
        scaled = pix.scaled(
            self.view_label.size(),
            QtCore.Qt.KeepAspectRatio,
            QtCore.Qt.SmoothTransformation,
        )
        self.view_label.setPixmap(scaled)

    def resizeEvent(self, event: QtGui.QResizeEvent):
        super().resizeEvent(event)
        self.overlay.setGeometry(self.view_label.rect())

    def start_region_selection(self):
        self.overlay.setGeometry(self.view_label.rect())
        self.overlay.start_selection()

    def grab_for_analysis(self, region: Optional[QRect]) -> QtGui.QPixmap:
        pix = self.view_label.grab()
        if pix.isNull():
            return pix

        dpr = pix.devicePixelRatio() or 1.0

        if region is None:
            return pix

        scaled_rect = QRect(
            int(region.x() * dpr),
            int(region.y() * dpr),
            int(region.width() * dpr),
            int(region.height() * dpr),
        )

        bounds = QRect(0, 0, pix.width(), pix.height())
        scaled_rect = scaled_rect & bounds

        if scaled_rect.width() <= 0 or scaled_rect.height() <= 0:
            return pix

        cropped = pix.copy(scaled_rect)
        cropped.setDevicePixelRatio(dpr)
        return cropped


# ===== Settings Dialog =====

class SettingsDialog(QtWidgets.QDialog):
    def __init__(self, parent=None,
                 default_url="",
                 dual_mode=False,
                 capture_source="web",
                 camera_index=0):
        super().__init__(parent)
        self.setWindowTitle("Preferences")
        self.setModal(True)
        self.resize(420, 260)

        layout = QtWidgets.QVBoxLayout(self)

        # URL
        url_label = QtWidgets.QLabel("Default blackjack URL:")
        self.url_edit = QtWidgets.QLineEdit(default_url)

        # Dual-region
        self.dual_checkbox = QtWidgets.QCheckBox(
            "Use separate regions for dealer and player"
        )
        self.dual_checkbox.setChecked(dual_mode)

        # Source
        source_group = QtWidgets.QGroupBox("Capture source")
        source_layout = QtWidgets.QVBoxLayout(source_group)
        self.web_radio = QtWidgets.QRadioButton("Web (online mode)")
        self.camera_radio = QtWidgets.QRadioButton("Camera (Continuity/USB/etc.)")
        if capture_source == "camera":
            self.camera_radio.setChecked(True)
        else:
            self.web_radio.setChecked(True)
        source_layout.addWidget(self.web_radio)
        source_layout.addWidget(self.camera_radio)

        # Camera index
        cam_label = QtWidgets.QLabel(
            "Camera index (0, 1, 2… – select the one that is your iPhone/Continuity Camera):"
        )
        self.cam_spin = QtWidgets.QSpinBox()
        self.cam_spin.setRange(0, 10)
        self.cam_spin.setValue(int(camera_index))

        layout.addWidget(url_label)
        layout.addWidget(self.url_edit)
        layout.addSpacing(6)
        layout.addWidget(self.dual_checkbox)
        layout.addSpacing(8)
        layout.addWidget(source_group)
        layout.addWidget(cam_label)
        layout.addWidget(self.cam_spin)
        layout.addStretch(1)

        btn_box = QtWidgets.QDialogButtonBox(
            QtWidgets.QDialogButtonBox.Ok | QtWidgets.QDialogButtonBox.Cancel
        )
        btn_box.accepted.connect(self.accept)
        btn_box.rejected.connect(self.reject)
        layout.addWidget(btn_box)

    def get_values(self):
        source = "camera" if self.camera_radio.isChecked() else "web"
        return (
            self.url_edit.text().strip(),
            self.dual_checkbox.isChecked(),
            source,
            self.cam_spin.value(),
        )


# ===== Main GUI =====

class BlackjackGUI(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Blackjack Vision Assistant")
        self.resize(1500, 850)

        self.settings = QSettings("BlackjackProject", "BlackjackVisionAssistant")
        self.default_url = self.settings.value("defaultUrl", "https://www.google.com")
        self.dual_region_mode = self.settings.value("dualRegionMode", False, type=bool)
        self.capture_source = self.settings.value("captureSource", "web")
        self.camera_index = int(self.settings.value("cameraIndex", 0))

        self.shoe = ShoeState(decks=6)

        self.capture_region: Optional[QRect] = None
        self.dealer_region: Optional[QRect] = None
        self.player_region: Optional[QRect] = None
        self.next_selection_target = "single"

        self.current_vision_raw: Optional[dict] = None
        self.last_result: Optional[dict] = None

        try:
            self.client = create_openai_client()
        except Exception as e:
            self.client = None
            QtWidgets.QMessageBox.warning(
                self,
                "OpenAI error",
                f"Could not initialise OpenAI client:\n{e}\n\n"
                "Scan will use a dummy local result instead.",
            )

        self._build_ui()
        self._apply_settings_to_ui()

    # ----- Helpers -----

    def active_frame(self):
        return self.camera_frame if self.capture_source == "camera" else self.web_frame

    # ----- UI build -----

    def _build_ui(self):
        central = QtWidgets.QWidget()
        self.setCentralWidget(central)
        main_layout = QtWidgets.QHBoxLayout(central)

        # Menu
        settings_action = QtWidgets.QAction("Preferences…", self)
        settings_action.triggered.connect(self.open_settings_dialog)
        menu_bar = self.menuBar()
        settings_menu = menu_bar.addMenu("Settings")
        settings_menu.addAction(settings_action)

        # Left: source stack (web + camera)
        left_layout = QtWidgets.QVBoxLayout()
        toolbar_layout = QtWidgets.QHBoxLayout()

        self.url_edit = QtWidgets.QLineEdit()
        self.url_edit.setPlaceholderText("Enter blackjack site URL…")
        self.url_edit.returnPressed.connect(self.load_url)

        load_button = QtWidgets.QPushButton("Go")
        load_button.clicked.connect(self.load_url)

        self.select_region_button = QtWidgets.QPushButton("Select region")
        self.select_region_button.clicked.connect(self.start_region_selection)

        capture_button = QtWidgets.QPushButton("Scan table")
        capture_button.clicked.connect(self.scan_table)

        reshuffle_button = QtWidgets.QPushButton("Reshuffle shoe")
        reshuffle_button.clicked.connect(self.reshuffle_shoe)

        toolbar_layout.addWidget(self.url_edit, 1)
        toolbar_layout.addWidget(load_button)
        toolbar_layout.addWidget(self.select_region_button)
        toolbar_layout.addWidget(capture_button)
        toolbar_layout.addWidget(reshuffle_button)

        # Source stack
        self.web_frame = BrowserFrame()
        self.camera_frame = CameraFrame()

        self.web_frame.regionSelected.connect(self.on_region_selected)
        self.camera_frame.regionSelected.connect(self.on_region_selected)

        self.source_stack = QtWidgets.QStackedWidget()
        self.source_stack.addWidget(self.web_frame)
        self.source_stack.addWidget(self.camera_frame)

        self.selection_mode_label = QtWidgets.QLabel("")
        self.selection_mode_label.setStyleSheet("color: #9ca3af; font-size: 10pt;")

        left_layout.addLayout(toolbar_layout)
        left_layout.addWidget(self.source_stack, 1)
        left_layout.addWidget(self.selection_mode_label)

        # Load default URL
        self.web_frame.load_url(self.default_url)

        # Right: dashboard
        right_layout = QtWidgets.QVBoxLayout()

        # Cards group
        cards_group = QtWidgets.QGroupBox("Cards")
        cards_layout = QtWidgets.QVBoxLayout(cards_group)

        # Dealer row
        dealer_row = QtWidgets.QHBoxLayout()
        dealer_label = QtWidgets.QLabel("Dealer:")
        dealer_label.setStyleSheet("font-weight: 600;")
        dealer_row.addWidget(dealer_label)

        self.dealer_cards_layout = QtWidgets.QHBoxLayout()
        self.dealer_cards_layout.setSpacing(6)
        dealer_row.addLayout(self.dealer_cards_layout)
        dealer_row.addStretch(1)

        self.dealer_total_label = QtWidgets.QLabel("Total: –")
        self.dealer_total_label.setStyleSheet("color: #9ca3af;")
        dealer_row.addWidget(self.dealer_total_label)

        self.dealer_add_button = QtWidgets.QPushButton("+")
        self.dealer_add_button.setFixedSize(26, 26)
        self.dealer_add_button.clicked.connect(self.on_add_dealer_card)
        dealer_row.addWidget(self.dealer_add_button)

        # Player row
        player_row = QtWidgets.QVBoxLayout()
        player_header_row = QtWidgets.QHBoxLayout()
        player_label = QtWidgets.QLabel("Player:")
        player_label.setStyleSheet("font-weight: 600;")
        player_header_row.addWidget(player_label)
        player_header_row.addStretch(1)

        self.player_total_label = QtWidgets.QLabel("Total: –")
        self.player_total_label.setStyleSheet("color: #9ca3af;")
        player_header_row.addWidget(self.player_total_label)

        self.player_add_button = QtWidgets.QPushButton("+")
        self.player_add_button.setFixedSize(26, 26)
        self.player_add_button.clicked.connect(self.on_add_player_card)
        player_header_row.addWidget(self.player_add_button)

        self.player_main_cards_layout = QtWidgets.QHBoxLayout()
        self.player_main_cards_layout.setSpacing(6)
        self.player_extra_hands_layout = QtWidgets.QVBoxLayout()

        player_row.addLayout(player_header_row)
        player_row.addLayout(self.player_main_cards_layout)
        player_row.addLayout(self.player_extra_hands_layout)

        cards_layout.addLayout(dealer_row)
        cards_layout.addSpacing(10)
        cards_layout.addLayout(player_row)

        # Stats group
        stats_group = QtWidgets.QGroupBox("Stats & Advice")
        stats_layout = QtWidgets.QVBoxLayout(stats_group)

        self.action_button = QtWidgets.QPushButton("Action: –")
        self.action_button.setEnabled(False)
        self.action_button.setMinimumHeight(40)

        self.winprob_button = QtWidgets.QPushButton("Win chance: –")
        self.winprob_button.setEnabled(False)
        self.winprob_button.setMinimumHeight(36)

        self.count_label = QtWidgets.QLabel(
            "Running count: 0 | True count: 0.00 | Decks remaining: 6.0"
        )

        stats_layout.addWidget(self.action_button)
        stats_layout.addWidget(self.winprob_button)
        stats_layout.addSpacing(8)
        stats_layout.addWidget(self.count_label)
        stats_layout.addStretch(1)

        # Raw JSON + explanation
        json_group = QtWidgets.QGroupBox("Raw model JSON (debug)")
        json_layout = QtWidgets.QVBoxLayout(json_group)
        self.cards_text = QtWidgets.QPlainTextEdit()
        self.cards_text.setReadOnly(True)
        json_layout.addWidget(self.cards_text)

        explanation_group = QtWidgets.QGroupBox("Why this move?")
        explanation_layout = QtWidgets.QVBoxLayout(explanation_group)
        self.explanation_label = QtWidgets.QLabel("Explanation: –")
        self.explanation_label.setWordWrap(True)
        self.explanation_label.setStyleSheet("color: #9ca3af;")
        explanation_layout.addWidget(self.explanation_label)

        right_layout.addWidget(cards_group, 2)
        right_layout.addWidget(stats_group, 1)
        right_layout.addWidget(json_group, 2)
        right_layout.addWidget(explanation_group, 1)

        main_layout.addLayout(left_layout, 3)
        main_layout.addLayout(right_layout, 2)

        self.statusBar().showMessage("Ready")

    # ----- Settings -----

    def _apply_settings_to_ui(self):
        self.url_edit.setText(self.default_url)
        if self.dual_region_mode:
            self.next_selection_target = "dealer"
        else:
            self.next_selection_target = "single"
        self._update_capture_source_ui()

    def _update_capture_source_ui(self):
        if self.capture_source == "camera":
            self.source_stack.setCurrentWidget(self.camera_frame)
            self.camera_frame.set_camera_index(self.camera_index)
            self.camera_frame.start_camera()
            base = "Camera mode"
        else:
            self.source_stack.setCurrentWidget(self.web_frame)
            self.camera_frame.stop_camera()
            base = "Web mode"

        if self.dual_region_mode:
            self.selection_mode_label.setText(
                f"Dual-region mode: next selection = {self.next_selection_target.capitalize()} area ({base})"
            )
        else:
            self.selection_mode_label.setText(
                f"Single-region mode ({base})"
            )

    def open_settings_dialog(self):
        dlg = SettingsDialog(
            self,
            default_url=self.default_url,
            dual_mode=self.dual_region_mode,
            capture_source=self.capture_source,
            camera_index=self.camera_index,
        )
        if dlg.exec_() == QtWidgets.QDialog.Accepted:
            url, dual, source, cam_index = dlg.get_values()
            if url:
                self.default_url = url
                self.web_frame.load_url(url)
                self.url_edit.setText(url)
            self.dual_region_mode = dual
            self.capture_source = source
            self.camera_index = int(cam_index)

            self.settings.setValue("defaultUrl", self.default_url)
            self.settings.setValue("dualRegionMode", self.dual_region_mode)
            self.settings.setValue("captureSource", self.capture_source)
            self.settings.setValue("cameraIndex", self.camera_index)

            self._apply_settings_to_ui()

    # ----- Card layout helpers -----

    def clear_layout(self, layout: QtWidgets.QLayout):
        while layout.count():
            item = layout.takeAt(0)
            w = item.widget()
            if w is not None:
                w.deleteLater()

    def _make_card_button(self, card: str,
                          owner: str,
                          index: int,
                          hand_index: Optional[int] = None) -> CardButton:
        btn = CardButton(card, self)

        def handler(new_card: str, owner=owner, idx=index, h_idx=hand_index):
            self.on_card_edited(owner, idx, h_idx, new_card)

        btn.cardEdited.connect(handler)
        return btn

    def update_cards_display(self, vision_raw: dict):
        dealer_up = vision_raw.get("dealer_up")
        dealer_down = vision_raw.get("dealer_down_known")
        dealer_all = vision_raw.get("dealer_all_cards") or []

        if not dealer_all:
            if dealer_up:
                dealer_all.append(dealer_up)
            if dealer_down:
                dealer_all.append(dealer_down)
            vision_raw["dealer_all_cards"] = dealer_all

        player = vision_raw.get("player") or []
        extra_hands = vision_raw.get("extra_player_hands") or []

        # Dealer
        self.clear_layout(self.dealer_cards_layout)
        if dealer_all:
            for i, c in enumerate(dealer_all):
                btn = self._make_card_button(c, owner="dealer", index=i)
                self.dealer_cards_layout.addWidget(btn)
        else:
            self.dealer_cards_layout.addWidget(QtWidgets.QLabel("–"))

        dealer_total = vision_raw.get("dealer_total") or format_total_string(dealer_all)
        vision_raw["dealer_total"] = dealer_total
        self.dealer_total_label.setText(f"Total: {dealer_total}")

        # Player main
        self.clear_layout(self.player_main_cards_layout)
        if player:
            for i, c in enumerate(player):
                btn = self._make_card_button(c, owner="player", index=i)
                self.player_main_cards_layout.addWidget(btn)
        else:
            self.player_main_cards_layout.addWidget(QtWidgets.QLabel("–"))

        player_total = vision_raw.get("player_total") or format_total_string(player)
        vision_raw["player_total"] = player_total
        self.player_total_label.setText(f"Total: {player_total}")

        # Extra hands
        self.clear_layout(self.player_extra_hands_layout)
        for h_idx, hand in enumerate(extra_hands):
            if not hand:
                continue
            row = QtWidgets.QHBoxLayout()
            hand_label = QtWidgets.QLabel(f"Hand {h_idx + 2}:")
            hand_label.setStyleSheet("font-size: 12px; color: #9ca3af;")
            row.addWidget(hand_label)
            for i, c in enumerate(hand):
                btn = self._make_card_button(c, owner="extra", index=i, hand_index=h_idx)
                row.addWidget(btn)
            row.addStretch(1)
            self.player_extra_hands_layout.addLayout(row)

    # ----- Card editing / adding -----

    def on_card_edited(self, owner: str, index: int,
                       hand_index: Optional[int], new_card: str):
        if self.current_vision_raw is None:
            return
        vr = self.current_vision_raw

        if owner == "dealer":
            dealer_all = vr.get("dealer_all_cards") or []
            while len(dealer_all) <= index:
                dealer_all.append("A♠")
            dealer_all[index] = new_card
            vr["dealer_all_cards"] = dealer_all
            vr["dealer_up"] = dealer_all[0] if dealer_all else None
            vr["dealer_down_known"] = dealer_all[1] if len(dealer_all) > 1 else None

        elif owner == "player":
            player = vr.get("player") or []
            while len(player) <= index:
                player.append("A♠")
            player[index] = new_card
            vr["player"] = player

        elif owner == "extra":
            extra = vr.get("extra_player_hands") or []
            while len(extra) <= (hand_index or 0):
                extra.append([])
            hand = extra[hand_index]
            while len(hand) <= index:
                hand.append("A♠")
            hand[index] = new_card
            extra[hand_index] = hand
            vr["extra_player_hands"] = extra

        self.recompute_from_current_cards()

    def on_add_dealer_card(self):
        if self.current_vision_raw is None:
            self.current_vision_raw = {
                "dealer_up": None,
                "dealer_down_known": None,
                "dealer_all_cards": [],
                "player": [],
                "extra_player_hands": [],
            }
        vr = self.current_vision_raw
        dealer_all = vr.get("dealer_all_cards") or []
        dlg = CardEditDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_card = dlg.get_card()
        dealer_all.append(new_card)
        vr["dealer_all_cards"] = dealer_all
        vr["dealer_up"] = dealer_all[0] if dealer_all else None
        vr["dealer_down_known"] = dealer_all[1] if len(dealer_all) > 1 else None
        self.recompute_from_current_cards()

    def on_add_player_card(self):
        if self.current_vision_raw is None:
            self.current_vision_raw = {
                "dealer_up": None,
                "dealer_down_known": None,
                "dealer_all_cards": [],
                "player": [],
                "extra_player_hands": [],
            }
        vr = self.current_vision_raw
        player = vr.get("player") or []
        dlg = CardEditDialog(self)
        if dlg.exec_() != QtWidgets.QDialog.Accepted:
            return
        new_card = dlg.get_card()
        player.append(new_card)
        vr["player"] = player
        self.recompute_from_current_cards()

    def recompute_from_current_cards(self):
        if self.current_vision_raw is None:
            return

        vision_copy = json.loads(json.dumps(self.current_vision_raw))

        if self.client is None:
            self.update_cards_display(vision_copy)
            return

        result = recommend_from_cards(
            vision_copy,
            self.shoe,
            self.client,
            include_explanation=True,
            update_shoe=False,
        )

        self.current_vision_raw = result.get("vision_raw", vision_copy)
        self.last_result = result
        self.update_from_result(result)

    # ----- Actions -----

    def load_url(self):
        url_text = self.url_edit.text().strip()
        if not url_text:
            return
        if not (url_text.startswith("http://") or url_text.startswith("https://")):
            url_text = "https://" + url_text
        self.web_frame.load_url(url_text)
        self.statusBar().showMessage(f"Loading {url_text}…", 3000)

    def start_region_selection(self):
        self.active_frame().start_region_selection()
        if self.dual_region_mode:
            self.statusBar().showMessage(
                f"Drag to select {self.next_selection_target} region…", 5000
            )
        else:
            self.statusBar().showMessage("Drag to select table region…", 5000)

    def on_region_selected(self, rect: QRect):
        if not self.dual_region_mode:
            self.capture_region = rect
            self.selection_mode_label.setText(
                f"Single-region mode ({'Camera' if self.capture_source=='camera' else 'Web'}): "
                f"region set (w={rect.width()}, h={rect.height()})"
            )
            self.statusBar().showMessage("Region selected (single)", 4000)
            return

        if self.next_selection_target == "dealer":
            self.dealer_region = rect
            self.next_selection_target = "player"
            self.selection_mode_label.setText(
                "Dual-region mode: Dealer region set. Next = Player region."
            )
            self.statusBar().showMessage("Dealer region selected", 4000)
        else:
            self.player_region = rect
            self.next_selection_target = "dealer"
            self.selection_mode_label.setText(
                "Dual-region mode: Player region set. Next = Dealer region."
            )
            self.statusBar().showMessage("Player region selected", 4000)

    def reshuffle_shoe(self):
        self.shoe = ShoeState(decks=6)
        self.count_label.setText(
            f"Running count: {self.shoe.running_count} | "
            f"True count: {self.shoe.true_count():.2f} | "
            f"Decks remaining: {self.shoe.decks_remaining():.1f}"
        )
        self.winprob_button.setText("Win chance: –")
        self.winprob_button.setStyleSheet(self._win_button_style("#4b5563"))
        self.action_button.setText("Action: –")
        self.action_button.setStyleSheet(self._action_button_style("#4b5563"))
        self.cards_text.clear()
        self.clear_layout(self.dealer_cards_layout)
        self.clear_layout(self.player_main_cards_layout)
        self.clear_layout(self.player_extra_hands_layout)
        self.dealer_total_label.setText("Total: –")
        self.player_total_label.setText("Total: –")
        self.explanation_label.setText("Explanation: –")
        self.current_vision_raw = None
        self.last_result = None
        self.statusBar().showMessage("Shoe reset", 3000)

    def _composite_for_dual_mode(self, frame_widget) -> Optional[QtGui.QPixmap]:
        if self.dealer_region is None or self.player_region is None:
            return None

        dealer_pix = frame_widget.grab_for_analysis(self.dealer_region)
        player_pix = frame_widget.grab_for_analysis(self.player_region)
        if dealer_pix.isNull() or player_pix.isNull():
            return None

        w = dealer_pix.width() + player_pix.width()
        h = max(dealer_pix.height(), player_pix.height())

        composite = QtGui.QPixmap(w, h)
        composite.fill(QtCore.Qt.transparent)
        p = QtGui.QPainter(composite)
        p.drawPixmap(0, 0, dealer_pix)
        p.drawPixmap(dealer_pix.width(), 0, player_pix)
        p.end()
        return composite

    def scan_table(self):
        frame_widget = self.active_frame()

        if self.dual_region_mode:
            composite = self._composite_for_dual_mode(frame_widget)
            if composite is None:
                QtWidgets.QMessageBox.warning(
                    self, "Error", "Dealer and player regions not fully set."
                )
                return
            pixmap = composite
        else:
            pixmap = frame_widget.grab_for_analysis(self.capture_region)

        if pixmap.isNull():
            QtWidgets.QMessageBox.warning(
                self, "Error", "Could not grab frame for analysis."
            )
            return

        pil_img = qpixmap_to_pil(pixmap)

        if self.client is None:
            result = self.fake_analyze_image(pil_img)
        else:
            try:
                result = recommend_from_image(pil_img, self.shoe, self.client, include_explanation=True)
            except Exception as e:
                QtWidgets.QMessageBox.warning(
                    self,
                    "Analysis error",
                    f"Error while calling model:\n{e}\n\n"
                    "Using dummy analysis instead.",
                )
                result = self.fake_analyze_image(pil_img)

        self.current_vision_raw = json.loads(
            json.dumps(result.get("vision_raw", {}))
        )
        result["vision_raw"] = self.current_vision_raw
        self.last_result = result

        self.update_from_result(result)
        self.statusBar().showMessage("Scan complete", 3000)

    def fake_analyze_image(self, img: Image.Image) -> dict:
        self.shoe.running_count += 1
        self.shoe.cards_seen += 2

        tc = round(self.shoe.true_count(), 2)
        decks_rem = self.shoe.decks_remaining()
        win_prob = 0.43 + 0.015 * tc

        player_cards = ["A♠", "3♦"]
        dealer_cards = ["10♣"]

        dealer_total = format_total_string(dealer_cards)
        player_total = format_total_string(player_cards)

        return {
            "vision_raw": {
                "dealer_up": "10♣",
                "dealer_down_known": None,
                "player": player_cards,
                "extra_player_hands": [],
                "dealer_all_cards": dealer_cards,
                "dealer_total": dealer_total,
                "player_total": player_total,
            },
            "counting": {
                "running_count": self.shoe.running_count,
                "true_count": tc,
                "decks_remaining_est": round(decks_rem, 2),
                "cards_seen": self.shoe.cards_seen,
            },
            "recommendation": {
                "bet_units": 1 if tc <= 0 else 4,
                "action": "H" if tc < 1 else "S",
                "insurance": (tc >= 3),
            },
            "win_probability": max(0.25, min(0.70, win_prob)),
            "explanation": "Dummy explanation: You have a soft 14 (A+3) vs dealer 10, "
                           "so hitting keeps your bust risk low while improving your hand.",
        }

    # ----- Styles -----

    def _action_button_style(self, bg: str) -> str:
        return f"""
        QPushButton {{
            background-color: {bg};
            border-radius: 8px;
            padding: 8px 14px;
            border: 1px solid #475569;
            color: #e5e7eb;
            font-weight: 600;
            font-size: 13px;
            text-align: left;
        }}
        QPushButton:disabled {{
            opacity: 0.9;
        }}
        """

    def _win_button_style(self, bg: str) -> str:
        return f"""
        QPushButton {{
            background-color: {bg};
            border-radius: 999px;
            padding: 6px 14px;
            border: 1px solid #1f2933;
            color: #0b1120;
            font-weight: 600;
            font-size: 12px;
        }}
        QPushButton:disabled {{
            opacity: 0.95;
        }}
        """

    # ----- Update from result -----

    def update_from_result(self, result: dict):
        vision_raw = result.get("vision_raw", {}) or {}
        counting = result.get("counting", {})
        recommendation = result.get("recommendation", {})

        self.update_cards_display(vision_raw)

        self.cards_text.setPlainText(
            json.dumps(vision_raw, indent=2, ensure_ascii=False)
        )

        action_code = recommendation.get("action", None)
        bet_units = recommendation.get("bet_units", None)
        insurance = recommendation.get("insurance", False)

        action_map = {
            "H": "Hit",
            "S": "Stand",
            "D": "Double",
            "P": "Split",
            "R": "Surrender",
            None: "–",
        }
        action_text = action_map.get(action_code, "–")

        ins_text = "Yes" if insurance else "No"
        bet_text = f"{bet_units} units" if bet_units is not None else "–"
        button_label = f"{action_text}  •  Bet: {bet_text}  •  Insurance: {ins_text}"
        self.action_button.setText(button_label)

        if action_text == "Hit":
            bg = "#2563eb"
        elif action_text == "Stand":
            bg = "#16a34a"
        elif action_text == "Double":
            bg = "#7c3aed"
        elif action_text == "Split":
            bg = "#0d9488"
        elif action_text == "Surrender":
            bg = "#dc2626"
        else:
            bg = "#4b5563"
        self.action_button.setStyleSheet(self._action_button_style(bg))

        rc = counting.get("running_count", 0)
        tc = counting.get("true_count", 0.0)
        decks_rem = counting.get("decks_remaining_est", 6.0)
        self.count_label.setText(
            f"Running count: {rc} | True count: {tc:.2f} | Decks remaining: {decks_rem:.1f}"
        )

        win_prob = result.get("win_probability", None)
        if win_prob is not None:
            pct = win_prob * 100.0
            self.winprob_button.setText(f"{pct:.1f}% win chance")
            if pct < 45.0:
                bg = "#f87171"
            elif pct < 55.0:
                bg = "#facc15"
            else:
                bg = "#4ade80"
            self.winprob_button.setStyleSheet(self._win_button_style(bg))
        else:
            self.winprob_button.setText("Win chance: –")
            self.winprob_button.setStyleSheet(self._win_button_style("#4b5563"))

        expl = result.get("explanation") or "No explanation available."
        self.explanation_label.setText(
            expl if expl.startswith("Explanation:") else f"Explanation: {expl}"
        )

    def closeEvent(self, event: QtGui.QCloseEvent):

        try:
            self.camera_frame.stop_camera()
        except Exception:
            pass
        super().closeEvent(event)


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)

    app.setStyleSheet(
        """
        QMainWindow {
            background-color: #020617;
        }
        QWidget {
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #e5e7eb;
            font-size: 11pt;
        }
        QLineEdit, QPlainTextEdit {
            background-color: #020617;
            border: 1px solid #334155;
            border-radius: 6px;
            padding: 4px 8px;
            selection-background-color: #4b5563;
        }
        QPlainTextEdit {
            font-family: "JetBrains Mono", "Fira Code", "Consolas", monospace;
            font-size: 10pt;
        }
        QPushButton {
            background-color: #0f172a;
            border-radius: 6px;
            padding: 6px 12px;
            border: 1px solid #475569;
        }
        QPushButton:hover {
            background-color: #1e293b;
        }
        QPushButton:pressed {
            background-color: #020617;
        }
        QGroupBox {
            border: 1px solid #334155;
            border-radius: 10px;
            margin-top: 8px;
            background-color: #020617;
        }
        QGroupBox::title {
            subcontrol-origin: margin;
            subcontrol-position: top left;
            padding: 0 6px;
            color: #9ca3af;
            font-weight: 600;
        }
        QStatusBar {
            background-color: #020617;
            color: #9ca3af;
        }
        """
    )

    window = BlackjackGUI()
    window.show()
    sys.exit(app.exec_())
