"""Ada Lovelace — PyQt6 desktop UI."""
from __future__ import annotations

import logging
import os
from datetime import datetime
from typing import Callable, Optional

from PyQt6.QtCore import (
    Qt, QSize, QTimer, QSizeF, pyqtSignal,
)
from PyQt6.QtGui import QColor, QPalette, QTextCursor
from PyQt6.QtWidgets import (
    QApplication, QButtonGroup, QComboBox, QFileDialog, QFrame,
    QHBoxLayout, QLabel, QLineEdit, QMainWindow, QPushButton,
    QRadioButton, QScrollArea, QSizePolicy, QSplitter,
    QTableWidget, QTableWidgetItem, QTabWidget, QTextBrowser,
    QTextEdit, QVBoxLayout, QWidget,
)

from ui.workers import ApiWorker, ChatWorker

logger = logging.getLogger(__name__)

# ── palette ───────────────────────────────────────────────
_BG      = "#090912"
_SURF    = "#0d0d1a"
_PANEL   = "#111120"
_CARD    = "#181830"
_BDR     = "#1a1a30"
_BDR2    = "#242445"
_ACCENT  = "#00c9a7"
_PURPLE  = "#7b68ee"
_DANGER  = "#ff5f5f"
_TEXT    = "#ddddf0"
_TEXT2   = "#a0a0c0"
_MUTED   = "#606080"
_USER_BG = "#14143a"
_USER_BD = "#22225a"

DARK_QSS = f"""
QMainWindow  {{ background: {_BG}; }}
QWidget      {{ background: {_BG}; color: {_TEXT};
               font-family: "Segoe UI", "SF Pro Display", sans-serif;
               font-size: 13px; }}
QFrame       {{ background: transparent; border: none; }}

/* header */
QWidget#header      {{ background: {_SURF}; border-bottom: 1px solid {_BDR}; }}
QLabel#logoGem      {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 {_ACCENT}, stop:1 {_PURPLE});
                       color: #000; font-size: 15px; font-weight: 800;
                       border-radius: 8px; qproperty-alignment: AlignCenter; }}
QLabel#logoName     {{ color: {_TEXT}; font-size: 15px; font-weight: 700;
                       background: transparent; }}
QLabel#logoTag      {{ color: {_MUTED}; font-size: 11px; background: transparent; }}
QLabel#pill         {{ color: {_MUTED}; border: 1px solid {_BDR2};
                       border-radius: 10px; padding: 2px 8px;
                       font-size: 10px; font-weight: 700; background: transparent; }}
QLabel#livePill     {{ color: {_ACCENT}; border: 1px solid rgba(0,201,167,100);
                       border-radius: 10px; padding: 2px 8px;
                       font-size: 10px; font-weight: 700;
                       background: rgba(0,201,167,25); }}

/* chat */
QWidget#chatPanel   {{ background: {_BG}; }}
QWidget#msgList     {{ background: {_BG}; }}
QScrollArea#chatScroll {{ background: {_BG}; border: none; }}

QFrame#asstBubble   {{ background: {_PANEL}; border: 1px solid {_BDR};
                       border-radius: 10px; }}
QFrame#userBubble   {{ background: {_USER_BG}; border: 1px solid {_USER_BD};
                       border-radius: 10px; }}

QLabel#aAvatar      {{ background: qlineargradient(x1:0,y1:0,x2:1,y2:1,
                        stop:0 {_ACCENT}, stop:1 {_PURPLE});
                       color: #000; font-weight: 800; font-size: 12px;
                       border-radius: 7px; qproperty-alignment: AlignCenter; }}
QLabel#uAvatar      {{ background: {_CARD}; color: {_MUTED};
                       font-weight: 700; font-size: 12px; border-radius: 7px;
                       border: 1px solid {_BDR2};
                       qproperty-alignment: AlignCenter; }}
QLabel#msgMeta      {{ color: {_MUTED}; font-size: 11px; background: transparent; }}
QLabel#userText     {{ color: {_TEXT}; font-size: 14px; background: transparent; }}
QTextBrowser#msgBrowser {{ background: transparent; border: none;
                           color: {_TEXT}; font-size: 13px; }}

/* input */
QWidget#inputArea   {{ background: {_SURF}; border-top: 1px solid {_BDR}; }}
QTextEdit#msgInput  {{ background: {_BG}; border: 1px solid {_BDR2};
                       border-radius: 10px; color: {_TEXT};
                       font-size: 14px; padding: 6px; }}
QTextEdit#msgInput:focus {{ border-color: {_ACCENT}; }}

/* buttons */
QPushButton#primaryBtn {{
  background: {_ACCENT}; color: #000; border: none;
  border-radius: 8px; padding: 8px 18px; font-weight: 600; font-size: 13px; }}
QPushButton#primaryBtn:hover    {{ background: #00ddb8; }}
QPushButton#primaryBtn:disabled {{ background: #1a4035; color: #4a8a77; }}
QPushButton#ghostBtn {{
  background: transparent; color: {_TEXT2};
  border: 1px solid {_BDR2}; border-radius: 8px;
  padding: 8px 18px; font-size: 13px; }}
QPushButton#ghostBtn:hover {{ background: {_CARD}; color: {_TEXT}; }}
QPushButton#chipBtn {{
  background: {_SURF}; color: {_TEXT2};
  border: 1px solid {_BDR2}; border-radius: 14px;
  padding: 5px 13px; font-size: 12px; }}
QPushButton#chipBtn:hover {{
  background: rgba(0,201,167,20); color: {_ACCENT};
  border-color: {_ACCENT}; }}

/* tools panel */
QTabWidget#toolsPanel           {{ background: {_SURF}; }}
QTabWidget#toolsPanel::pane     {{ background: {_SURF}; border: none;
                                   border-top: 1px solid {_BDR};
                                   border-left: 1px solid {_BDR}; }}
QTabBar                         {{ background: {_SURF}; }}
QTabBar::tab                    {{ background: {_SURF}; color: {_MUTED};
                                   padding: 10px 13px; border: none;
                                   border-bottom: 2px solid transparent;
                                   font-size: 12px; font-weight: 500;
                                   min-width: 58px; }}
QTabBar::tab:selected           {{ color: {_ACCENT};
                                   border-bottom: 2px solid {_ACCENT}; }}
QTabBar::tab:hover:!selected    {{ color: {_TEXT2}; }}

QScrollArea#toolScroll  {{ background: {_SURF}; border: none; }}
QWidget#toolContent     {{ background: {_SURF}; }}

/* form */
QLabel#sectionLabel {{ color: {_MUTED}; font-size: 10px; font-weight: 700;
                       background: transparent;
                       padding-bottom: 5px; border-bottom: 1px solid {_BDR}; }}
QLabel#fieldLabel   {{ color: {_MUTED}; font-size: 11px; font-weight: 600;
                       background: transparent; }}
QLineEdit {{
  background: {_BG}; border: 1px solid {_BDR2}; border-radius: 6px;
  color: {_TEXT}; padding: 7px 10px; font-size: 13px; }}
QLineEdit:focus {{ border-color: {_ACCENT}; }}
QComboBox {{
  background: {_BG}; border: 1px solid {_BDR2}; border-radius: 6px;
  color: {_TEXT}; padding: 7px 10px; font-size: 13px; }}
QComboBox::drop-down {{ border: none; padding-right: 6px; }}
QComboBox::down-arrow {{ image: none; border: none; width: 0; height: 0; }}
QComboBox QAbstractItemView {{
  background: {_CARD}; border: 1px solid {_BDR2};
  color: {_TEXT}; selection-background-color: {_BDR2}; outline: none; }}
QTextEdit#toolInput {{
  background: {_BG}; border: 1px solid {_BDR2}; border-radius: 6px;
  color: {_TEXT}; padding: 6px 10px; font-size: 13px; }}
QTextEdit#toolInput:focus {{ border-color: {_ACCENT}; }}
QTextEdit#resultEdit {{
  background: {_BG}; border: 1px solid {_BDR}; border-radius: 6px;
  color: {_TEXT2}; padding: 6px 10px;
  font-family: "Cascadia Code", "Consolas", "Courier New", monospace;
  font-size: 12px; }}

/* table */
QTableWidget {{
  background: {_BG}; gridline-color: {_BDR};
  border: 1px solid {_BDR}; border-radius: 6px;
  color: {_TEXT}; font-size: 12px; outline: none; }}
QTableWidget::item         {{ padding: 6px 8px; border-bottom: 1px solid {_BDR}; }}
QTableWidget::item:selected {{ background: {_BDR2}; }}
QHeaderView               {{ background: {_PANEL}; }}
QHeaderView::section      {{
  background: {_PANEL}; color: {_MUTED};
  font-size: 10px; font-weight: 700; padding: 6px 8px;
  border: none; border-bottom: 1px solid {_BDR}; }}

/* radio */
QRadioButton {{ color: {_TEXT2}; spacing: 6px; background: transparent; }}
QRadioButton::indicator {{
  width: 14px; height: 14px; border-radius: 7px;
  border: 1.5px solid {_BDR2}; background: {_BG}; }}
QRadioButton::indicator:checked {{ background: {_ACCENT}; border-color: {_ACCENT}; }}
QRadioButton:checked {{ color: {_ACCENT}; }}

/* scrollbars */
QScrollBar:vertical   {{ background: transparent; width: 4px;  border: none; margin: 0; }}
QScrollBar:horizontal {{ background: transparent; height: 4px; border: none; margin: 0; }}
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {{
  background: {_BDR2}; border-radius: 2px;
  min-height: 20px; min-width: 20px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; border: none; }}
QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}

/* splitter */
QSplitter::handle:horizontal {{ background: {_BDR}; width: 1px; }}
"""


# ── helpers ───────────────────────────────────────────────

def _section_label(title: str) -> QLabel:
    lbl = QLabel(title.upper())
    lbl.setObjectName("sectionLabel")
    return lbl


def _field_wrap(label_text: str, widget: QWidget) -> QWidget:
    w = QWidget()
    w.setStyleSheet("background: transparent;")
    lay = QVBoxLayout(w)
    lay.setContentsMargins(0, 0, 0, 0)
    lay.setSpacing(4)
    lbl = QLabel(label_text)
    lbl.setObjectName("fieldLabel")
    lay.addWidget(lbl)
    lay.addWidget(widget)
    return w


def _result_edit(height: int = 100) -> QTextEdit:
    t = QTextEdit()
    t.setObjectName("resultEdit")
    t.setReadOnly(True)
    t.setFixedHeight(height)
    return t


def _run_worker(fn: Callable, on_result: Callable, on_error: Callable,
                store: list) -> ApiWorker:
    w = ApiWorker(fn)
    w.result.connect(on_result)
    w.error_occurred.connect(on_error)
    w.start()
    store.append(w)
    return w


# ── adaptive text browser ─────────────────────────────────

class AdaptiveBrowser(QTextBrowser):
    """QTextBrowser that grows to fit its markdown content (no inner scrollbar)."""

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("msgBrowser")
        self.setReadOnly(True)
        self.setOpenExternalLinks(True)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.document().documentLayout().documentSizeChanged.connect(
            self._on_doc_size_changed
        )

    def _on_doc_size_changed(self, size: QSizeF) -> None:
        h = max(24, int(size.height()) + 10)
        self.setFixedHeight(h)
        self.updateGeometry()

    def sizeHint(self) -> QSize:
        w = max(self.width(), 100)
        doc = self.document()
        doc.setTextWidth(w)
        h = max(24, int(doc.size().height()) + 10)
        return QSize(w, h)

    def minimumSizeHint(self) -> QSize:
        return QSize(80, 24)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._on_doc_size_changed(self.document().size())


# ── chat input ────────────────────────────────────────────

class ChatInput(QTextEdit):
    """Enter sends, Shift+Enter inserts newline."""

    submit_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("msgInput")
        self.setPlaceholderText(
            "Ask Ada about macro, filings, market data, or analysis…"
            "  (Enter to send · Shift+Enter for new line)"
        )
        self.setAcceptRichText(False)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Minimum)
        self.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        self.setFixedHeight(44)
        self.document().contentsChanged.connect(self._adjust_height)

    def _adjust_height(self) -> None:
        h = int(self.document().size().height()) + 18
        self.setFixedHeight(max(44, min(h, 140)))

    def keyPressEvent(self, event) -> None:
        if (event.key() in (Qt.Key.Key_Return, Qt.Key.Key_Enter)
                and not (event.modifiers() & Qt.KeyboardModifier.ShiftModifier)):
            self.submit_requested.emit()
            return
        super().keyPressEvent(event)


# ── message bubble ────────────────────────────────────────

class MessageWidget(QFrame):
    """A single chat message (user or assistant)."""

    def __init__(self, role: str, parent=None) -> None:
        super().__init__(parent)
        self.role = role
        self._buffer = ""
        self._setup_ui()

    def _setup_ui(self) -> None:
        outer = QHBoxLayout(self)
        outer.setContentsMargins(8, 4, 8, 4)
        outer.setSpacing(10)

        now = datetime.now().strftime("%H:%M")

        if self.role == "assistant":
            avatar = QLabel("A")
            avatar.setObjectName("aAvatar")
            avatar.setFixedSize(28, 28)

            bubble = QFrame()
            bubble.setObjectName("asstBubble")
            b_lay = QVBoxLayout(bubble)
            b_lay.setContentsMargins(12, 8, 12, 8)
            b_lay.setSpacing(4)

            meta = QLabel(f"Ada  ·  {now}")
            meta.setObjectName("msgMeta")

            self._browser = AdaptiveBrowser()
            b_lay.addWidget(meta)
            b_lay.addWidget(self._browser)

            outer.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)
            outer.addWidget(bubble, 1)

        else:
            avatar = QLabel("U")
            avatar.setObjectName("uAvatar")
            avatar.setFixedSize(28, 28)

            bubble = QFrame()
            bubble.setObjectName("userBubble")
            b_lay = QVBoxLayout(bubble)
            b_lay.setContentsMargins(12, 8, 12, 8)
            b_lay.setSpacing(4)

            meta = QLabel(f"You  ·  {now}")
            meta.setObjectName("msgMeta")
            meta.setAlignment(Qt.AlignmentFlag.AlignRight)

            self._text_lbl = QLabel()
            self._text_lbl.setObjectName("userText")
            self._text_lbl.setWordWrap(True)
            self._text_lbl.setTextInteractionFlags(
                Qt.TextInteractionFlag.TextSelectableByMouse
            )
            b_lay.addWidget(meta)
            b_lay.addWidget(self._text_lbl)

            bubble.setMaximumWidth(520)
            outer.addStretch(1)
            outer.addWidget(bubble)
            outer.addWidget(avatar, 0, Qt.AlignmentFlag.AlignTop)

    def set_content(self, text: str) -> None:
        self._buffer = text
        if self.role == "assistant":
            self._browser.setMarkdown(text)
        else:
            self._text_lbl.setText(text)

    def append_token(self, token: str) -> None:
        self._buffer += token
        cursor = self._browser.textCursor()
        cursor.movePosition(QTextCursor.MoveOperation.End)
        cursor.insertText(token)
        self._browser.setTextCursor(cursor)

    def finalize(self) -> None:
        self._browser.setMarkdown(self._buffer)


# ── welcome screen ────────────────────────────────────────

class WelcomeWidget(QWidget):
    chip_clicked = pyqtSignal(str)

    _PROMPTS = [
        "Macro snapshot — recession risk?",
        "Summarize AAPL latest 10-K",
        "Analyze NVDA earnings",
        "What's the yield curve saying?",
        "Compare MSFT vs GOOGL",
        "What's in my knowledge base?",
    ]

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setAlignment(Qt.AlignmentFlag.AlignCenter)
        lay.setSpacing(16)
        lay.setContentsMargins(40, 60, 40, 60)

        gem = QLabel("◆")
        gem.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gem.setStyleSheet(f"""
            font-size: 28px; color: {_ACCENT};
            background: rgba(0,201,167,20);
            border: 1px solid rgba(0,201,167,70);
            border-radius: 18px; padding: 14px;
        """)
        gem.setFixedSize(72, 72)

        title = QLabel("Ask Ada anything")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet(
            f"font-size: 22px; font-weight: 700; color: {_TEXT}; background: transparent;"
        )

        sub = QLabel(
            "Grounded in live macro data, SEC filings, and market quotes.\n"
            "Ask for analysis, summaries, or data lookups."
        )
        sub.setAlignment(Qt.AlignmentFlag.AlignCenter)
        sub.setStyleSheet(
            f"font-size: 13px; color: {_TEXT2}; background: transparent; line-height: 1.6;"
        )
        sub.setWordWrap(True)

        chips_w = QWidget()
        chips_w.setStyleSheet("background: transparent;")
        rows = QVBoxLayout(chips_w)
        rows.setSpacing(6)
        rows.setContentsMargins(0, 0, 0, 0)
        for i in range(0, len(self._PROMPTS), 3):
            row = QHBoxLayout()
            row.setSpacing(8)
            row.setAlignment(Qt.AlignmentFlag.AlignCenter)
            for prompt in self._PROMPTS[i:i + 3]:
                btn = QPushButton(prompt)
                btn.setObjectName("chipBtn")
                btn.clicked.connect(lambda _, p=prompt: self.chip_clicked.emit(p))
                row.addWidget(btn)
            rows.addLayout(row)

        lay.addStretch(1)
        lay.addWidget(gem, 0, Qt.AlignmentFlag.AlignCenter)
        lay.addWidget(title)
        lay.addWidget(sub)
        lay.addWidget(chips_w)
        lay.addStretch(1)


# ── chat panel ────────────────────────────────────────────

class ChatPanel(QWidget):
    message_ready = pyqtSignal(str)
    clear_requested = pyqtSignal()

    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("chatPanel")
        self._messages: list[MessageWidget] = []
        self._welcome: Optional[WelcomeWidget] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(0)

        self._scroll = QScrollArea()
        self._scroll.setObjectName("chatScroll")
        self._scroll.setWidgetResizable(True)
        self._scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        self._container = QWidget()
        self._container.setObjectName("msgList")
        self._msg_lay = QVBoxLayout(self._container)
        self._msg_lay.setContentsMargins(12, 12, 12, 12)
        self._msg_lay.setSpacing(8)

        self._welcome = WelcomeWidget()
        self._welcome.chip_clicked.connect(self._on_chip)
        self._msg_lay.addWidget(self._welcome)
        self._msg_lay.addStretch(1)

        self._scroll.setWidget(self._container)
        lay.addWidget(self._scroll, 1)

        # Input area
        input_area = QWidget()
        input_area.setObjectName("inputArea")
        in_lay = QVBoxLayout(input_area)
        in_lay.setContentsMargins(16, 12, 16, 14)
        in_lay.setSpacing(0)

        row = QHBoxLayout()
        row.setSpacing(8)

        self._input = ChatInput()
        self._input.submit_requested.connect(self._on_submit)

        self._send_btn = QPushButton("Send")
        self._send_btn.setObjectName("primaryBtn")
        self._send_btn.setFixedSize(80, 44)
        self._send_btn.clicked.connect(self._on_submit)

        self._clear_btn = QPushButton("Clear")
        self._clear_btn.setObjectName("ghostBtn")
        self._clear_btn.setFixedSize(70, 44)
        self._clear_btn.clicked.connect(self._on_clear)

        row.addWidget(self._input, 1)
        row.addWidget(self._send_btn)
        row.addWidget(self._clear_btn)
        in_lay.addLayout(row)
        lay.addWidget(input_area)

    def _on_chip(self, text: str) -> None:
        self._input.setPlainText(text)
        self._input.setFocus()

    def _on_submit(self) -> None:
        text = self._input.toPlainText().strip()
        if not text:
            return
        self._input.clear()
        self._input.setFixedHeight(44)
        self.message_ready.emit(text)

    def _on_clear(self) -> None:
        for msg in self._messages:
            self._msg_lay.removeWidget(msg)
            msg.deleteLater()
        self._messages.clear()

        self._welcome = WelcomeWidget()
        self._welcome.chip_clicked.connect(self._on_chip)
        self._msg_lay.insertWidget(0, self._welcome)
        self.clear_requested.emit()

    def _hide_welcome(self) -> None:
        if self._welcome and self._welcome.isVisible():
            self._welcome.hide()
            self._msg_lay.removeWidget(self._welcome)

    def add_user_message(self, text: str) -> None:
        self._hide_welcome()
        w = MessageWidget("user")
        w.set_content(text)
        self._messages.append(w)
        self._msg_lay.insertWidget(self._msg_lay.count() - 1, w)
        self._scroll_bottom()

    def add_assistant_placeholder(self) -> MessageWidget:
        w = MessageWidget("assistant")
        self._messages.append(w)
        self._msg_lay.insertWidget(self._msg_lay.count() - 1, w)
        self._scroll_bottom()
        return w

    def _scroll_bottom(self) -> None:
        QTimer.singleShot(30, lambda: (
            self._scroll.verticalScrollBar().setValue(
                self._scroll.verticalScrollBar().maximum()
            )
        ))

    def set_input_enabled(self, enabled: bool) -> None:
        self._input.setEnabled(enabled)
        self._send_btn.setEnabled(enabled)
        self._send_btn.setText("Send" if enabled else "…")


# ── tool tab: market data ─────────────────────────────────

class MarketTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        c = QWidget()
        c.setObjectName("toolContent")
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(_section_label("Stock Quote"))
        self._q_ticker = QLineEdit()
        self._q_ticker.setPlaceholderText("AAPL")
        self._q_ticker.textChanged.connect(lambda t: self._q_ticker.setText(t.upper()))
        self._q_ticker.returnPressed.connect(self._fetch_quote)
        lay.addWidget(_field_wrap("Ticker", self._q_ticker))
        self._q_btn = QPushButton("Get Quote")
        self._q_btn.setObjectName("primaryBtn")
        self._q_btn.clicked.connect(self._fetch_quote)
        lay.addWidget(self._q_btn)
        self._q_result = _result_edit(110)
        lay.addWidget(self._q_result)

        lay.addWidget(_section_label("Market News"))
        self._n_ticker = QLineEdit()
        self._n_ticker.setPlaceholderText("Leave blank for market-wide news")
        self._n_ticker.textChanged.connect(lambda t: self._n_ticker.setText(t.upper()))
        self._n_ticker.returnPressed.connect(self._fetch_news)
        lay.addWidget(_field_wrap("Ticker (optional)", self._n_ticker))
        self._n_btn = QPushButton("Get News")
        self._n_btn.setObjectName("primaryBtn")
        self._n_btn.clicked.connect(self._fetch_news)
        lay.addWidget(self._n_btn)
        self._n_result = _result_edit(180)
        lay.addWidget(self._n_result)

        lay.addStretch(1)
        scroll.setWidget(c)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _fetch_quote(self) -> None:
        ticker = self._q_ticker.text().strip().upper()
        if not ticker:
            return
        self._q_btn.setEnabled(False)
        self._q_btn.setText("Fetching…")
        self._q_result.clear()

        def do():
            from data_sources.polygon_client import PolygonClient
            return PolygonClient().format_quote_text(ticker)

        def done():
            self._q_btn.setEnabled(True)
            self._q_btn.setText("Get Quote")

        w = _run_worker(do,
                        lambda r: self._q_result.setPlainText(r),
                        lambda e: self._q_result.setPlainText(f"Error: {e}"),
                        self._workers)
        w.finished.connect(done)

    def _fetch_news(self) -> None:
        ticker = self._n_ticker.text().strip().upper() or None
        self._n_btn.setEnabled(False)
        self._n_btn.setText("Fetching…")
        self._n_result.clear()

        def do():
            from data_sources.polygon_client import PolygonClient
            return PolygonClient().format_news_text(ticker)

        def done():
            self._n_btn.setEnabled(True)
            self._n_btn.setText("Get News")

        w = _run_worker(do,
                        lambda r: self._n_result.setPlainText(r),
                        lambda e: self._n_result.setPlainText(f"Error: {e}"),
                        self._workers)
        w.finished.connect(done)


# ── tool tab: knowledge base ──────────────────────────────

class KnowledgeTab(QWidget):
    def __init__(self, get_agent: Callable, parent=None) -> None:
        super().__init__(parent)
        self._get_agent = get_agent
        self._workers: list = []
        self._pending_file: Optional[str] = None
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        c = QWidget()
        c.setObjectName("toolContent")
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        # Filing ingestion
        lay.addWidget(_section_label("SEC Filing Ingestion"))
        self._f_ticker = QLineEdit()
        self._f_ticker.setPlaceholderText("AAPL")
        self._f_ticker.textChanged.connect(lambda t: self._f_ticker.setText(t.upper()))
        lay.addWidget(_field_wrap("Ticker", self._f_ticker))
        self._f_form = QComboBox()
        self._f_form.addItems(["10-K — Annual Report", "10-Q — Quarterly Report"])
        lay.addWidget(_field_wrap("Form Type", self._f_form))
        self._f_btn = QPushButton("Fetch & Ingest")
        self._f_btn.setObjectName("primaryBtn")
        self._f_btn.clicked.connect(self._ingest_filing)
        lay.addWidget(self._f_btn)
        self._f_result = _result_edit(80)
        lay.addWidget(self._f_result)

        # File upload
        lay.addWidget(_section_label("Upload Document"))
        self._file_btn = QPushButton("Choose File  (PDF · TXT · MD · CSV)")
        self._file_btn.setObjectName("ghostBtn")
        self._file_btn.clicked.connect(self._choose_file)
        lay.addWidget(self._file_btn)
        self._file_name = QLabel("No file selected")
        self._file_name.setObjectName("fieldLabel")
        lay.addWidget(self._file_name)
        self._file_ingest_btn = QPushButton("Ingest File")
        self._file_ingest_btn.setObjectName("primaryBtn")
        self._file_ingest_btn.setEnabled(False)
        self._file_ingest_btn.clicked.connect(self._ingest_file)
        lay.addWidget(self._file_ingest_btn)
        self._file_result = _result_edit(80)
        lay.addWidget(self._file_result)

        # Paste text
        lay.addWidget(_section_label("Paste Text"))
        self._text_input = QTextEdit()
        self._text_input.setObjectName("toolInput")
        self._text_input.setPlaceholderText(
            "Paste research notes, earnings transcripts, etc."
        )
        self._text_input.setFixedHeight(90)
        lay.addWidget(_field_wrap("Content", self._text_input))
        self._text_src = QLineEdit()
        self._text_src.setPlaceholderText("e.g. Q3 earnings call")
        lay.addWidget(_field_wrap("Source Label", self._text_src))
        self._text_btn = QPushButton("Ingest Text")
        self._text_btn.setObjectName("primaryBtn")
        self._text_btn.clicked.connect(self._ingest_text)
        lay.addWidget(self._text_btn)
        self._text_result = _result_edit(60)
        lay.addWidget(self._text_result)

        # Stats
        lay.addWidget(_section_label("Knowledge Base"))
        self._stats_btn = QPushButton("Refresh Stats")
        self._stats_btn.setObjectName("ghostBtn")
        self._stats_btn.clicked.connect(self._refresh_stats)
        lay.addWidget(self._stats_btn)
        self._stats_result = _result_edit(50)
        lay.addWidget(self._stats_result)

        lay.addStretch(1)
        scroll.setWidget(c)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _ingest_filing(self) -> None:
        ticker = self._f_ticker.text().strip().upper()
        if not ticker:
            self._f_result.setPlainText("Enter a ticker first.")
            return
        form = self._f_form.currentText().split(" ")[0]
        self._f_btn.setEnabled(False)
        self._f_btn.setText("Fetching…")

        def do():
            agent = self._get_agent()
            result = agent.ingest_filing(ticker, form)
            stats = agent.knowledge_base_stats()
            return f"{result}\n{stats}"

        def done():
            self._f_btn.setEnabled(True)
            self._f_btn.setText("Fetch & Ingest")

        w = _run_worker(do,
                        lambda r: self._f_result.setPlainText(r),
                        lambda e: self._f_result.setPlainText(f"Error: {e}"),
                        self._workers)
        w.finished.connect(done)

    def _choose_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "Select Document", "",
            "Documents (*.pdf *.txt *.md *.csv);;All Files (*)"
        )
        if path:
            self._pending_file = path
            self._file_name.setText(f"→ {os.path.basename(path)}")
            self._file_ingest_btn.setEnabled(True)

    def _ingest_file(self) -> None:
        if not self._pending_file:
            return
        path = self._pending_file
        self._file_ingest_btn.setEnabled(False)
        self._file_ingest_btn.setText("Ingesting…")

        def do():
            agent = self._get_agent()
            if path.lower().endswith(".pdf"):
                result = agent.ingest_pdf(path)
            else:
                with open(path, "r", encoding="utf-8", errors="replace") as fh:
                    text = fh.read()
                result = agent.ingest_text(text, source=os.path.basename(path))
            return f"{result}\n{agent.knowledge_base_stats()}"

        def done():
            self._file_ingest_btn.setEnabled(True)
            self._file_ingest_btn.setText("Ingest File")

        w = _run_worker(do,
                        lambda r: self._file_result.setPlainText(r),
                        lambda e: self._file_result.setPlainText(f"Error: {e}"),
                        self._workers)
        w.finished.connect(done)

    def _ingest_text(self) -> None:
        text = self._text_input.toPlainText().strip()
        if not text:
            self._text_result.setPlainText("Paste some text first.")
            return
        source = self._text_src.text().strip() or "manual"
        self._text_btn.setEnabled(False)
        self._text_btn.setText("Ingesting…")

        def do():
            agent = self._get_agent()
            result = agent.ingest_text(text, source=source)
            return f"{result}\n{agent.knowledge_base_stats()}"

        def done():
            self._text_btn.setEnabled(True)
            self._text_btn.setText("Ingest Text")
            self._text_input.clear()

        w = _run_worker(do,
                        lambda r: self._text_result.setPlainText(r),
                        lambda e: self._text_result.setPlainText(f"Error: {e}"),
                        self._workers)
        w.finished.connect(done)

    def _refresh_stats(self) -> None:
        self._stats_btn.setEnabled(False)
        w = _run_worker(
            lambda: self._get_agent().knowledge_base_stats(),
            lambda r: self._stats_result.setPlainText(r),
            lambda e: self._stats_result.setPlainText(f"Error: {e}"),
            self._workers,
        )
        w.finished.connect(lambda: self._stats_btn.setEnabled(True))


# ── tool tab: macro dashboard ─────────────────────────────

class MacroTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        lay = QVBoxLayout(self)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        hdr = QHBoxLayout()
        hdr.addWidget(_section_label("Live FRED Indicators"))
        hdr.addStretch(1)
        self._refresh_btn = QPushButton("Refresh")
        self._refresh_btn.setObjectName("primaryBtn")
        self._refresh_btn.setFixedHeight(30)
        self._refresh_btn.clicked.connect(self._fetch)
        hdr.addWidget(self._refresh_btn)
        lay.addLayout(hdr)

        self._table = QTableWidget(0, 3)
        self._table.setHorizontalHeaderLabels(["Indicator", "Series ID", "Latest"])
        self._table.horizontalHeader().setStretchLastSection(False)
        self._table.horizontalHeader().setMinimumSectionSize(60)
        self._table.setColumnWidth(0, 180)
        self._table.setColumnWidth(1, 90)
        self._table.setColumnWidth(2, 70)
        self._table.verticalHeader().setVisible(False)
        self._table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self._table.setSelectionBehavior(
            QTableWidget.SelectionBehavior.SelectRows
        )
        lay.addWidget(self._table, 1)

    def _fetch(self) -> None:
        self._refresh_btn.setEnabled(False)
        self._refresh_btn.setText("Loading…")

        def do():
            from data_sources import FREDClient
            fred = FREDClient()
            snap = fred.macro_snapshot()
            return [
                (info["label"], info["series_id"],
                 f"{info['value']:.2f}" if info["value"] is not None else "N/A")
                for info in snap.values()
            ]

        def on_result(rows):
            self._table.setRowCount(len(rows))
            for r, (label, sid, val) in enumerate(rows):
                self._table.setItem(r, 0, QTableWidgetItem(label))
                self._table.setItem(r, 1, QTableWidgetItem(sid))
                val_item = QTableWidgetItem(val)
                val_item.setForeground(QColor(_ACCENT))
                self._table.setItem(r, 2, val_item)

        def done():
            self._refresh_btn.setEnabled(True)
            self._refresh_btn.setText("Refresh")

        w = _run_worker(do, on_result,
                        lambda e: logger.error("Macro fetch: %s", e),
                        self._workers)
        w.finished.connect(done)


# ── tool tab: deep analysis ───────────────────────────────

class AnalysisTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._workers: list = []
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        c = QWidget()
        c.setObjectName("toolContent")
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(10)

        lay.addWidget(_section_label("Market Bridge Pipeline"))

        self._ticker = QLineEdit()
        self._ticker.setPlaceholderText("AAPL")
        self._ticker.textChanged.connect(lambda t: self._ticker.setText(t.upper()))
        lay.addWidget(_field_wrap("Ticker", self._ticker))

        # Type radios
        type_w = QWidget()
        type_w.setStyleSheet("background: transparent;")
        type_lay = QVBoxLayout(type_w)
        type_lay.setContentsMargins(0, 0, 0, 0)
        type_lay.setSpacing(4)
        lbl = QLabel("Analysis Type")
        lbl.setObjectName("fieldLabel")
        type_lay.addWidget(lbl)

        self._type_group = QButtonGroup(self)
        for label, value in [
            ("Earnings (8-K)", "earnings"),
            ("Annual (10-K)", "annual"),
            ("Quarterly (10-Q)", "quarterly"),
            ("Custom Query", "custom"),
        ]:
            rb = QRadioButton(label)
            rb.setProperty("value", value)
            self._type_group.addButton(rb)
            type_lay.addWidget(rb)
        self._type_group.buttons()[0].setChecked(True)
        self._type_group.buttonClicked.connect(self._on_type_change)
        lay.addWidget(type_w)

        self._q_wrap = QWidget()
        self._q_wrap.setStyleSheet("background: transparent;")
        q_lay = QVBoxLayout(self._q_wrap)
        q_lay.setContentsMargins(0, 0, 0, 0)
        self._question = QLineEdit()
        self._question.setPlaceholderText("What are the main AI revenue drivers?")
        q_lay.addWidget(_field_wrap("Custom Question", self._question))
        self._q_wrap.setVisible(False)
        lay.addWidget(self._q_wrap)

        self._run_btn = QPushButton("Run Analysis")
        self._run_btn.setObjectName("primaryBtn")
        self._run_btn.clicked.connect(self._run)
        lay.addWidget(self._run_btn)

        # Result area
        self._result_frame = QFrame()
        self._result_frame.setObjectName("asstBubble")
        self._result_frame.setVisible(False)
        rf_lay = QVBoxLayout(self._result_frame)
        rf_lay.setContentsMargins(12, 10, 12, 10)
        rf_lay.setSpacing(6)

        self._signal_lbl = QLabel()
        self._signal_lbl.setStyleSheet("font-weight: 700; font-size: 14px; background: transparent;")
        self._meta_lbl = QLabel()
        self._meta_lbl.setObjectName("msgMeta")
        self._summary = AdaptiveBrowser()
        rf_lay.addWidget(self._signal_lbl)
        rf_lay.addWidget(self._meta_lbl)
        rf_lay.addWidget(self._summary)
        lay.addWidget(self._result_frame)

        lay.addStretch(1)
        scroll.setWidget(c)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)

    def _on_type_change(self, btn: QRadioButton) -> None:
        self._q_wrap.setVisible(btn.property("value") == "custom")

    def _run(self) -> None:
        ticker = self._ticker.text().strip().upper()
        if not ticker:
            return
        checked = self._type_group.checkedButton()
        atype = checked.property("value") if checked else "earnings"
        question = self._question.text().strip()
        if atype == "custom" and not question:
            self._signal_lbl.setText("Enter a custom question first.")
            return

        self._run_btn.setEnabled(False)
        self._run_btn.setText("Analyzing…  (30–60 s)")
        self._result_frame.setVisible(False)

        def do():
            from market_bridge.core.pipeline import MarketBridgePipeline
            pipeline = MarketBridgePipeline()
            if atype == "earnings":
                return pipeline.analyze_earnings(ticker)
            elif atype == "annual":
                return pipeline.analyze_annual(ticker)
            elif atype == "quarterly":
                return pipeline.analyze_quarterly(ticker)
            else:
                return pipeline.custom_query(ticker, question)

        def on_result(result):
            a = result.analysis
            f = result.filing
            color = {
                "bullish": _ACCENT,
                "bearish": _DANGER,
                "neutral": _PURPLE,
            }.get(a.signal.lower(), _PURPLE)
            self._signal_lbl.setText(f"● {a.signal.upper()}  —  Conviction: {a.conviction}")
            self._signal_lbl.setStyleSheet(
                f"color: {color}; font-weight: 700; font-size: 13px; background: transparent;"
            )
            date_str = f.filed_at if f else "N/A"
            self._meta_lbl.setText(
                f"{ticker}  ·  Filed {date_str}  ·  {a.chunks_used} chunks  ·  {a.model}"
            )
            self._summary.setMarkdown(a.summary)
            self._result_frame.setVisible(True)

        def on_error(msg):
            self._signal_lbl.setStyleSheet(
                f"color: {_DANGER}; font-weight: 600; font-size: 13px; background: transparent;"
            )
            self._signal_lbl.setText(f"Error: {msg}")
            self._result_frame.setVisible(True)

        def done():
            self._run_btn.setEnabled(True)
            self._run_btn.setText("Run Analysis")

        w = _run_worker(do, on_result, on_error, self._workers)
        w.finished.connect(done)


# ── tool tab: config ──────────────────────────────────────

class ConfigTab(QWidget):
    def __init__(self, parent=None) -> None:
        super().__init__(parent)
        self._setup_ui()

    def _setup_ui(self) -> None:
        scroll = QScrollArea()
        scroll.setObjectName("toolScroll")
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        c = QWidget()
        c.setObjectName("toolContent")
        lay = QVBoxLayout(c)
        lay.setContentsMargins(16, 16, 16, 16)
        lay.setSpacing(8)

        from config import Config

        lay.addWidget(_section_label("API Keys"))
        for name, key in [
            ("Anthropic API", Config.ANTHROPIC_API_KEY),
            ("FRED API", Config.FRED_API_KEY),
            ("Polygon API", Config.POLYGON_API_KEY),
            ("SEC API (Market Bridge)", getattr(Config, "SEC_API_KEY", "")),
        ]:
            row = QHBoxLayout()
            k = QLabel(name)
            k.setObjectName("fieldLabel")
            v = QLabel("✓  Set" if key else "✗  Not set")
            v.setStyleSheet(
                f"color: {_ACCENT if key else _DANGER}; font-weight: 600;"
                " background: transparent;"
            )
            row.addWidget(k)
            row.addStretch(1)
            row.addWidget(v)
            lay.addLayout(row)

        lay.addWidget(_section_label("Settings"))
        for name, val in [
            ("Model", Config.LLM_MODEL),
            ("Max Tokens", str(Config.LLM_MAX_TOKENS)),
            ("ChromaDB Dir", Config.CHROMA_PERSIST_DIR),
            ("Chunk Size", str(Config.CHUNK_SIZE)),
            ("Chunk Overlap", str(Config.CHUNK_OVERLAP)),
            ("Top-K Results", str(Config.TOP_K_RESULTS)),
        ]:
            row = QHBoxLayout()
            k = QLabel(name)
            k.setObjectName("fieldLabel")
            v = QLabel(val)
            v.setStyleSheet(
                f"color: {_TEXT2}; font-family: Consolas, monospace;"
                " font-size: 12px; background: transparent;"
            )
            row.addWidget(k)
            row.addStretch(1)
            row.addWidget(v)
            lay.addLayout(row)

        lay.addWidget(_section_label("Setup"))
        note = QLabel(
            "1. Add API keys to  .env\n"
            "2. pip install -r requirements.txt\n"
            "3. python desktop.py"
        )
        note.setStyleSheet(
            f"color: {_TEXT2}; font-size: 12px; background: transparent; line-height: 1.8;"
        )
        note.setWordWrap(True)
        lay.addWidget(note)

        lay.addStretch(1)
        scroll.setWidget(c)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(scroll)


# ── tools panel ───────────────────────────────────────────

class ToolsPanel(QTabWidget):
    def __init__(self, get_agent: Callable, parent=None) -> None:
        super().__init__(parent)
        self.setObjectName("toolsPanel")
        self.setDocumentMode(True)

        self.addTab(MarketTab(),              "📈  Market")
        self.addTab(KnowledgeTab(get_agent),  "📚  Knowledge")
        self.addTab(MacroTab(),               "📊  Macro")
        self.addTab(AnalysisTab(),            "🔬  Analysis")
        self.addTab(ConfigTab(),              "⚙  Config")


# ── main window ───────────────────────────────────────────

class MainWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self._agent = None
        self._chat_worker: Optional[ChatWorker] = None
        self._current_asst_widget: Optional[MessageWidget] = None
        self._setup_palette()
        self._setup_ui()

    # ── agent (lazy) ──────────────────────────────────────
    def _get_agent(self):
        if self._agent is None:
            from core import FinanceAgent
            self._agent = FinanceAgent()
            logger.info("FinanceAgent initialized")
        return self._agent

    # ── palette ───────────────────────────────────────────
    def _setup_palette(self) -> None:
        palette = QPalette()
        palette.setColor(QPalette.ColorRole.Window,          QColor(_BG))
        palette.setColor(QPalette.ColorRole.WindowText,      QColor(_TEXT))
        palette.setColor(QPalette.ColorRole.Base,            QColor(_BG))
        palette.setColor(QPalette.ColorRole.AlternateBase,   QColor(_SURF))
        palette.setColor(QPalette.ColorRole.Text,            QColor(_TEXT))
        palette.setColor(QPalette.ColorRole.Button,          QColor(_CARD))
        palette.setColor(QPalette.ColorRole.ButtonText,      QColor(_TEXT))
        palette.setColor(QPalette.ColorRole.Highlight,       QColor(_BDR2))
        palette.setColor(QPalette.ColorRole.HighlightedText, QColor(_TEXT))
        QApplication.instance().setPalette(palette)

    # ── UI setup ──────────────────────────────────────────
    def _setup_ui(self) -> None:
        self.setWindowTitle("Ada Lovelace — Finance Research AI")
        self.resize(1300, 840)
        self.setMinimumSize(900, 600)
        self.setStyleSheet(DARK_QSS)

        root = QWidget()
        self.setCentralWidget(root)
        root_lay = QVBoxLayout(root)
        root_lay.setContentsMargins(0, 0, 0, 0)
        root_lay.setSpacing(0)

        # Header
        header = QWidget()
        header.setObjectName("header")
        header.setFixedHeight(52)
        h_lay = QHBoxLayout(header)
        h_lay.setContentsMargins(20, 0, 20, 0)
        h_lay.setSpacing(10)

        gem = QLabel("A")
        gem.setObjectName("logoGem")
        gem.setFixedSize(32, 32)

        name = QLabel("Ada Lovelace")
        name.setObjectName("logoName")
        tag = QLabel("Finance Research AI")
        tag.setObjectName("logoTag")

        h_lay.addWidget(gem)
        h_lay.addWidget(name)
        h_lay.addWidget(tag)
        h_lay.addStretch(1)
        for text, obj in [("● Live", "livePill"), ("FRED", "pill"),
                          ("EDGAR", "pill"), ("Polygon", "pill"), ("RAG", "pill")]:
            p = QLabel(text)
            p.setObjectName(obj)
            h_lay.addWidget(p)

        root_lay.addWidget(header)

        # Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setHandleWidth(1)
        splitter.setChildrenCollapsible(False)

        self._chat = ChatPanel()
        self._chat.message_ready.connect(self._on_user_message)
        self._chat.clear_requested.connect(self._on_clear)

        self._tools = ToolsPanel(self._get_agent)

        splitter.addWidget(self._chat)
        splitter.addWidget(self._tools)
        splitter.setSizes([820, 480])

        root_lay.addWidget(splitter, 1)

    # ── chat flow ─────────────────────────────────────────
    def _on_user_message(self, text: str) -> None:
        self._chat.add_user_message(text)
        self._current_asst_widget = self._chat.add_assistant_placeholder()
        self._chat.set_input_enabled(False)

        worker = ChatWorker(self._get_agent(), text)
        worker.token.connect(self._on_token)
        worker.error_occurred.connect(self._on_chat_error)
        worker.finished.connect(self._on_chat_done)
        self._chat_worker = worker
        worker.start()

    def _on_token(self, token: str) -> None:
        if self._current_asst_widget:
            self._current_asst_widget.append_token(token)
            self._chat._scroll_bottom()

    def _on_chat_done(self) -> None:
        if self._current_asst_widget:
            self._current_asst_widget.finalize()
            self._chat._scroll_bottom()
        self._chat.set_input_enabled(True)
        self._current_asst_widget = None

    def _on_chat_error(self, msg: str) -> None:
        if self._current_asst_widget:
            self._current_asst_widget.set_content(f"⚠️ Error: {msg}")
        self._chat.set_input_enabled(True)

    def _on_clear(self) -> None:
        if self._agent:
            self._agent.clear_history()

    def closeEvent(self, event) -> None:
        if self._chat_worker and self._chat_worker.isRunning():
            self._chat_worker.requestInterruption()
            self._chat_worker.wait(2000)
        super().closeEvent(event)
