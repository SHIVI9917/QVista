"""
The look of the application.

Everything visual is decided here and nowhere else: the colours, the
corner radii, the spacing rhythm, and the one stylesheet that dresses
every widget in the program. A dialog should never have to describe its
own appearance -- it says what a control *is* (a primary action, a
warning) and this module decides what that looks like.

The palette is a deep indigo rather than a neutral grey. Grey chrome
around a 3D viewport reads as an instrument panel; a cool, slightly blue
ground reads as depth, and it lets a single warm accent do the work of
pointing at whatever matters on screen. Two accents carry the interface:
**iris** for anything interactive and **aqua** for anything that
succeeded or is safe to press. Amber and rose are kept for warnings and
errors, so a colour that is not one of those two never appears without
meaning something.

Nothing here is square. Corners are rounded on a three-step scale --
cards, controls, and small indicators -- which is what stops a dense
scientific interface from reading as a spreadsheet.

Qt's stylesheet dialect has no variables, so the sheet is written with
``$NAME`` placeholders and substituted from :data:`TOKENS`. ``$`` does
not occur in CSS, which is why it, rather than ``{}``, is the marker.
"""

from __future__ import annotations

from string import Template

from qvista.gui.icons import url as icon

# ----------------------------------------------------------------
# Palette
# ----------------------------------------------------------------

#: The window itself, top and bottom of a very shallow gradient. The
#: difference is deliberately almost invisible: it should read as light
#: falling on a surface, not as a background effect.
BG_TOP = "#171d2c"
BG_BOTTOM = "#0e1119"

#: Panels and docks sit on this.
SURFACE = "#181d2b"

#: Cards -- group boxes, list rows, the raised things inside a panel.
SURFACE_2 = "#1f2637"

#: Anything you can type into or open.
SURFACE_3 = "#252e43"

#: Under the pointer.
HOVER = "#2f3a55"

#: Pressed, and the track behind a slider.
SUNKEN = "#141926"

#: Hairlines. STRONG is for the edge of something focused.
BORDER = "#2b344b"
BORDER_STRONG = "#3c4a68"

#: Type. FAINT is for things present but not being offered.
TEXT = "#e9edf8"
TEXT_DIM = "#9ba5be"
TEXT_FAINT = "#6d7893"

#: The interactive accent.
IRIS = "#7c8cff"
IRIS_DEEP = "#5c6ae0"
IRIS_GLOW = "#9aa6ff"

#: The affirmative accent.
AQUA = "#36cfc2"
AQUA_DEEP = "#22a89d"

#: Caution, and failure.
AMBER = "#f2a65a"
AMBER_DEEP = "#d0843a"
ROSE = "#ef6076"
ROSE_DEEP = "#cf4459"

#: Text drawn on top of a saturated accent.
ON_ACCENT = "#0c1020"

#: Corner radii: cards, controls, indicators.
RADIUS_LG = "12px"
RADIUS_MD = "9px"
RADIUS_SM = "6px"

#: The UI face and the one used wherever columns have to line up.
FONT_FAMILY = '"Segoe UI Variable Text", "Segoe UI", "Inter", sans-serif'
MONO_FAMILY = '"Cascadia Code", "Consolas", monospace'
FONT_SIZE = "10pt"
MONO_SIZE = "9.5pt"


TOKENS = {
    "BG_TOP": BG_TOP,
    "BG_BOTTOM": BG_BOTTOM,
    "SURFACE": SURFACE,
    "SURFACE_2": SURFACE_2,
    "SURFACE_3": SURFACE_3,
    "HOVER": HOVER,
    "SUNKEN": SUNKEN,
    "BORDER": BORDER,
    "BORDER_STRONG": BORDER_STRONG,
    "TEXT": TEXT,
    "TEXT_DIM": TEXT_DIM,
    "TEXT_FAINT": TEXT_FAINT,
    "IRIS": IRIS,
    "IRIS_DEEP": IRIS_DEEP,
    "IRIS_GLOW": IRIS_GLOW,
    "AQUA": AQUA,
    "AQUA_DEEP": AQUA_DEEP,
    "AMBER": AMBER,
    "AMBER_DEEP": AMBER_DEEP,
    "ROSE": ROSE,
    "ROSE_DEEP": ROSE_DEEP,
    "ON_ACCENT": ON_ACCENT,
    "RADIUS_LG": RADIUS_LG,
    "RADIUS_MD": RADIUS_MD,
    "RADIUS_SM": RADIUS_SM,
    "FONT_FAMILY": FONT_FAMILY,
    "MONO_FAMILY": MONO_FAMILY,
    "FONT_SIZE": FONT_SIZE,
    "MONO_SIZE": MONO_SIZE,
    "CHEVRON_DOWN": icon("chevron_down.svg"),
    "CHEVRON_DOWN_ON": icon("chevron_down_active.svg"),
    "CHEVRON_UP": icon("chevron_up.svg"),
    "CHEVRON_UP_ON": icon("chevron_up_active.svg"),
    "CHEVRON_RIGHT": icon("chevron_right.svg"),
    "CHEVRON_RIGHT_ON": icon("chevron_right_active.svg"),
    "CHECK": icon("check.svg"),
}


# ----------------------------------------------------------------
# Inline text styles
#
# For the odd label that has to carry a state. Kept here so a warning is
# the same amber everywhere.
# ----------------------------------------------------------------

ERROR_STYLE = f"color: {ROSE};"
WARNING_STYLE = f"color: {AMBER};"
INFO_STYLE = f"color: {TEXT_DIM};"
SUCCESS_STYLE = f"color: {AQUA};"

#: Monospace blocks -- reports, previews, anything in columns.
MONO_STYLE = (
    f"font-family: {MONO_FAMILY}; font-size: {MONO_SIZE}; "
    f"color: {TEXT};"
)

#: The explanatory line under a control.
CAPTION_STYLE = f"color: {TEXT_DIM}; font-size: 9pt;"

#: A heading inside a dialog.
HEADING_STYLE = (
    f"color: {TEXT}; font-size: 12pt; font-weight: 600;"
)

#: The line under a heading that says what the thing is for.
SUBHEADING_STYLE = f"color: {TEXT_DIM}; font-size: 9.5pt;"


# ----------------------------------------------------------------
# Buttons
# ----------------------------------------------------------------

#: ``role -> (top, bottom, text)``. The two-stop gradient is what keeps a
#: flat colour from looking like a coloured rectangle.
_ROLES = {
    "primary": (IRIS, IRIS_DEEP, "#ffffff"),
    "info": (SURFACE_3, "#1e2637", TEXT),
    "success": (AQUA, AQUA_DEEP, ON_ACCENT),
    "warning": (AMBER, AMBER_DEEP, ON_ACCENT),
    "danger": (ROSE, ROSE_DEEP, "#ffffff"),
}


def button_style(role: str = "info") -> str:
    """
    A complete stylesheet for one button.

    Self-contained on purpose. A per-widget stylesheet replaces the
    global rules for that widget rather than adding to them, so a button
    given only a background colour would also lose its radius, its hover
    state and -- the one that matters -- its disabled state, and a dead
    button that still looks pressable reads as a broken program.
    """

    top, bottom, text = _ROLES.get(role, _ROLES["info"])

    return Template(
        """
        QPushButton {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 $TOP, stop:1 $BOTTOM);
            color: $TEXT_ON;
            border: 1px solid $BORDER;
            border-radius: $RADIUS_MD;
            padding: 7px 16px;
            font-weight: 600;
        }
        QPushButton:hover {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                        stop:0 $TOP, stop:1 $TOP);
            border-color: $BORDER_STRONG;
        }
        QPushButton:pressed {
            background: $BOTTOM;
            padding-top: 8px;
            padding-bottom: 6px;
        }
        QPushButton:disabled {
            background: $SUNKEN;
            color: $TEXT_FAINT;
            border-color: $BORDER;
        }
        """
    ).substitute(
        TOP=top,
        BOTTOM=bottom,
        TEXT_ON=text,
        BORDER=BORDER,
        BORDER_STRONG=BORDER_STRONG,
        SUNKEN=SUNKEN,
        TEXT_FAINT=TEXT_FAINT,
        RADIUS_MD=RADIUS_MD,
    )


PRIMARY_BUTTON_STYLE = button_style("primary")
INFO_BUTTON_STYLE = button_style("info")
SUCCESS_BUTTON_STYLE = button_style("success")
WARNING_BUTTON_STYLE = button_style("warning")
DANGER_BUTTON_STYLE = button_style("danger")


# ----------------------------------------------------------------
# The sheet
# ----------------------------------------------------------------

_SHEET = Template(
    """
/* ---------------------------------------------------------- */
/* Ground                                                      */
/* ---------------------------------------------------------- */

QWidget {
    background: transparent;
    color: $TEXT;
    font-family: $FONT_FAMILY;
    font-size: $FONT_SIZE;
}

QMainWindow, QDialog {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 $BG_TOP, stop:1 $BG_BOTTOM);
}

QDialog {
    border-radius: $RADIUS_LG;
}

QMessageBox, QInputDialog, QFileDialog {
    background: $BG_BOTTOM;
}

QLabel {
    background: transparent;
    color: $TEXT;
}

QLabel:disabled {
    color: $TEXT_FAINT;
}

/* ---------------------------------------------------------- */
/* Menu bar: a strip of chips rather than a ribbon             */
/* ---------------------------------------------------------- */

QMenuBar {
    background: transparent;
    padding: 6px 8px;
    border-bottom: 1px solid $BORDER;
}

QMenuBar::item {
    background: transparent;
    color: $TEXT_DIM;
    padding: 6px 13px;
    margin: 0px 2px;
    border-radius: $RADIUS_SM;
}

QMenuBar::item:selected {
    background: $SURFACE_3;
    color: $TEXT;
}

QMenuBar::item:pressed {
    background: $IRIS;
    color: #ffffff;
}

QMenu {
    background: $SURFACE_2;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 6px;
}

QMenu::item {
    padding: 7px 26px 7px 16px;
    border-radius: $RADIUS_SM;
    color: $TEXT;
}

QMenu::item:selected {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 $IRIS_DEEP, stop:1 $IRIS);
    color: #ffffff;
}

QMenu::item:disabled {
    color: $TEXT_FAINT;
}

QMenu::separator {
    height: 1px;
    background: $BORDER;
    margin: 6px 10px;
}

QMenu::right-arrow {
    image: $CHEVRON_RIGHT;
    width: 11px;
    height: 11px;
    margin-right: 10px;
}

QMenu::right-arrow:selected { image: $CHEVRON_RIGHT_ON; }

/* ---------------------------------------------------------- */
/* Docks: the title is a caption, not a title bar              */
/* ---------------------------------------------------------- */

QDockWidget {
    color: $TEXT_DIM;
    titlebar-close-icon: none;
    titlebar-normal-icon: none;
}

QDockWidget::title {
    background: transparent;
    padding: 9px 14px 7px 14px;
    color: $TEXT_DIM;
    font-weight: 600;
}

QMainWindow::separator {
    background: transparent;
    width: 6px;
    height: 6px;
}

QMainWindow::separator:hover {
    background: $IRIS_DEEP;
}

/* ---------------------------------------------------------- */
/* Group boxes as cards                                        */
/* ---------------------------------------------------------- */

/*
 * margin-top has to clear the whole title, because the title is drawn
 * in that margin and the frame starts underneath it. At 16px it was
 * two pixels short of the text plus its own padding, so every group
 * box in the application had its heading sliced through by its own
 * top border.
 */
QGroupBox {
    background: $SURFACE_2;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_LG;
    margin-top: 24px;
    padding: 10px 12px 12px 12px;
    font-weight: 600;
}

QGroupBox::title {
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 14px;
    top: 0px;
    padding: 2px 8px;
    color: $IRIS_GLOW;
    background: transparent;
}

QFrame[frameShape="4"], QFrame[frameShape="5"] {
    background: $BORDER;
    border: none;
    max-height: 1px;
}

/* ---------------------------------------------------------- */
/* Buttons                                                     */
/* ---------------------------------------------------------- */

QPushButton {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 $SURFACE_3, stop:1 #1e2637);
    color: $TEXT;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 7px 16px;
    font-weight: 600;
}

QPushButton:hover {
    background: $HOVER;
    border-color: $BORDER_STRONG;
}

QPushButton:pressed {
    background: $SUNKEN;
    padding-top: 8px;
    padding-bottom: 6px;
}

QPushButton:disabled {
    background: $SUNKEN;
    color: $TEXT_FAINT;
    border-color: $BORDER;
}

QPushButton:default {
    border-color: $IRIS_DEEP;
}

/* ---------------------------------------------------------- */
/* Cards and toggle chips                                      */
/*                                                             */
/* A card is a box you choose *a kind of thing* with -- one of  */
/* seven calculations. A chip is a switch: on or off, and it    */
/* says so by filling with the accent rather than by moving.    */
/* ---------------------------------------------------------- */

QFrame#card {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 $SURFACE_2, stop:1 #1a2031);
    border: 1px solid $BORDER;
    border-radius: $RADIUS_LG;
}

QFrame#card:hover {
    background: $SURFACE_3;
    border-color: $BORDER_STRONG;
}

QFrame#card[on="true"] {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                stop:0 #2a3258, stop:1 #212840);
    border: 1px solid $IRIS;
}

QLabel#cardTitle {
    background: transparent;
    color: $TEXT;
    font-size: 11pt;
    font-weight: 700;
}

QFrame#card[on="true"] QLabel#cardTitle {
    color: $IRIS_GLOW;
}

QLabel#cardHint {
    background: transparent;
    color: $TEXT_FAINT;
    font-size: 8.5pt;
    font-weight: 400;
}

QFrame#card[on="true"] QLabel#cardHint {
    color: $TEXT_DIM;
}

QPushButton#chip {
    background: $SURFACE_2;
    color: $TEXT_DIM;
    border: 1px solid $BORDER;
    border-radius: 14px;
    padding: 6px 14px;
    font-weight: 600;
}

QPushButton#chip:hover {
    background: $HOVER;
    color: $TEXT;
    border-color: $BORDER_STRONG;
}

QPushButton#chip:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 $IRIS_DEEP, stop:1 $IRIS);
    color: #ffffff;
    border-color: $IRIS_GLOW;
}

QPushButton#mode {
    background: $SURFACE_2;
    color: $TEXT_DIM;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 7px 14px;
    font-weight: 600;
}

QPushButton#mode:hover {
    background: $HOVER;
    color: $TEXT;
    border-color: $BORDER_STRONG;
}

QPushButton#mode:checked {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 $AQUA_DEEP, stop:1 $AQUA);
    color: $ON_ACCENT;
    border-color: $AQUA;
}

QToolButton {
    background: transparent;
    border: 1px solid transparent;
    border-radius: $RADIUS_SM;
    padding: 5px 8px;
}

QToolButton:hover {
    background: $SURFACE_3;
    border-color: $BORDER;
}

/* ---------------------------------------------------------- */
/* Text entry                                                  */
/* ---------------------------------------------------------- */

QLineEdit, QPlainTextEdit, QTextEdit, QAbstractSpinBox {
    background: $SURFACE_3;
    color: $TEXT;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 6px 10px;
    selection-background-color: $IRIS_DEEP;
    selection-color: #ffffff;
}

QLineEdit:hover, QAbstractSpinBox:hover {
    border-color: $BORDER_STRONG;
}

QLineEdit:focus, QPlainTextEdit:focus, QTextEdit:focus,
QAbstractSpinBox:focus {
    border-color: $IRIS;
    background: $SURFACE_2;
}

QLineEdit:disabled, QAbstractSpinBox:disabled {
    background: $SUNKEN;
    color: $TEXT_FAINT;
}

QPlainTextEdit, QTextEdit {
    font-family: $MONO_FAMILY;
    font-size: $MONO_SIZE;
    padding: 10px 12px;
}

/* The arrows themselves are left to Fusion, which draws them from the
   palette. Faking them with the CSS border-triangle trick puts a grey
   rectangle in the corner instead: Qt sizes a sub-control box first and
   the zero-size trick has nothing to collapse. */

QAbstractSpinBox::up-button, QAbstractSpinBox::down-button {
    background: transparent;
    border: none;
    width: 20px;
    margin-right: 3px;
    border-radius: 4px;
}

QAbstractSpinBox::up-button:hover, QAbstractSpinBox::down-button:hover {
    background: $HOVER;
}

QAbstractSpinBox::up-arrow { image: $CHEVRON_UP; width: 11px; height: 11px; }
QAbstractSpinBox::down-arrow { image: $CHEVRON_DOWN; width: 11px; height: 11px; }

QAbstractSpinBox::up-arrow:hover { image: $CHEVRON_UP_ON; }
QAbstractSpinBox::down-arrow:hover { image: $CHEVRON_DOWN_ON; }

QAbstractSpinBox::up-arrow:disabled, QAbstractSpinBox::down-arrow:disabled {
    image: none;
}

/* ---------------------------------------------------------- */
/* Dropdowns                                                   */
/* ---------------------------------------------------------- */

QComboBox {
    background: $SURFACE_3;
    color: $TEXT;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 6px 10px;
    min-height: 20px;
}

QComboBox:hover {
    border-color: $BORDER_STRONG;
    background: $HOVER;
}

QComboBox:focus, QComboBox:on {
    border-color: $IRIS;
}

QComboBox:disabled {
    background: $SUNKEN;
    color: $TEXT_FAINT;
}

QComboBox::drop-down {
    border: none;
    width: 26px;
    subcontrol-origin: padding;
    subcontrol-position: center right;
}

QComboBox::down-arrow { image: $CHEVRON_DOWN; width: 12px; height: 12px; }
QComboBox::down-arrow:on, QComboBox::down-arrow:hover {
    image: $CHEVRON_DOWN_ON;
}

QComboBox QAbstractItemView {
    background: $SURFACE_2;
    border: 1px solid $BORDER_STRONG;
    border-radius: $RADIUS_MD;
    padding: 5px;
    outline: none;
    selection-background-color: $IRIS_DEEP;
    selection-color: #ffffff;
}

QComboBox QAbstractItemView::item {
    padding: 6px 10px;
    border-radius: $RADIUS_SM;
    min-height: 22px;
}

/* ---------------------------------------------------------- */
/* Ticks and dots                                              */
/* ---------------------------------------------------------- */

QCheckBox, QRadioButton {
    spacing: 9px;
    padding: 3px 0px;
    background: transparent;
}

QCheckBox::indicator, QRadioButton::indicator {
    width: 17px;
    height: 17px;
    border: 1px solid $BORDER_STRONG;
    background: $SUNKEN;
}

QCheckBox::indicator { border-radius: $RADIUS_SM; }
QRadioButton::indicator { border-radius: 9px; }

QCheckBox::indicator:hover, QRadioButton::indicator:hover {
    border-color: $IRIS;
}

QCheckBox::indicator:checked {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 $IRIS_GLOW, stop:1 $IRIS_DEEP);
    border-color: $IRIS;
    image: $CHECK;
}

QRadioButton::indicator:checked {
    background: qradialgradient(cx:0.5, cy:0.5, radius:0.5,
                fx:0.5, fy:0.5,
                stop:0 $IRIS_GLOW, stop:0.42 $IRIS_DEEP,
                stop:0.46 $SUNKEN, stop:1 $SUNKEN);
    border: 1px solid $IRIS;
}

QCheckBox::indicator:disabled, QRadioButton::indicator:disabled {
    border-color: $BORDER;
    background: $SUNKEN;
}

/* ---------------------------------------------------------- */
/* Sliders                                                     */
/* ---------------------------------------------------------- */

QSlider::groove:horizontal {
    height: 6px;
    background: $SUNKEN;
    border: 1px solid $BORDER;
    border-radius: 4px;
}

QSlider::sub-page:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 $IRIS_DEEP, stop:1 $IRIS_GLOW);
    border: 1px solid $IRIS_DEEP;
    border-radius: 4px;
}

QSlider::handle:horizontal {
    background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #ffffff, stop:1 #cdd5f5);
    border: 2px solid $IRIS;
    width: 15px;
    margin: -7px 0px;
    border-radius: 9px;
}

QSlider::handle:horizontal:hover {
    border-color: $IRIS_GLOW;
    background: #ffffff;
}

QSlider::tick-mark { background: $BORDER_STRONG; }

/* ---------------------------------------------------------- */
/* Scrollbars: thin, and out of the way until wanted           */
/* ---------------------------------------------------------- */

QScrollBar:vertical {
    background: transparent;
    width: 12px;
    margin: 2px;
}

QScrollBar::handle:vertical {
    background: $BORDER_STRONG;
    border-radius: 4px;
    min-height: 32px;
}

QScrollBar:horizontal {
    background: transparent;
    height: 12px;
    margin: 2px;
}

QScrollBar::handle:horizontal {
    background: $BORDER_STRONG;
    border-radius: 4px;
    min-width: 32px;
}

QScrollBar::handle:hover { background: $IRIS_DEEP; }

QScrollBar::add-line, QScrollBar::sub-line {
    height: 0px; width: 0px; border: none; background: none;
}

QScrollBar::add-page, QScrollBar::sub-page { background: none; }

QScrollArea {
    background: transparent;
    border: none;
}

/* ---------------------------------------------------------- */
/* Splitters                                                   */
/* ---------------------------------------------------------- */

QSplitter::handle {
    background: transparent;
}

QSplitter::handle:horizontal { width: 8px; }
QSplitter::handle:vertical { height: 8px; }

QSplitter::handle:hover { background: $IRIS_DEEP; }

/* ---------------------------------------------------------- */
/* Lists, trees, tables                                        */
/* ---------------------------------------------------------- */

QAbstractItemView {
    background: $SURFACE_2;
    alternate-background-color: $SURFACE;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_MD;
    padding: 4px;
    outline: none;
    selection-background-color: $IRIS_DEEP;
    selection-color: #ffffff;
}

QAbstractItemView::item {
    padding: 5px 8px;
    border-radius: $RADIUS_SM;
}

QAbstractItemView::item:hover {
    background: $HOVER;
}

QHeaderView::section {
    background: $SURFACE;
    color: $TEXT_DIM;
    border: none;
    border-bottom: 1px solid $BORDER;
    padding: 7px 10px;
    font-weight: 600;
}

QTableView {
    gridline-color: $BORDER;
}

/* ---------------------------------------------------------- */
/* Tabs                                                        */
/* ---------------------------------------------------------- */

QTabWidget::pane {
    background: $SURFACE_2;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_LG;
    top: -1px;
}

QTabBar::tab {
    background: transparent;
    color: $TEXT_DIM;
    padding: 8px 18px;
    margin-right: 4px;
    border-radius: $RADIUS_SM;
}

QTabBar::tab:hover { background: $SURFACE_3; color: $TEXT; }

QTabBar::tab:selected {
    background: $SURFACE_2;
    color: $IRIS_GLOW;
    border: 1px solid $BORDER;
    border-bottom-color: $SURFACE_2;
}

/* ---------------------------------------------------------- */
/* Progress and status                                         */
/* ---------------------------------------------------------- */

QProgressBar {
    background: $SUNKEN;
    border: 1px solid $BORDER;
    border-radius: $RADIUS_SM;
    height: 8px;
    text-align: center;
    color: $TEXT_DIM;
}

QProgressBar::chunk {
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 $IRIS, stop:1 $AQUA);
    border-radius: 5px;
}

QStatusBar {
    background: transparent;
    color: $TEXT_DIM;
    border-top: 1px solid $BORDER;
    padding: 3px 10px;
}

QStatusBar::item { border: none; }

QToolTip {
    background: $SURFACE_2;
    color: $TEXT;
    border: 1px solid $IRIS_DEEP;
    border-radius: $RADIUS_SM;
    padding: 6px 10px;
}
"""
)


def stylesheet() -> str:
    """The whole application sheet, tokens substituted."""

    return _SHEET.substitute(TOKENS)


# ----------------------------------------------------------------
# Applying it
# ----------------------------------------------------------------


def _qcolor(value: str):
    from PySide6.QtGui import QColor

    return QColor(value)


def palette():
    """
    A matching :class:`QPalette`.

    The stylesheet does not reach everything: native file dialogs, the
    text cursor, and the frames Qt draws for itself all read the palette
    instead. Without this they come back light grey in the middle of a
    dark application.
    """

    from PySide6.QtGui import QPalette

    result = QPalette()

    result.setColor(QPalette.Window, _qcolor(BG_BOTTOM))
    result.setColor(QPalette.WindowText, _qcolor(TEXT))
    result.setColor(QPalette.Base, _qcolor(SURFACE_3))
    result.setColor(QPalette.AlternateBase, _qcolor(SURFACE_2))
    result.setColor(QPalette.Text, _qcolor(TEXT))
    result.setColor(QPalette.Button, _qcolor(SURFACE_3))
    result.setColor(QPalette.ButtonText, _qcolor(TEXT))
    result.setColor(QPalette.BrightText, _qcolor(ROSE))
    result.setColor(QPalette.Highlight, _qcolor(IRIS_DEEP))
    result.setColor(QPalette.HighlightedText, _qcolor("#ffffff"))
    result.setColor(QPalette.Link, _qcolor(IRIS))
    result.setColor(QPalette.ToolTipBase, _qcolor(SURFACE_2))
    result.setColor(QPalette.ToolTipText, _qcolor(TEXT))
    result.setColor(QPalette.PlaceholderText, _qcolor(TEXT_FAINT))

    for group in (QPalette.Disabled,):
        result.setColor(group, QPalette.Text, _qcolor(TEXT_FAINT))
        result.setColor(group, QPalette.ButtonText, _qcolor(TEXT_FAINT))
        result.setColor(group, QPalette.WindowText, _qcolor(TEXT_FAINT))

    return result


def apply(app) -> None:
    """
    Dress the application.

    Fusion first: the native Windows style draws several controls itself
    and ignores half of what a stylesheet asks for, so the sheet would
    apply unevenly on top of it. Fusion honours all of it.
    """

    from PySide6.QtGui import QFont

    from qvista.gui.icons import app_icon

    app.setStyle("Fusion")
    app.setPalette(palette())
    app.setWindowIcon(app_icon())

    font = QFont("Segoe UI Variable Text", 10)

    if not font.exactMatch():
        font = QFont("Segoe UI", 10)

    app.setFont(font)

    app.setStyleSheet(stylesheet())
