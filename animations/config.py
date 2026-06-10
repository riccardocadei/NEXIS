# ── Palette (white-background version) ────────────────────────────────────────
from manim import WHITE

BG            = WHITE
WHITE_TEXT    = "#111111"   # near-black text on white background
GRAY_TEXT     = "#666666"
DIM_GRAY      = "#CCCCCC"

BLUE_LIGHT    = "#1A8EC9"   # satellite / foundation-model stream
GREEN_LIGHT   = "#2E9A4A"   # survey data
YELLOW_LIGHT  = "#A07800"   # titles (darkened so amber is readable on white)
PURPLE_LIGHT  = "#8B50B8"   # SAE sparse neurons
TEAL_LIGHT    = "#1A9E8A"   # output labels / interpretation
RED_LIGHT     = "#CC3333"   # rejection / error
ORANGE_LIGHT  = "#C06010"   # model box accent

# ── Typography ────────────────────────────────────────────────────────────────
TITLE_SCALE  = 0.72
BODY_SCALE   = 0.48
SMALL_SCALE  = 0.38
LABEL_SCALE  = 0.32

# ── Bottom Y anchor ───────────────────────────────────────────────────────────
BOT_Y = -3.3


# ── Shared helpers ────────────────────────────────────────────────────────────
from manim import Text, RoundedRectangle, Arrow, DOWN, UP, RIGHT, LEFT


def slide_title(text, color=WHITE_TEXT):
    return Text(text, color=color).scale(TITLE_SCALE).to_edge(UP, buff=0.4)


def subtitle(text, color=GRAY_TEXT):
    return Text(text, color=color).scale(BODY_SCALE)


def label(text, color=GRAY_TEXT):
    return Text(text, color=color).scale(LABEL_SCALE)


def make_box(lines, width, height, box_col, txt_scale=None, corner=0.10):
    if txt_scale is None:
        txt_scale = SMALL_SCALE
    from manim import VGroup
    rect = RoundedRectangle(
        width=width, height=height, corner_radius=corner,
        color=box_col, stroke_width=2.0,
        fill_color=box_col, fill_opacity=0.13,
    )
    if isinstance(lines, str):
        lines = [lines]
    texts = VGroup(
        *[Text(l, color=WHITE_TEXT).scale(txt_scale) for l in lines]
    )
    texts.arrange(DOWN, buff=0.07).move_to(rect)
    return VGroup(rect, texts)


def make_arrow(start, end, color=GRAY_TEXT, stroke=2.0):
    return Arrow(
        start, end, buff=0.12, color=color,
        stroke_width=stroke,
        max_tip_length_to_length_ratio=0.15,
    )
