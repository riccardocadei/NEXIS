# ── Palette (matches manim-eci) ───────────────────────────────────────────────
from manim import BLACK

BG            = BLACK
WHITE_TEXT    = "#F2F2F2"
GRAY_TEXT     = "#AAAAAA"
DIM_GRAY      = "#3A3A3A"

BLUE_LIGHT    = "#5BC4F5"   # satellite / foundation-model stream
GREEN_LIGHT   = "#6FD18A"   # survey data
YELLOW_LIGHT  = "#F5C842"   # concat emphasis / titles
PURPLE_LIGHT  = "#C39BD3"   # SAE sparse neurons
TEAL_LIGHT    = "#5BCFB5"   # output labels / interpretation
RED_LIGHT     = "#F47C7C"   # rejection / error
ORANGE_LIGHT  = "#F5A050"   # model box accent

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
