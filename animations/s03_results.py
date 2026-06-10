# Render (preview):  cd animations && conda run -n manim manim -pql s03_results.py Results
# Render (high):     cd animations && conda run -n manim manim -qh  s03_results.py Results
# Duration: ~65s

from manim import *
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BG, WHITE_TEXT, GRAY_TEXT, DIM_GRAY,
    BLUE_LIGHT, GREEN_LIGHT, TEAL_LIGHT,
    SMALL_SCALE, LABEL_SCALE,
)

DOCS = Path(__file__).parent.parent / "docs" / "assets"

UGA_MAP = DOCS / "uganda_treatment_map.png"
GHA_MAP = DOCS / "ghana_treatment_map.png"

LX    = -3.55
RX    = +3.55
DIV_X =  0.0

UGA_COL = BLUE_LIGHT
GHA_COL = TEAL_LIGHT
SEL_COL = GREEN_LIGHT
SEC_COL = WHITE_TEXT    # section header color

# Treatment map dot colors — exact website gradient endpoints
MAP_CTRL_COL = "#2C7BB6"   # rgb(44, 123, 182) — control
MAP_TRT_COL  = "#D73027"   # rgb(215, 48, 39)  — treated


# ── helpers ───────────────────────────────────────────────────────────────────

def _img(path, max_w, max_h):
    p = Path(path)
    if p.exists():
        mob = ImageMobject(str(p))
        sw = max_w / mob.width
        sh = max_h / mob.height
        mob.scale(min(sw, sh))
        return mob
    rect = Rectangle(width=max_w * 0.7, height=max_h * 0.55,
                     fill_color="#888", fill_opacity=0.10,
                     stroke_color=DIM_GRAY, stroke_width=1.0)
    lbl = Text(p.name, color=DIM_GRAY).scale(0.20).move_to(rect)
    return VGroup(rect, lbl)


def _row(label, value, cx, y, val_col=WHITE_TEXT):
    lbl = Text(label, color=GRAY_TEXT).scale(SMALL_SCALE)
    val = Text(value, color=val_col).scale(SMALL_SCALE)
    grp = VGroup(lbl, val).arrange(RIGHT, buff=0.20)
    grp.move_to([cx, y, 0])
    return grp


def _section_header(txt, col, cx, y, sc=0.38):
    return Text(txt, color=col, weight=BOLD).scale(sc).move_to([cx, y, 0])


def _hrule(y, x0, x1, col=DIM_GRAY, lw=0.6):
    return Line([x0, y, 0], [x1, y, 0], color=col, stroke_width=lw)


# ── scene ─────────────────────────────────────────────────────────────────────

class Results(Scene):
    def construct(self):                              # noqa: C901
        self.camera.background_color = BG

        divider = Line([DIV_X, 3.10, 0], [DIV_X, -3.80, 0],
                       color=DIM_GRAY, stroke_width=0.8)

        # ══════════════════════════════════════════════════════════════════
        # ACT 1 — Column headers
        # ══════════════════════════════════════════════════════════════════
        col_uga = (Text("Uganda YOP",       color=UGA_COL, weight=BOLD)
                   .scale(0.52).move_to([LX, 3.65, 0]))
        col_gha = (Text("Ghana LEAP 1000",  color=GHA_COL, weight=BOLD)
                   .scale(0.52).move_to([RX, 3.65, 0]))
        sub_uga = (Text("N=2,082  ·  331 communities  ·  2008", color=GRAY_TEXT)
                   .scale(0.27).next_to(col_uga, DOWN, buff=0.10))
        sub_gha = (Text("N=2,331  ·  162 communities  ·  2015–17", color=GRAY_TEXT)
                   .scale(0.27).next_to(col_gha, DOWN, buff=0.10))

        self.play(
            FadeIn(divider),
            FadeIn(col_uga), FadeIn(sub_uga),
            FadeIn(col_gha), FadeIn(sub_gha),
            run_time=0.70,
        )
        self.wait(0.30)

        # ══════════════════════════════════════════════════════════════════
        # ACT 2 — Treatment maps
        # ══════════════════════════════════════════════════════════════════
        # Load maps then align their tops to the same y
        map_uga = _img(UGA_MAP, 5.2, 2.2)
        map_gha = _img(GHA_MAP, 4.0, 3.2)
        MAP_TOP = 3.00   # shared top y for both maps
        map_uga.move_to([LX, MAP_TOP - map_uga.height / 2, 0])
        map_gha.move_to([RX, MAP_TOP - map_gha.height / 2, 0])

        # Legend: vertical stack (control above treated), right-aligned beside each map
        leg_trt = Text("● treated", color=MAP_TRT_COL).scale(LABEL_SCALE)
        leg_ctr = Text("● control", color=MAP_CTRL_COL).scale(LABEL_SCALE)

        leg_uga = VGroup(leg_ctr.copy(), leg_trt.copy()).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        leg_uga.next_to(map_uga, RIGHT, buff=0.12).align_to(map_uga, UP)

        leg_gha = VGroup(leg_ctr.copy(), leg_trt.copy()).arrange(DOWN, buff=0.12, aligned_edge=LEFT)
        leg_gha.next_to(map_gha, RIGHT, buff=0.12).align_to(map_gha, UP)

        self.play(
            FadeIn(map_uga, shift=UP * 0.07),
            FadeIn(map_gha, shift=UP * 0.07),
            run_time=1.10,
        )
        self.play(FadeIn(leg_uga), FadeIn(leg_gha), run_time=0.40)
        self.wait(1.20)

        # ══════════════════════════════════════════════════════════════════
        # ACT 3 — Uganda outcome 1  +  Ghana outcome  (in sync)
        # ══════════════════════════════════════════════════════════════════

        # Anchor outcome sections below the lower map bottom
        map_uga_bot = map_uga.get_bottom()[1]
        map_gha_bot = map_gha.get_bottom()[1]
        sec_top = min(map_uga_bot, map_gha_bot) - 0.28

        # Uganda section 1
        rule_u1  = _hrule(sec_top + 0.12, -6.5, -0.1)
        hdr_u1   = _section_header("Skilled Employment", UGA_COL, LX, sec_top - 0.10)
        ate_u1   = _row("ATE",        "+0.31 pp",                     LX, sec_top - 0.40)
        sel_u1   = _row("candidates", "170  →  5 selected",           LX, sec_top - 0.68, SEL_COL)
        fac_u1   = _row("factors",    "language  ·  river  ·  vegetation", LX, sec_top - 0.96)

        # Ghana outcome
        rule_g1  = _hrule(sec_top + 0.12, 0.1, 6.5)
        hdr_g1   = _section_header("Household Consumption", GHA_COL, RX, sec_top - 0.10)
        ate_g1   = _row("ATE",        "+7.4 GH₵/mo",                  RX, sec_top - 0.40)
        sel_g1   = _row("candidates", "155  →  2 selected",           RX, sec_top - 0.68, SEL_COL)
        fac_g1   = _row("factors",    "ephemeral waterways  ·  forest", RX, sec_top - 0.96)

        self.play(Create(rule_u1), Create(rule_g1), run_time=0.35)
        self.play(FadeIn(hdr_u1), FadeIn(hdr_g1), run_time=0.40)

        for u, g in [(ate_u1, ate_g1), (sel_u1, sel_g1), (fac_u1, fac_g1)]:
            self.play(
                FadeIn(u, shift=RIGHT * 0.04),
                FadeIn(g, shift=RIGHT * 0.04),
                run_time=0.45,
            )
            self.wait(0.20)

        self.wait(0.80)

        # ══════════════════════════════════════════════════════════════════
        # ACT 4 — Uganda outcome 2  (Uganda only)
        # ══════════════════════════════════════════════════════════════════
        sec2_top = sec_top - 1.25   # start just below factors row

        rule_u2  = _hrule(sec2_top + 0.06, -6.5, -0.1)
        hdr_u2   = _section_header("Business Assets", UGA_COL, LX, sec2_top - 0.16)
        ate_u2   = _row("ATE",        "+0.61 log USD",                LX, sec2_top - 0.45)
        sel_u2   = _row("candidates", "170  →  2 selected",           LX, sec2_top - 0.73, SEL_COL)
        fac_u2   = _row("factors",    "NDVI  ·  structured agriculture", LX, sec2_top - 1.01)

        self.play(Create(rule_u2), run_time=0.30)
        self.play(FadeIn(hdr_u2), run_time=0.35)
        for mob in [ate_u2, sel_u2, fac_u2]:
            self.play(FadeIn(mob, shift=RIGHT * 0.04), run_time=0.40)
            self.wait(0.18)
        self.wait(0.80)

        # ══════════════════════════════════════════════════════════════════
        # ACT 5 — Takeaway
        # ══════════════════════════════════════════════════════════════════
        sep_line = Line([-6.5, -3.25, 0], [6.5, -3.25, 0],
                        color=DIM_GRAY, stroke_width=0.8)
        takeaway = (
            Text(
                "Environmental context—invisible to surveys—determines who benefits most.",
                color=WHITE_TEXT,
            )
            .scale(0.37)
            .move_to([0, -3.58, 0])
        )

        self.play(Create(sep_line), run_time=0.35)
        self.play(FadeIn(takeaway, shift=UP * 0.06), run_time=0.70)
        self.wait(5.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.50)
