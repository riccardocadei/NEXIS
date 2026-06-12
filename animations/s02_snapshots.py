# Render all three snapshots:
#   cd animations && conda run -n manim manim -sqh s02_snapshots.py Snapshot1 Snapshot2 Snapshot3
# Or one at a time (low-quality preview):
#   conda run -n manim manim -sql s02_snapshots.py Snapshot1

from manim import *
import numpy as np
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BG, WHITE_TEXT, GRAY_TEXT, DIM_GRAY,
    BLUE_LIGHT, GREEN_LIGHT, PURPLE_LIGHT,
    RED_LIGHT, YELLOW_LIGHT,
    TITLE_SCALE, LABEL_SCALE, SMALL_SCALE,
)

# White-background palette overrides (same as s02_method.py)
BG          = "#FFFFFF"
WHITE_TEXT  = "#1a1a1a"
GRAY_TEXT   = "#666666"
DIM_GRAY    = "#BBBBBB"
GREEN_LIGHT = "#2d8a4e"
RED_LIGHT   = "#c0392b"

# ── Layout constants (identical to s02_method.py) ────────────────────────────
_NODE_R   = 0.30
_STROKE_W = 2.5
LABEL_Y   = 2.30
FORM_Y    = -3.30
FWD_TEST_Y = FORM_Y + 0.70
BWD_TEST_Y = FORM_Y + 0.28
SLBL_Y     = FORM_Y + 1.10


# ── Helpers (copied verbatim from s02_method.py) ─────────────────────────────

def _node(label, color=WHITE_TEXT):
    circ = Circle(radius=_NODE_R, color=color, stroke_width=_STROKE_W,
                  fill_color=color, fill_opacity=0.0)
    lbl  = Text(label, color=color).scale(0.46)
    return VGroup(circ, lbl)


def _causal_arrow(start_mob, end_mob, color=WHITE_TEXT):
    return Arrow(
        start_mob.get_center(), end_mob.get_center(),
        color=color, buff=_NODE_R + 0.05, stroke_width=2.0,
        tip_length=0.18,
    )


def _sankey_band(src, dst, color, width=0.10, opacity=0.35):
    s = src.get_right() + RIGHT * 0.02
    d = dst.get_left()  + LEFT  * 0.02
    mid_x = (s[0] + d[0]) / 2.0
    hw = width / 2.0
    band = VMobject()
    band.start_new_path(s + UP * hw)
    band.add_cubic_bezier_curve_to(
        np.array([mid_x, s[1] + hw, 0]),
        np.array([mid_x, d[1] + hw, 0]),
        d + UP * hw,
    )
    band.add_line_to(d + DOWN * hw)
    band.add_cubic_bezier_curve_to(
        np.array([mid_x, d[1] - hw, 0]),
        np.array([mid_x, s[1] - hw, 0]),
        s + DOWN * hw,
    )
    band.add_line_to(s + UP * hw)
    band.set_fill(color, opacity=opacity)
    band.set_stroke(color, width=0.8, opacity=opacity * 0.8)
    return band


def _cross_on(mob, color=RED_LIGHT, size=0.22):
    c = mob.get_center()
    l1 = Line(c + UP * size + LEFT  * size, c + DOWN * size + RIGHT * size,
              color=color, stroke_width=4.0)
    l2 = Line(c + UP * size + RIGHT * size, c + DOWN * size + LEFT  * size,
              color=color, stroke_width=4.0)
    return VGroup(l1, l2)


def _pval(val, color=WHITE_TEXT, sc=0.34):
    return MarkupText(f"<i>p</i>-value = {val}", color=color).scale(sc)


def _step_num(n):
    return (Text(f"Step  {n}", color=GRAY_TEXT, weight="BOLD")
            .scale(SMALL_SCALE)
            .to_corner(UL, buff=0.55)
            .set_y(LABEL_Y))


def _s_label(members):
    txt = "S  =  ∅" if not members else "S  =  {" + ",  ".join(members) + "}"
    return (Text(txt, color=WHITE_TEXT)
            .scale(SMALL_SCALE)
            .move_to([0, SLBL_Y, 0]))


# ── Shared scene builder ──────────────────────────────────────────────────────

def _build_base(scene):
    """Add all static DAG elements and return a dict of mobjects."""
    scene.camera.background_color = BG

    title = Text("Neural EXposure Interaction Search",
                 color=WHITE_TEXT).scale(TITLE_SCALE).to_edge(UP, buff=0.35)

    Z1 = _node("Z₁"); Z2 = _node("Z₂"); Z3 = _node("Z₃")
    W1 = _node("W₁"); W2 = _node("W₂")
    T  = _node("T");  Y  = _node("Y")

    col_Z = -2.8; col_W = -0.1; col_Y = 2.8; dy = 0.0

    Z1.move_to([col_Z,  1.55 + dy, 0])
    Z2.move_to([col_Z,  0.00 + dy, 0])
    Z3.move_to([col_Z, -1.55 + dy, 0])
    W1.move_to([col_W,  0.95 + dy, 0])
    W2.move_to([col_W, -0.55 + dy, 0])
    T.move_to([(col_W + col_Y) / 2, -1.55 + dy, 0])
    Y.move_to([col_Y,   0.20 + dy, 0])

    a_T_Y  = _causal_arrow(T,  Y)
    a_W1_Y = _causal_arrow(W1, Y)
    a_W2_Y = _causal_arrow(W2, Y)

    bnd_Z1_W1 = _sankey_band(Z1, W1, BLUE_LIGHT,   width=0.14, opacity=0.40)
    bnd_Z1_W2 = _sankey_band(Z1, W2, BLUE_LIGHT,   width=0.06, opacity=0.18)
    bnd_Z3_W2 = _sankey_band(Z3, W2, PURPLE_LIGHT, width=0.14, opacity=0.40)
    bnd_Z3_W1 = _sankey_band(Z3, W1, PURPLE_LIGHT, width=0.06, opacity=0.18)
    bnd_Z2_W1 = _sankey_band(Z2, W1, GRAY_TEXT,    width=0.06, opacity=0.18)
    bnd_Z2_W2 = _sankey_band(Z2, W2, GRAY_TEXT,    width=0.06, opacity=0.18)

    lbl_sc = LABEL_SCALE * 0.85
    top_y  = LABEL_Y

    lbl_Z = VGroup(
        Text("candidate neurons",       color=GRAY_TEXT).scale(lbl_sc),
        Text("(sparse representation)", color=GRAY_TEXT).scale(lbl_sc * 0.9),
    ).arrange(DOWN, buff=0.04).move_to([col_Z, top_y, 0])

    lbl_W = VGroup(
        Text("direct effect modifiers", color=GRAY_TEXT).scale(lbl_sc),
        Text("(ground truth)",          color=GRAY_TEXT).scale(lbl_sc * 0.9),
    ).arrange(DOWN, buff=0.04).move_to([col_W, top_y, 0])

    lbl_Y = Text("outcome",   color=GRAY_TEXT).scale(lbl_sc).move_to([col_Y, top_y, 0])
    lbl_T = Text("treatment", color=GRAY_TEXT).scale(lbl_sc).next_to(T, LEFT, buff=0.20)

    # compute content x anchor (identical logic to s02_method.py)
    sc_t    = 0.37
    buff_lc = 0.22
    fwd_lbl = Text("a.  Forward test:", color=GRAY_TEXT).scale(sc_t)
    bwd_lbl = Text("b.  Backward test:", color=GRAY_TEXT).scale(sc_t)
    max_lbl_w = max(fwd_lbl.width, bwd_lbl.width)
    _ref = Text(
        "H₀(j | Z₁, Z₃) :   E[τ | Zⱼ, Z₁, Z₃]  =  E[τ | Z₁, Z₃]",
        color=WHITE_TEXT).scale(sc_t)
    lbl_left = -(max_lbl_w + buff_lc + _ref.width) / 2
    fwd_lbl.move_to([lbl_left + fwd_lbl.width / 2, FWD_TEST_Y, 0])
    bwd_lbl.move_to([lbl_left + bwd_lbl.width / 2, BWD_TEST_Y, 0])
    ctt_x = lbl_left + max_lbl_w + buff_lc

    scene.add(
        title,
        T, lbl_T, Y, lbl_Y, a_T_Y, a_W1_Y, a_W2_Y,
        W1, W2, lbl_W,
        Z1, Z2, Z3, lbl_Z,
        bnd_Z1_W1, bnd_Z1_W2,
        bnd_Z3_W2, bnd_Z3_W1,
        bnd_Z2_W1, bnd_Z2_W2,
        fwd_lbl, bwd_lbl,
    )

    return dict(
        Z1=Z1, Z2=Z2, Z3=Z3, W1=W1, W2=W2, T=T, Y=Y,
        a_W1_Y=a_W1_Y, a_W2_Y=a_W2_Y,
        bnd_Z1_W1=bnd_Z1_W1, bnd_Z1_W2=bnd_Z1_W2,
        bnd_Z3_W2=bnd_Z3_W2, bnd_Z3_W1=bnd_Z3_W1,
        ctt_x=ctt_x, sc_t=sc_t,
    )


def _mk_fwd_ctt(s, ctt_x, sc_t):
    if s == "∅":
        m = Text("H₀ :   E[τ | Zⱼ]  =  E[τ]   ∀ j ∈ {1,2,3}", color=WHITE_TEXT).scale(sc_t)
    elif s == "Z₁":
        m = Text("H₀ :   E[τ | Zⱼ, Z₁]  =  E[τ | Z₁]   ∀ j ∈ {2,3}", color=WHITE_TEXT).scale(sc_t)
    else:  # "Z₁, Z₃"
        m = Text("H₀ :   E[τ | Z₁, Z₂, Z₃]  =  E[τ | Z₁, Z₃]", color=WHITE_TEXT).scale(sc_t)
    m.move_to([ctt_x + m.width / 2, FWD_TEST_Y, 0])
    return m


def _mk_bwd_ctt(mode, ctt_x, sc_t):
    """mode: 'step1' | 'step2' | 'stop'"""
    if mode == "step1":
        m = Text("H₀ :   E[τ | Z₁]  =  E[τ]", color=WHITE_TEXT).scale(sc_t)
    elif mode == "step2":
        m = MarkupText(
            "H₀ :   E[τ | Z<sub>S</sub>]"
            "  =  E[τ | Z<sub>S∖j</sub>]   ∀ j ∈ {1,3}",
            color=WHITE_TEXT).scale(sc_t)
    else:  # "stop"
        m = Text("STOP", color=RED_LIGHT, weight="BOLD").scale(sc_t * 1.1)
    m.move_to([ctt_x + m.width / 2, BWD_TEST_Y, 0])
    return m


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot 1 — Step 1, S just updated to {Z₁}
#   forward p-values still visible; backward test written
# ══════════════════════════════════════════════════════════════════════════════
class Snapshot1(Scene):
    def construct(self):
        m = _build_base(self)
        Z1, Z2, Z3 = m["Z1"], m["Z2"], m["Z3"]
        ctt_x, sc_t = m["ctt_x"], m["sc_t"]

        # Z₁ selected (green)
        Z1[0].set_stroke(GREEN_LIGHT, width=3.5)
        Z1[1].set_color(GREEN_LIGHT)

        # S label
        self.add(_s_label(["Z₁"]))
        self.add(_step_num(1))

        # Forward test content
        self.add(_mk_fwd_ctt("∅", ctt_x, sc_t))

        # Forward p-values (still visible)
        self.add(_pval("0.001", color=GREEN_LIGHT).next_to(Z1, LEFT, buff=0.45))
        self.add(_pval("0.011").next_to(Z2, LEFT, buff=0.45))
        self.add(_pval("0.008").next_to(Z3, LEFT, buff=0.45))

        # Backward test already written
        self.add(_mk_bwd_ctt("step1", ctt_x, sc_t))

        self.wait(1)


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot 2 — Step 2, S just updated to {Z₁, Z₃}
#   forward p-values still visible; backward test written
# ══════════════════════════════════════════════════════════════════════════════
class Snapshot2(Scene):
    def construct(self):
        m = _build_base(self)
        Z1, Z2, Z3, W1 = m["Z1"], m["Z2"], m["Z3"], m["W1"]
        bnd_Z1_W1, bnd_Z1_W2 = m["bnd_Z1_W1"], m["bnd_Z1_W2"]
        ctt_x, sc_t = m["ctt_x"], m["sc_t"]

        # W₁ dimmed (already eliminated in step 1)
        W1[0].set_stroke(opacity=0.20)
        W1[1].set_opacity(0.20)
        bnd_Z1_W1.set_fill(opacity=0.08).set_stroke(opacity=0.06)
        bnd_Z1_W2.set_fill(opacity=0.05).set_stroke(opacity=0.03)
        self.add(_cross_on(W1))

        # Z₁ and Z₃ selected (green)
        Z1[0].set_stroke(GREEN_LIGHT, width=3.5)
        Z1[1].set_color(GREEN_LIGHT)
        Z3[0].set_stroke(GREEN_LIGHT, width=3.5)
        Z3[1].set_color(GREEN_LIGHT)

        # S label
        self.add(_s_label(["Z₁", "Z₃"]))
        self.add(_step_num(2))

        # Forward test content
        self.add(_mk_fwd_ctt("Z₁", ctt_x, sc_t))

        # Forward p-values (still visible)
        self.add(_pval("0.038", sc=0.32).next_to(Z2, LEFT, buff=0.45))
        self.add(_pval("0.012", color=GREEN_LIGHT, sc=0.32).next_to(Z3, LEFT, buff=0.45))

        # Backward test already written
        self.add(_mk_bwd_ctt("step2", ctt_x, sc_t))

        self.wait(1)


# ══════════════════════════════════════════════════════════════════════════════
# Snapshot 3 — Step 3, Z₂ failed (S stays {Z₁, Z₃})
#   forward p-value for Z₂ visible (dimmed); backward test → STOP
# ══════════════════════════════════════════════════════════════════════════════
class Snapshot3(Scene):
    def construct(self):
        m = _build_base(self)
        Z1, Z2, Z3 = m["Z1"], m["Z2"], m["Z3"]
        W1, W2 = m["W1"], m["W2"]
        bnd_Z1_W1, bnd_Z1_W2 = m["bnd_Z1_W1"], m["bnd_Z1_W2"]
        bnd_Z3_W2, bnd_Z3_W1 = m["bnd_Z3_W2"], m["bnd_Z3_W1"]
        ctt_x, sc_t = m["ctt_x"], m["sc_t"]

        # W₁ dimmed
        W1[0].set_stroke(opacity=0.20)
        W1[1].set_opacity(0.20)
        bnd_Z1_W1.set_fill(opacity=0.08).set_stroke(opacity=0.06)
        bnd_Z1_W2.set_fill(opacity=0.05).set_stroke(opacity=0.03)
        self.add(_cross_on(W1))

        # W₂ dimmed
        W2[0].set_stroke(opacity=0.20)
        W2[1].set_opacity(0.20)
        bnd_Z3_W2.set_fill(opacity=0.08).set_stroke(opacity=0.06)
        bnd_Z3_W1.set_fill(opacity=0.05).set_stroke(opacity=0.03)
        self.add(_cross_on(W2))

        # Z₁ and Z₃ remain green (in S)
        Z1[0].set_stroke(GREEN_LIGHT, width=3.5)
        Z1[1].set_color(GREEN_LIGHT)
        Z3[0].set_stroke(GREEN_LIGHT, width=3.5)
        Z3[1].set_color(GREEN_LIGHT)

        # Z₂ dimmed (failed)
        Z2[0].set_stroke(DIM_GRAY, width=2.0)
        Z2[1].set_color(DIM_GRAY)

        # S label (unchanged from step 2)
        self.add(_s_label(["Z₁", "Z₃"]))
        self.add(_step_num(3))

        # Forward test content
        self.add(_mk_fwd_ctt("Z₁, Z₃", ctt_x, sc_t))

        # Forward p-value for Z₂ (dimmed — failed)
        self.add(_pval("0.214", color=DIM_GRAY, sc=0.30).next_to(Z2, LEFT, buff=0.45))

        # Backward test → STOP
        self.add(_mk_bwd_ctt("stop", ctt_x, sc_t))

        self.wait(1)
