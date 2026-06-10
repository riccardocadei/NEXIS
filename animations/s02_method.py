# Render (preview):  cd animations && conda run -n manim manim -pql s02_method.py Selection
# Render (Twitter):  cd animations && conda run -n manim manim -qh  s02_method.py Selection
# Duration: ~70s

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

# White-background palette overrides
BG         = "#FFFFFF"
WHITE_TEXT = "#1a1a1a"
GRAY_TEXT  = "#666666"
DIM_GRAY   = "#BBBBBB"
GREEN_LIGHT = "#2d8a4e"
RED_LIGHT   = "#c0392b"

# ── Layout constants ──────────────────────────────────────────────────────────

_NODE_R   = 0.30
_STROKE_W = 2.5
LABEL_Y   = 2.30      # y-centre of column labels (candidate neurons row)
FORM_Y     = -3.30     # bottom reference for test block
FWD_TEST_Y = FORM_Y + 0.70   # forward test line  (-2.60)
BWD_TEST_Y = FORM_Y + 0.28   # backward test line (-3.02)
SLBL_Y     = FORM_Y + 1.10   # S = {} label       (-2.20)


# ── Helpers ───────────────────────────────────────────────────────────────────

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
    """Curved filled band from src center-right to dst center-left."""
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
    """Step counter at top-left, vertically aligned with column labels."""
    return (Text(f"Step  {n}", color=GRAY_TEXT, weight="BOLD")
            .scale(SMALL_SCALE)
            .to_corner(UL, buff=0.55)
            .set_y(LABEL_Y))


def _s_label(members):
    """S set indicator, centred above the test block."""
    txt = "S  =  ∅" if not members else "S  =  {" + ",  ".join(members) + "}"
    return (Text(txt, color=WHITE_TEXT)
            .scale(SMALL_SCALE)
            .move_to([0, SLBL_Y, 0]))


def _fwd_test_line(s_display=None):
    sc = 0.37
    if s_display is None:
        return Text("a.  Forward test:   None", color=DIM_GRAY).scale(sc).move_to([0, FWD_TEST_Y, 0])
    if s_display == "∅":
        txt = "a.  Forward test:   H₀(j) :   E[τ | Zⱼ]  =  E[τ]"
    else:
        txt = (f"a.  Forward test:   H₀(j | {s_display}) :"
               f"   E[τ | Zⱼ, {s_display}]  =  E[τ | {s_display}]")
    return Text(txt, color=WHITE_TEXT).scale(sc).move_to([0, FWD_TEST_Y, 0])


def _bwd_test_line(active=False):
    sc = 0.37
    if active:
        mob = MarkupText(
            "b.  Backward test:   H₀(j | Z<sub>S∖j</sub>) :"
            "   E[τ | Z<sub>S</sub>]  =  E[τ | Z<sub>S∖j</sub>]",
            color=WHITE_TEXT).scale(sc)
    else:
        mob = Text("b.  Backward test:   None", color=DIM_GRAY).scale(sc)
    return mob.move_to([0, BWD_TEST_Y, 0])


# ─────────────────────────────────────────────────────────────────────────────
class Selection(Scene):
    def construct(self):
        self.camera.background_color = BG

        # ── Title ─────────────────────────────────────────────────────────────
        title = Text("Neural EXposure Interaction Search",
                     color=WHITE_TEXT).scale(TITLE_SCALE).to_edge(UP, buff=0.35)

        # ── DAG nodes ────────────────────────────────────────────────────────
        Z1 = _node("Z₁");  Z2 = _node("Z₂");  Z3 = _node("Z₃")
        W1 = _node("W₁");  W2 = _node("W₂")
        T  = _node("T");        Y  = _node("Y")

        col_Z = -2.8;  col_W = -0.1;  col_Y = 2.8
        dy = 0.0

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

        # ── Title + DAG appear together ───────────────────────────────────────
        # ── Test block layout (computed before first frame) ───────────────────
        sc_t     = 0.37
        buff_lc  = 0.22

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

        s_lbl = _s_label([])

        self.play(
            FadeIn(title),
            FadeIn(T), FadeIn(lbl_T), FadeIn(Y), FadeIn(lbl_Y),
            FadeIn(a_T_Y), FadeIn(a_W1_Y), FadeIn(a_W2_Y),
            FadeIn(W1), FadeIn(W2), FadeIn(lbl_W),
            FadeIn(Z1), FadeIn(Z2), FadeIn(Z3), FadeIn(lbl_Z),
            FadeIn(bnd_Z1_W1), FadeIn(bnd_Z1_W2),
            FadeIn(bnd_Z3_W2), FadeIn(bnd_Z3_W1),
            FadeIn(bnd_Z2_W1), FadeIn(bnd_Z2_W2),
            FadeIn(s_lbl),
            FadeIn(fwd_lbl), FadeIn(bwd_lbl),
            run_time=1.8,
        )
        self.wait(1.0)

        def _mk_fwd(s=None):
            if s is None:
                m = Text("", color=DIM_GRAY).scale(sc_t)
            elif s == "∅":
                m = Text("H₀ :   E[τ | Zⱼ]  =  E[τ]   ∀ j ∈ {1,2,3}", color=WHITE_TEXT).scale(sc_t)
            elif s == "Z₁":
                m = Text("H₀ :   E[τ | Zⱼ, Z₁]  =  E[τ | Z₁]   ∀ j ∈ {2,3}", color=WHITE_TEXT).scale(sc_t)
            else:  # s == "Z₁, Z₃" — only j=2 remains
                m = Text("H₀ :   E[τ | Z₁, Z₂, Z₃]  =  E[τ | Z₁, Z₃]", color=WHITE_TEXT).scale(sc_t)
            m.move_to([ctt_x + m.width / 2, FWD_TEST_Y, 0])
            return m

        def _mk_bwd(active=False, step1=False):
            if active and step1:
                m = Text("H₀ :   E[τ | Z₁]  =  E[τ]", color=WHITE_TEXT).scale(sc_t)
            elif active:
                m = MarkupText(
                    "H₀ :   E[τ | Z<sub>S</sub>]"
                    "  =  E[τ | Z<sub>S∖j</sub>]   ∀ j ∈ {1,3}",
                    color=WHITE_TEXT).scale(sc_t)
            else:
                m = Text("", color=DIM_GRAY).scale(sc_t)
            m.move_to([ctt_x + m.width / 2, BWD_TEST_Y, 0])
            return m

        # ══════════════════════════════════════════════════════════════════════
        # STEP 1 — S = ∅ : test each neuron unconditionally; select Z₁
        # ══════════════════════════════════════════════════════════════════════
        step_num_tgt = _step_num(1)
        step_big = (Text("Step  1", color=WHITE_TEXT, weight="BOLD")
                    .scale(1.0).move_to(ORIGIN))
        overlay1 = Rectangle(width=16, height=9,
                              fill_color=BG, fill_opacity=0.82,
                              stroke_width=0).move_to(ORIGIN)

        self.play(FadeIn(overlay1), FadeIn(step_big), run_time=0.45)
        self.wait(0.40)
        self.play(Transform(step_big, step_num_tgt), FadeOut(overlay1), run_time=0.60)
        step_num = step_big
        self.wait(0.2)

        # empty content slots (labels already visible)
        fwd_ctt = _mk_fwd(None)
        bwd_ctt = _mk_bwd(False)
        self.add(fwd_ctt, bwd_ctt)
        self.wait(0.3)

        # a. forward fills in
        new_fwd = _mk_fwd("∅")
        self.play(FadeOut(fwd_ctt), FadeIn(new_fwd), run_time=0.55)
        fwd_ctt = new_fwd
        self.wait(0.45)

        # forward p-vals appear all at once
        val_z1 = _pval("0.001").next_to(Z1, LEFT, buff=0.45)
        val_z2 = _pval("0.011").next_to(Z2, LEFT, buff=0.45)
        val_z3 = _pval("0.008").next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(val_z1), FadeIn(val_z2), FadeIn(val_z3), run_time=0.8)
        self.wait(0.7)

        # Z₁ selected (lowest p-val)
        self.play(
            val_z1.animate.set_color(GREEN_LIGHT),
            Z1[0].animate.set_stroke(GREEN_LIGHT, width=3.5),
            Z1[1].animate.set_color(GREEN_LIGHT),
            run_time=0.40,
        )
        self.wait(0.20)

        # append Z₁ to S
        s_lbl_1 = _s_label(["Z₁"])
        ghost1 = Z1[1].copy()
        self.add(ghost1)
        self.play(ghost1.animate.move_to(s_lbl_1.get_center()), run_time=0.42)
        self.play(FadeOut(VGroup(ghost1, s_lbl)), FadeIn(s_lbl_1), run_time=0.35)
        s_lbl = s_lbl_1
        self.wait(0.3)

        # b. backward: clear fwd p-vals, fill in H₀(Z₁) (S∖Z₁ = ∅)
        new_bwd = _mk_bwd(active=True, step1=True)
        self.play(
            FadeOut(val_z1), FadeOut(val_z2), FadeOut(val_z3),
            FadeOut(bwd_ctt), FadeIn(new_bwd),
            run_time=0.55,
        )
        bwd_ctt = new_bwd
        self.wait(0.45)

        # p-val on Z₁ only (the selected node) — appears already green
        bwd1_z1 = _pval("0.001", color=GREEN_LIGHT, sc=0.32).next_to(Z1, LEFT, buff=0.45)
        self.play(FadeIn(bwd1_z1), run_time=0.7)
        self.wait(0.7)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 2 — beat 1: clear + step num + graph dim; beat 2: fill fwd def
        # ══════════════════════════════════════════════════════════════════════
        x_W1          = _cross_on(W1)
        step_num_tgt2 = _step_num(2)
        step_big_2    = (Text("Step  2", color=WHITE_TEXT, weight="BOLD")
                         .scale(1.0).move_to(ORIGIN))
        overlay2 = Rectangle(width=16, height=9,
                              fill_color=BG, fill_opacity=0.82,
                              stroke_width=0).move_to(ORIGIN)

        self.play(
            FadeOut(bwd1_z1), FadeOut(bwd_ctt),
            FadeOut(step_num), FadeOut(fwd_ctt),
            W1[0].animate.set_stroke(opacity=0.20),
            W1[1].animate.set_opacity(0.20),
            bnd_Z1_W1.animate.set_fill(opacity=0.08).set_stroke(opacity=0.06),
            bnd_Z1_W2.animate.set_fill(opacity=0.05).set_stroke(opacity=0.03),
            FadeIn(x_W1),
            FadeIn(overlay2), FadeIn(step_big_2),
            run_time=0.9,
        )
        self.wait(0.40)
        self.play(Transform(step_big_2, step_num_tgt2), FadeOut(overlay2), run_time=0.60)
        step_num = step_big_2
        bwd_ctt  = _mk_bwd(False)
        self.add(bwd_ctt)
        self.wait(0.25)

        new_fwd_2 = _mk_fwd("Z₁")
        self.play(FadeIn(new_fwd_2), run_time=0.5)
        fwd_ctt = new_fwd_2
        self.wait(0.35)

        # forward p-vals appear all at once
        val2_z2 = _pval("0.038", sc=0.32).next_to(Z2, LEFT, buff=0.45)
        val2_z3 = _pval("0.012", sc=0.32).next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(val2_z2), FadeIn(val2_z3), run_time=0.8)
        self.wait(0.7)

        # Z₃ selected
        self.play(
            val2_z3.animate.set_color(GREEN_LIGHT),
            Z3[0].animate.set_stroke(GREEN_LIGHT, width=3.5),
            Z3[1].animate.set_color(GREEN_LIGHT),
            run_time=0.40,
        )
        self.wait(0.20)

        # append Z₃ to S
        s_lbl_2 = _s_label(["Z₁", "Z₃"])
        ghost2 = Z3[1].copy()
        self.add(ghost2)
        self.play(ghost2.animate.move_to(s_lbl_2.get_center()), run_time=0.42)
        self.play(FadeOut(VGroup(ghost2, s_lbl)), FadeIn(s_lbl_2), run_time=0.35)
        s_lbl = s_lbl_2
        self.wait(0.3)

        # b. backward: clear fwd p-vals, fill in active backward def
        new_bwd = _mk_bwd(True)
        self.play(
            FadeOut(val2_z2), FadeOut(val2_z3),
            FadeOut(bwd_ctt), FadeIn(new_bwd),
            run_time=0.55,
        )
        bwd_ctt = new_bwd
        self.wait(0.45)

        # p-vals on Z₁ and Z₃ (both selected) — appear already green
        bwd_z1 = _pval("0.004", color=GREEN_LIGHT, sc=0.32).next_to(Z1, LEFT, buff=0.45)
        bwd_z3 = _pval("0.011", color=GREEN_LIGHT, sc=0.32).next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(bwd_z1), FadeIn(bwd_z3), run_time=0.8)
        self.wait(0.7)

        # ══════════════════════════════════════════════════════════════════════
        # STEP 3 — beat 1: clear + step num + graph dim; beat 2: fill fwd def
        # ══════════════════════════════════════════════════════════════════════
        x_W2          = _cross_on(W2)
        step_num_tgt3 = _step_num(3)
        step_big_3    = (Text("Step  3", color=WHITE_TEXT, weight="BOLD")
                         .scale(1.0).move_to(ORIGIN))
        overlay3 = Rectangle(width=16, height=9,
                              fill_color=BG, fill_opacity=0.82,
                              stroke_width=0).move_to(ORIGIN)

        self.play(
            FadeOut(bwd_z1), FadeOut(bwd_z3), FadeOut(bwd_ctt),
            FadeOut(step_num), FadeOut(fwd_ctt),
            W2[0].animate.set_stroke(opacity=0.20),
            W2[1].animate.set_opacity(0.20),
            bnd_Z3_W2.animate.set_fill(opacity=0.08).set_stroke(opacity=0.06),
            bnd_Z3_W1.animate.set_fill(opacity=0.05).set_stroke(opacity=0.03),
            FadeIn(x_W2),
            FadeIn(overlay3), FadeIn(step_big_3),
            run_time=0.9,
        )
        self.wait(0.40)
        self.play(Transform(step_big_3, step_num_tgt3), FadeOut(overlay3), run_time=0.60)
        step_num = step_big_3
        bwd_ctt  = _mk_bwd(False)
        self.add(bwd_ctt)
        self.wait(0.25)

        new_fwd_3 = _mk_fwd("Z₁, Z₃")
        self.play(FadeIn(new_fwd_3), run_time=0.5)
        fwd_ctt = new_fwd_3
        self.wait(0.35)

        # single forward p-val for Z₂
        val3_z2 = _pval("0.214", sc=0.30).next_to(Z2, LEFT, buff=0.45)
        self.play(FadeIn(val3_z2), run_time=0.8)
        self.wait(0.85)

        # Z₂ fails — p-val > α/k
        self.play(
            val3_z2.animate.set_color(DIM_GRAY),
            Z2[0].animate.set_stroke(DIM_GRAY, width=2.0),
            Z2[1].animate.set_color(DIM_GRAY),
            run_time=0.75,
        )
        self.wait(0.5)
        self.play(FadeOut(val3_z2), run_time=0.5)
        self.wait(0.5)

        # ─── FINALE ─────────────────────────────────────────────────────────────
        # Proxy subgraph — Z₁, Z₃ replace W₁, W₂; dashed arrows to Y
        proxy1 = DashedLine(
            Z1.get_right(), Y.get_left(),
            dash_length=0.14, color=WHITE_TEXT, stroke_width=2.0,
        )
        proxy1.add_tip(tip_length=0.12)

        proxy3 = DashedLine(
            Z3.get_right(), Y.get_left(),
            dash_length=0.14, color=WHITE_TEXT, stroke_width=2.0,
        )
        proxy3.add_tip(tip_length=0.12)

        keep = {title, Z1, Z3, T, Y, a_T_Y}
        fade_group = Group(*[m for m in self.mobjects if m not in keep])

        self.play(
            FadeOut(fade_group),
            Z1[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            Z1[1].animate.set_color(WHITE_TEXT),
            Z3[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            Z3[1].animate.set_color(WHITE_TEXT),
            T[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            T[1].animate.set_color(WHITE_TEXT),
            Y[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            Y[1].animate.set_color(WHITE_TEXT),
            a_T_Y.animate.set_color(WHITE_TEXT),
            run_time=1.0,
        )
        self.play(FadeIn(proxy1), FadeIn(proxy3), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
