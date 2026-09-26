# NEXIS (Algorithm 1 of the paper) on a toy dictionary of three candidate neurons.
#
# Forward steps admit the argmin p-value while it passes the gate alpha/|S_bar|;
# once the forward step stops, one terminal backward step keeps j only if
# p_j(A) <= alpha/m for every subset A of the other selected coordinates.
#
# Render (preview):  cd animations && conda run -n manim manim -pql s02_method.py Selection
# Render (website):  cd animations && conda run -n manim manim -qh  s02_method.py Selection
#                    cp media/videos/s02_method/1080p30/Selection.mp4 ../docs/assets/nexis_method.mp4
# Duration: ~35s

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

# ── Toy numbers (alpha = 0.05, m = 3 candidates) ──────────────────────────────
# Forward step k tests j in S_bar given S and admits argmin p if p <= alpha/|S_bar|.
# Terminal backward step keeps j if max_{A subset S\{j}} p_j(A) <= alpha/m = 0.05/3.
# p_j(A) is one test per (j, A): the forward and backward tables below agree
# (p_1(∅)=0.001, p_3(∅)=0.008, p_3({1})=0.012).
ALPHA = 0.05
M     = 3

# ── Layout constants ──────────────────────────────────────────────────────────

_NODE_R   = 0.30
_STROKE_W = 2.5
LABEL_Y   = 2.30      # y-centre of column labels (candidate neurons row)
FORM_Y     = -3.30     # bottom reference for test block
FWD_TEST_Y = FORM_Y + 0.70   # forward step line   (-2.60)
BWD_TEST_Y = FORM_Y + 0.28   # backward step line  (-3.02)
SLBL_Y     = FORM_Y + 1.10   # S = {} label        (-2.20)
SC_T       = 0.37            # test-block text scale


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


def pval(val, color=WHITE_TEXT, sc=0.34):
    return MarkupText(f"<i>p</i>-value = {val}", color=color).scale(sc)


def subset_pvals(j, rows, color=WHITE_TEXT, sc=0.30):
    """Stack of p_j(A) lines for the terminal backward step, rows = [(A, p)]."""
    lines = [MarkupText(f"<i>p</i><sub>{j}</sub>({a}) = {p}", color=color).scale(sc)
             for a, p in rows]
    return VGroup(*lines).arrange(DOWN, buff=0.10, aligned_edge=LEFT)


def step_tag(txt):
    """Step indicator at top-left, vertically aligned with column labels."""
    return (Text(txt, color=GRAY_TEXT, weight="BOLD")
            .scale(SMALL_SCALE)
            .to_corner(UL, buff=0.55)
            .set_y(LABEL_Y))


def s_label(members, suffix=None):
    """Selection indicator, centred above the test block."""
    txt = "S  =  ∅" if not members else "S  =  {" + ",  ".join(members) + "}"
    mob = Text(txt, color=WHITE_TEXT).scale(SMALL_SCALE)
    if suffix:
        mob = VGroup(mob, Text(suffix, color=GRAY_TEXT).scale(SMALL_SCALE * 0.9)
                     ).arrange(RIGHT, buff=0.35)
    return mob.move_to([0, SLBL_Y, 0])


# Forward step contents: H0 of the round and its gate alpha/|S_bar|.
FWD_TXT = {
    1: "H₀(j | ∅) :   E[τ | Zⱼ]  =  E[τ],   j ∈ {1, 2, 3}        gate  α/|S̄|  =  0.05/3",
    2: "H₀(j | Z₁) :   E[τ | Zⱼ, Z₁]  =  E[τ | Z₁],   j ∈ {2, 3}        gate  α/|S̄|  =  0.05/2",
    3: "H₀(2 | Z₁, Z₃) :   E[τ | Z₁, Z₂, Z₃]  =  E[τ | Z₁, Z₃]        gate  α/|S̄|  =  0.05/1",
}
BWD_MARKUP = ("keep  j ∈ S  only if  <i>p</i><sub>j</sub>(A)  ≤  α/m  =  0.05/3"
              "   for every  A ⊆ S ∖ {j}")


def build_base(scene):
    """Add the static DAG and test-block labels to `scene`; return the mobjects."""
    scene.camera.background_color = BG

    title = Text("Neural EXposure Interaction Search",
                 color=WHITE_TEXT).scale(TITLE_SCALE).to_edge(UP, buff=0.35)

    Z1 = _node("Z₁");  Z2 = _node("Z₂");  Z3 = _node("Z₃")
    W1 = _node("W₁");  W2 = _node("W₂")
    T  = _node("T");   Y  = _node("Y")

    col_Z = -2.8;  col_W = -0.1;  col_Y = 2.8

    Z1.move_to([col_Z,  1.55, 0])
    Z2.move_to([col_Z,  0.00, 0])
    Z3.move_to([col_Z, -1.55, 0])
    W1.move_to([col_W,  0.95, 0])
    W2.move_to([col_W, -0.55, 0])
    T.move_to([(col_W + col_Y) / 2, -1.55, 0])
    Y.move_to([col_Y,   0.20, 0])

    a_T_Y  = _causal_arrow(T,  Y)
    a_W1_Y = _causal_arrow(W1, Y)
    a_W2_Y = _causal_arrow(W2, Y)

    # Z₁ and Z₃ are the principal proxies of W₁ and W₂; Z₂ is entangled with both.
    bands = dict(
        Z1_W1=_sankey_band(Z1, W1, BLUE_LIGHT,   width=0.14, opacity=0.40),
        Z1_W2=_sankey_band(Z1, W2, BLUE_LIGHT,   width=0.06, opacity=0.18),
        Z3_W2=_sankey_band(Z3, W2, PURPLE_LIGHT, width=0.14, opacity=0.40),
        Z3_W1=_sankey_band(Z3, W1, PURPLE_LIGHT, width=0.06, opacity=0.18),
        Z2_W1=_sankey_band(Z2, W1, GRAY_TEXT,    width=0.06, opacity=0.18),
        Z2_W2=_sankey_band(Z2, W2, GRAY_TEXT,    width=0.06, opacity=0.18),
    )

    lbl_sc = LABEL_SCALE * 0.85
    lbl_Z = VGroup(
        Text("candidate neurons",        color=GRAY_TEXT).scale(lbl_sc),
        Text("(learned representation)", color=GRAY_TEXT).scale(lbl_sc * 0.9),
    ).arrange(DOWN, buff=0.04).move_to([col_Z, LABEL_Y, 0])
    lbl_W = VGroup(
        Text("direct effect modifiers", color=GRAY_TEXT).scale(lbl_sc),
        Text("(unobserved)",            color=GRAY_TEXT).scale(lbl_sc * 0.9),
    ).arrange(DOWN, buff=0.04).move_to([col_W, LABEL_Y, 0])
    lbl_Y = Text("outcome",   color=GRAY_TEXT).scale(lbl_sc).move_to([col_Y, LABEL_Y, 0])
    lbl_T = Text("treatment", color=GRAY_TEXT).scale(lbl_sc).next_to(T, LEFT, buff=0.20)

    # Test block: labels left-aligned, contents left-aligned at ctt_x, block centred.
    fwd_lbl = Text("a.  Forward step:", color=GRAY_TEXT).scale(SC_T)
    bwd_lbl = Text("b.  Terminal backward step:", color=GRAY_TEXT).scale(SC_T)
    max_lbl_w = max(fwd_lbl.width, bwd_lbl.width)
    widest = max([Text(t).scale(SC_T).width for t in FWD_TXT.values()]
                 + [MarkupText(BWD_MARKUP).scale(SC_T).width])
    buff_lc = 0.22
    lbl_left = -(max_lbl_w + buff_lc + widest) / 2
    fwd_lbl.move_to([lbl_left + fwd_lbl.width / 2, FWD_TEST_Y, 0])
    bwd_lbl.move_to([lbl_left + bwd_lbl.width / 2, BWD_TEST_Y, 0])
    ctt_x = lbl_left + max_lbl_w + buff_lc

    static = [title, T, lbl_T, Y, lbl_Y, a_T_Y, a_W1_Y, a_W2_Y,
              W1, W2, lbl_W, Z1, Z2, Z3, lbl_Z, *bands.values(), fwd_lbl, bwd_lbl]
    return dict(title=title, Z1=Z1, Z2=Z2, Z3=Z3, W1=W1, W2=W2, T=T, Y=Y,
                a_T_Y=a_T_Y, bands=bands, ctt_x=ctt_x, static=static)


def fwd_content(k, ctt_x):
    m = Text(FWD_TXT[k], color=WHITE_TEXT).scale(SC_T)
    return m.move_to([ctt_x + m.width / 2, FWD_TEST_Y, 0])


def bwd_content(ctt_x):
    m = MarkupText(BWD_MARKUP, color=WHITE_TEXT).scale(SC_T)
    return m.move_to([ctt_x + m.width / 2, BWD_TEST_Y, 0])


def select_style(node, color=GREEN_LIGHT, width=3.5):
    """(stroke, label) targets for colouring a node; used with .animate or directly."""
    node[0].set_stroke(color, width=width)
    node[1].set_color(color)
    return node


def explained_style(W, bands):
    """Dim a direct modifier once its principal proxy is in S."""
    W[0].set_stroke(opacity=0.20)
    W[1].set_opacity(0.20)
    bands[0].set_fill(opacity=0.08).set_stroke(opacity=0.06)
    bands[1].set_fill(opacity=0.05).set_stroke(opacity=0.03)


# ─────────────────────────────────────────────────────────────────────────────
class Selection(Scene):
    def _big_step(self, big_txt, tag_txt, extra_anims=()):
        """Full-screen step banner that shrinks into the top-left step tag."""
        big = Text(big_txt, color=WHITE_TEXT, weight="BOLD").scale(1.0).move_to(ORIGIN)
        overlay = Rectangle(width=16, height=9, fill_color=BG, fill_opacity=0.82,
                            stroke_width=0).move_to(ORIGIN)
        self.play(FadeIn(overlay), FadeIn(big), *extra_anims, run_time=0.8)
        self.wait(0.40)
        self.play(Transform(big, step_tag(tag_txt)), FadeOut(overlay), run_time=0.60)
        return big

    def _append_to_S(self, node, new_lbl, old_lbl):
        ghost = node[1].copy()
        self.add(ghost)
        self.play(ghost.animate.move_to(new_lbl.get_center()), FadeOut(old_lbl),
                  run_time=0.5)
        self.play(ReplacementTransform(ghost, new_lbl), run_time=0.35)
        return new_lbl

    def construct(self):
        b = build_base(self)
        Z1, Z2, Z3, W1, W2 = b["Z1"], b["Z2"], b["Z3"], b["W1"], b["W2"]
        bands, ctt_x = b["bands"], b["ctt_x"]

        s_lbl = s_label([])
        self.play(*[FadeIn(m) for m in b["static"]], FadeIn(s_lbl), run_time=1.8)
        self.wait(1.0)

        # ══════════════════════════════════════════════════════════════════════
        # FORWARD 1 — S = ∅: marginal tests; argmin Z₁ passes 0.05/3
        # ══════════════════════════════════════════════════════════════════════
        tag = self._big_step("Forward step  1", "Forward  1")
        fwd = fwd_content(1, ctt_x)
        self.play(FadeIn(fwd), run_time=0.55)
        self.wait(0.45)

        p1 = pval("0.001").next_to(Z1, LEFT, buff=0.45)
        p2 = pval("0.011").next_to(Z2, LEFT, buff=0.45)
        p3 = pval("0.008").next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(p1), FadeIn(p2), FadeIn(p3), run_time=0.8)
        self.wait(0.9)

        # argmin p = 0.001 <= 0.05/3: Z₁ enters S
        self.play(p1.animate.set_color(GREEN_LIGHT),
                  Z1[0].animate.set_stroke(GREEN_LIGHT, width=3.5),
                  Z1[1].animate.set_color(GREEN_LIGHT), run_time=0.40)
        self.wait(0.20)
        s_lbl = self._append_to_S(Z1, s_label(["1"]), s_lbl)
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════════════
        # FORWARD 2 — S = {1}: W₁ is screened off by Z₁; argmin Z₃ passes 0.05/2
        # ══════════════════════════════════════════════════════════════════════
        x_W1 = _cross_on(W1)
        tag_old = tag
        tag = self._big_step("Forward step  2", "Forward  2", extra_anims=(
            FadeOut(p1), FadeOut(p2), FadeOut(p3), FadeOut(fwd), FadeOut(tag_old),
            W1[0].animate.set_stroke(opacity=0.20), W1[1].animate.set_opacity(0.20),
            bands["Z1_W1"].animate.set_fill(opacity=0.08).set_stroke(opacity=0.06),
            bands["Z1_W2"].animate.set_fill(opacity=0.05).set_stroke(opacity=0.03),
            FadeIn(x_W1)))
        fwd = fwd_content(2, ctt_x)
        self.play(FadeIn(fwd), run_time=0.5)
        self.wait(0.45)

        p2 = pval("0.038", sc=0.32).next_to(Z2, LEFT, buff=0.45)
        p3 = pval("0.012", sc=0.32).next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(p2), FadeIn(p3), run_time=0.8)
        self.wait(0.9)

        self.play(p3.animate.set_color(GREEN_LIGHT),
                  Z3[0].animate.set_stroke(GREEN_LIGHT, width=3.5),
                  Z3[1].animate.set_color(GREEN_LIGHT), run_time=0.40)
        self.wait(0.20)
        s_lbl = self._append_to_S(Z3, s_label(["1", "3"]), s_lbl)
        self.wait(0.6)

        # ══════════════════════════════════════════════════════════════════════
        # FORWARD 3 — S = {1, 3}: W₂ screened off; Z₂ fails 0.05/1 → forward stops
        # ══════════════════════════════════════════════════════════════════════
        x_W2 = _cross_on(W2)
        tag_old = tag
        tag = self._big_step("Forward step  3", "Forward  3", extra_anims=(
            FadeOut(p2), FadeOut(p3), FadeOut(fwd), FadeOut(tag_old),
            W2[0].animate.set_stroke(opacity=0.20), W2[1].animate.set_opacity(0.20),
            bands["Z3_W2"].animate.set_fill(opacity=0.08).set_stroke(opacity=0.06),
            bands["Z3_W1"].animate.set_fill(opacity=0.05).set_stroke(opacity=0.03),
            FadeIn(x_W2)))
        fwd = fwd_content(3, ctt_x)
        self.play(FadeIn(fwd), run_time=0.5)
        self.wait(0.45)

        p2 = pval("0.214", sc=0.30).next_to(Z2, LEFT, buff=0.45)
        self.play(FadeIn(p2), run_time=0.8)
        self.wait(0.9)

        # 0.214 > 0.05: nothing enters, the forward step stops at S = {1, 3}
        stop_lbl = s_label(["1", "3"], suffix="forward step stops")
        self.play(p2.animate.set_color(DIM_GRAY),
                  Z2[0].animate.set_stroke(DIM_GRAY, width=2.0),
                  Z2[1].animate.set_color(DIM_GRAY),
                  FadeOut(s_lbl), FadeIn(stop_lbl), run_time=0.75)
        s_lbl = stop_lbl
        self.wait(1.2)

        # ══════════════════════════════════════════════════════════════════════
        # TERMINAL BACKWARD STEP — every subset A of the other selected coordinates,
        # level α/m = 0.05/3; both Z₁ and Z₃ survive
        # ══════════════════════════════════════════════════════════════════════
        tag_old = tag
        tag = self._big_step("Terminal backward step", "Backward", extra_anims=(
            FadeOut(p2), FadeOut(fwd), FadeOut(tag_old)))
        bwd = bwd_content(ctt_x)
        self.play(FadeIn(bwd), run_time=0.55)
        self.wait(0.9)

        q1 = subset_pvals(1, [("∅", "0.001"), ("{3}", "0.004")]).next_to(Z1, LEFT, buff=0.45)
        q3 = subset_pvals(3, [("∅", "0.008"), ("{1}", "0.012")]).next_to(Z3, LEFT, buff=0.45)
        self.play(FadeIn(q1), FadeIn(q3), run_time=0.9)
        self.wait(1.4)

        # all p_j(A) <= 0.05/3: both coordinates are kept
        self.play(q1.animate.set_color(GREEN_LIGHT), q3.animate.set_color(GREEN_LIGHT),
                  run_time=0.5)
        for _ in range(2):
            self.play(Z1.animate.scale(1.18), Z3.animate.scale(1.18), run_time=0.25)
            self.play(Z1.animate.scale(1 / 1.18), Z3.animate.scale(1 / 1.18), run_time=0.25)
        self.wait(1.0)

        # ─── FINALE ─────────────────────────────────────────────────────────────
        # Principal proxies Z₁, Z₃ stand in for the latent W₁, W₂; arrows to Y
        T, Y, a_T_Y = b["T"], b["Y"], b["a_T_Y"]
        proxy1 = _causal_arrow(Z1, Y)
        proxy3 = _causal_arrow(Z3, Y)

        keep = {b["title"], Z1, Z3, T, Y, a_T_Y}
        fade_group = Group(*[m for m in self.mobjects if m not in keep])
        self.play(
            FadeOut(fade_group),
            Z1[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            Z1[1].animate.set_color(WHITE_TEXT),
            Z3[0].animate.set_stroke(WHITE_TEXT, width=2.5),
            Z3[1].animate.set_color(WHITE_TEXT),
            run_time=1.0,
        )
        self.play(FadeIn(proxy1), FadeIn(proxy3), run_time=0.7)
        self.wait(1.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.0)
