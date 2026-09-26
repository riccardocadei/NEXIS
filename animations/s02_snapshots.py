# Still frames of the NEXIS toy example (same layout and numbers as s02_method.py).
#
# Render all four snapshots (PNG):
#   cd animations && conda run -n manim manim -sqh s02_snapshots.py Snapshot1 Snapshot2 Snapshot3 Snapshot4
# Or one at a time (low-quality preview):
#   conda run -n manim manim -sql s02_snapshots.py Snapshot1

from manim import *
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from s02_method import (
    GREEN_LIGHT, DIM_GRAY,
    build_base, fwd_content, bwd_content, pval, subset_pvals,
    step_tag, fwd_tag, BWD_TAG, s_label, select_style, explained_style, _cross_on,
)


def _base(scene):
    b = build_base(scene)
    scene.add(*b["static"])
    return b


def _explain(scene, b, which):
    """Dim W₁ (after Z₁ enters) and/or W₂ (after Z₃ enters)."""
    if 1 in which:
        explained_style(b["W1"], [b["bands"]["Z1_W1"], b["bands"]["Z1_W2"]])
        scene.add(_cross_on(b["W1"]))
    if 3 in which:
        explained_style(b["W2"], [b["bands"]["Z3_W2"], b["bands"]["Z3_W1"]])
        scene.add(_cross_on(b["W2"]))


# Snapshot 1 — forward step 1: marginal p-values, Z₁ admitted (0.001 <= 0.05/3)
class Snapshot1(Scene):
    def construct(self):
        b = _base(self)
        select_style(b["Z1"])
        self.add(step_tag(fwd_tag(1)), s_label(["1"]), fwd_content(1))
        self.add(pval("0.001", color=GREEN_LIGHT).next_to(b["Z1"], LEFT, buff=0.45))
        self.add(pval("0.011").next_to(b["Z2"], LEFT, buff=0.45))
        self.add(pval("0.008").next_to(b["Z3"], LEFT, buff=0.45))
        self.wait(1)


# Snapshot 2 — forward step 2: given Z₁, Z₃ admitted (0.012 <= 0.05/2)
class Snapshot2(Scene):
    def construct(self):
        b = _base(self)
        _explain(self, b, {1})
        select_style(b["Z1"]); select_style(b["Z3"])
        self.add(step_tag(fwd_tag(2)), s_label(["1", "3"]), fwd_content(2))
        self.add(pval("0.038", sc=0.32).next_to(b["Z2"], LEFT, buff=0.45))
        self.add(pval("0.012", color=GREEN_LIGHT, sc=0.32).next_to(b["Z3"], LEFT, buff=0.45))
        self.wait(1)


# Snapshot 3 — forward step 3: Z₂ fails 0.05/1, the forward step stops
class Snapshot3(Scene):
    def construct(self):
        b = _base(self)
        _explain(self, b, {1, 3})
        select_style(b["Z1"]); select_style(b["Z3"])
        select_style(b["Z2"], color=DIM_GRAY, width=2.0)
        self.add(step_tag(fwd_tag(3)),
                 s_label(["1", "3"], suffix="forward step stops"),
                 fwd_content(3))
        self.add(pval("0.214", color=DIM_GRAY, sc=0.30).next_to(b["Z2"], LEFT, buff=0.45))
        self.wait(1)


# Snapshot 4 — terminal backward step: every p_j(A) <= 0.05/3, Z₁ and Z₃ kept
class Snapshot4(Scene):
    def construct(self):
        b = _base(self)
        _explain(self, b, {1, 3})
        select_style(b["Z1"]); select_style(b["Z3"])
        select_style(b["Z2"], color=DIM_GRAY, width=2.0)
        self.add(step_tag(BWD_TAG),
                 s_label(["1", "3"], suffix="forward step stops"),
                 bwd_content())
        self.add(subset_pvals(1, [("∅", "0.001"), ("{3}", "0.004")], color=GREEN_LIGHT)
                 .next_to(b["Z1"], LEFT, buff=0.45))
        self.add(subset_pvals(3, [("∅", "0.008"), ("{1}", "0.012")], color=GREEN_LIGHT)
                 .next_to(b["Z3"], LEFT, buff=0.45))
        self.wait(1)
