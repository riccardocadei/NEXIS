# Render (preview):  cd animations && conda run -n manim manim -pql s01_pipeline.py Pipeline
# Render (Twitter):  cd animations && conda run -n manim manim -qh  s01_pipeline.py Pipeline
# Duration: ~65s

from manim import *
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BG, WHITE_TEXT, GRAY_TEXT, DIM_GRAY,
    BLUE_LIGHT, GREEN_LIGHT, YELLOW_LIGHT, PURPLE_LIGHT, TEAL_LIGHT,
    LABEL_SCALE,
    make_arrow,
)

ASSETS   = Path(__file__).parent / "assets"
TILE_PNG = ASSETS / "sample_tile.png"
TILE_TIF = Path(
    "/nfs/scistore19/locatgrp/rcadei/NEXIS/data/ghana/satellite/tif/ghana_comm0014.tif"
)


# ── Tile helper ────────────────────────────────────────────────────────────────
def _make_tile_png() -> bool:
    if TILE_PNG.exists():
        return True
    ASSETS.mkdir(exist_ok=True)
    if not TILE_TIF.exists():
        return False
    try:
        import rasterio
        from PIL import Image as PILImage
        with rasterio.open(TILE_TIF) as src:
            data = src.read()
        rgb = np.transpose(data[[2, 1, 0]], (1, 2, 0)).astype(float)
        for b in range(3):
            p2, p98 = np.percentile(rgb[:, :, b], [2, 98])
            rgb[:, :, b] = np.clip((rgb[:, :, b] - p2) / max(p98 - p2, 1e-9), 0, 1)
        PILImage.fromarray((rgb * 255).astype(np.uint8)) \
            .resize((256, 256), PILImage.LANCZOS).save(str(TILE_PNG))
        return True
    except Exception as e:
        print(f"[tile PNG: {e}]")
        return False


# ── Scene helpers ──────────────────────────────────────────────────────────────
def dot_col(n, color, h=1.55, r=0.060, opacity=0.65):
    step = h / max(n - 1, 1)
    grp  = VGroup()
    for i in range(n):
        grp.add(Dot(radius=r, color=color, fill_opacity=opacity)
                .move_to(UP * (h / 2 - i * step)))
    return grp


def survey_card(rows=4, w=1.38, rh=0.25, seed=7):
    fields = ["treatment", "outcome", "age", "assets"]
    rng    = np.random.default_rng(seed)
    card   = VGroup()
    for i, f in enumerate(fields[:rows]):
        y   = (rows / 2 - .5 - i) * (rh + 0.04)
        bg  = Rectangle(width=w, height=rh, fill_color="#FFFFF0", fill_opacity=1.0,
                        stroke_color=YELLOW_LIGHT, stroke_width=0.8).move_to(UP * y)
        ft  = Text(f, color=GRAY_TEXT).scale(0.22).next_to(bg.get_left(), RIGHT, buff=0.10)
        bw  = float(rng.uniform(0.3, 0.85)) * (w * 0.48)
        bar = Rectangle(width=bw, height=rh * 0.40, fill_color=YELLOW_LIGHT,
                        fill_opacity=0.75, stroke_width=0)
        bar.move_to(bg.get_right() + LEFT * (0.10 + bw / 2))
        card.add(bg, ft, bar)
    card.add(Rectangle(width=w + 0.06, height=rows * (rh + 0.04) + 0.04,
                       color=YELLOW_LIGHT, stroke_width=2.0, fill_opacity=0))
    return card


def cand_node(pos, color, lbl_text, r=0.27, sc=0.38):
    circ = Circle(radius=r, color=color, fill_color=color,
                  fill_opacity=0.15, stroke_width=2.0).move_to(pos)
    lbl  = Text(lbl_text, color=color).scale(sc).move_to(circ)
    return VGroup(circ, lbl)


def pill(txt, col, sc=0.30):
    t  = Text(txt, color=col).scale(sc)
    bg = RoundedRectangle(
        width=t.width + 0.20, height=t.height + 0.14, corner_radius=0.08,
        fill_color=col, fill_opacity=0.12, stroke_color=col, stroke_width=1.0,
    )
    return VGroup(bg, t.move_to(bg))


def _unit(a, b):
    d = b - a
    return d / np.linalg.norm(d)


# ── Scene ─────────────────────────────────────────────────────────────────────
class Pipeline(Scene):
    def construct(self):                                   # noqa: C901
        self.camera.background_color = BG

        # ── Layout constants ───────────────────────────────────────────────
        X0, X1, X2 = -5.80, -3.65, -2.10   # tile | dense | Z column
        Y_S, Y_V   =  1.30, -1.45           # satellite arm y | survey arm y

        # Z nodes (SAE): vertical column at X2, evenly spaced
        N_Z   = 8
        Z_R   = 0.22
        Z_y_c = 1.70
        Z_stp = 0.54
        Z_ys  = [Z_y_c + (3.5 - i) * Z_stp for i in range(N_Z)]
        Z_lbls = ["Z₁","Z₂","Z₃","Z₄","Z₅","Z₆","Z₇","Z₈"]

        # X nodes (survey): same size and spacing as Z nodes, 4 nodes
        N_X   = 4
        X_R   = 0.22
        X_y_c = -1.65
        X_stp = 0.54
        X_ys  = [X_y_c + (1.5 - i) * X_stp for i in range(N_X)]
        X_lbls = ["Z₉","Z₁₀","Z₁₁","Z₁₂"]

        T_pos = np.array([2.00, -2.80, 0.0])
        Y_pos = np.array([5.20, -0.30, 0.0])
        TY_R  = 0.28

        SEL_Z = {1, 4}   # Z₂, Z₅  (2 SAE neurons)
        SEL_X = {1}      # Z₁₀     (1 survey neuron)

        # ══════════════════════════════════════════════════════════════════
        # ACT 1 — Pipeline: tile → Prithvi-EO → SAE (14) → 8 activate
        #                   survey card → 4 feature dots
        # ══════════════════════════════════════════════════════════════════
        have_png = _make_tile_png()
        if have_png:
            tile = ImageMobject(str(TILE_PNG)).set_width(1.75)
        else:
            tile = Square(side_length=1.75, fill_color="#4A7A4A",
                          fill_opacity=0.9, stroke_width=0)
        tile.move_to([X0, Y_S, 0])

        border  = Square(side_length=1.82, color=BLUE_LIGHT,
                         stroke_width=2.5, fill_opacity=0).move_to([X0, Y_S, 0])
        sat_lbl = Text("satellite", color=BLUE_LIGHT).scale(LABEL_SCALE)
        sat_lbl.next_to(border, DOWN, buff=0.12)

        self.play(FadeIn(tile), Create(border), FadeIn(sat_lbl), run_time=0.9)
        self.wait(0.4)

        # tile → Prithvi-EO (label overlaid on tile) → dense (10 dots)
        dense = dot_col(10, BLUE_LIGHT, h=1.55).move_to([X1, Y_S, 0])
        a1    = make_arrow(border.get_right(), dense.get_left() + LEFT * 0.04,
                           color=BLUE_LIGHT)
        lbl1  = Text("Prithvi-EO  ❄", color=BLUE_LIGHT).scale(0.30)
        lbl1.move_to(border.get_center())

        self.play(GrowArrow(a1), FadeIn(lbl1), run_time=0.55)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dense],
                               lag_ratio=0.06, run_time=1.10))
        self.wait(0.30)

        # dense → SAE (14 nodes) — same blue stream
        N_SAE    = 14
        N_ACTIVE = 8
        sparse = dot_col(N_SAE, BLUE_LIGHT, h=2.80).move_to([X2, Y_S, 0])
        a2     = make_arrow(dense.get_right(), sparse.get_left() + LEFT * 0.04,
                            color=BLUE_LIGHT)
        lbl2   = Text("SAE", color=BLUE_LIGHT).scale(0.30)
        lbl2.next_to(a2, UP, buff=0.09)

        self.play(GrowArrow(a2), FadeIn(lbl2), run_time=0.50)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in sparse],
                               lag_ratio=0.04, run_time=0.90))
        self.wait(0.30)

        # 8 neurons activate, 6 dim out
        act_idx   = [0, 2, 4, 6, 7, 9, 11, 13]
        inact_idx = [i for i in range(N_SAE) if i not in act_idx]

        self.play(
            *[sparse[i].animate.set_opacity(1.0).scale(1.35) for i in act_idx],
            *[sparse[i].animate.set_opacity(0.10)            for i in inact_idx],
            run_time=0.60,
        )
        self.wait(0.40)

        # survey card → N_X feature dots (gold)
        card    = survey_card().move_to([X0, Y_V, 0])
        srv_lbl = Text("survey", color=YELLOW_LIGHT).scale(LABEL_SCALE)
        srv_lbl.next_to(card, DOWN, buff=0.12)

        self.play(FadeIn(card), FadeIn(srv_lbl), run_time=0.70)
        self.wait(0.25)

        sfeat = dot_col(N_X, YELLOW_LIGHT, h=(N_X - 1) * X_stp).move_to([X2, Y_V, 0])
        a3    = make_arrow(card.get_right(), sfeat.get_left() + LEFT * 0.04,
                           color=YELLOW_LIGHT)

        self.play(GrowArrow(a3), run_time=0.45)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in sfeat],
                               lag_ratio=0.12, run_time=0.55))
        self.wait(0.60)

        # ══════════════════════════════════════════════════════════════════
        # ACT 2 — In-column morph: activated dots → Z nodes; sfeat → X nodes
        # ══════════════════════════════════════════════════════════════════
        Z_nodes = [
            cand_node(np.array([X2, yy, 0.0]), BLUE_LIGHT, lbl, r=Z_R, sc=0.35)
            for yy, lbl in zip(Z_ys, Z_lbls)
        ]
        X_nodes = [
            cand_node(np.array([X2, yy, 0.0]), YELLOW_LIGHT, lbl, r=X_R, sc=0.32)
            for yy, lbl in zip(X_ys, X_lbls)
        ]

        # Pre-build faint arrows from every node to Y (revealed in ACT 3)
        all_nodes = Z_nodes + X_nodes
        all_ZtoY = [
            Arrow(
                n.get_center() + _unit(n.get_center(), Y_pos) * (Z_R + 0.05),
                Y_pos           - _unit(n.get_center(), Y_pos) * (TY_R + 0.05),
                buff=0, color=DIM_GRAY, stroke_width=1.0,
                max_tip_length_to_length_ratio=0.08,
            ).set_opacity(0.25)
            for n in all_nodes
        ]

        self.play(
            *[ReplacementTransform(sparse[act_idx[i]], Z_nodes[i]) for i in range(N_ACTIVE)],
            *[FadeOut(sparse[j]) for j in inact_idx],
            *[ReplacementTransform(sfeat[i], X_nodes[i]) for i in range(N_X)],
            run_time=0.90,
        )
        self.wait(0.30)

        # ══════════════════════════════════════════════════════════════════
        # ACT 3 — Fade pipeline arrows (keep satellite & survey);
        #         T, Y (black), and light Z→Y arrows appear together
        # ══════════════════════════════════════════════════════════════════
        self.play(
            FadeOut(a1), FadeOut(lbl1), FadeOut(dense),
            FadeOut(a2), FadeOut(lbl2),
            FadeOut(a3),
            run_time=0.80,
        )
        self.wait(0.20)

        T_node = cand_node(T_pos, WHITE_TEXT, "T", r=TY_R, sc=0.48)
        Y_node = cand_node(Y_pos, WHITE_TEXT, "Y", r=TY_R, sc=0.48)
        TY_arr = Arrow(T_pos, Y_pos, buff=TY_R + 0.05,
                       color=WHITE_TEXT, stroke_width=3.0,
                       max_tip_length_to_length_ratio=0.055)

        self.play(
            GrowFromCenter(T_node),
            GrowFromCenter(Y_node),
            LaggedStart(*[FadeIn(a) for a in all_ZtoY], lag_ratio=0.03),
            run_time=0.90,
        )
        self.play(GrowArrow(TY_arr), run_time=0.50)
        self.wait(0.30)

        for _ in range(2):
            self.play(T_node.animate.scale(1.33), Y_node.animate.scale(1.33), run_time=0.33)
            self.play(T_node.animate.scale(1/1.33), Y_node.animate.scale(1/1.33), run_time=0.33)
        self.wait(0.35)

        # ══════════════════════════════════════════════════════════════════
        # ACT 4 — NEXIS: scan nodes, saturate to select (2 SAE + 1 survey)
        # ══════════════════════════════════════════════════════════════════
        self.play(
            *[n.animate.set_opacity(0.28) for n in Z_nodes + X_nodes],
            *[a.animate.set_opacity(0.12) for a in all_ZtoY],
            run_time=0.40,
        )

        # (target_src, target_idx, [scan_z_idxs])
        scan_rounds = [
            ("z", 1, [0, 3]),    # scan Z₁,Z₄  →  select Z₂  (SAE)
            ("z", 4, [2, 5]),    # scan Z₃,Z₆  →  select Z₅  (SAE)
            ("x", 1, [6, 7]),    # scan Z₇,Z₈  →  select Z₁₀ (survey)
        ]

        sel_z_idxs = []
        sel_x_idxs = []

        for src, target_idx, scan_z_idxs in scan_rounds:
            for s_idx in scan_z_idxs:
                if s_idx not in sel_z_idxs:
                    self.play(
                        Z_nodes[s_idx].animate.set_opacity(0.72),
                        all_ZtoY[s_idx].animate.set_opacity(0.50),
                        run_time=0.16,
                    )
                    self.play(
                        Z_nodes[s_idx].animate.set_opacity(0.28),
                        all_ZtoY[s_idx].animate.set_opacity(0.12),
                        run_time=0.14,
                    )

            if src == "z":
                self.play(
                    Z_nodes[target_idx].animate.set_opacity(1.0),
                    all_ZtoY[target_idx].animate.set_opacity(1.0).set_color(BLUE_LIGHT),
                    run_time=0.38,
                )
                sel_z_idxs.append(target_idx)
            else:
                x_ai = N_Z + target_idx
                self.play(
                    X_nodes[target_idx].animate.set_opacity(1.0),
                    all_ZtoY[x_ai].animate.set_opacity(1.0).set_color(YELLOW_LIGHT),
                    run_time=0.38,
                )
                sel_x_idxs.append(target_idx)
            self.wait(0.32)

        # Backward check: pulse selected nodes
        for idx in sorted(sel_z_idxs):
            self.play(Z_nodes[idx].animate.scale(1.16), run_time=0.20)
            self.play(Z_nodes[idx].animate.scale(1/1.16), run_time=0.20)
        for idx in sorted(sel_x_idxs):
            self.play(X_nodes[idx].animate.scale(1.16), run_time=0.20)
            self.play(X_nodes[idx].animate.scale(1/1.16), run_time=0.20)
        self.wait(0.20)

        # ══════════════════════════════════════════════════════════════════
        # ACT 5 — Fade non-selected; pulse winners
        # ══════════════════════════════════════════════════════════════════
        sel_nodes    = ([Z_nodes[i] for i in sorted(SEL_Z)]
                        + [X_nodes[i] for i in sorted(SEL_X)])
        non_sel_Z    = [Z_nodes[i] for i in range(N_Z) if i not in SEL_Z]
        non_sel_X    = [X_nodes[i] for i in range(N_X) if i not in SEL_X]
        non_sel_arrs = ([all_ZtoY[i] for i in range(N_Z) if i not in SEL_Z]
                        + [all_ZtoY[N_Z + i] for i in range(N_X) if i not in SEL_X])

        self.play(
            *[n.animate.set_opacity(0.07) for n in non_sel_Z + non_sel_X],
            *[a.animate.set_opacity(0.03) for a in non_sel_arrs],
            run_time=0.65,
        )
        self.wait(0.25)

        for _ in range(2):
            self.play(*[w.animate.scale(1.22) for w in sel_nodes], run_time=0.28)
            self.play(*[w.animate.scale(1/1.22) for w in sel_nodes], run_time=0.28)
        self.wait(0.35)

        # ══════════════════════════════════════════════════════════════════
        # ACT 6 — VLM interpretation pills + tagline
        # ══════════════════════════════════════════════════════════════════
        p_tree   = pill("tree cover",   TEAL_LIGHT)
        p_road   = pill("road access",  TEAL_LIGHT)
        p_assets = pill("asset wealth", TEAL_LIGHT)

        p_tree.next_to(sel_nodes[0],  LEFT, buff=0.18)
        p_road.next_to(sel_nodes[1],  LEFT, buff=0.18)
        p_assets.next_to(sel_nodes[2], LEFT, buff=0.18)

        self.play(LaggedStart(
            FadeIn(p_tree,   shift=LEFT * 0.08),
            FadeIn(p_road,   shift=LEFT * 0.08),
            FadeIn(p_assets, shift=LEFT * 0.08),
            lag_ratio=0.32, run_time=1.70,
        ))
        self.wait(1.0)

        tagline = VGroup(
            Text("interpretable", color=TEAL_LIGHT  ).scale(0.40),
            Text("·",             color=GRAY_TEXT   ).scale(0.40),
            Text("causal",        color=YELLOW_LIGHT).scale(0.40),
            Text("·",             color=GRAY_TEXT   ).scale(0.40),
            Text("sparse",        color=BLUE_LIGHT  ).scale(0.40),
        ).arrange(RIGHT, buff=0.20)
        tagline.to_edge(DOWN, buff=0.42)

        self.play(FadeIn(tagline, shift=UP * 0.10), run_time=0.90)
        self.wait(4.0)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.5)
