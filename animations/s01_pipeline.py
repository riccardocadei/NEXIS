# Render (preview):  conda run -n manim manim -pql s01_pipeline.py Pipeline
# Render (Twitter):  conda run -n manim manim -qh  s01_pipeline.py Pipeline
# Duration: ~70 s

from manim import *
import numpy as np
from pathlib import Path
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from config import (
    BG, WHITE_TEXT, GRAY_TEXT, DIM_GRAY,
    BLUE_LIGHT, GREEN_LIGHT, YELLOW_LIGHT, PURPLE_LIGHT, TEAL_LIGHT,
    SMALL_SCALE, LABEL_SCALE,
    make_box, make_arrow,
)

ASSETS  = Path(__file__).parent / "assets"
TILE_PNG = ASSETS / "sample_tile.png"

DATA_ROOT = Path("/nfs/scistore19/locatgrp/rcadei/NEXIS/data/ghana/satellite/tif")
TILE_TIF  = DATA_ROOT / "ghana_comm0014.tif"


# ─── Satellite tile PNG (cached) ─────────────────────────────────────────────

def _make_tile_png() -> bool:
    """Convert one HLS GeoTIFF (6 bands, float64) to an RGB PNG for Manim."""
    if TILE_PNG.exists():
        return True
    ASSETS.mkdir(exist_ok=True)
    if not TILE_TIF.exists():
        return False
    try:
        import rasterio
        from PIL import Image as PILImage
        with rasterio.open(TILE_TIF) as src:
            data = src.read()                        # (6, H, W) float64
        # Natural colour: Red(3), Green(2), Blue(1)  →  0-indexed bands 2,1,0
        rgb = np.transpose(data[[2, 1, 0]], (1, 2, 0)).astype(float)
        for b in range(3):
            p2, p98 = np.percentile(rgb[:, :, b], [2, 98])
            rgb[:, :, b] = np.clip((rgb[:, :, b] - p2) / max(p98 - p2, 1e-9), 0, 1)
        out = PILImage.fromarray((rgb * 255).astype(np.uint8))
        out = out.resize((256, 256), PILImage.LANCZOS)
        out.save(str(TILE_PNG))
        return True
    except Exception as exc:
        print(f"[tile PNG prep failed: {exc}]")
        return False


# ─── Scene helpers ────────────────────────────────────────────────────────────

def dot_col(n, color, bright=None, dim=DIM_GRAY, h=1.55, r=0.055):
    grp  = VGroup()
    step = h / max(n - 1, 1)
    for i in range(n):
        c = color if (bright is None or i in bright) else dim
        d = Dot(radius=r, color=c, fill_opacity=1.0)
        d.move_to(UP * (h / 2 - i * step))
        grp.add(d)
    return grp


def proc_tile(sz=1.8, n=7, seed=42):
    """Earth-tone procedural fallback if TIFF unavailable."""
    pal = ["#3D6B4A","#6B9B5A","#C4A84E","#8A7250",
           "#4F8060","#92B05C","#BDB06A","#5E8A6A",
           "#2D5A3A","#A0AA55","#7E9060","#4A7030"]
    p   = sz / n
    rng = np.random.default_rng(seed)
    patches = VGroup(*[
        Square(side_length=p * 0.95,
               fill_color=pal[int(rng.integers(len(pal)))],
               fill_opacity=1.0, stroke_width=0)
        .move_to(RIGHT * (col - n/2 + .5) * p + UP * (n/2 - .5 - row) * p)
        for row in range(n) for col in range(n)
    ])
    border = Square(side_length=sz, color=BLUE_LIGHT,
                    stroke_width=2.5, fill_opacity=0)
    return VGroup(patches, border)


def survey_card(rows=5, w=1.5, rh=0.26, seed=7):
    fields = ["age", "education", "assets", "gender", "region"]
    rng    = np.random.default_rng(seed)
    card   = VGroup()
    for i, f in enumerate(fields[:rows]):
        y  = (rows / 2 - .5 - i) * (rh + 0.04)
        bg = Rectangle(width=w, height=rh, fill_color="#0D1A0D", fill_opacity=1.0,
                       stroke_color=GREEN_LIGHT, stroke_width=0.8).move_to(UP * y)
        ft = Text(f, color=GRAY_TEXT).scale(0.23)
        ft.next_to(bg.get_left(), RIGHT, buff=0.10)
        bw  = float(rng.uniform(0.3, 0.85)) * (w * 0.48)
        bar = Rectangle(width=bw, height=rh * 0.40, fill_color=GREEN_LIGHT,
                        fill_opacity=0.75, stroke_width=0)
        bar.move_to(bg.get_right() + LEFT * (0.10 + bw / 2))
        card.add(bg, ft, bar)
    card.add(Rectangle(width=w + 0.06, height=rows * (rh + 0.04) + 0.04,
                       color=GREEN_LIGHT, stroke_width=2.0, fill_opacity=0))
    return card


def dag_node(pos, color, tex, radius=0.27, tex_sc=0.34):
    circ = Circle(radius=radius, color=color, fill_color=color,
                  fill_opacity=0.18, stroke_width=2.0)
    circ.move_to(pos)
    lbl = Text(tex, color=color).scale(tex_sc).move_to(circ)
    return VGroup(circ, lbl), circ


# ─── Main scene ───────────────────────────────────────────────────────────────

class Pipeline(Scene):
    """
    Pipeline animation – no title slide.
    Flow:  satellite  →  FM (frozen)  →  dense emb  →  SAE  →  sparse Z
                                                                      ↘
                                         survey card  →  survey feats  →  NEXIS  →  labels
    """

    def construct(self):                                          # noqa: C901
        self.camera.background_color = BG

        # ── Coordinate anchors ────────────────────────────────────────────
        X_IN    = -5.6   # input column (tile + card)
        X_FM    = -3.6   # foundation-model box
        X_DENSE = -2.05  # dense embedding column
        X_SAE   = -0.75  # SAE box
        X_SP    =  0.65  # sparse-Z column  (SAE output)
        Y_SAT   =  1.35  # satellite arm
        Y_SRV   = -1.50  # survey arm
        X_SF    =  X_SP  # survey-feature column (same x, different y)

        # NEXIS area
        NX, NY  = 3.15, 0.0

        N_DENSE   = 14
        N_SP      = N_DENSE
        BRIGHT_SP = {1, 4, 7, 10, 12}   # 5 active neurons after SAE
        N_SURV    = 5

        # ── Prepare satellite PNG once ────────────────────────────────────
        have_png = _make_tile_png()

        # ──────────────────────────────────────────────────────────────────
        # ACT 1 – Satellite branch
        # ──────────────────────────────────────────────────────────────────
        if have_png:
            tile = ImageMobject(str(TILE_PNG)).set_width(1.80)
            tile.move_to([X_IN, Y_SAT, 0])
            tile_obj = tile
        else:
            tile_obj = proc_tile().move_to([X_IN, Y_SAT, 0])

        border = Square(side_length=1.84, color=BLUE_LIGHT,
                        stroke_width=2.5, fill_opacity=0)
        border.move_to([X_IN, Y_SAT, 0])

        sat_lbl = Text("satellite imagery", color=BLUE_LIGHT).scale(LABEL_SCALE)
        sat_lbl.next_to(border, DOWN, buff=0.13)

        self.play(FadeIn(tile_obj), run_time=0.9)
        self.play(Create(border), FadeIn(sat_lbl), run_time=0.55)
        self.wait(0.3)

        # Foundation Model box  (❄ = frozen weights)
        fm_box = make_box(
            ["Foundation Model", "❄  (e.g., Prithvi-EO)"],
            width=2.2, height=0.82,
            box_col=BLUE_LIGHT, txt_scale=0.27,
        )
        fm_box.move_to([X_FM, Y_SAT, 0])
        a_tile_fm = make_arrow(border.get_right(), fm_box.get_left(), color=BLUE_LIGHT)

        self.play(GrowArrow(a_tile_fm), run_time=0.6)
        self.play(FadeIn(fm_box), run_time=0.65)
        self.wait(0.25)

        # Dense embedding
        dense = dot_col(N_DENSE, BLUE_LIGHT).move_to([X_DENSE, Y_SAT, 0])
        a_fm_dense = make_arrow(fm_box.get_right(), dense.get_left() + LEFT * 0.04,
                                color=BLUE_LIGHT)

        self.play(GrowArrow(a_fm_dense), run_time=0.45)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in dense],
                               lag_ratio=0.06, run_time=1.4))
        self.wait(0.3)

        # SAE  (satellite branch only – survey joins LATER)
        sae_box = make_box("SAE", width=0.95, height=0.70,
                           box_col=PURPLE_LIGHT, txt_scale=0.40)
        sae_box.move_to([X_SAE, Y_SAT, 0])
        a_dense_sae = make_arrow(dense.get_right(), sae_box.get_left(), color=PURPLE_LIGHT)

        self.play(GrowArrow(a_dense_sae), run_time=0.45)
        self.play(FadeIn(sae_box), run_time=0.55)
        self.wait(0.2)

        # Sparse-Z column
        sparse = dot_col(N_SP, PURPLE_LIGHT).move_to([X_SP, Y_SAT, 0])
        a_sae_sp = make_arrow(sae_box.get_right(), sparse.get_left() + LEFT * 0.04,
                              color=PURPLE_LIGHT)

        self.play(GrowArrow(a_sae_sp), run_time=0.45)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in sparse],
                               lag_ratio=0.04, run_time=0.75))
        self.wait(0.2)

        # Sparsify: dim inactive neurons, pulse survivors
        inactive_sp = [d for i, d in enumerate(sparse) if i not in BRIGHT_SP]
        active_sp   = [d for i, d in enumerate(sparse) if i in BRIGHT_SP]

        self.play(
            LaggedStart(
                *[d.animate.set_color(DIM_GRAY).set_fill(DIM_GRAY, opacity=0.18)
                  for d in inactive_sp],
                lag_ratio=0.04, run_time=1.5,
            ),
        )
        self.play(
            LaggedStart(
                *[Succession(d.animate.scale(1.5), d.animate.scale(1 / 1.5))
                  for d in active_sp],
                lag_ratio=0.14, run_time=0.95,
            ),
        )
        self.wait(0.6)

        # ──────────────────────────────────────────────────────────────────
        # ACT 2 – Survey branch  (appears after SAE)
        # ──────────────────────────────────────────────────────────────────
        card    = survey_card().move_to([X_IN, Y_SRV, 0])
        srv_lbl = Text("survey data", color=GREEN_LIGHT).scale(LABEL_SCALE)
        srv_lbl.next_to(card, DOWN, buff=0.13)

        self.play(FadeIn(card), run_time=0.75)
        self.play(FadeIn(srv_lbl), run_time=0.40)
        self.wait(0.3)

        surv_feat = dot_col(N_SURV, GREEN_LIGHT, h=0.65).move_to([X_SF, Y_SRV, 0])
        a_card_sf = make_arrow(card.get_right(),
                               surv_feat.get_left() + LEFT * 0.04,
                               color=GREEN_LIGHT)
        sf_lbl = Text("survey features", color=GRAY_TEXT).scale(0.24)
        sf_lbl.next_to(surv_feat, DOWN, buff=0.10)

        self.play(GrowArrow(a_card_sf), run_time=0.65)
        self.play(LaggedStart(*[GrowFromCenter(d) for d in surv_feat],
                               lag_ratio=0.12, run_time=0.65))
        self.play(FadeIn(sf_lbl), run_time=0.30)
        self.wait(0.6)

        # Dim upstream pipeline internals – keep SAE arrows and both columns prominent
        to_dim = [tile_obj, border, sat_lbl, a_tile_fm, fm_box,
                  a_fm_dense, dense, a_dense_sae, sae_box,
                  card, srv_lbl, a_card_sf, sf_lbl]
        self.play(
            *[m.animate.set_opacity(0.20) for m in to_dim],
            run_time=0.9,
        )

        # ──────────────────────────────────────────────────────────────────
        # ACT 3 – NEXIS causal model
        # ──────────────────────────────────────────────────────────────────

        # Convergence arrows from both input columns into the NEXIS area
        a_sp_nx = make_arrow(
            np.array([X_SP + 0.12, Y_SAT, 0.0]),
            np.array([NX - 1.65, NY + 0.35, 0.0]),
            color=PURPLE_LIGHT,
        )
        a_sf_nx = make_arrow(
            np.array([X_SF + 0.12, Y_SRV, 0.0]),
            np.array([NX - 1.65, NY - 0.35, 0.0]),
            color=GREEN_LIGHT,
        )
        self.play(GrowArrow(a_sp_nx), GrowArrow(a_sf_nx), run_time=0.85)
        self.wait(0.3)

        # T and Y nodes
        T_pos = np.array([NX - 1.05, NY + 0.60, 0.0])
        Y_pos = np.array([NX + 1.05, NY + 0.60, 0.0])
        mid_TY = (T_pos + Y_pos) / 2

        T_grp, T_circ = dag_node(T_pos, BLUE_LIGHT,  "T")
        Y_grp, Y_circ = dag_node(Y_pos, GREEN_LIGHT, "Y")
        TY_arr = make_arrow(
            T_pos + RIGHT * 0.28, Y_pos + LEFT * 0.28,
            color=WHITE_TEXT, stroke=2.5,
        )

        self.play(GrowFromCenter(T_grp), GrowFromCenter(Y_grp), run_time=0.75)
        self.play(GrowArrow(TY_arr), run_time=0.55)
        self.wait(0.25)

        # Candidate Z nodes (3 SAE neurons + 1 survey feature)
        z_y = NY - 0.52
        z_defs = [
            (NX - 1.05, z_y, PURPLE_LIGHT, "Z₁"),   # SAE neuron
            (NX - 0.35, z_y, PURPLE_LIGHT, "Z₂"),   # SAE neuron  (will be rejected)
            (NX + 0.35, z_y, PURPLE_LIGHT, "Z₃"),   # SAE neuron
            (NX + 1.05, z_y, GREEN_LIGHT,  "X₁"),   # survey feature
        ]
        z_grps  = []
        z_circs = []
        for xp, yp, col, tex in z_defs:
            grp, circ = dag_node(np.array([xp, yp, 0.0]), col, tex,
                                 radius=0.22, tex_sc=0.28)
            z_grps.append(grp)
            z_circs.append(circ)

        self.play(
            LaggedStart(*[GrowFromCenter(g) for g in z_grps],
                        lag_ratio=0.18, run_time=0.95),
        )

        # Interaction arrows: each Z_i → midpoint of T→Y  (moderation)
        interact_arrs = []
        for (xp, yp, col, _) in z_defs:
            arr = Arrow(
                np.array([xp, yp + 0.23, 0.0]),
                mid_TY + DOWN * 0.06,
                buff=0.06, color=col,
                stroke_width=1.1,
                max_tip_length_to_length_ratio=0.12,
            )
            interact_arrs.append(arr)

        self.play(
            LaggedStart(*[Create(a) for a in interact_arrs],
                        lag_ratio=0.16, run_time=0.9),
        )
        self.wait(0.3)

        # T and Y pulsate  →  "these are the quantities of interest"
        for _ in range(2):
            self.play(
                T_circ.animate.scale(1.38),
                Y_circ.animate.scale(1.38),
                run_time=0.40,
            )
            self.play(
                T_circ.animate.scale(1 / 1.38),
                Y_circ.animate.scale(1 / 1.38),
                run_time=0.40,
            )
        self.wait(0.35)

        # Selection: Z_2 (index 1) is rejected
        REJECT = 1
        self.play(
            z_grps[REJECT].animate.set_opacity(0.16),
            interact_arrs[REJECT].animate.set_opacity(0.10),
            run_time=0.75,
        )
        self.wait(0.25)

        # NEXIS box draws around {T, Y, Z_1, Z_2(dim), Z_3, X_1} + T→Y arrow
        nexis_content = VGroup(T_grp, Y_grp, TY_arr, *z_grps, *interact_arrs)
        nexis_border  = SurroundingRectangle(
            nexis_content, buff=0.30,
            color=YELLOW_LIGHT, stroke_width=2.0, corner_radius=0.15,
        )
        nexis_lbl = Text("NEXIS", color=YELLOW_LIGHT).scale(0.36)
        nexis_lbl.next_to(nexis_border, UP, buff=0.09)

        self.play(Create(nexis_border), run_time=0.85)
        self.play(Write(nexis_lbl), run_time=0.55)
        self.wait(0.8)

        # ──────────────────────────────────────────────────────────────────
        # ACT 4 – Expand selected neurons: VLM interpretation + survey label
        # ──────────────────────────────────────────────────────────────────

        # VLM annotation (appears before labels for SAE neurons)
        vlm_ann = Text("VLM interpretation", color=TEAL_LIGHT).scale(0.26)
        vlm_ann.move_to([5.35, 1.10, 0])
        self.play(FadeIn(vlm_ann, shift=DOWN * 0.05), run_time=0.50)

        # Label pills for selected: Z_1, Z_3 (SAE, teal) and X_1 (survey, green)
        label_defs = [
            (z_grps[0], "tree cover",   TEAL_LIGHT),
            (z_grps[2], "road access",  TEAL_LIGHT),
            (z_grps[3], "asset wealth", GREEN_LIGHT),
        ]

        X_OUT = 5.45
        pills  = VGroup()
        p_arrs = VGroup()

        for z_grp, txt, col in label_defs:
            src_y = z_grp.get_center()[1]
            pill_txt = Text(txt, color=col).scale(0.29)
            pill_bg  = RoundedRectangle(
                width=pill_txt.width + 0.22, height=pill_txt.height + 0.14,
                corner_radius=0.08,
                fill_color=col, fill_opacity=0.12,
                stroke_color=col, stroke_width=1.0,
            )
            pill = VGroup(pill_bg, pill_txt.move_to(pill_bg))
            pill.move_to([X_OUT + pill_bg.width / 2 - 0.1, src_y, 0])

            arr = make_arrow(
                z_grp.get_right() + RIGHT * 0.05,
                pill.get_left() + LEFT * 0.04,
                color=col,
            )
            pills.add(pill)
            p_arrs.add(arr)

        self.play(
            LaggedStart(
                *[AnimationGroup(GrowArrow(a), FadeIn(p, shift=RIGHT * 0.06))
                  for a, p in zip(p_arrs, pills)],
                lag_ratio=0.30, run_time=2.6,
            ),
        )
        self.wait(1.0)

        # ── Finale ────────────────────────────────────────────────────────
        tagline = VGroup(
            Text("interpretable", color=TEAL_LIGHT  ).scale(0.40),
            Text("·",             color=GRAY_TEXT   ).scale(0.40),
            Text("causal",        color=YELLOW_LIGHT).scale(0.40),
            Text("·",             color=GRAY_TEXT   ).scale(0.40),
            Text("sparse",        color=PURPLE_LIGHT).scale(0.40),
        ).arrange(RIGHT, buff=0.20)
        tagline.to_edge(DOWN, buff=0.42)

        self.play(FadeIn(tagline, shift=UP * 0.10), run_time=0.90)
        self.wait(3.5)
        self.play(FadeOut(Group(*self.mobjects)), run_time=1.5)
