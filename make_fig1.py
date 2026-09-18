"""
Regenerate the Figure 1 pipeline schematic for the breast-cancer revision.
Fixes the SEED equation: ln n0 = a0 + a1*Xsize + a2*Xnodes  ->  ln(1+Xnodes).
Vector output (PDF) for direct LaTeX inclusion, plus a PNG preview.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib.lines import Line2D
from pathlib import Path
import numpy as np

BLUE = "#1f5fa8"
BLUE_FILL = "#eaf1fb"
RED = "#b8272c"
RED_FILL = "#fbeaea"
GREEN = "#1a7a3c"
GREEN_FILL = "#eaf7ee"
PURPLE = "#5a2d91"
PURPLE_FILL = "#f5f0fb"
ORANGE = "#d9622b"
ORANGE_FILL = "#fdf1ea"
GREY = "#6b6b6b"
GREY_FILL = "#f2f2f2"
LOWRISK = "#2166ac"
HIGHRISK = "#c0392b"

fig = plt.figure(figsize=(16, 9), dpi=200)
fig.patch.set_facecolor("white")
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, 1)
ax.set_ylim(0, 1)
ax.axis("off")


def rbox(x, y, w, h, fc, ec, lw=1.6, rad=0.012, zorder=2):
    p = FancyBboxPatch((x, y), w, h,
                        boxstyle=f"round,pad=0,rounding_size={rad}",
                        linewidth=lw, edgecolor=ec, facecolor=fc,
                        zorder=zorder, mutation_aspect=1)
    ax.add_patch(p)
    return p


def arrow(x0, y0, x1, y1, color, lw=1.8, zorder=1):
    a = FancyArrowPatch((x0, y0), (x1, y1),
                         arrowstyle="-|>", mutation_scale=13,
                         linewidth=lw, color=color, zorder=zorder,
                         shrinkA=0, shrinkB=2)
    ax.add_patch(a)


# ---------------------------------------------------------------- column 1
col1_x, col1_w = 0.015, 0.175
obs = [
    ("Tumour size", BLUE),
    ("Lymph node count", BLUE),
    ("Histological grade", RED),
    ("ER status", RED),
    ("HER2 status", RED),
    ("PR status", RED),
]
n_obs = len(obs)
top, bottom = 0.955, 0.32
gap = 0.018
h_obs = (top - bottom - gap * (n_obs - 1)) / n_obs
obs_centers = []
for i, (label, color) in enumerate(obs):
    y = top - i * (h_obs + gap) - h_obs
    fc = BLUE_FILL if color == BLUE else RED_FILL
    rbox(col1_x, y, col1_w, h_obs, fc, color)
    ax.text(col1_x + 0.02, y + h_obs / 2, label, fontsize=11.5,
            va="center", ha="left", color="black", weight="bold")
    obs_centers.append((col1_x + col1_w, y + h_obs / 2, color))

ax.text(col1_x, bottom - 0.045,
        "measured once at diagnosis,\nunique to each patient.",
        fontsize=9.5, va="top", ha="left", color="#333333", style="italic")

# ---------------------------------------------------------------- column 2
col2_x, col2_w = 0.225, 0.175

# SOIL (future work, dashed)
soil_h = 0.145
soil_y = 0.80
p = FancyBboxPatch((col2_x, soil_y), col2_w, soil_h,
                    boxstyle="round,pad=0,rounding_size=0.012",
                    linewidth=1.6, edgecolor=GREY, facecolor="#f7f7f7",
                    linestyle=(0, (4, 3)), zorder=2)
ax.add_patch(p)
ax.text(col2_x + col2_w / 2, soil_y + soil_h - 0.032, "SOIL — future work",
        fontsize=12.5, ha="center", va="top", color=GREY, style="italic", weight="bold")
ax.text(col2_x + col2_w / 2, soil_y + soil_h / 2 - 0.02,
        "TILs · PD-L1 ·\nstromal markers",
        fontsize=10.5, ha="center", va="center", color=GREY)

# SEED
seed_y, seed_h = 0.535, 0.225
rbox(col2_x, seed_y, col2_w, seed_h, BLUE_FILL, BLUE, lw=2.0)
ax.text(col2_x + col2_w / 2, seed_y + seed_h - 0.032, "SEED", fontsize=17,
        ha="center", va="top", color=BLUE, weight="bold")
ax.text(col2_x + col2_w / 2, seed_y + seed_h - 0.062,
        "encodes initial\ntumour burden $n_0$",
        fontsize=10.5, ha="center", va="top", color="black")
ax.plot([col2_x + 0.012, col2_x + col2_w - 0.012],
        [seed_y + seed_h - 0.115] * 2, color=BLUE, lw=0.9)
ax.text(col2_x + col2_w / 2, seed_y + 0.05,
        r"$\ln n_0=\alpha_0+\alpha_1\cdot X_{size}$"
        "\n" r"$+\,\alpha_2\cdot\ln(1+X_{nodes})$",
        fontsize=11.5, ha="center", va="center", color="black")

# SPEED
speed_y, speed_h = 0.285, 0.235
rbox(col2_x, speed_y, col2_w, speed_h, RED_FILL, RED, lw=2.0)
ax.text(col2_x + col2_w / 2, speed_y + speed_h - 0.028, "SPEED", fontsize=17,
        ha="center", va="top", color=RED, weight="bold")
ax.text(col2_x + col2_w / 2, speed_y + speed_h - 0.058,
        "encodes covariate-adjusted\nrate $\\lambda_i$",
        fontsize=10.5, ha="center", va="top", color="black")
ax.plot([col2_x + 0.012, col2_x + col2_w - 0.012],
        [speed_y + speed_h - 0.108] * 2, color=RED, lw=0.9)
ax.text(col2_x + col2_w / 2, speed_y + 0.085,
        r"$\lambda=\beta_0+\beta_1\cdot X_{grade}$"
        "\n" r"$+\,\beta_2\cdot X_{ER}+\beta_3\cdot X_{HER2}$"
        "\n" r"$+\,\beta_4\cdot X_{PR}$",
        fontsize=10.3, ha="center", va="center", color="black")
ax.text(col2_x + col2_w / 2, speed_y + 0.016,
        r"reference anchor: $\lambda_{\rm ref}=0.1140$ month$^{-1}$"
        "\n" r"(185-day doubling time; $\beta_0$ implied)",
        fontsize=6.8, ha="center", va="bottom", color="#333333")

# arrows: observables -> SEED / SPEED
for i, (cx, cy, color) in enumerate(obs_centers):
    target_y = (seed_y + seed_h / 2) if color == BLUE else (speed_y + speed_h / 2)
    arrow(cx, cy, col2_x, target_y, color, lw=1.6)

# ---------------------------------------------------------------- column 3
col3_x, col3_w = 0.425, 0.075

n0_y, n0_h = 0.595, 0.14
rbox(col3_x, n0_y, col3_w, n0_h, GREEN_FILL, GREEN, lw=1.8)
ax.text(col3_x + col3_w / 2, n0_y + n0_h - 0.03, "$n_0$", fontsize=18,
        ha="center", va="top", color=GREEN, weight="bold")
ax.text(col3_x + col3_w / 2, n0_y + 0.04, "initial DTC\nburden",
        fontsize=9.5, ha="center", va="center", color="black")

lam_y, lam_h = 0.375, 0.14
rbox(col3_x, lam_y, col3_w, lam_h, GREEN_FILL, GREEN, lw=1.8)
ax.text(col3_x + col3_w / 2, lam_y + lam_h - 0.03, r"$\lambda$", fontsize=18,
        ha="center", va="top", color=GREEN, weight="bold")
ax.text(col3_x + col3_w / 2, lam_y + 0.04, "proliferative\nrate",
        fontsize=9.5, ha="center", va="center", color="black")

arrow(col2_x + col2_w, seed_y + seed_h / 2, col3_x, n0_y + n0_h / 2, BLUE, lw=1.8)
arrow(col2_x + col2_w, speed_y + speed_h / 2, col3_x, lam_y + lam_h / 2, RED, lw=1.8)

# ---------------------------------------------------------------- column 4
col4_x, col4_w = 0.525, 0.245
box4_bottom, box4_h = 0.255, 0.695
rbox(col4_x, box4_bottom, col4_w, box4_h, PURPLE_FILL, PURPLE, lw=2.0)
ax.text(col4_x + col4_w / 2, 0.92, "Exponential ODE", fontsize=15.5,
        ha="center", va="top", color=PURPLE, weight="bold")
ax.text(col4_x + col4_w / 2, 0.865, r"$\dfrac{dN}{dt}=\lambda N(t)$",
        fontsize=13.5, ha="center", va="center", color="black")
ax.text(col4_x + col4_w / 2, 0.815,
        r"$\Rightarrow\ N(t)=n_0e^{\lambda t},\ \ N(0)=n_0$",
        fontsize=11.5, ha="center", va="center", color="black")
ax.plot([col4_x + 0.02, col4_x + col4_w - 0.02], [0.785, 0.785],
        color=PURPLE, lw=0.9)

arrow(col3_x + col3_w, n0_y + n0_h / 2, col4_x, 0.60, GREEN, lw=1.8)
arrow(col3_x + col3_w, lam_y + lam_h / 2, col4_x, 0.60, GREEN, lw=1.8)

# inset growth curve (kept well clear of the box's bottom edge / caption)
gax = fig.add_axes([col4_x + 0.032, 0.415, col4_w - 0.064, 0.335])
t = np.linspace(0, 10, 200)
n0v, lam = 1.0, 0.7
Nt = n0v * np.exp(lam * t)
M = 22
Tpred = np.log(M / n0v) / lam
gax.plot(t, Nt, color=PURPLE, lw=2.2, zorder=3)
gax.axhline(M, color="black", lw=1, ls=(0, (5, 4)))
gax.plot([Tpred], [M], "o", color=PURPLE, ms=7, zorder=4)
gax.plot([0], [n0v], "o", color=PURPLE, ms=7, zorder=4)
gax.set_xlim(-0.3, 10.8)
gax.set_ylim(-0.5, M * 1.22)
gax.text(10.6, M + 0.8, "$M$  clinical detection\nthreshold\n($M\\approx10^9$ cells)",
          fontsize=8.1, ha="right", va="bottom", color="black")
gax.annotate("$n_0$", xy=(0, n0v), xytext=(9, -11), textcoords="offset points",
             fontsize=10.5, ha="left", va="top", color=PURPLE, weight="bold")
gax.annotate("$T_{\\rm pred}$", xy=(Tpred, 0), xytext=(0, -20), textcoords="offset points",
             fontsize=10.5, ha="center", va="top", color=PURPLE, weight="bold")
gax.axvline(Tpred, ymin=0, ymax=(M / (M * 1.22 + 0.5)), color=PURPLE, lw=0.9, ls=(0, (3, 3)))
gax.set_xticks([]); gax.set_yticks([])
gax.annotate("$t$ (time)", xy=(1, 0), xycoords="axes fraction",
             xytext=(0, -20), textcoords="offset points",
             fontsize=9.3, ha="right", va="top", color="black")
gax.set_ylabel("$N(t)$ (tumour cell count)", fontsize=8.8, labelpad=2)
for s in gax.spines.values():
    s.set_visible(False)
gax.annotate("", xy=(10.7, 0), xytext=(-0.3, 0),
             arrowprops=dict(arrowstyle="-|>", lw=1, color="black"))
gax.annotate("", xy=(0, M * 1.2), xytext=(0, -0.5),
             arrowprops=dict(arrowstyle="-|>", lw=1, color="black"))

ax.text(col4_x + col4_w / 2, box4_bottom - 0.02,
        r"$T_{\rm pred}$ is the time at which this patient's tumour reaches $M$ —"
        "\nit differs across patients through $n_0$ and covariate-adjusted $\\lambda_i$; $\\lambda_{\\rm ref}$ is fixed.",
        fontsize=9, ha="center", va="top", color="#333333", style="italic")

# ---------------------------------------------------------------- column 5
col5_x, col5_w = 0.79, 0.195
box5_bottom, box5_h = 0.255, 0.695
rbox(col5_x, box5_bottom, col5_w, box5_h, ORANGE_FILL, ORANGE, lw=2.0)
ax.text(col5_x + col5_w / 2, 0.92, r"$T_{\rm pred}$", fontsize=19,
        ha="center", va="top", color=ORANGE, weight="bold", style="italic")
ax.text(col5_x + col5_w / 2, 0.858,
        r"$T_{\rm pred}=\dfrac{\ln M-\ln n_0}{\lambda}$",
        fontsize=13, ha="center", va="center", color="black")
ax.text(col5_x + col5_w / 2, 0.808,
        "each patient's predicted relapse\ntime depends on their own $n_0$ and $\\lambda$.",
        fontsize=8.6, ha="center", va="top", color="#333333")
ax.text(col5_x + col5_w / 2, 0.752,
        "$M\\approx10^9$ cells (absorbed into\n$\\alpha_0$, constant across patients)",
        fontsize=8.6, ha="center", va="top", color="#333333")
ax.plot([col5_x + 0.02, col5_x + col5_w - 0.02], [0.715, 0.715],
        color=ORANGE, lw=0.9)
ax.text(col5_x + col5_w / 2, 0.700, "stratified by each patient's $T_{\\rm pred}$",
        fontsize=8.6, ha="center", va="top", color="#333333")

arrow(col4_x + col4_w, 0.60, col5_x, 0.60, PURPLE, lw=1.8)

# inset KM-style risk curves
kax = fig.add_axes([col5_x + 0.026, 0.415, col5_w - 0.052, 0.255])


def km(rng, hazard_scale):
    rs = np.random.RandomState(rng)
    tt = np.sort(rs.exponential(hazard_scale, 40))
    surv = 1 - np.arange(1, 41) / 42
    return np.concatenate([[0], tt]), np.concatenate([[1], surv])


t_lo, s_lo = km(3, 90)
t_hi, s_hi = km(7, 35)
kax.step(t_hi, s_hi, where="post", color=HIGHRISK, lw=2)
kax.step(t_lo, s_lo, where="post", color=LOWRISK, lw=2)
kax.axvline(45, color="black", lw=0.9, ls=(0, (4, 3)))
kax.set_xlim(0, 200)
kax.set_ylim(0, 1.05)
kax.set_xlabel("Time (months)", fontsize=8.8, labelpad=3)
kax.set_ylabel("Survival probability $S(t)$", fontsize=8.2, labelpad=2)
kax.tick_params(labelsize=7.5)
kax.text(197, 1.0, "Low Risk —", fontsize=7.8, color=LOWRISK, ha="right", va="top", weight="bold")
kax.text(197, 0.925, "$T_{\\rm pred}$ above median\n(slow-growing tumour,\nlong predicted relapse time)",
         fontsize=6.7, color=LOWRISK, ha="right", va="top")
kax.text(197, 0.60, "High Risk —", fontsize=7.8, color=HIGHRISK, ha="right", va="top", weight="bold")
kax.text(197, 0.525, "$T_{\\rm pred}$ below median\n(fast-growing tumour,\nshort predicted relapse time)",
         fontsize=6.7, color=HIGHRISK, ha="right", va="top")
for s in ["top", "right"]:
    kax.spines[s].set_visible(False)

# ---------------------------------------------------------------- bottom labels
labels = [
    (col1_x, col1_w, "1. Clinical Observables\n(Patient-Specific)"),
    (col2_x, col2_w, "2. Mechanistic\nBuckets"),
    (col3_x, col3_w, "3. Kinetic\nParameters"),
    (col4_x, col4_w, "4. Biophysical\nModel"),
    (col5_x, col5_w, "5. Patient Survival\nPrediction"),
]
lab_y, lab_h = 0.015, 0.075
for x, w, text in labels:
    rbox(x, lab_y, w, lab_h, GREY_FILL, "#bdbdbd", lw=1.2, rad=0.01)
    ax.text(x + w / 2, lab_y + lab_h / 2, text, fontsize=10, ha="center", va="center",
            color="#333333", weight="bold")

output_dir = Path(__file__).parent
fig.savefig(output_dir / "fig1_pipeline_revision.pdf", facecolor="white")
fig.savefig(output_dir / "fig1_pipeline_revision.png", facecolor="white", dpi=200)
print("done")
