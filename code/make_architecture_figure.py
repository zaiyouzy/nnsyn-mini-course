"""Create a clean, manuscript-ready architecture figure from the verified plan."""
from pathlib import Path
from matplotlib import pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "mini_course" / "figures" / "fig07_architecture_plainconvunet.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

INK = "#142630"
MUTED = "#5d6c73"
TEAL = "#2d7d78"
TEAL_DARK = "#1e625f"
NAVY = "#102b3a"
LINE = "#cfdad6"

channels = [32, 64, 128, 256, 320, 320]
spatial = [
    "48×192×224", "48×96×112", "24×48×56",
    "12×24×28", "6×12×14", "6×6×7",
]

fig, ax = plt.subplots(figsize=(12, 5.4), facecolor="white")
ax.set_xlim(0, 20)
ax.set_ylim(0, 8.2)
ax.axis("off")

enc_x = [2.7, 4.2, 5.7, 7.2, 8.7]
enc_y = [5.8, 4.95, 4.1, 3.25, 2.4]
bottleneck = (10.0, 1.45)
dec_x = [11.3, 12.8, 14.3, 15.8, 17.3]
dec_y = [2.4, 3.25, 4.1, 4.95, 5.8]


def stage_box(x, y, channel, color):
    patch = FancyBboxPatch(
        (x - 0.58, y - 0.38), 1.16, 0.76,
        boxstyle="round,pad=0.02,rounding_size=0.08",
        facecolor=color, edgecolor=color, linewidth=1.2, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, str(channel), ha="center", va="center",
            color="white", fontsize=15, fontweight="bold", zorder=3)


def endpoint_box(x, y, title):
    patch = FancyBboxPatch(
        (x - 0.90, y - 0.49), 1.80, 0.98,
        boxstyle="round,pad=0.025,rounding_size=0.09",
        facecolor="white", edgecolor=TEAL, linewidth=1.5, zorder=2,
    )
    ax.add_patch(patch)
    ax.text(x, y, title, ha="center", va="center",
            color=INK, fontsize=11, fontweight="bold", zorder=3)


def arrow(a, b, shrink_a=31, shrink_b=31, color=LINE, lw=1.8):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle="-|>", mutation_scale=12,
        linewidth=lw, color=color, shrinkA=shrink_a, shrinkB=shrink_b,
        zorder=0,
    ))


endpoint_box(1.0, 5.8, "MRI input")
endpoint_box(19.0, 5.8, "CT output")
ax.text(1.0, 5.12, "1 × 48 × 192 × 224", ha="center", va="top",
        color=MUTED, fontsize=8.7)
ax.text(19.0, 5.12, "1 × 48 × 192 × 224", ha="center", va="top",
        color=MUTED, fontsize=8.7)

for i, (x, y) in enumerate(zip(enc_x, enc_y)):
    stage_box(x, y, channels[i], TEAL)
    ax.text(x, y - 0.58, spatial[i], ha="center", va="top",
            color=MUTED, fontsize=9.5)

for i, (x, y) in enumerate(zip(dec_x, dec_y)):
    source_i = 4 - i
    stage_box(x, y, channels[source_i], TEAL_DARK)
    ax.text(x, y - 0.58, spatial[source_i], ha="center", va="top",
            color=MUTED, fontsize=9.5)

stage_box(*bottleneck, channels[5], NAVY)
ax.text(bottleneck[0], bottleneck[1] - 0.58, spatial[5],
        ha="center", va="top", color=MUTED, fontsize=9.5)

points = [(1.0, 5.8)] + list(zip(enc_x, enc_y)) + [bottleneck] + list(zip(dec_x, dec_y)) + [(19.0, 5.8)]
for i, (a, b) in enumerate(zip(points[:-1], points[1:])):
    shrink_a = 40 if i == 0 else 31
    shrink_b = 40 if i == len(points) - 2 else 31
    arrow(a, b, shrink_a=shrink_a, shrink_b=shrink_b)

# Skip arrows stop before the decoder blocks. Blocks are drawn above all lines.
for i in range(5):
    target_i = 4 - i
    start = (enc_x[i] + 0.68, enc_y[i])
    end = (dec_x[target_i] - 0.85, dec_y[target_i])
    ax.add_patch(FancyArrowPatch(
        start, end, arrowstyle="-|>", mutation_scale=10,
        linewidth=1.25, linestyle=(0, (5, 4)), color=TEAL,
        alpha=0.68, shrinkA=0, shrinkB=0, zorder=0,
    ))

ax.text(5.7, 6.95, "ENCODER", ha="center", color=TEAL,
        fontsize=12, fontweight="bold")
ax.text(14.3, 6.95, "DECODER", ha="center", color=TEAL_DARK,
        fontsize=12, fontweight="bold")
ax.text(10.0, 7.75, "3D PlainConvUNet planned for the SynthRAD2025 abdomen data",
        ha="center", color=INK, fontsize=20, fontweight="bold")
ax.text(10.0, 7.28,
        "The encoder builds context; the decoder restores resolution; dashed arrows carry spatial detail.",
        ha="center", color=MUTED, fontsize=12)

# Plain reading key: no nested frame, so it remains clean at web and print sizes.
ax.text(
    10.0, 0.23,
    "Block number = feature channels     |     Label below = D × H × W voxels     |     Dashed arrow = skip connection",
    ha="center", va="center", color=MUTED, fontsize=9.8,
)

fig.subplots_adjust(left=0.015, right=0.985, bottom=0.05, top=0.98)
fig.savefig(OUT, dpi=220, facecolor="white")
plt.close(fig)
print(OUT)
