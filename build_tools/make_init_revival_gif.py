"""Animated GIF of EMeans on 2-D blobs, with and without k-means|| and greedy revival.

It shows the fits of `examples/plot_init_and_revival.py` step by step, and
writes `doc/_static/emeans_init_revival.gif`. Run it from the root of the
repository, with matplotlib and Pillow installed:

    python build_tools/make_init_revival_gif.py

To record the centers after each step, it replaces the kernel of
`sklearn_emeans._emeans` by one that runs the steps one at a time, which gives
the same centers on 2-D data, where the default revival is off.
"""

# SPDX-License-Identifier: BSD-3-Clause

import math
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Circle, FancyArrowPatch, Rectangle
from PIL import Image
from sklearn.cluster import KMeans
from sklearn.datasets import make_blobs
from sklearn.metrics import pairwise_distances_argmin_min

import sklearn_emeans._emeans as emeans_module
from sklearn_emeans import EMeans

OUT = sys.argv[1] if len(sys.argv) > 1 else "doc/_static/emeans_init_revival.gif"
N_CLUSTERS, N_EPOCHS, RANDOM_STATE = 64, 3, 9

BACKGROUND, PANEL, INK, MUTED = "#fcfcfb", "#f3f3f0", "#1f1f1f", "#5f5f5f"
# Colors that remain distinct with a color vision deficiency.
VARIANTS = [
    (
        "Default",
        "Default",
        dict(),
        "#D55E00",
        ["Random samples as initial centers.", "No revival on 2-D data."],
    ),
    (
        'init="k-means||"',
        "k-means||",
        dict(init="k-means||"),
        "#0072B2",
        ["Initial centers spread out.", "No revival on 2-D data."],
    ),
    (
        'revival="greedy"',
        "greedy",
        dict(revival="greedy"),
        "#009E73",
        ["Random samples as initial centers.", "Moves centers to lower the inertia."],
    ),
    (
        "Both",
        "Both",
        dict(init="k-means||", revival="greedy"),
        "#CC79A7",
        ["k-means|| initial centers,", "and greedy revival."],
    ),
]


def make_data():
    """The data of examples/plot_init_and_revival.py."""
    rng = np.random.RandomState(3)
    grid = np.array([(i, j) for i in range(8) for j in range(8)], dtype=float)
    blob_centers = grid + rng.uniform(-0.25, 0.25, size=grid.shape)
    blob_sizes = np.round(np.exp(rng.uniform(np.log(20), np.log(1_000), 64)))
    blob_stds = rng.uniform(0.07, 0.12, 64)
    return make_blobs(
        n_samples=blob_sizes.astype(int),
        centers=blob_centers,
        cluster_std=blob_stds,
        random_state=rng,
    )


def record_fit(estimator, X):
    """Fit `estimator` and return the centers after each step.

    Returns a list of (step, centers, kind), with kind "init", "step",
    "greedy" (after a greedy revival) or "final".
    """
    snapshots = []
    emeans_steps = emeans_module._emeans_steps
    greedy_call = emeans_module.GreedyRevival.__call__

    def one_step_at_a_time(
        X,
        rows,
        batch_size,
        centers,
        weight_ema,
        step_sizes,
        epoch_sums,
        epoch_counts,
        first_step,
        revival_period,
        revive,
        n_threads,
    ):
        # With a revival period, the kernel revives the clusters between its
        # steps: running them one at a time would change the result.
        assert revival_period == 0
        if first_step == 0:
            snapshots.append((0, centers.copy(), "init"))
        for i in range(len(step_sizes)):
            emeans_steps(
                X,
                rows[i * batch_size : (i + 1) * batch_size],
                batch_size,
                centers,
                weight_ema,
                step_sizes[i : i + 1],
                epoch_sums,
                epoch_counts,
                first_step + i,
                0,
                revive,
                n_threads,
            )
            snapshots.append((first_step + i + 1, centers.copy(), "step"))

    def recorded_greedy_call(self, centers, weight_ema):
        n_revived = greedy_call(self, centers, weight_ema)
        if n_revived:
            snapshots.append((snapshots[-1][0], centers.copy(), "greedy"))
        return n_revived

    emeans_module._emeans_steps = one_step_at_a_time
    emeans_module.GreedyRevival.__call__ = recorded_greedy_call
    try:
        estimator.fit(X)
    finally:
        emeans_module._emeans_steps = emeans_steps
        emeans_module.GreedyRevival.__call__ = greedy_call
    snapshots.append((snapshots[-1][0], estimator.cluster_centers_.copy(), "final"))
    return snapshots


X, y = make_data()
blob_means = np.array([X[y == i].mean(axis=0) for i in range(N_CLUSTERS)])
blob_radii = np.array([3 * X[y == i].std(axis=0).mean() for i in range(N_CLUSTERS)])
reference_inertia = KMeans(N_CLUSTERS, init=blob_means, n_init=1).fit(X).inertia_
kmeans_10_inits = (
    KMeans(N_CLUSTERS, n_init=10, random_state=0).fit(X).inertia_ / reference_inertia
)


def relative_inertia(centers):
    return (pairwise_distances_argmin_min(X, centers)[1] ** 2).sum() / reference_inertia


def empty_blobs(centers):
    """Blobs that are not the closest blob of any center."""
    closest_blobs = pairwise_distances_argmin_min(centers, blob_means)[0]
    return set(range(N_CLUSTERS)) - set(closest_blobs)


runs = []
for name, short_name, params, color, description in VARIANTS:
    estimator = EMeans(
        N_CLUSTERS, n_epochs=N_EPOCHS, random_state=RANDOM_STATE, **params
    )
    snapshots = record_fit(estimator, X)
    steps_per_epoch = math.ceil(len(X) / estimator.batch_size_)
    runs.append(
        dict(
            name=name,
            short_name=short_name,
            color=color,
            description=description,
            snapshots=snapshots,
            final_inertia=estimator.inertia_ / reference_inertia,
            inertias=np.array([relative_inertia(c) for _, c, _ in snapshots]),
            epochs=np.array([step / steps_per_epoch for step, _, _ in snapshots]),
        )
    )
    print(f"{name}: {estimator.inertia_ / reference_inertia:.3f} x the reference")
n_steps = runs[0]["snapshots"][-1][0]
assert all(run["snapshots"][-1][0] == n_steps for run in runs)

# More frames at the start, where the centers move the most.
frame_steps = np.unique(np.round(n_steps * np.linspace(0, 1, 110) ** 1.6).astype(int))

# -- Layout --------------------------------------------------------------------

plt.rcParams.update({"font.family": ["Noto Sans", "DejaVu Sans"], "text.color": INK})
WIDTH, HEIGHT = 12.0, 8.4
fig = plt.figure(figsize=(WIDTH, HEIGHT), dpi=100, facecolor=BACKGROUND)
fig.text(
    0.035,
    0.955,
    "e-means on 2-D data: initialization and revival matter",
    fontsize=22,
    fontweight="bold",
    va="center",
)
fig.text(
    0.035,
    0.915,
    f"{len(X):,} points in 64 blobs of 20 to 1,000 points, k = 64. "
    "Reference: one center per blob.",
    fontsize=12.5,
    color=MUTED,
    va="center",
)

column_width, gap, left = 0.215, 0.025, 0.035
panels = []
for j, run in enumerate(runs):
    x0 = left + j * (column_width + gap)
    fig.patches.append(
        Rectangle(
            (x0, 0.852),
            0.011,
            0.016,
            transform=fig.transFigure,
            facecolor=run["color"],
            edgecolor="none",
        )
    )
    fig.text(x0 + 0.018, 0.86, run["name"], fontsize=14, fontweight="bold", va="center")
    for i, line in enumerate(run["description"]):
        fig.text(x0, 0.825 - 0.026 * i, line, fontsize=10, color=MUTED, va="center")
    ax = fig.add_axes([x0, 0.46, column_width, column_width * WIDTH / HEIGHT])
    ax.set_facecolor(PANEL)
    ax.scatter(X[:, 0], X[:, 1], s=0.6, color="#9a9a96", alpha=0.6, linewidths=0)
    ax.set_xlim(X[:, 0].min() - 0.3, X[:, 0].max() + 0.3)
    ax.set_ylim(X[:, 1].min() - 0.3, X[:, 1].max() + 0.3)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    ax.add_patch(
        Rectangle((0, 0), 1, 1, transform=ax.transAxes, color=PANEL, zorder=-1)
    )
    rings = []
    for mean, radius in zip(blob_means, blob_radii):
        ring = Circle(
            mean,
            radius,
            fill=False,
            linestyle=(0, (2, 1.5)),
            linewidth=1.1,
            edgecolor=INK,
            visible=False,
        )
        ax.add_patch(ring)
        rings.append(ring)
    centers = ax.scatter(
        [], [], s=26, color=run["color"], edgecolors="white", linewidths=0.8, zorder=3
    )
    status = fig.text(x0, 0.43, "", fontsize=11.5, va="center")
    n_empty = fig.text(x0, 0.403, "", fontsize=11.5, va="center", color=MUTED)
    panels.append(
        dict(ax=ax, rings=rings, centers=centers, status=status, n_empty=n_empty)
    )

fig.text(
    0.035,
    0.365,
    "Inertia during the fit",
    fontsize=14,
    fontweight="bold",
    va="center",
)
fig.text(
    0.965,
    0.365,
    "dashed rings: blobs without a center",
    fontsize=10.5,
    color=MUTED,
    ha="right",
    va="center",
)
curve_ax = fig.add_axes([0.11, 0.115, 0.75, 0.22])
curve_ax.set_facecolor(BACKGROUND)
curve_ax.set_yscale("log")
curve_ax.set_ylim(0.93, 6)
curve_ax.set_xlim(0, N_EPOCHS)
curve_ax.set_yticks([1, 1.5, 2, 3, 5], ["1×", "1.5×", "2×", "3×", "5×"])
curve_ax.minorticks_off()
curve_ax.grid(axis="y", color="#e2e2de", linewidth=0.8)
for side in ["top", "right", "left"]:
    curve_ax.spines[side].set_visible(False)
curve_ax.spines["bottom"].set_color("#bdbdb8")
curve_ax.tick_params(colors=MUTED, labelsize=10, length=0)
curve_ax.set_xlabel("Epochs (passes over the data)", color=MUTED, fontsize=10.5)
curve_ax.set_ylabel("Inertia relative\nto the reference", color=MUTED, fontsize=10.5)
curve_ax.axhline(kmeans_10_inits, color=MUTED, linestyle=(0, (4, 3)), linewidth=1)
kmeans_label = curve_ax.text(
    N_EPOCHS + 0.06,
    kmeans_10_inits,
    "KMeans, 10 inits",
    color=MUTED,
    fontsize=9.5,
    va="center",
)
curves = []
for run in runs:
    # Until their first revival, the fits with greedy revival follow the same
    # path as the ones without: draw the latter on top.
    (line,) = curve_ax.plot(
        [],
        [],
        color=run["color"],
        linewidth=2,
        zorder=3 if run["short_name"] in ("greedy", "Both") else 4,
        solid_capstyle="round",
    )
    (dot,) = curve_ax.plot(
        [],
        [],
        "o",
        color=run["color"],
        markersize=6,
        markeredgecolor=BACKGROUND,
        markeredgewidth=1.2,
    )
    label = curve_ax.text(
        0, 0, run["short_name"], fontsize=10, fontweight="bold", va="center"
    )
    curves.append((line, dot, label))

fig.text(
    0.035,
    0.03,
    'init="k-means||" costs about 2 to 3 more passes over the data, and '
    'revival="greedy" makes the fit longer.  '
    f"EMeans(n_clusters=64, n_epochs={N_EPOCHS}, random_state={RANDOM_STATE}).",
    fontsize=9,
    color=MUTED,
    va="center",
)


def spread_labels(positions, min_gap):
    """Push apart the label positions closer than `min_gap`."""
    positions = np.array(positions, dtype=float)
    for _ in range(50):
        order = np.argsort(positions, kind="stable")
        overlaps = min_gap - np.diff(positions[order])
        if (overlaps <= 1e-9).all():
            break
        for a, b, overlap in zip(order[:-1], order[1:], overlaps):
            if overlap > 1e-9:
                positions[a] -= overlap / 2
                positions[b] += overlap / 2
    return positions


ARROW_STEPS = 25  # Number of steps during which a greedy move stays drawn.


def draw(frame_step, final):
    label_positions = []
    for run, panel, (line, dot, label) in zip(runs, panels, curves):
        snapshots = run["snapshots"]
        if final:
            index = len(snapshots) - 1
        else:
            index = max(
                i
                for i, (step, _, kind) in enumerate(snapshots)
                if step <= frame_step and kind != "final"
            )
        centers = snapshots[index][1]
        panel["centers"].set_offsets(centers)
        empty = empty_blobs(centers)
        for blob, ring in enumerate(panel["rings"]):
            ring.set_visible(blob in empty)

        # Arrows from the old to the new position of the centers moved by the
        # recent greedy revivals.
        for patch in list(panel["ax"].patches):
            if isinstance(patch, FancyArrowPatch):
                patch.remove()
        for i, (step, after, kind) in enumerate(snapshots[: index + 1]):
            age = frame_step - step
            if kind != "greedy" or final or age > ARROW_STEPS:
                continue
            before = snapshots[i - 1][1]
            for c in np.flatnonzero(np.linalg.norm(after - before, axis=1) > 0.35):
                panel["ax"].add_patch(
                    FancyArrowPatch(
                        before[c],
                        after[c],
                        arrowstyle="-|>",
                        mutation_scale=9,
                        color=INK,
                        linewidth=1,
                        alpha=1 - 0.7 * age / ARROW_STEPS,
                        zorder=4,
                        shrinkB=3,
                    )
                )

        inertias = run["inertias"][: index + 1].copy()
        if final:
            inertias[-1] = run["final_inertia"]
            panel["status"].set_text(f"Done: {inertias[-1]:.2f}× the reference")
            panel["status"].set_fontweight("bold")
        else:
            panel["status"].set_text(f"Epoch {run['epochs'][index]:.1f} of {N_EPOCHS}")
            panel["status"].set_fontweight("normal")
        panel["n_empty"].set_text(
            f"{len(empty)} empty blob{'s' if len(empty) != 1 else ''}"
            if empty
            else "No empty blob"
        )
        epochs = run["epochs"][: index + 1]
        line.set_data(epochs, inertias)
        dot.set_data([epochs[-1]], [inertias[-1]])
        label_positions.append(math.log(min(inertias[-1], 5.5)))
        x_end = epochs[-1]

    # Near the end, the label of KMeans is in the way of the others.
    near_end = bool(x_end > N_EPOCHS - 1)
    positions = spread_labels(
        label_positions + [math.log(kmeans_10_inits)] * near_end, min_gap=0.16
    )
    for (_, _, label), position in zip(curves, positions):
        label.set_position((x_end + 0.06, math.exp(position)))
        label.set_visible(x_end > 0)
    kmeans_label.set_y(math.exp(positions[-1]) if near_end else kmeans_10_inits)


def render():
    fig.canvas.draw()
    return Image.fromarray(np.asarray(fig.canvas.buffer_rgba())[..., :3].copy())


frames, durations = [], []
for n, step in enumerate(frame_steps):
    draw(step, final=False)
    frames.append(render())
    durations.append(1500 if n == 0 else 70)
draw(n_steps, final=True)
frames.append(render())
durations.append(6000)

# One palette for all the frames, so that the colors don't flicker, plus a
# transparent color for the pixels that don't change from the previous frame,
# which makes the file ~4x smaller.
N_COLORS = 96
palette_image = frames[-1].quantize(colors=N_COLORS, method=Image.Quantize.MEDIANCUT)
palette = palette_image.getpalette()[: 3 * N_COLORS] + [255, 0, 255]
palette_image.putpalette(palette)
indexed = [
    np.asarray(frame.quantize(palette=palette_image, dither=Image.Dither.NONE))
    for frame in frames
]
gif_frames = []
for i, pixels in enumerate(indexed):
    if i > 0:
        pixels = np.where(pixels == indexed[i - 1], N_COLORS, pixels).astype(np.uint8)
    image = Image.fromarray(pixels, mode="P")
    image.putpalette(palette)
    gif_frames.append(image)
gif_frames[0].save(
    OUT,
    save_all=True,
    append_images=gif_frames[1:],
    duration=durations,
    loop=0,
    disposal=1,
    transparency=N_COLORS,
)
print(f"{OUT}: {len(gif_frames)} frames")
