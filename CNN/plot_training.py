"""
Plot afterstate-agent training progress for the README: score, tile-reach rates, and loss.

Reads the per-episode metrics CSV and the training stdout log (loss is only printed there,
every --log-interval episodes), and writes a light and a dark PNG.

  python -m CNN.plot_training --train-csv CNN/logs/cnn_after_pilot_train.csv \
      --log CNN/logs/cnn_after_pilot.out --out-dir docs/images
"""

import argparse
import csv
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import FuncFormatter, PercentFormatter

THEMES = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", muted="#898781",
                  grid="#e1e0d9", axis="#c3c2b7", series=["#2a78d6", "#eb6834", "#1baf7a"]),
    "dark": dict(surface="#1a1a19", ink="#f0efec", ink2="#c3c2b7", muted="#898781",
                 grid="#2c2c2a", axis="#383835", series=["#3987e5", "#d95926", "#199e70"]),
}
RATE_TILES = (1024, 2048, 4096)
LOSS_RE = re.compile(r"Episode\s+(\d+):.*loss=([0-9.]+)")


def load_train(path: str):
    with open(path) as f:
        rows = list(csv.DictReader(f))
    episodes = np.array([int(r["episode"]) for r in rows])
    scores = np.array([int(r["score"]) for r in rows])
    tiles = np.array([int(r["max_tile"]) for r in rows])
    return episodes, scores, tiles


def load_loss(path: str):
    points = {}
    with open(path) as f:
        for line in f:
            m = LOSS_RE.search(line)
            if m and float(m.group(2)) > 0:  # loss=0 before the replay warmup ends
                points[int(m.group(1))] = float(m.group(2))
    eps = np.array(sorted(points))
    return eps, np.array([points[e] for e in eps])


def trailing_mean(x: np.ndarray, w: int) -> np.ndarray:
    c = np.cumsum(np.insert(x.astype(float), 0, 0.0))
    return (c[w:] - c[:-w]) / w  # value i covers x[i : i + w]


def compact(v, _pos=None) -> str:
    return f"{v / 1000:.0f}K" if abs(v) >= 1000 else f"{v:.0f}"


def style_axis(ax, t, title: str) -> None:
    ax.set_facecolor(t["surface"])
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(t["axis"])
    ax.grid(axis="y", color=t["grid"], linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    ax.tick_params(colors=t["muted"], labelsize=9, length=0, pad=6)
    ax.set_title(title, loc="left", color=t["ink2"], fontsize=10.5, pad=8)


def end_label(ax, x, y, text, t) -> None:
    ax.annotate(text, (x, y), xytext=(8, 0), textcoords="offset points", va="center",
                color=t["ink2"], fontsize=9, annotation_clip=False)


def plot(theme: str, train, loss, out: Path, score_window: int, rate_window: int) -> None:
    t = THEMES[theme]
    episodes, scores, tiles = train
    loss_eps, loss_vals = loss
    plt.rcParams["font.family"] = ["Helvetica Neue", "Arial", "DejaVu Sans"]

    fig, (ax_s, ax_r, ax_l) = plt.subplots(
        3, 1, figsize=(9, 9.6), sharex=True, gridspec_kw=dict(height_ratios=[1.25, 1, 0.8], hspace=0.38))
    fig.patch.set_facecolor(t["surface"])
    x_max = episodes[-1]

    # 1) score: every game as a faint dot, trailing mean as the line
    style_axis(ax_s, t, f"Score per game  ·  dots = single games, line = {score_window}-game average")
    ax_s.scatter(episodes, scores, s=3, color=t["muted"], alpha=0.25, linewidths=0, zorder=2)
    sm = trailing_mean(scores, score_window)
    ax_s.plot(episodes[score_window - 1:], sm, color=t["series"][0], linewidth=2, solid_capstyle="round", zorder=3)
    end_label(ax_s, x_max, sm[-1], compact(sm[-1]), t)
    ax_s.yaxis.set_major_formatter(FuncFormatter(compact))
    ax_s.set_ylim(0, np.percentile(scores, 99.5) * 1.05)

    # 2) share of games reaching each tile, trailing window
    style_axis(ax_r, t, f"Games reaching tile  ·  {rate_window}-game rolling share")
    for tile, color in zip(RATE_TILES, t["series"]):
        rate = trailing_mean((tiles >= tile).astype(float), rate_window)
        ax_r.plot(episodes[rate_window - 1:], rate, color=color, linewidth=2, solid_capstyle="round",
                  label=f"{tile}", zorder=3)
        end_label(ax_r, x_max, rate[-1], f"{tile}  {rate[-1] * 100:.0f}%", t)
    ax_r.yaxis.set_major_formatter(PercentFormatter(1.0, decimals=0))
    ax_r.set_ylim(0, 1)
    leg = ax_r.legend(loc="upper left", frameon=False, fontsize=9, ncol=3, handlelength=1.6,
                      title="max tile ≥", title_fontsize=9, alignment="left")
    for txt in leg.get_texts() + [leg.get_title()]:
        txt.set_color(t["ink2"])

    # 3) loss
    style_axis(ax_l, t, "Training loss  ·  MSE on scaled returns, mean of last 1000 updates")
    ax_l.plot(loss_eps, loss_vals, color=t["series"][0], linewidth=2, solid_capstyle="round", zorder=3)
    end_label(ax_l, loss_eps[-1], loss_vals[-1], f"{loss_vals[-1]:.3f}", t)
    ax_l.set_ylim(0, loss_vals.max() * 1.15)
    ax_l.set_xlabel("Episode (games played)", color=t["ink2"], fontsize=9.5, labelpad=8)
    ax_l.xaxis.set_major_formatter(FuncFormatter(lambda v, _p: f"{v:,.0f}"))
    ax_l.set_xlim(0, x_max)

    fig.suptitle("Afterstate CNN · training progress", x=0.075, y=0.975, ha="left",
                 color=t["ink"], fontsize=14, fontweight="semibold")
    fig.subplots_adjust(left=0.075, right=0.86, top=0.92, bottom=0.07)
    fig.savefig(out, dpi=150, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--train-csv", default="CNN/logs/cnn_after_pilot_train.csv")
    p.add_argument("--log", default="CNN/logs/cnn_after_pilot.out")
    p.add_argument("--out-dir", default="docs/images")
    p.add_argument("--score-window", type=int, default=100)
    p.add_argument("--rate-window", type=int, default=500)
    args = p.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    train, loss = load_train(args.train_csv), load_loss(args.log)
    for theme in THEMES:
        plot(theme, train, loss, out_dir / f"training_progress_{theme}.png", args.score_window, args.rate_window)


if __name__ == "__main__":
    main()
