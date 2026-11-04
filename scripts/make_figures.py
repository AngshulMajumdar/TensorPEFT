"""Render every figure in docs/figures from the data in benchmarks/.

  python scripts/make_figures.py
"""
import json
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle  # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "docs", "figures")
B = os.path.join(ROOT, "benchmarks")
os.makedirs(OUT, exist_ok=True)

# ---- design tokens (light surface) -----------------------------------------------------------
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9"
BLUE, ORANGE, AQUA, YELLOW, MAGENTA = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
TP = BLUE
METHOD_COLOR = {"TensorPEFT": BLUE, "prompt": ORANGE, "LayerNorm": AQUA, "LoRA": YELLOW, "(IA)": MAGENTA}
LIGHT_BLUE = "#cde2fb"

plt.rcParams.update({
    "font.family": "Inter", "font.size": 12, "axes.titlesize": 15, "axes.titleweight": "semibold",
    "axes.labelsize": 12, "axes.labelcolor": INK2, "axes.edgecolor": MUTED, "axes.linewidth": 0.8,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "figure.facecolor": SURF,
    "axes.facecolor": SURF, "savefig.facecolor": SURF, "axes.spines.top": False, "axes.spines.right": False,
    "axes.titlelocation": "left", "axes.titlepad": 14, "legend.frameon": False,
})


def color_of(name):
    for k, c in METHOD_COLOR.items():
        if k.lower() in name.lower():
            return c
    return MUTED


def grid(ax, axis="y"):
    ax.grid(axis=axis, color=GRID, lw=0.8)
    ax.set_axisbelow(True)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=160, bbox_inches="tight", pad_inches=0.25)
    plt.close(fig)
    print("wrote", name)


def box(ax, x, y, w, h, text, fc, ec=None, tc=INK, fs=12, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.12",
                                fc=fc, ec=ec or fc, lw=1.2))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", color=tc, fontsize=fs, weight=weight, wrap=True)


def arrow(ax, x0, y0, x1, y1, c=INK2):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=16, color=c, lw=1.6))


R = json.load(open(os.path.join(B, "results.json")))
BASE = R["setup"]["base_val"]


# ---- 1. LoRA vs TensorPEFT concept -------------------------------------------------------------
def fig_concept():
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4))
    for ax in axes:
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 8)
        ax.axis("off")
    ax = axes[0]
    ax.set_title("LoRA: every matrix gets its own update", color=INK)
    roles = ["q", "k", "v", "o", "in", "out"]
    for li in range(6):
        for ri, r in enumerate(roles):
            x, y = 0.6 + ri * 1.5, 6.6 - li * 1.15
            ax.add_patch(Rectangle((x, y), 0.95, 0.8, fc="#eceae4", ec=MUTED, lw=0.6))
            ax.add_patch(Rectangle((x + 1.0, y), 0.12, 0.8, fc=YELLOW, ec="none"))
            ax.add_patch(Rectangle((x, y - 0.17), 0.95, 0.12, fc=YELLOW, ec="none"))
            if li == 0:
                ax.text(x + 0.48, 7.65, r, ha="center", color=INK2, fontsize=11)
    ax.text(5.0, -0.25, "6 matrices × 12 layers = 72 separate low-rank pairs\nrank 1 everywhere on GPT-2: 165,888 parameters",
            ha="center", va="top", color=INK2, fontsize=11)
    ax = axes[1]
    ax.set_title("TensorPEFT: one shared update for the whole stack", color=INK)
    for k in range(8):
        off = k * 0.22
        ax.add_patch(Rectangle((1.0 + off, 1.4 + off), 3.6, 3.6, fc="#eceae4", ec=MUTED, lw=0.6))
    ax.text(2.8 + 0.8, 7.0, "all layers × {q, k, v, o}\nstacked into one block", ha="center", color=INK2, fontsize=11)
    # the four vectors of one atom
    ax.add_patch(Rectangle((6.3, 6.3), 3.2, 0.32, fc=TP, ec="none"))
    ax.text(9.6, 6.46, "", va="center")
    ax.text(6.3, 6.85, "layer weights (12)", color=INK, fontsize=11)
    ax.add_patch(Rectangle((6.3, 5.15), 1.1, 0.32, fc=TP, ec="none"))
    ax.text(6.3, 5.7, "role weights (q,k,v,o)", color=INK, fontsize=11)
    ax.add_patch(Rectangle((6.3, 1.4), 0.32, 3.2, fc=TP, ec="none"))
    ax.text(6.8, 3.0, "input\ndirection\n(768)", color=INK, fontsize=11, va="center")
    ax.add_patch(Rectangle((8.2, 4.28), 1.5, 0.32, fc=TP, ec="none"))
    ax.text(8.2, 3.6, "output\ndirection\n(768)", color=INK, fontsize=11, va="center")
    ax.text(5.0, -0.25, "one atom = 4 small vectors that touch every layer\n12 + 4 + 768 + 768 + 1 = 1,553 parameters",
            ha="center", va="top", color=INK2, fontsize=11)
    save(fig, "01_lora_vs_tensorpeft.png")


# ---- 2. workflow ---------------------------------------------------------------------------------
def fig_workflow():
    fig, ax = plt.subplots(figsize=(12, 3.4))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 3)
    ax.axis("off")
    steps = [("1  Wrap", "get_tensorpeft_model\n(nothing changes yet)"),
             ("2  Probe", "one backward pass\non 4 batches"),
             ("3  Pick atoms", "steepest rank-one\ngradient directions"),
             ("4  Train", "AdamW or Trainer,\njust like LoRA"),
             ("5  Merge / share", "merge_and_unload()\nor push_to_hub()")]
    for i, (t, s) in enumerate(steps):
        x = 0.15 + i * 2.4
        fc = TP if i in (1, 2) else "#eceae4"
        tc = "white" if i in (1, 2) else INK
        box(ax, x, 1.15, 1.95, 1.45, "", fc)
        ax.text(x + 0.975, 2.18, t, ha="center", color=tc, fontsize=13, weight="semibold")
        ax.text(x + 0.975, 1.62, s, ha="center", va="center", color=tc, fontsize=10)
        if i < 4:
            arrow(ax, x + 2.0, 1.88, x + 2.35, 1.88)
    ax.text(0.15 + 2.4, 0.55, "the only step LoRA does not have: about as long as one training step",
            color=TP, fontsize=11)
    save(fig, "02_workflow.png")


# ---- 3. stacking -> tensor ---------------------------------------------------------------------
def fig_stacking():
    fig, ax = plt.subplots(figsize=(13, 4.6))
    ax.set_xlim(0, 13)
    ax.set_ylim(0, 5)
    ax.axis("off")
    roles = ["q", "k", "v", "o"]
    for li in range(4):
        for ri, r in enumerate(roles):
            x, y = 0.3 + ri * 0.95, 3.7 - li * 0.95
            ax.add_patch(Rectangle((x, y), 0.8, 0.8, fc="#eceae4", ec=MUTED, lw=0.6))
            ax.text(x + 0.4, y + 0.4, f"{r}{li + 1}", ha="center", va="center", fontsize=10, color=INK2)
    ax.text(2.2, 0.35, "…  12 layers × 4 roles, each 768 × 768", ha="center", color=INK2, fontsize=11)
    arrow(ax, 4.4, 2.6, 5.5, 2.6)
    for k in range(10):
        off = k * 0.13
        ax.add_patch(Rectangle((5.6 + off, 1.0 + off), 2.6, 2.6, fc="#eceae4" if k < 9 else "#e3e1da", ec=MUTED, lw=0.6))
    ax.text(7.5, 0.45, "one tensor: 12 × 4 × 768 × 768", ha="center", color=INK2, fontsize=11)
    ax.text(9.9, 3.6, "Groups are formed automatically\nby matrix shape. GPT-2 has three:", color=INK, fontsize=11)
    ax.text(9.9, 2.35, "• q, k, v, o: 12 × 4 × 768 × 768\n• MLP-in: 12 × 768 × 3072\n• MLP-out: 12 × 3072 × 768",
            color=INK2, fontsize=11)
    ax.text(9.9, 1.3, "Llama-style models get\n{q,o}, {k,v}, {gate,up}, {down}.", color=INK2, fontsize=11)
    save(fig, "03_stacking.png")


# ---- 4. parameter cost -------------------------------------------------------------------------
def fig_param_cost():
    rows = [("LoRA r=4, all matrices", 663552, YELLOW), ("LoRA r=1, all matrices", 165888, YELLOW),
            ("LoRA r=1, q and v only", 36864, YELLOW), ("TensorPEFT, 3 atoms (one per group)", 9259, TP),
            ("TensorPEFT, 1 MLP atom", 3853, TP), ("TensorPEFT, 1 attention atom", 1553, TP)]
    fig, ax = plt.subplots(figsize=(11, 4.4))
    y = list(range(len(rows)))[::-1]
    for yi, (n, v, c) in zip(y, rows):
        ax.barh(yi, v, height=0.56, color=c)
        ax.text(v * 1.12, yi, f"{v:,}", va="center", color=INK, fontsize=11)
    ax.set_yticks(y, [r[0] for r in rows])
    ax.set_xscale("log")
    ax.set_xlim(800, 3e6)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_title("Adapter sizes on GPT-2 (124M parameters)")
    grid(ax, "x")
    ax.tick_params(axis="y", length=0)
    save(fig, "04_parameter_cost.png")


# ---- 5. descent spectrum -----------------------------------------------------------------------
def fig_spectrum():
    d = json.load(open(os.path.join(B, "atoms_gpt2_bengali.json")))
    names = {"attn": "q, k, v, o tensor", "mlp.c_fc": "MLP-in tensor", "mlp.c_proj": "MLP-out tensor"}
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.8), sharey=True)
    for ax, (k, v) in zip(axes, d["spectrum"].items()):
        label = next(n for p, n in names.items() if k.startswith(p))
        xs = range(1, len(v) + 1)
        ax.bar(xs, v, width=0.62, color=[TP] + [LIGHT_BLUE] * (len(v) - 1))
        ax.set_title(label, fontsize=13)
        ax.set_xticks(list(xs))
        ax.set_xlabel("atom")
        grid(ax)
    axes[0].set_ylabel("initial loss decrease\nper unit step")
    fig.suptitle("One gradient probe: the first atom of each tensor is the steepest", x=0.06, ha="left",
                 fontsize=15, weight="semibold", y=1.10)
    save(fig, "05_descent_spectrum.png")


# ---- 6. what an atom learned -------------------------------------------------------------------
def fig_anatomy():
    d = json.load(open(os.path.join(B, "atoms_gpt2_bengali.json")))
    a = d["atoms"]["0"]
    layer = [abs(x) for x in a["factor0"][0]]
    roles = d["groups"][0]["roles"]
    role = {r.split(".")[-1].replace("c_proj", "o"): abs(x) for r, x in zip(roles, a["factor1"][0])}
    order = ["q", "k", "v", "o"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 3.9), gridspec_kw={"width_ratios": [3, 1.3]})
    ax = axes[0]
    ax.bar(range(1, 13), layer, width=0.62, color=TP)
    ax.set_xticks(range(1, 13))
    ax.set_xlabel("transformer layer")
    ax.set_ylabel("weight (absolute value)")
    ax.set_title("Attention atom after training: where it acts")
    grid(ax)
    ax = axes[1]
    ax.bar(order, [role[r] for r in order], width=0.6, color=TP)
    ax.set_xlabel("projection")
    ax.set_title("…and on which projection")
    grid(ax)
    save(fig, "06_atom_anatomy.png")


# ---- 7. race at 9k -----------------------------------------------------------------------------
def fig_race():
    p = os.path.join(B, "race_9k.json")
    if not os.path.exists(p):
        print("skip race (no data)")
        return
    d = json.load(open(p))
    fig, ax = plt.subplots(figsize=(11, 5.6))
    finals = []
    for r in d["runs"]:
        xs, ys = zip(*r["curve"])
        c = color_of(r["name"])
        ax.plot(xs, ys, color=c, lw=3.0 if r["name"].startswith("TensorPEFT") else 2.0, zorder=3 if c == TP else 2)
        finals.append((ys[-1], r["name"], c))
    ax.axhline(d["base"], color=MUTED, ls="--", lw=1)
    ax.text(4, d["base"] + 0.006, f"GPT-2 before fine-tuning ({d['base']:.3f})", color=INK2, fontsize=10)
    finals.sort()
    ypos, last = [], None
    for y, n, c in finals:          # de-overlap direct labels
        y2 = y if last is None or y - last > 0.022 else last + 0.022
        ypos.append(y2)
        last = y2
    for (y, n, c), y2 in zip(finals, ypos):
        short = n.replace(" · ", ", ")
        ax.annotate(f"{short}  {y:.3f}", (300, y), (308, y2), color=INK, fontsize=10.5, va="center",
                    weight="semibold" if c == TP else "normal",
                    arrowprops=dict(arrowstyle="-", color=c, lw=1.2))
        ax.plot([300], [y], "o", color=c, ms=7, zorder=4)
    ax.set_xlim(0, 300)
    ax.set_ylim(1.30, max(1.72, d["base"] + 0.02))
    ax.set_xlabel("training step")
    ax.set_ylabel("validation loss (nats/token)  —  lower is better")
    ax.set_title("Race at ≈9,200 trainable parameters (GPT-2 → Bengali, 300 steps)")
    grid(ax)
    save(fig, "07_race_9k.png")


# ---- 8. loss vs parameters ---------------------------------------------------------------------
def fig_frontier():
    tiers = R["tiers_seed0"]
    tp = sorted([(t["params"], t["val"]) for t in tiers if t["method"].startswith("TensorPEFT")])
    others = [t for t in tiers if not t["method"].startswith("TensorPEFT")] + \
             [dict(budget="", method=m["method"], params=m["params"], val=m["val"]) for m in R["larger_budgets_lora"]]
    fig, ax = plt.subplots(figsize=(11.5, 6.2))
    ax.axvspan(700, 60000, color=LIGHT_BLUE, alpha=0.35, lw=0)
    ax.text(900, 1.115, "TensorPEFT best or tied", color=TP, fontsize=11, weight="semibold")
    for t in others:
        ax.plot(t["params"], t["val"], "o", color=MUTED, ms=8, mec=SURF, mew=1.5, zorder=3)
    place = {  # label position in data coordinates (None = just right of the point)
        "LoRA r=1 (q,v, last 3 layers)": (13000, 1.442), "(IA)3 (values)": (13000, 1.468),
        "LoRA r=1 (q,v)": (110000, 1.352), "prompt tuning (48 tokens)": (110000, 1.374),
        "QLoRA r=1 (q,v), NF4 base": (110000, 1.396), "LayerNorm tuning (all)": (21000, 1.212),
        "LoRA r=1 (q,k,v,o,MLP-in)": (130000, 1.218), "LoRA r=4 (all matrices)": (420000, 1.150),
    }
    for t in others:
        name = t["method"].replace("(IA)3", "(IA)³")
        if t["method"] in place:
            tx, ty = place[t["method"]]
            ax.annotate(name, (t["params"], t["val"]), (tx, ty), fontsize=9.5, color=INK2, va="center",
                        ha="left" if tx > t["params"] else "right",
                        arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8, shrinkA=3, shrinkB=5,
                                        relpos=(0, 0.5) if tx > t["params"] else (1, 0.5)))
        else:
            ax.annotate(name, (t["params"], t["val"]), (8, 0), textcoords="offset points", fontsize=9.5,
                        color=INK2, va="center")
    xs, ys = zip(*tp)
    ax.plot(xs, ys, "-o", color=TP, lw=3, ms=9, mec=SURF, mew=1.5, zorder=5)
    ax.text(950, 1.395, "TensorPEFT", color=TP, fontsize=13, weight="semibold")
    ax.set_xscale("log")
    ax.set_xlim(500, 1.5e6)
    ax.set_xlabel("trainable parameters (log scale)")
    ax.set_ylabel("validation loss (nats/token)  —  lower is better")
    ax.set_title("Loss vs. trainable parameters: every method, same data, 300 steps")
    grid(ax, "both")
    save(fig, "08_loss_vs_parameters.png")


# ---- 9. ranking per budget ---------------------------------------------------------------------
def fig_rankings():
    budgets = ["~1.5k", "~9k", "~37k", "~55k"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 7.6))
    for ax, b in zip(axes.flat, budgets):
        rows = sorted([t for t in R["tiers_seed0"] if t["budget"] == b], key=lambda t: t["val"])[::-1]
        y = range(len(rows))
        lo, hi = min(t["val"] for t in rows), max(t["val"] for t in rows)
        x0, x1 = lo - 0.25 * (hi - lo) - 0.01, hi + 0.30 * (hi - lo) + 0.02
        for yi, t in zip(y, rows):
            c = TP if t["method"].startswith("TensorPEFT") else MUTED
            ax.plot([x0, t["val"]], [yi, yi], color=GRID, lw=1, zorder=1)
            ax.plot(t["val"], yi, "o", color=c, ms=10, zorder=3)
            ax.text(t["val"] + 0.035 * (x1 - x0), yi, f"{t['val']:.3f}", va="center", fontsize=10,
                    color=INK, weight="semibold" if c == TP else "normal")
        ax.set_yticks(list(y), [t["method"].replace("(IA)3", "(IA)³") for t in rows], fontsize=10)
        ax.set_xlim(x0, x1)
        ax.set_title(f"{b.replace('~', '≈')} parameters", fontsize=13)
        ax.tick_params(axis="y", length=0)
        grid(ax, "x")
    for ax in axes[1]:
        ax.set_xlabel("validation loss — lower is better")
    fig.suptitle("Ranking at each budget (seed 0)", x=0.02, ha="left", fontsize=15, weight="semibold")
    fig.tight_layout()
    save(fig, "09_rankings.png")


# ---- 10. three seeds ---------------------------------------------------------------------------
def fig_seeds():
    S = R["three_seeds"]
    fig, axes = plt.subplots(1, 3, figsize=(12, 4.2))
    for ax, s in zip(axes, S):
        for xi, (vals, c, lab) in enumerate([(s["TensorPEFT"], TP, "TensorPEFT"),
                                             (s["baseline_vals"], MUTED, s["baseline"].split(" (")[0].replace("(IA)3", "(IA)³"))]):
            ax.plot([xi] * 3, vals, "o", color=c, ms=8, alpha=0.9)
            m = sum(vals) / 3
            ax.plot([xi - 0.22, xi + 0.22], [m, m], color=c, lw=3)
        ax.set_xticks([0, 1], ["TensorPEFT", s["baseline"].split(" (")[0].replace("(IA)3", "(IA)³")], fontsize=10.5)
        ax.set_xlim(-0.6, 1.6)
        verdict = "tie" if s["p_paired_t"] > 0.05 else ("TensorPEFT better" if s["mean_diff"] < 0 else "baseline better")
        ax.set_title(f"{s['budget'].replace('~', '≈')} params\n{verdict}  (Δ {s['mean_diff']:+.3f}, p = {s['p_paired_t']:.3f})",
                     fontsize=12)
        grid(ax)
    axes[0].set_ylabel("validation loss — lower is better")
    fig.suptitle("Three random seeds against the best baseline at each budget (dots = seeds, bar = mean)",
                 x=0.02, ha="left", fontsize=14, weight="semibold", y=1.06)
    fig.tight_layout()
    save(fig, "10_three_seeds.png")


# ---- 11. schedule ablation ---------------------------------------------------------------------
def fig_schedule():
    S = R["schedule_ablation"]
    fig, ax = plt.subplots(figsize=(10, 3.6))
    for i, s in enumerate(S):
        a, b = s["atoms_at_start"], s["atoms_added_in_stages"]
        ax.plot([a, b], [i, i], color=GRID, lw=3, zorder=1)
        ax.plot(b, i, "o", color=MUTED, ms=11, zorder=3)
        ax.plot(a, i, "o", color=TP, ms=11, zorder=3)
        ax.text(b + 0.004, i, f"{b:.3f}  atoms added in stages", va="center", fontsize=10.5, color=INK2)
        ax.text(a - 0.004, i, f"all atoms at the start  {a:.3f}", va="center", ha="right", fontsize=10.5,
                color=INK, weight="semibold")
    ax.set_yticks(range(len(S)), [s["budget"].replace("~", "≈") + " params" for s in S])
    ax.set_ylim(-0.6, len(S) - 0.4)
    ax.set_xlim(1.14, 1.40)
    ax.set_xlabel("validation loss — lower is better")
    ax.set_title("Choose all atoms up front from one probe")
    ax.tick_params(axis="y", length=0)
    grid(ax, "x")
    save(fig, "11_schedule.png")


if __name__ == "__main__":
    fig_concept()
    fig_workflow()
    fig_stacking()
    fig_param_cost()
    fig_spectrum()
    fig_anatomy()
    fig_race()
    fig_frontier()
    fig_rankings()
    fig_seeds()
    fig_schedule()
