"""
Figures 1-4 for paper 1 (vehicles). Numbers for Figs 2-4 are the 3-seed means in paper1_vehicles.md Tables 4 and 6
(themselves from logs/angle_all/ and logs/angle_heldout_class/). Fig 1 reads the image caches.

Output: figures/paper1/fig{1..4}.{pdf,png}
Usage: venv/bin/python scripts/make_paper1_figures.py
"""
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np

OUT = Path('figures/paper1'); OUT.mkdir(parents=True, exist_ok=True)
CACHE = Path('/Volumes/Machine Learning/atr_cache')
INK, INK2, MUTED, GRID = '#0b0b0b', '#52514e', '#898781', '#e4e3df'
C = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100']          # validated categorical slots 1-4
MARK = ['o', 's', '^', 'D']
plt.rcParams.update({'font.family': 'Helvetica Neue', 'font.size': 8.5, 'axes.edgecolor': MUTED, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.linewidth': 0.8, 'savefig.dpi': 300, 'legend.frameon': False})


def save(fig, n):
    for ext in ('pdf', 'png'):
        fig.savefig(OUT / f'fig{n}.{ext}', bbox_inches='tight')
    plt.close(fig)


# ---------- Figure 1: example inputs ----------
def stretch(img, vis):
    v = img[vis]; lo, hi = np.percentile(v, [1, 99])
    out = np.clip((img - lo) / max(hi - lo, 1e-6), 0, 1); out[~vis] = 0; return out


def square(cx, cy, S, s=128):
    m = np.zeros((s, s), bool)
    x0 = int(np.clip(round(cx - S / 2), 0, s - S)); y0 = int(np.clip(round(cy - S / 2), 0, s - S))
    m[y0:y0 + S, x0:x0 + S] = True; return m


def fig1():
    rows = []
    d = np.load(CACHE / 'mstar_test.npz'); X, y = d['X'], d['y']
    i = int(np.where(y == 0)[0][5]); img = X[i].astype(np.float32)
    rows.append(('MSTAR (2S1)', img, 64, 64, 96, 64))
    for name, lab in [('atr_soc_test', 'ATRNet-STAR SOC'), ('atr_scene_test', 'ATRNet-STAR EOC-scene test')]:
        d = np.load(CACHE / f'{name}.npz')
        g = d['group'] if name == 'atr_scene_test' else None
        cand = np.where(g == 'urban')[0] if g is not None and (g == 'urban').any() else np.arange(len(d['y']))
        i = int(cand[len(cand) // 3]); img = d['X'][i].astype(np.float32)
        img = np.log10(img + 1.0)
        rows.append((lab, img, float(d['cx'][i]), float(d['cy'][i]), 80, 80))
    fig, ax = plt.subplots(len(rows), 3, figsize=(5.4, 1.85 * len(rows)))
    for r, (lab, img, cx, cy, Sbg, Stg) in enumerate(rows):
        full = np.ones_like(img, bool); bg = ~square(cx, cy, Sbg); tg = square(cx, cy, Stg)
        for c, (vis, t) in enumerate([(full, 'Full'), (bg, 'Background only'), (tg, 'Target only')]):
            a = ax[r, c]; a.imshow(stretch(img, vis), cmap='gray', vmin=0, vmax=1); a.set_xticks([]); a.set_yticks([])
            for sp in a.spines.values(): sp.set_visible(False)
            if r == 0: a.set_title(t, fontsize=8.5, color=INK)
            if c == 0: a.set_ylabel(f'{lab}\nS = {Sbg} / {Stg}' if Sbg != Stg else f'{lab}\nS = {Sbg}', fontsize=8, color=INK2)
    fig.tight_layout(h_pad=0.6, w_pad=0.4); save(fig, 1)


# ---------- Figure 2: accuracy vs gap ----------
T4 = {  # train set -> {test angle: acc}
    (15,): {30: 54.6, 45: 41.0, 60: 22.4}, (30,): {15: 58.5, 45: 71.0, 60: 30.0}, (45,): {15: 47.4, 30: 71.9, 60: 41.6},
    (60,): {15: 29.7, 30: 38.7, 45: 49.8}, (15, 30): {45: 63.9, 60: 29.7}, (15, 45): {30: 72.5, 60: 38.6},
    (15, 60): {30: 57.9, 45: 56.9}, (30, 45): {15: 55.7, 60: 39.0}, (30, 60): {15: 54.9, 45: 71.3},
    (45, 60): {15: 43.8, 30: 64.3}, (15, 30, 45): {60: 36.7}, (15, 30, 60): {45: 66.6}, (15, 45, 60): {30: 68.1},
    (30, 45, 60): {15: 53.8}}


def fig2():
    fig, a = plt.subplots(figsize=(4.6, 3.0))
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    pts = [(min(abs(t - x) for x in A), t, acc, len(A)) for A, d in T4.items() for t, acc in d.items()]
    for k, ang in enumerate([15, 30, 45, 60]):
        p = [q for q in pts if q[1] == ang]
        xs = [q[0] + (k - 1.5) * 1.6 for q in p]
        a.scatter(xs, [q[2] for q in p], s=34, marker=MARK[k], color=C[k], edgecolor='#fcfcfb', lw=1.2, zorder=3,
                  label=f'test {ang}°')
    for gap in (15, 30, 45):
        m = np.mean([q[2] for q in pts if q[0] == gap])
        a.plot([gap - 5, gap + 5], [m, m], color=INK, lw=2, solid_capstyle='round', zorder=4)
        a.text(gap + 6, m, f'mean {m:.1f}%', va='center', fontsize=8, color=INK)
    a.set_xticks([15, 30, 45]); a.set_xticklabels(['15°', '30°', '45°']); a.set_xlim(5, 58); a.set_ylim(0, 80)
    a.set_xlabel('Gap to nearest training angle'); a.set_ylabel('Test accuracy (%)')
    a.axhline(2.5, color=MUTED, lw=0.8, ls=(0, (3, 3))); a.text(57, 4, 'chance 2.5%', ha='right', fontsize=7.5, color=MUTED)
    a.legend(loc='upper right', ncol=2, fontsize=7.5, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(); save(fig, 2)


# ---------- Figure 3: single-angle heatmap ----------
def fig3():
    ang = [15, 30, 45, 60]
    M = np.full((4, 4), np.nan)
    for i, tr in enumerate(ang):
        for j, te in enumerate(ang):
            if tr != te: M[i, j] = T4[(tr,)][te]
    fig, a = plt.subplots(figsize=(3.4, 3.0))
    cmap = matplotlib.colors.LinearSegmentedColormap.from_list('blue', ['#cde2fb', '#86b6ef', '#2a78d6', '#184f95', '#0d366b'])
    cmap.set_bad('#f0efec')
    im = a.imshow(M, cmap=cmap, vmin=20, vmax=75)
    for i in range(4):
        for j in range(4):
            if i == j: a.text(j, i, 'train', ha='center', va='center', fontsize=7.5, color=MUTED)
            else: a.text(j, i, f'{M[i, j]:.1f}', ha='center', va='center', fontsize=8.5,
                         color='#ffffff' if M[i, j] > 50 else INK)
    a.set_xticks(range(4)); a.set_xticklabels([f'{x}°' for x in ang]); a.set_yticks(range(4)); a.set_yticklabels([f'{x}°' for x in ang])
    a.set_xlabel('Test depression angle'); a.set_ylabel('Training depression angle')
    for sp in a.spines.values(): sp.set_visible(False)
    a.tick_params(length=0)
    cb = fig.colorbar(im, ax=a, fraction=0.046, pad=0.04); cb.set_label('Accuracy (%)'); cb.outline.set_visible(False)
    fig.tight_layout(); save(fig, 3)


# ---------- Figure 4: partial library ----------
def fig4():
    gaps = ['15° gap\ntrain 30°, test 45°', '30° gap\ntrain 15°, test 45°', '45° gap\ntrain 15°, test 60°']
    S = [('No class at test angle', [71.0, 41.0, 22.4], [0, 0, 0]),
         ('Partial library: missing classes', [68.6, 28.7, 9.0], [0.6, 1.2, 0.4]),
         ('Partial library: library classes', [85.6, 81.1, 68.6], [0.4, 0.7, 0.5])]
    fig, a = plt.subplots(figsize=(4.6, 3.0))
    a.yaxis.grid(True, color=GRID, lw=0.6); a.set_axisbelow(True)
    w = 0.26; x = np.arange(3)
    for k, (lab, v, e) in enumerate(S):
        xs = x + (k - 1) * (w + 0.02)
        a.bar(xs, v, w, color=C[k], label=lab, edgecolor='#fcfcfb', lw=1.0, zorder=3)
        if any(e): a.errorbar(xs, v, yerr=e, fmt='none', ecolor=INK, elinewidth=0.8, capsize=2, zorder=4)
        if k == 1:
            for xi, vi in zip(xs, v): a.text(xi, vi + 2, f'{vi:.1f}', ha='center', fontsize=7.5, color=INK)
    a.set_xticks(x); a.set_xticklabels(gaps); a.set_ylim(0, 100); a.set_ylabel('Test accuracy at the test angle (%)')
    a.legend(loc='lower left', bbox_to_anchor=(0, 1.0), ncol=1, fontsize=7.5)
    fig.tight_layout(); save(fig, 4)


if __name__ == '__main__':
    fig2(); fig3(); fig4(); fig1()
    print('written to', OUT)
