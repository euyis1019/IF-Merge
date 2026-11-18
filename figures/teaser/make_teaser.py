"""README teaser: IF-Merge in prediction space (CLIP ViT-B/32, 8 tasks).

    python figures/teaser/make_teaser.py            ->  figures/teaser.png

Plane: classical MDS of the square-root Jensen-Shannon divergence between the predictions of real
models (512 test images per task; `pred.npz`). Height and colour: accuracy of those models on the
same images (`pred.json`). Curves: IF-Merge round by round from ten existing merges. The surface is
a thin-plate fit through the real models, faded to the floor away from them (display only).
The accuracy ranges in the legend are the full-test-set numbers reported in the paper.
Every number in PRESETS (and every E("NAME", default) below) can be overridden by an environment variable.
"""
import json, os, sys, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
from matplotlib import font_manager as fm
from matplotlib.colors import LinearSegmentedColormap, LightSource, to_rgb
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from scipy.interpolate import RBFInterpolator, PchipInterpolator
from scipy.ndimage import gaussian_filter
E = lambda k, d: type(d)(os.environ.get(k, d))
PRESETS = {
    # higher accuracy is lower (a valley), seen from the valley floor, with projections on the ground
    "valley": dict(INV=1, AZIM=20, ELEV=32, VEX=0.35, LAZ=200, LALT=60, GRID=400, FH=4.3, ZOOM=0.585, PAD=0.08, ZASP=0.40,
                   ZFLOOR=0.38, FLOORMAP=1, SHADOW=1, AY=0.005, AX=0.0, LGY=0.955, LGD=0.055, AMODE="wgt"),
    # higher accuracy is higher (a peak), seen from the side of the initializations
    "peak": dict(INV=0, AZIM=-132, ELEV=26, GRID=320, FH=3.95, ZOOM=0.675, PAD=0.08, ZASP=0.42, AY=0.04, AX=0.02, AMODE="hn"),
}
MODE = sys.argv[1] if len(sys.argv) > 1 else "valley"
for _k, _v in PRESETS[MODE].items():
    os.environ.setdefault(_k, str(_v))
if len(sys.argv) > 2:
    os.environ.setdefault("CMAP", sys.argv[2])
for f in fm.findSystemFonts():
    if "texgyretermes" in f.lower(): fm.fontManager.addfont(f)
plt.rcParams.update({"font.family": ["TeX Gyre Termes", "STIXGeneral"], "mathtext.fontset": "stix", "pdf.fonttype": 42})
D = os.path.dirname(os.path.abspath(__file__))
J = json.load(open(f"{D}/pred.json")); Z = np.load(f"{D}/pred.npz")
names, meta = J["names"], J["meta"]; ix = {n: i for i, n in enumerate(names)}
acc = np.array([m["acc"] for m in meta])
STARTS = ["wa_plain", "ta", "ties", "regmean", "adamerging", "regmeanpp", "tsv_m", "wudi", "iso_c", "iso_cts"]
PRETTY = {"wa_plain": "Weight Averaging", "ta": "Task Arithmetic", "ties": "TIES", "regmean": "RegMean", "adamerging": "AdaMerging",
          "regmeanpp": "RegMean++", "tsv_m": "TSV-M", "wudi": "WUDI", "iso_c": "Iso-C", "iso_cts": "Iso-CTS"}
COL = {"wa_plain": "#d1495b", "ta": "#e2703a", "ties": "#edae49", "regmean": "#8f6bb3", "adamerging": "#c05c9e",
       "regmeanpp": "#6b74c4", "tsv_m": "#2a9d8f", "wudi": "#5aa85e", "iso_c": "#3a86c8", "iso_cts": "#22a7d6"}
END_COL = E("END_COL", "#16233f"); INK = "#2a2a2a"; INK2 = "#6b6660"
keep = [i for i, m in enumerate(meta) if m["kind"] in ("if", "seg", "mix", "experts", "joint", "mixj")]
M = Z["js"]; M = 0.5 * (M + M.T)
Dm = np.sqrt(np.maximum(M[np.ix_(keep, keep)], 0)); n = len(keep)
Jc = np.eye(n) - 1 / n; w, v = np.linalg.eigh(-0.5 * Jc @ (Dm ** 2) @ Jc); o = np.argsort(w)[::-1]; w, v = w[o], v[:, o]
XY = v[:, :2] * np.sqrt(w[:2]); expl = w[:2].sum() / w[w > 0].sum()
pos = {names[i]: XY[k] for k, i in enumerate(keep)}
end_name = lambda s: f"if:{s}:r4" if f"if:{s}:r4" in ix else f"if:{s}:end"
c0 = np.array([pos[f"if:{s}:r0"] for s in STARTS]).mean(0); tgt = np.array([pos[end_name(s)] for s in STARTS]).mean(0) - c0
th = -np.arctan2(tgt[1], tgt[0]); R = np.array([[np.cos(th), -np.sin(th)], [np.sin(th), np.cos(th)]])
XY = (XY - c0) @ R.T
# the sign of the second MDS axis is arbitrary: fix it so that Weight Averaging starts at positive y
if XY[keep.index(ix["if:wa_plain:r0"]), 1] < 0: XY[:, 1] *= -1
if E("FLIP", 0): XY[:, 1] *= -1
pos = {names[i]: XY[k] for k, i in enumerate(keep)}; a = acc[keep]
G, ALO, AHI = E("GAMMA", 1.8), E("ALO", 62.0), E("AHI", 91.5)
hz = lambda x: np.clip((np.asarray(x, float) - ALO) / (AHI - ALO), 0, None) ** G
traj = np.array([meta[i]["kind"] in ("if", "seg", "experts", "joint") for i in keep])
rbf = RBFInterpolator(XY, a, kernel="thin_plate_spline", smoothing=np.where(traj, 0.003, E("SMOOTH_MIX", 0.15)), degree=1)
pad, N = E("PAD", 0.14), E("GRID", 320)
lo, hi = XY.min(0) - pad, XY.max(0) + pad
gx, gy = np.meshgrid(np.linspace(lo[0], hi[0], N), np.linspace(lo[1], hi[1], N))
Q = np.c_[gx.ravel(), gy.ravel()]
dnn = np.sqrt(((Q[:, None] - XY[None]) ** 2).sum(-1)).min(1)
wgt = np.exp(-(np.maximum(0, dnn - E("R0", 0.02)) / E("SIG", 0.05)) ** 2).reshape(gx.shape)
wgt = gaussian_filter(wgt, E("GS", 5.0) * N / 240)
A = ALO + (np.clip(rbf(Q), ALO, AHI).reshape(gx.shape) - ALO) * wgt
H = gaussian_filter(hz(A), E("GH", 2.0) * N / 240)
INV = E("INV", 1); ZT = float(hz(AHI))
zf = (lambda x: ZT - hz(x)) if INV else hz
HS = ZT - H if INV else H
CM = {"sandw": ["#ffffff", "#eef4f1", "#cfe5df", "#a3cfcf", "#72aec6", "#4a86b8", "#2f5c9c"],
      "dawnw": ["#ffffff", "#fbf1df", "#f6d9ae", "#eeb58e", "#d08f97", "#9a7cab", "#5f66a6"],
      "icew": ["#ffffff", "#f1f5f9", "#dbe8f3", "#bcd5ec", "#97bfe3", "#709fd4", "#4a7fbd"]}[E("CMAP", "sandw")]
cmap = LinearSegmentedColormap.from_list("t", CM)
FW, FH = E("FW", 6.875), E("FH", 3.7)
fig = plt.figure(figsize=(FW, FH), dpi=E("DPI", 450))
S_ = E("SQ", 1.3) * FW; AW_, AH_ = S_ / FW, S_ / FH
ax = fig.add_axes([(1 - AW_) / 2 + E("AX", 0.0), (1 - AH_) / 2 + E("AY", 0.0), AW_, AH_], projection="3d", computed_zorder=False)
ax.patch.set_alpha(0)
ax.view_init(elev=E("ELEV", 40.0), azim=E("AZIM", -132.0))
HN = H / hz(AHI)
rgb = LightSource(azdeg=E("LAZ", 300.0), altdeg=E("LALT", 50.0)).shade_rgb(cmap(np.clip(HN, 0, 1))[..., :3], elevation=HS / ZT, blend_mode="soft", vert_exag=E("VEX", 1.0))
rgb = np.concatenate([rgb, np.ones_like(rgb[..., :1])], -1)
if E("AMODE", "wgt") == "wgt":
    t_ = np.clip((wgt - E("A0", 0.45)) / (E("A1", 0.8) - E("A0", 0.45)), 0, 1); rgb[..., 3] = t_ * t_ * (3 - 2 * t_)
else:
    rgb[..., 3] = np.clip(HN / E("AFADE", 0.035), 0, 1) ** 1.5
x0, x1, y0, y1 = gx.min(), gx.max(), gy.min(), gy.max()
ZFL = -E("ZFLOOR", 0.0) * ZT
if E("FLOORMAP", 0):
    ax.contourf(gx, gy, np.clip(HN, 0, 1), zdir="z", offset=ZFL, levels=np.r_[0.02, hz(np.array([70, 75, 80, 84, 87, 89, 90.3])) / ZT, 1.0], cmap=LinearSegmentedColormap.from_list("f", [tuple(1 - E("FML", 0.45) * (1 - np.array(to_rgb(c)))) for c in CM]), vmin=0, vmax=1, zorder=0.1, antialiased=True)
if E("FLOORGRID", 1):
    gc = "#e4e1db"
    for xv in np.linspace(x0, x1, E("NGX", 15)): ax.plot([xv, xv], [y0, y1], [ZFL, ZFL], color=gc, lw=0.45, zorder=0.2)
    for yv in np.linspace(y0, y1, E("NGY", 11)): ax.plot([x0, x1], [yv, yv], [ZFL, ZFL], color=gc, lw=0.45, zorder=0.2)
    ax.plot([x0, x1, x1, x0, x0], [y0, y0, y1, y1, y0], [ZFL] * 5, color="#bcb7af", lw=0.7, zorder=0.3)
ax.plot_surface(gx, gy, HS, facecolors=rgb, rstride=1, cstride=1, linewidth=0, antialiased=False, shade=False, zorder=1, rasterized=True)
LEV = [70, 75, 80, 84, 87, 89, 90.3]
ax.contour(gx, gy, np.where(rgb[..., 3] > E("CMASK", 0.6), HS, np.nan), levels=sorted(zf(LEV)), colors="white", linewidths=0.45, alpha=0.85, zorder=2)
def path(s):
    pts = [(0.0, f"if:{s}:r0")]
    if f"if:{s}:r4" in ix:
        pts += [(t, f"seg:{s}:r0-r1:{t}") for t in (0.125, 0.25, 0.5, 0.75)] + [(1.0, f"if:{s}:r1"), (1.5, f"seg:{s}:r1-r2:0.5")]
        pts += [(float(r), f"if:{s}:r{r}") for r in (2, 3, 4)]
    else:
        pts += [(4 * t, f"seg:{s}:start-end:{t}") for t in (0.125, 0.25, 0.5, 0.75)] + [(4.0, f"if:{s}:end")]
    P = np.array([[*pos[nm], zf(acc[ix[nm]])] for _, nm in pts])
    d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1) + 1e-6)]
    return PchipInterpolator(d, P)(np.linspace(0, d[-1], 200)), P
LIFT, LW, MS = E("LIFT", 0.012), E("LW", 1.5), E("MS", 26.0)
ends3 = []
for s in sorted(STARTS, key=lambda s: -pos[f"if:{s}:r0"][1] * np.sin(np.deg2rad(E("AZIM", -132.0))) - pos[f"if:{s}:r0"][0] * np.cos(np.deg2rad(E("AZIM", -132.0)))):
    C, P = path(s); C[:, 2] += LIFT
    c0_, c1_ = np.array(to_rgb(COL[s])), np.array(to_rgb(END_COL))
    frac = np.clip(np.linspace(0, 1, len(C) - 1) * E("CGS", 1.1), 0, 1) ** E("CG", 2.2)
    cols = c0_[None] * (1 - frac[:, None]) + c1_[None] * frac[:, None]
    seg = np.stack([C[:-1], C[1:]], 1)
    if E("SHADOW", 0):
        sh = seg.copy(); sh[..., 2] = ZFL + 0.002
        ax.add_collection3d(Line3DCollection(sh, colors=np.c_[cols, np.full(len(cols), E("SHA", 0.75))], linewidths=LW * 0.7, zorder=0.6, capstyle="round"))
        ax.scatter(P[0, 0], P[0, 1], ZFL + 0.003, s=MS * 0.35, color=COL[s], edgecolors="white", linewidths=0.4, depthshade=False, zorder=0.7)
    ax.add_collection3d(Line3DCollection(seg, colors="white", linewidths=LW + 1.3, zorder=5, capstyle="round"))
    ax.add_collection3d(Line3DCollection(seg, colors=cols, linewidths=LW, zorder=6, capstyle="round"))
    ax.scatter(*P[0, :2], P[0, 2] + LIFT, s=MS, color=COL[s], edgecolors="white", linewidths=0.9, depthshade=False, zorder=8)
    ends3.append(P[-1] + [0, 0, LIFT])
for p in ends3:
    ax.scatter(*p, s=MS * 0.62, color=END_COL, edgecolors="white", linewidths=0.7, depthshade=False, zorder=9)
pe_ = pos["experts"]; ze = zf(acc[ix["experts"]]) + LIFT
ax.scatter(pe_[0], pe_[1], ze, marker="*", s=E("STAR", 150.0), color="#f6c445", edgecolors="#8a6500", linewidths=0.6, depthshade=False, zorder=10)
MTL = [nm for nm in names if nm.startswith("joint:") and nm in pos]; MTL_COL = E("MTL_COL", "#2e8b57")
for nm in MTL:
    q = pos[nm]; ax.scatter(q[0], q[1], zf(acc[ix[nm]]) + LIFT, marker="D", s=E("MTLS", 34.0), color=MTL_COL, edgecolors="white", linewidths=0.8, depthshade=False, zorder=10)
    if E("SHADOW", 0): ax.scatter(q[0], q[1], ZFL + 0.003, marker="D", s=E("MTLS", 34.0) * 0.3, color=MTL_COL, edgecolors="white", linewidths=0.3, depthshade=False, zorder=0.7)
if E("SHADOW", 0): ax.scatter(pe_[0], pe_[1], ZFL + 0.003, marker="*", s=E("STAR", 150.0) * 0.3, color="#f6c445", edgecolors="#8a6500", linewidths=0.3, depthshade=False, zorder=0.7)
x0, x1, y0, y1 = gx.min(), gx.max(), gy.min(), gy.max()
ax.set_xlim(x0, x1); ax.set_ylim(y0, y1); ax.set_zlim(ZFL, ZT); ax.set_axis_off()
ax.set_box_aspect((x1 - x0, y1 - y0, E("ZASP", 0.36) * (x1 - x0) * (ZT - ZFL) / ZT), zoom=E("ZOOM", 1.0))
# vertical accuracy axis
fig.canvas.draw(); Mp = ax.get_proj()
p2 = lambda x, y, z: proj3d.proj_transform(x, y, z, Mp)[:2]
corners = [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]
axp = np.array(min(corners, key=lambda c: p2(c[0], c[1], ZFL)[0])) if E("ZCORNER", -1) < 0 else np.array(corners[E("ZCORNER", -1)])
stroke = [pe.withStroke(linewidth=2.0, foreground="white")]
if E("ZAXIS", 1):
    TK = [70, 80, 85, 88, 90]
    ax.plot([axp[0]] * 2, [axp[1]] * 2, [ZFL, ZT if INV else hz(91.2)], color=INK2, lw=0.8, zorder=3)
    for t in TK:
        u = p2(axp[0], axp[1], zf(t))
        ax.annotate("", xy=u, xytext=(-3.5, 0), textcoords="offset points", arrowprops=dict(arrowstyle="-", color=INK2, lw=0.8), zorder=3)
        ax.annotate(f"{t}", xy=u, xytext=(-5.5, 0), textcoords="offset points", ha="right", va="center", fontsize=6.5, color=INK2, zorder=3)
    ax.annotate("accuracy (%)", xy=p2(axp[0], axp[1], ZT if INV else hz(91.2)), xytext=(-12, 4), textcoords="offset points", ha="left", va="bottom", fontsize=7, color=INK2)
LAB = {"wa_plain": (-6, 2, "right", "center"), "ta": (-5, -7, "right", "center"), "ties": (4, -8, "left", "center"),
       "adamerging": (-3, -8, "center", "top"), "regmean": (-5, 5, "right", "bottom"), "regmeanpp": (-4, 6, "right", "bottom"),
       "tsv_m": (-6, -2, "right", "center"), "wudi": (7, -4, "left", "center"), "iso_c": (-6, -4, "right", "center"), "iso_cts": (5, -5, "left", "top")}
if E("DEBUG", 0):
    for cx, cy in ((x0, y0), (x1, y0), (x1, y1), (x0, y1)):
        print("corner", (cx == x1, cy == y1), ax.transData.transform(p2(cx, cy, 0)) / fig.dpi)
if INV:
    LAB = {"wa_plain": (8, 0, "left", "center"), "ta": (2, 7, "center", "bottom"), "ties": (-7, -3, "right", "center"),
           "adamerging": (7, 1, "left", "center"), "regmean": (-6, -5, "right", "top"), "regmeanpp": (-7, -3, "right", "center"),
           "tsv_m": (7, 3, "left", "center"), "wudi": (7, 3, "left", "center"), "iso_c": (3, 7, "left", "bottom"), "iso_cts": (-7, 3, "right", "center")}
if E("LABELS", 1):
    Dis = Z["dis"]; si = [ix[f"if:{s}:r0"] for s in STARTS]; ei = [ix[end_name(s)] for s in STARTS]
    d0 = Dis[np.ix_(si, si)].sum() / 90 * 100; d1 = Dis[np.ix_(ei, ei)].sum() / 90 * 100
    ea_ = acc[ei]; sa_ = acc[si]; em = np.array(ends3).mean(0)
    print(f"disagreement starts {d0:.1f} ends {d1:.1f}")
    lx, ly, dy_ = E("LGX", 0.165), E("LGY", 0.94), E("LGD", 0.062)
    la = fig.add_axes([0, 0, 1, 1]); la.set_axis_off(); la.set_xlim(0, 1); la.set_ylim(0, 1); la.patch.set_alpha(0)
    for k, s_ in enumerate(["wa_plain", "ties", "adamerging", "tsv_m", "iso_cts"]):
        la.scatter(lx + 0.011 * k, ly, s=15, color=COL[s_], edgecolors="white", linewidths=0.5, zorder=3 + k, clip_on=False)
    la.text(lx + 0.064, ly, E("LEGEND0", f"10 existing merges:  65.5\u201385.1% accuracy,  predictions disagree on {d0:.1f}% of images"), ha="left", va="center", fontsize=7.3, color=INK)
    la.plot([lx - 0.004, lx + 0.03], [ly - dy_] * 2, color=END_COL, lw=1.5, solid_capstyle="round")
    la.scatter(lx + 0.044, ly - dy_, s=15, color=END_COL, edgecolors="white", linewidths=0.5, zorder=5, clip_on=False)
    la.text(lx + 0.064, ly - dy_, E("LEGEND1", f"after IF-Merge:  86.9\u201388.7% accuracy,  predictions disagree on {d1:.1f}% of images"), ha="left", va="center", fontsize=7.3, color=END_COL)
    pc = [np.array(ax.transData.transform(p2(c[0], c[1], ZFL))) for c in corners]
    kb = int(np.argmin([q[1] for q in pc])); cand = [(kb - 1) % 4, (kb + 1) % 4]
    ko = cand[E("FEDGE", 0)]; tA, tB = (pc[ko], pc[kb]) if pc[ko][0] < pc[kb][0] else (pc[kb], pc[ko])
    ang = np.degrees(np.arctan2(tB[1] - tA[1], tB[0] - tA[0])); mid = tA + E("FPOS", 0.5) * (tB - tA)
    nrm = np.array([tB[1] - tA[1], -(tB[0] - tA[0])]); nrm = nrm / np.linalg.norm(nrm) * E("FOFF", 9.0) * fig.dpi / 72
    ax.annotate("prediction space (2-D embedding)", xy=mid + nrm, xycoords="figure pixels", rotation=ang, rotation_mode="anchor",
                ha="center", va="center", fontsize=6.8, color=INK2)
    for s in STARTS:
        nm = f"if:{s}:r0"; p = pos[nm]; dx, dy, ha, va = LAB[s]
        ax.annotate(f"{PRETTY[s]}", xy=p2(p[0], p[1], zf(acc[ix[nm]]) + LIFT), xytext=(dx, dy), textcoords="offset points",
                    ha=ha, va=va, fontsize=6.6, color=INK, zorder=20, path_effects=stroke)
    edx, edy, eha, eva = [(-9, -2, "right", "center") if INV else (7, 5, "left", "bottom")][0]
    ax.annotate("8 task experts", xy=p2(pe_[0], pe_[1], ze), xytext=(E("EXDX", float(edx)), E("EXDY", float(edy))), textcoords="offset points",
                ha=eha, va=eva, fontsize=7, color="#7a5a00", zorder=20, path_effects=stroke)
    if MTL:
        ma = np.mean([acc[ix[nm]] for nm in MTL]); qm = np.mean([pos[nm] for nm in MTL], 0)
        mdx, mdy, mha, mva = [(9, -4, "left", "center") if INV else (10, -12, "left", "top")][0]
        ax.annotate("multi-task training", xy=p2(qm[0], qm[1], zf(ma) + LIFT), xytext=(E("MTDX", float(mdx)), E("MTDY", float(mdy))), textcoords="offset points",
                    ha=mha, va=mva, fontsize=7, color="#1d5c39", zorder=20, path_effects=stroke)
OUT = os.path.join(os.path.dirname(D), "teaser.png")
fig.savefig(OUT, dpi=E("DPI", 300))
from PIL import Image, ImageChops, ImageDraw  # trim the margins; round the corners so the card also sits well on dark pages
im = Image.open(OUT).convert("RGB"); box = ImageChops.difference(im, Image.new("RGB", im.size, "white")).getbbox()
im = im.crop((max(box[0] - 40, 0), max(box[1] - 40, 0), min(box[2] + 40, im.width), min(box[3] + 40, im.height)))
mask = Image.new("L", im.size, 0); ImageDraw.Draw(mask).rounded_rectangle((0, 0, im.width - 1, im.height - 1), radius=28, fill=255)
im.putalpha(mask); im.save(OUT, optimize=True)
ea = [acc[ix[end_name(s)]] for s in STARTS]; sa = [acc[ix[f"if:{s}:r0"]] for s in STARTS]
print(f"explained {expl:.3f}; start acc {min(sa):.1f}-{max(sa):.1f}; end acc {min(ea):.1f}-{max(ea):.1f}; experts {acc[ix['experts']]:.1f}")
