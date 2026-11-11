"""Krok 28: model velkého skoku Cα (dokument oddíly 32 a 33.8). Generativní model p(Cα(t + Δ) | Cα(t), torze(t), sekvence, Δ)
pro Δ = 1–10 ns, naučený na dvojicích snímků ATLAS. Cíl pro hybrid: vylosovat polohy Cα o Δ dál, ty pak řízená MD
(oracle28.py, řízení 100 ps) přenese do systému ve vodě a pokračuje MD (orákulum ukázalo, že s pravým cílem to funguje).

Model: podmíněný flow matching. Start z = Cα(t) + σ·ε, cíl x1 = Cα(t + Δ) přeložený (Kabsch) na Cα(t), přímá cesta
x_τ = (1 − τ)·z + τ·x1, síť předpovídá rychlost x1 − z. Síť je invariantní v uzlech a hranách (vzdálenosti v x_τ,
v Cα(t) a křížové |x_τ,i − Cα(t)_j|, sekvenční odstup, typ rezidua, sin/cos φ, ψ, χ1, τ, log Δ) a ekvivariantní
na výstupu (rychlost = vážený součet rozdílových vektorů), takže nezávisí na otočení a posunutí. Hustá verze bez grafu
(proteiny do ~150 reziduí). Vzorkování: Euler v --steps krocích.

S --tors (výchozí) model zároveň losuje torze φ, ψ, χ1 v čase t + Δ: flow matching na toru (start a(t) + σ_a·ε,
cesta po kratším oblouku, rychlost wrap(a1 − z)); uzly dostávají sin/cos právě vzorkovaných torzí, hlava vel_a
předpovídá jejich rychlost. Orákulum ukázalo, že bez χ1 boční řetězce po skoku nenásledují (JS χ1 2×).
S --rot (výchozí) se χ1 neučí spojitě, ale jako rotamer (g+, t, g−): hlava rot dává pravděpodobnosti rotameru v t + Δ
(křížová entropie při všech τ), při vzorkování se rotamer vylosuje z posledního kroku, tedy podle vylosované struktury.
Důvod: spojitý tok z a(t) ± 0,2 rad ke dvěma symetrickým rotamerům ±120° má v průměru nulovou rychlost a změny
rotamerů téměř nevzorkuje (syntetický test: změny χ1 0,1 % proti skutečným 9–50 %). Nový rotamer = jeho střed
(60°, 180°, −60°), stejný rotamer = současný χ1.

    python jump28.py train --atlas public/atlas/cache --train-lists public/atlas/list_atlas_train_40-100_*.json \
        --val-lists public/atlas/list_atlas_val_40-100_*.json --lags-ns 1 2 5 10 --steps 20000 --out model_j28
    python jump28.py eval --model model_j28 --atlas public/atlas/cache --names 7jfl_C 6ono_C --lags-ns 1 10 \
        --out j28_eval.json

eval: pro dvojice (t, t + Δ) z ATLAS porovná S vylosovaných skoků se skutečným: energy score (správné skórovací
pravidlo, menší = lepší) proti dvěma základům (žádný pohyb; izotropní šum se správnou velikostí), velikost skoku,
korelaci pohyblivosti reziduí, geometrii (Cα–Cα sousedů 0,38 nm, srážky < 0,38 nm).

v4 (krok 29): model v1–v3 dělá skoky o 25–35 % menší, než jsou skutečné, i na proteinech z tréninku (poddisperze).
--prior mob: malá síť pohyblivosti (MobNet) předpovídá pro každé reziduum a Δ směrodatnou odchylku posunu Cα s_i
(Gaussova NLL na skutečném posunu, dobře podmíněná úloha bez kolapsu k průměru). Start toku pak není x0 + σ·ε, ale
x0 + k·s_i·ε, takže velikost a rozložení pohybu po řetězci má už start a tok doplní směr a souhru reziduí; log s_i
dostane i síť toku jako vstup. --warm <model v3>: převezme váhy výchozího modelu (vstup uzlů rozšířený o nulový
sloupec), nová je jen MobNet. eval/vzorkování --scale λ: vylosovaný posun x0 + λ·(x̂ − x0) (kalibrace bez tréninku).
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

N_TYPES = 32
N_RBF = 24
MAX_SEP = 32


# ---------------------------------------------------------------- data
def read_lists(files):
    names = []
    for f in files or []:
        names += [n for n in json.loads(Path(f).read_text()) if n not in names]
    return names


def load_reps(root, names, reps=None, t_max_ns=None):
    """Repliky ze cache; t_max_ns = jen snímky do tohoto času (doučení na krátké vlastní MD)."""
    out = []
    for name in names:
        d = Path(root) / name
        if not (d / "meta.json").exists():
            print(f"  {name}: v cache není, přeskakuji", flush=True)
            continue
        meta = json.loads((d / "meta.json").read_text())
        types = np.minimum(np.array(meta["types"], np.int64), N_TYPES - 1)
        dt = float(meta["dt_ps"])
        n = int(round(t_max_ns * 1000 / dt)) + 1 if t_max_ns else None
        for i, _ in enumerate(meta["reps"], 1):
            if reps and i not in reps:
                continue
            out.append(dict(name=f"{name}_R{i}", prot=name, types=types, dt=dt,
                            ang=np.load(d / f"R{i}_ang.npy", mmap_mode="r")[:n],
                            bb=np.load(d / f"R{i}_bb.npy", mmap_mode="r")[:n]))
    return out


def kabsch_onto(P, Q):
    """P (N, 3) přeložené rotací a posunem na Q (N, 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (P - pc) @ R.T + qc


def ang9(a):
    m = ~np.isnan(a)
    a0 = np.where(m, a, 0.0)
    return np.concatenate([np.sin(a0) * m, np.cos(a0) * m, m.astype(np.float32)], -1).astype(np.float32)


def pair_sample(p, lag_ns, t=None, rng=None):
    """(typy, Cα(t) vycentrované, Cα(t + Δ) přeložené na Cα(t), ang9(t)) z jedné repliky; None když je moc krátká."""
    lf = int(round(lag_ns * 1000 / p["dt"]))
    T = p["bb"].shape[0]
    if lf >= T:
        return None
    if t is None:
        t = int(rng.integers(0, T - lf))
    x0 = np.asarray(p["bb"][t][:, 1], np.float64)
    x1 = np.asarray(p["bb"][t + lf][:, 1], np.float64)
    x0 = x0 - x0.mean(0)
    x1 = kabsch_onto(x1, x0)
    a0 = np.asarray(p["ang"][t], np.float32)
    a1 = np.asarray(p["ang"][t + lf], np.float32)
    return p["types"], x0.astype(np.float32), x1.astype(np.float32), ang9(a0), a0, a1


def collate(samples, lags):
    B = len(samples)
    L = max(len(s[0]) for s in samples)
    types = np.zeros((B, L), np.int64)
    x0 = np.zeros((B, L, 3), np.float32)
    x1 = np.zeros((B, L, 3), np.float32)
    a9 = np.zeros((B, L, 9), np.float32)
    mask = np.zeros((B, L), np.float32)
    a0 = np.zeros((B, L, 3), np.float32)
    a1 = np.zeros((B, L, 3), np.float32)
    am = np.zeros((B, L, 3), np.float32)
    for b, (ty, p0, p1, a, r0, r1) in enumerate(samples):
        n = len(ty)
        types[b, :n], x0[b, :n], x1[b, :n], a9[b, :n], mask[b, :n] = ty, p0, p1, a, 1.0
        ok = ~(np.isnan(r0) | np.isnan(r1))
        a0[b, :n], a1[b, :n], am[b, :n] = np.where(ok, r0, 0.0), np.where(ok, r1, 0.0), ok
    rot1 = np.where(am[..., 2] > 0, np.floor(np.mod(a1[..., 2], 2 * np.pi) / (2 * np.pi / 3)), -1).clip(-1, 2)
    return dict(rot1=torch.from_numpy(rot1.astype(np.int64)), types=torch.from_numpy(types), x0=torch.from_numpy(x0), x1=torch.from_numpy(x1),
                a9=torch.from_numpy(a9), mask=torch.from_numpy(mask), lag=torch.tensor(lags, dtype=torch.float32),
                a0=torch.from_numpy(a0), a1=torch.from_numpy(a1), am=torch.from_numpy(am))


def batch(reps, B, lags_ns, rng):
    out, lg = [], []
    T_min = min(p["bb"].shape[0] for p in reps)
    usable = [l for l in lags_ns if any(int(round(l * 1000 / p["dt"])) < p["bb"].shape[0] for p in reps)]
    if not usable:
        raise SystemExit(f"žádný lag {lags_ns} se nevejde do dat ({T_min} snímků)")
    while len(out) < B:
        p = reps[int(rng.integers(len(reps)))]
        lag = float(rng.choice(usable))
        s = pair_sample(p, lag, rng=rng)
        if s is not None:
            out.append(s)
            lg.append(lag)
    return collate(out, lg)


# ---------------------------------------------------------------- síť
def pdist(a, b):
    """Vzdálenosti (B, L, L) bez torch.cdist (zpětný chod na MPS) a s konečným gradientem v nule."""
    return torch.sqrt(((a[:, :, None] - b[:, None]) ** 2).sum(-1) + 1e-8)


def rbf(d, lo=0.0, hi=2.4, n=N_RBF):
    c = torch.linspace(lo, hi, n, device=d.device)
    w = (hi - lo) / (n - 1)
    return torch.exp(-((d[..., None] - c) / w) ** 2)


def tfeat(x, n=8):
    f = torch.exp(torch.arange(n, device=x.device) * (math.log(1000.0) / max(n - 1, 1)))
    return torch.cat([torch.sin(x[..., None] * f), torch.cos(x[..., None] * f)], -1)


def wrap(a):
    return torch.atan2(torch.sin(a), torch.cos(a))


def mlp(i, h, o, n=2):
    layers, d = [], i
    for _ in range(n - 1):
        layers += [nn.Linear(d, h), nn.SiLU()]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, o))


class MobNet(nn.Module):
    """Pohyblivost reziduí: log směrodatné odchylky posunu Cα (nm na složku) o Δ z Cα(t), typů a torzí (B, L)."""

    def __init__(self, hidden=64, edge=32, layers=3):
        super().__init__()
        self.emb = nn.Embedding(N_TYPES, hidden)
        self.node_in = mlp(9 + 16, hidden, hidden)
        self.edge_in = mlp(N_RBF + MAX_SEP + 1, edge, edge)
        self.msg = nn.ModuleList([mlp(2 * hidden + edge, edge, edge) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(hidden + edge, hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.out = mlp(hidden, hidden, 1)

    def forward(self, x0, types, a9, mask, lag):
        B, L, _ = x0.shape
        mm = mask[:, :, None] * mask[:, None, :]
        idx = torch.arange(L, device=x0.device)
        sep = (idx[None, :] - idx[:, None]).abs().clamp(max=MAX_SEP)
        sep1 = torch.nn.functional.one_hot(sep, MAX_SEP + 1).float().expand(B, L, L, MAX_SEP + 1)
        e = self.edge_in(torch.cat([rbf(pdist(x0, x0)), sep1], -1))
        cond = tfeat(torch.log1p(lag), 8)[:, None, :].expand(B, L, 16)
        h = self.emb(types) + self.node_in(torch.cat([a9, cond], -1))
        n_eff = mask.sum(1).clamp(min=1)[:, None, None]
        for k in range(len(self.msg)):
            m = self.msg[k](torch.cat([h[:, :, None].expand(B, L, L, -1), h[:, None].expand(B, L, L, -1), e], -1))
            m = m * mm[..., None]
            e = e + m
            h = self.norm[k](h + self.upd[k](torch.cat([h, m.sum(2) / n_eff.sqrt()], -1)))
        return (self.out(h)[..., 0] - 2.0).clamp(-4.6, 0.0)  # s v (0,01; 1) nm, start kolem 0,14 nm


class JumpNet(nn.Module):
    def __init__(self, hidden=128, edge=64, layers=6, tors=False, rot=False, mob=False, mob_k=1.0):
        super().__init__()
        self.cfg = dict(hidden=hidden, edge=edge, layers=layers, tors=tors, rot=rot, mob=mob, mob_k=mob_k)
        self.tors = tors
        self.rot = tors and rot
        self.mob = MobNet() if mob else None
        self.mob_k = mob_k
        self.emb = nn.Embedding(N_TYPES, hidden)
        self.node_in = mlp(9 + 32 + (6 if tors else 0) + (1 if mob else 0), hidden, hidden)
        self.edge_in = mlp(3 * N_RBF + MAX_SEP + 1, edge, edge)
        self.msg = nn.ModuleList([mlp(2 * hidden + edge, edge, edge) for _ in range(layers)])
        self.upd = nn.ModuleList([mlp(hidden + edge, hidden, hidden) for _ in range(layers)])
        self.norm = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.vel_e = mlp(edge, edge, 3)
        self.vel_n = mlp(hidden, hidden, 1)
        if tors:
            self.vel_a = mlp(hidden, hidden, 3)
        if self.rot:
            self.rot_head = mlp(hidden, hidden, 3)

    def base_scale(self, x0, types, a9, mask, lag, sigma):
        """Směrodatná odchylka startu toku (B, L): k·s_i z MobNet, jinak σ; vrací (s, log s_i nebo None)."""
        if self.mob is None:
            return torch.full(x0.shape[:2], float(sigma), device=x0.device), None
        ls = self.mob(x0, types, a9, mask, lag)
        return self.mob_k * torch.exp(ls), ls

    def forward(self, xt, x0, types, a9, mask, tau, lag, at=None, am=None, ls=None):
        """Vrací rychlost Cα (B, L, 3); s tors (rychlost Cα, rychlost torzí (B, L, 3) pro torze at s maskou am,
        logity rotameru χ1 (B, L, 3) nebo None)."""
        B, L, _ = xt.shape
        mm = mask[:, :, None] * mask[:, None, :]
        dt_ = pdist(xt, xt)
        d0 = pdist(x0, x0)
        dx = pdist(xt, x0)
        idx = torch.arange(L, device=xt.device)
        sep = (idx[None, :] - idx[:, None]).abs().clamp(max=MAX_SEP)
        sep1 = torch.nn.functional.one_hot(sep, MAX_SEP + 1).float().expand(B, L, L, MAX_SEP + 1)
        e = self.edge_in(torch.cat([rbf(dt_), rbf(d0), rbf(dx), sep1], -1))
        cond = torch.cat([tfeat(tau, 8), tfeat(torch.log1p(lag), 8)], -1)[:, None, :].expand(B, L, 32)
        feats = [a9, cond]
        if self.tors:
            feats += [torch.sin(at) * am, torch.cos(at) * am]
        if self.mob is not None:
            feats += [(ls.detach() + 2.0)[..., None] * mask[..., None]]
        h = self.emb(types) + self.node_in(torch.cat(feats, -1))
        n_eff = mask.sum(1).clamp(min=1)[:, None, None]
        for k in range(len(self.msg)):
            m = self.msg[k](torch.cat([h[:, :, None].expand(B, L, L, -1), h[:, None].expand(B, L, L, -1), e], -1))
            m = m * mm[..., None]
            e = e + m
            h = self.norm[k](h + self.upd[k](torch.cat([h, m.sum(2) / n_eff.sqrt()], -1)))
        w = self.vel_e(e) * mm[..., None]
        rt = xt[:, :, None] - xt[:, None]
        r0 = x0[:, :, None] - x0[:, None]
        rx = xt[:, :, None] - x0[:, None]
        v = (w[..., 0:1] * rt + w[..., 1:2] * r0 + w[..., 2:3] * rx).sum(2) / n_eff
        v = v + self.vel_n(h) * (x0 - xt)
        if self.tors:
            return v * mask[..., None], self.vel_a(h) * am, (self.rot_head(h) if self.rot else None)
        return v * mask[..., None]


def mob_nll(ls, x0, x1, mask):
    """Gaussova NLL posunu Cα se směrodatnou odchylkou exp(ls) na složku (na reziduum, bez konstanty)."""
    d2 = ((x1 - x0) ** 2).sum(-1)
    return ((3 * ls + d2 / (2 * torch.exp(2 * ls))) * mask).sum() / mask.sum()


def fm_loss(net, b, sigma, dev, gen=None, sigma_a=0.2, w_a=0.05, w_r=0.02, w_m=0.05, parts=False):
    """Ztráta flow matching (nm² na složku); s torzemi + w_a · ztráta torzí (rad²), s rotamery + w_r · křížová
    entropie rotameru χ1, s MobNet + w_m · NLL pohyblivosti. parts=True vrací (celkem, Cα, torze, rotamer)."""
    x0, x1, mask = b["x0"].to(dev), b["x1"].to(dev), b["mask"].to(dev)
    B = x0.shape[0]
    ty, a9, lag = b["types"].to(dev), b["a9"].to(dev), b["lag"].to(dev)
    s, ls = net.base_scale(x0, ty, a9, mask, lag, sigma)
    z = x0 + s.detach()[..., None] * torch.randn(x0.shape, device=dev, generator=gen)
    tau = torch.rand(B, device=dev, generator=gen)
    xt = (1 - tau[:, None, None]) * z + tau[:, None, None] * x1
    lm = mob_nll(ls, x0, x1, mask) if ls is not None else None
    if not net.tors:
        v = net(xt, x0, ty, a9, mask, tau, lag, ls=ls)
        lx = (((v - (x1 - z)) ** 2).sum(-1) * mask).sum() / mask.sum() / 3
        tot = lx + w_m * lm if lm is not None else lx
        return (tot, lx, None, None, lm) if parts else tot
    a0, a1, am = b["a0"].to(dev), b["a1"].to(dev), b["am"].to(dev)
    if net.rot:  # χ1 jako rotamer, ne ve spojitém toku
        am = am * torch.tensor([1.0, 1.0, 0.0], device=dev)
    za = wrap(a0 + sigma_a * torch.randn(a0.shape, device=dev, generator=gen)) * am
    ua = wrap(a1 - za) * am
    at = wrap(za + tau[:, None, None] * ua) * am
    v, va, lg = net(xt, x0, ty, a9, mask, tau, lag, at, am, ls=ls)
    lx = (((v - (x1 - z)) ** 2).sum(-1) * mask).sum() / mask.sum() / 3
    la = (((va - ua) ** 2) * am).sum() / am.sum().clamp(min=1)
    tot = lx + w_a * la
    if lm is not None:
        tot = tot + w_m * lm
    lr = None
    if lg is not None:
        r1 = b["rot1"].to(dev)
        lr = nn.functional.cross_entropy(lg.reshape(-1, 3), r1.reshape(-1), ignore_index=-1)
        tot = tot + w_r * lr
    return (tot, lx, la, lr, lm) if parts else tot


@torch.no_grad()
def sample(net, x0, types, a9, mask, lag, sigma, steps, gen=None, a0=None, am=None, sigma_a=0.2, scale=1.0):
    """x0 (B, L, 3) vycentrované; vrací vylosované Cα(t + Δ) (B, L, 3), s torzemi (Cα, torze (B, L, 3)).
    scale λ ≠ 1: posun Cα x0 + λ·(x̂ − x0)."""
    dev = x0.device
    B = x0.shape[0]
    s, ls = net.base_scale(x0, types, a9, mask, lag, sigma)
    x = (x0 + s[..., None] * torch.randn(x0.shape, device=dev, generator=gen)) * mask[..., None]
    sc = lambda y: (x0 + scale * (y - x0)) * mask[..., None] if scale != 1.0 else y
    if not net.tors:
        for k in range(steps):
            tau = torch.full((B,), k / steps, device=dev)
            x = x + net(x, x0, types, a9, mask, tau, lag, ls=ls) / steps
        return sc(x)
    amf = am * torch.tensor([1.0, 1.0, 0.0], device=dev) if net.rot else am
    a = wrap(a0 + sigma_a * torch.randn(a0.shape, device=dev, generator=gen)) * amf
    for k in range(steps):
        tau = torch.full((B,), k / steps, device=dev)
        v, va, lg = net(x, x0, types, a9, mask, tau, lag, a, amf, ls=ls)
        x = x + v / steps
        a = wrap(a + va / steps) * amf
    if net.rot:  # rotamer χ1 podle vylosované struktury (poslední krok)
        _, _, lg = net(x, x0, types, a9, mask, torch.full((B,), 1.0 - 0.5 / steps, device=dev), lag, a, amf, ls=ls)
        r = torch.multinomial(torch.softmax(lg.reshape(-1, 3).float(), -1), 1).reshape(lg.shape[:2])
        r0 = torch.floor(torch.remainder(a0[..., 2], 2 * math.pi) / (2 * math.pi / 3)).clamp(0, 2).long()
        centre = torch.tensor([math.radians(60.0), math.pi, math.radians(-60.0)], device=dev)[r]
        a = a.clone()
        a[..., 2] = torch.where(r == r0, a0[..., 2], centre) * am[..., 2]
    return sc(x), a


def pick_device(name):
    if name:
        return torch.device(name)
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def save_model(net, out, extra):
    out.mkdir(parents=True, exist_ok=True)
    tmp = out / "jump28.pt.tmp"
    torch.save(dict(state=net.state_dict(), cfg=net.cfg, **extra), tmp)
    tmp.replace(out / "jump28.pt")


def load_jump(model_dir, dev):
    ck = torch.load(Path(model_dir) / "jump28.pt", map_location=dev, weights_only=False)
    net = JumpNet(**ck["cfg"]).to(dev)
    ck.setdefault("sigma_a", 0.2)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


def fix_ca_bonds(x, ref, iters=30, max_ref=0.45):
    """Sousední Cα v x vrátí na délky z ref (současná struktura), mimo přerušení řetězce (ref > max_ref nm)."""
    x = np.array(x, np.float64)
    d_ref = np.linalg.norm(ref[1:] - ref[:-1], axis=-1)
    ok = d_ref < max_ref
    for _ in range(iters):
        v = x[1:] - x[:-1]
        d = np.linalg.norm(v, axis=-1)
        corr = np.where(ok, (d - d_ref) / np.maximum(d, 1e-6), 0.0)[:, None] * v * 0.5
        x[:-1] += corr
        x[1:] -= corr
    return x


def sample_jump(net, ck, x0_np, types_np, ang_np, lag_ns, dev, n=1, steps=None, seed=0, scale=1.0):
    """Pro hybrid: Cα (L, 3) v nm, typy (L,), úhly (L, 3) → (n vylosovaných Cα (n, L, 3) přeložených na x0,
    torze (n, L, 3) s NaN kde úhel není, nebo None u modelu jen Cα)."""
    c = x0_np.mean(0)
    ang_np = np.asarray(ang_np, np.float32)
    x0 = torch.tensor(np.repeat((x0_np - c)[None], n, 0), dtype=torch.float32, device=dev)
    ty = torch.tensor(np.repeat(np.minimum(types_np, N_TYPES - 1)[None], n, 0), device=dev)
    a9 = torch.tensor(np.repeat(ang9(ang_np)[None], n, 0), device=dev)
    mask = torch.ones(x0.shape[:2], device=dev)
    lag = torch.full((n,), float(lag_ns), device=dev)
    torch.manual_seed(seed)
    ok = ~np.isnan(ang_np)
    a0 = torch.tensor(np.repeat(np.where(ok, ang_np, 0.0)[None], n, 0), dtype=torch.float32, device=dev)
    am = torch.tensor(np.repeat(ok[None], n, 0), dtype=torch.float32, device=dev)
    out = sample(net, x0, ty, a9, mask, lag, ck["sigma"], steps or ck["sample_steps"], a0=a0, am=am,
                 sigma_a=ck["sigma_a"], scale=scale)
    xs, ang = (out if net.tors else (out, None))
    xs = xs.cpu().numpy().astype(np.float64)
    xs = np.stack([kabsch_onto(x, x0_np - c) + c for x in xs])
    if ang is not None:
        ang = np.where(ok[None], ang.cpu().numpy().astype(np.float64), np.nan)
    return xs, ang


# ---------------------------------------------------------------- trénink
def train(args):
    dev = pick_device(args.device)
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    tr = load_reps(args.atlas, args.names or read_lists(args.train_lists), args.reps, args.t_max_ns)
    va = load_reps(args.atlas, read_lists(args.val_lists)) if args.val_lists else []
    if args.max_len:
        tr = [p for p in tr if len(p["types"]) <= args.max_len]
        va = [p for p in va if len(p["types"]) <= args.max_len]
    print(f"trénink {len(tr)} replik ({len({p['prot'] for p in tr})} proteinů), validace {len(va)}; zařízení {dev}; "
          f"lagy {args.lags_ns} ns, σ {args.sigma} nm", flush=True)
    if args.init:  # doučení: architektura a σ z výchozího modelu
        net, ck0 = load_jump(args.init, dev)
        net.train()
        args.sigma, args.sigma_a = ck0["sigma"], ck0["sigma_a"]
        print(f"doučení z {args.init} (krok {ck0.get('step')}, val {ck0.get('val')}), σ {args.sigma}", flush=True)
    else:
        net = JumpNet(args.hidden, args.edge, args.layers, tors=args.tors, rot=args.rot,
                      mob=args.prior == "mob", mob_k=args.mob_k).to(dev)
        if args.warm:  # převzít váhy modelu v1–v3; vstup uzlů rozšířený o nulové sloupce
            ckw = torch.load(Path(args.warm) / "jump28.pt", map_location=dev, weights_only=False)
            own = net.state_dict()
            n_copy, n_pad = 0, 0
            for k, v in ckw["state"].items():
                if k not in own:
                    continue
                if own[k].shape == v.shape:
                    own[k] = v
                    n_copy += 1
                elif v.dim() == 2 and own[k].shape[0] == v.shape[0] and own[k].shape[1] > v.shape[1]:
                    w = torch.zeros_like(own[k])
                    w[:, :v.shape[1]] = v
                    own[k] = w
                    n_pad += 1
            net.load_state_dict(own)
            print(f"teplý start z {args.warm}: převzato {n_copy} tenzorů, rozšířeno {n_pad}", flush=True)
    print(f"parametrů {sum(p.numel() for p in net.parameters()) / 1e6:.2f} M", flush=True)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    fl = dict(sigma_a=args.sigma_a, w_a=args.ang_weight, w_r=args.rot_weight, w_m=args.mob_weight)
    if args.time_budget_min:  # změřit rychlost kroku a počet kroků přizpůsobit času
        ts = None
        for i in range(18):
            if i == 3:
                if dev.type == "mps":
                    torch.mps.synchronize()
                ts = time.time()
            fm_loss(net, batch(tr, args.batch, args.lags_ns, rng), args.sigma, dev, **fl).backward()
            opt.zero_grad()
        if dev.type == "mps":
            torch.mps.synchronize()
        sps = (time.time() - ts) / 15
        n = int(args.time_budget_min * 60 * 0.85 / sps)
        print(f"rychlost {sps:.3f} s/krok; rozpočet {args.time_budget_min:g} min → {min(n, args.steps)} kroků "
              f"(strop {args.steps})", flush=True)
        args.steps = max(200, min(n, args.steps))
        args.log_every = min(args.log_every, max(50, args.steps // 20))
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    vrng = np.random.default_rng(12345)
    vbatches = [batch(va, args.batch, args.lags_ns, vrng) for _ in range(args.val_batches)] if va else []
    out = Path(args.out)
    best, hist, t0, run = float("inf"), [], time.time(), 0.0
    for step in range(1, args.steps + 1):
        net.train()
        loss = fm_loss(net, batch(tr, args.batch, args.lags_ns, rng), args.sigma, dev, **fl)
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        lv = loss.item()
        run = 0.98 * run + 0.02 * lv if step > 1 else lv
        if step % args.log_every == 0 or step == args.steps:
            net.eval()
            torch.manual_seed(7)  # stejný šum pro validaci v každém hodnocení
            with torch.no_grad():
                vp = [[float(x) if x is not None else float("nan") for x in fm_loss(net, vb, args.sigma, dev, parts=True, **fl)]
                      for vb in vbatches]
            vl, vx, va_, vr, vm = (float(np.mean([r[i] for r in vp])) for i in range(5)) if vp else \
                (run, np.nan, np.nan, np.nan, np.nan)
            torch.manual_seed(args.seed + step)
            hist.append(dict(step=step, train=round(run, 6), val=round(vl, 6), val_ca=round(vx, 6),
                             val_ang=round(va_, 5), val_rot=round(vr, 5), val_mob=round(vm, 4),
                             s=round(time.time() - t0, 1)))
            print(f"  krok {step}: trénink {run:.5f}, " + (f"validace {vl:.5f} (Cα {vx:.5f} nm², torze {va_:.4f} rad², "
                  f"rotamer {vr:.4f}" + (f", pohyblivost NLL {vm:.3f}" if vm == vm else "") + ")" if vp else
                  "bez validace") + f", {time.time() - t0:.0f} s", flush=True)
            if vl < best:
                best = vl
                save_model(net, out, dict(sigma=args.sigma, sigma_a=args.sigma_a, sample_steps=args.sample_steps,
                                          lags_ns=args.lags_ns, step=step, val=vl, args=vars(args)))
    (out / "train28.json").write_text(json.dumps(dict(args=vars(args), best_val=best, hist=hist), indent=1))
    print(f"Hotovo, nejlepší validace {best:.5f}, uloženo {out}", flush=True)


# ---------------------------------------------------------------- hodnocení
def rmsd(a, b, m):
    return np.sqrt((((a - b) ** 2).sum(-1) * m).sum(-1) / m.sum(-1))


def geometry(X, m):
    """X (S, L, 3): odchylka sousedních Cα od 0,38 nm a podíl srážek (|i − j| ≥ 3, < 0,38 nm)."""
    n = int(m.sum())
    X = X[:, :n]
    nb = np.linalg.norm(X[:, 1:] - X[:, :-1], axis=-1)
    D = np.linalg.norm(X[:, :, None] - X[:, None], axis=-1)
    iu = np.triu_indices(n, 3)
    return float(np.abs(nb - 0.38).mean()), float((D[:, iu[0], iu[1]] < 0.38).mean())


def bb_state(a):
    """Hrubý stav páteře: 0 α (ψ v (−120°, 50°)), 1 β/PPII, 2 aL (φ > 0); −1 chybí."""
    phi, psi = np.degrees(a[..., 0]), np.degrees(a[..., 1])
    st = np.where(phi > 0, 2, np.where((psi > -120) & (psi < 50), 0, 1))
    return np.where(np.isnan(phi) | np.isnan(psi), -1, st)


def rot_state(a):
    """Rotamer χ1: 0 g+ (0–120°), 1 t (120–240°), 2 g− (240–360°); −1 chybí."""
    c = np.degrees(a[..., 2]) % 360.0
    return np.where(np.isnan(c), -1, np.nan_to_num(c // 120).astype(np.int64))


def ang_metrics(A, a0, a1):
    """A (S, L, 3) vylosované torze, a0/a1 (L, 3) skutečné v t a t + Δ: shoda stavů s t + Δ (model, „bez pohybu“),
    u reziduí, která stav změnila, podíl vzorků, které trefí nový stav, a podíl změn v modelu proti skutečnosti."""
    out = {}
    for key, fn in (("bb", bb_state), ("chi", rot_state)):
        s0, s1, sg = fn(a0), fn(a1), fn(A)
        m = (s0 >= 0) & (s1 >= 0)
        if not m.any():
            continue
        ch = m & (s0 != s1)
        out[f"{key}_acc"] = float((sg[:, m] == s1[m]).mean())
        out[f"{key}_acc_stay"] = float((s0[m] == s1[m]).mean())
        out[f"{key}_chg_true"] = float(ch.sum() / m.sum())
        out[f"{key}_chg_model"] = float((sg[:, m] != s0[m]).mean())
        out[f"{key}_chg_hit"] = float((sg[:, ch] == s1[ch]).mean()) if ch.any() else float("nan")
    d = np.abs(np.arctan2(np.sin(A - a1[None]), np.cos(A - a1[None])))
    d0 = np.abs(np.arctan2(np.sin(a0 - a1), np.cos(a0 - a1)))
    out["phipsi_err_deg"] = float(np.degrees(np.nanmean(d[..., :2])))
    out["phipsi_err_stay_deg"] = float(np.degrees(np.nanmean(d0[..., :2])))
    out["chi_err_deg"] = float(np.degrees(np.nanmean(d[..., 2])))
    out["chi_err_stay_deg"] = float(np.degrees(np.nanmean(d0[..., 2])))
    return out


def evaluate(args):
    dev = pick_device(args.device)
    net, ck = load_jump(args.model, dev)
    names = args.names or read_lists(args.lists)
    rng = np.random.default_rng(args.seed)
    report = dict(model=args.model, sigma=ck["sigma"], steps=args.sample_steps or ck["sample_steps"], scale=args.scale,
                  proteins={})
    for name in names:
        reps = load_reps(args.atlas, [name], args.reps)
        if not reps:
            continue
        res = {}
        for lag in args.lags_ns:
            rows = []
            for _ in range(args.n_pairs):
                p = reps[int(rng.integers(len(reps)))]
                s = pair_sample(p, lag, rng=rng)
                if s is None:
                    continue
                ty, x0, y, a, r0, r1 = s
                S = args.n_samples
                b = collate([s] * S, [lag] * S)
                torch.manual_seed(int(rng.integers(1 << 30)))
                ok0 = torch.from_numpy(~np.isnan(np.repeat(r0[None], S, 0))).float().to(dev)
                X = sample(net, b["x0"].to(dev), b["types"].to(dev), b["a9"].to(dev), b["mask"].to(dev),
                           b["lag"].to(dev), ck["sigma"], args.sample_steps or ck["sample_steps"],
                           a0=torch.nan_to_num(torch.from_numpy(np.repeat(r0[None], S, 0))).to(dev), am=ok0,
                           sigma_a=ck["sigma_a"], scale=args.scale)
                A = None
                if net.tors:
                    X, A = X
                    A = np.where(np.isnan(r0)[None], np.nan, A.cpu().numpy().astype(np.float64))
                X = X.cpu().numpy()
                X = np.stack([kabsch_onto(x.astype(np.float64), x0.astype(np.float64)) for x in X])
                m = np.ones(len(ty))
                L = len(ty)
                true_r = float(rmsd(y, x0, m))
                gen_r = rmsd(X, x0[None], m[None])
                es = float(rmsd(X, y[None], m[None]).mean() - 0.5 * np.mean(
                    [rmsd(X[i], X[j], m) for i in range(S) for j in range(S) if i != j]))
                # izotropní základ se stejnou střední velikostí skoku (skutečnou, tedy zvýhodněný)
                iso = x0[None] + rng.normal(0, true_r / math.sqrt(3), (S, L, 3))
                es_iso = float(rmsd(iso, y[None], m[None]).mean() - 0.5 * np.mean(
                    [rmsd(iso[i], iso[j], m) for i in range(S) for j in range(S) if i != j]))
                mob_t = np.linalg.norm(y - x0, axis=-1)
                mob_g = np.linalg.norm(X - x0[None], axis=-1).mean(0)
                corr = float(np.corrcoef(mob_t, mob_g)[0, 1]) if mob_t.std() > 0 and mob_g.std() > 0 else float("nan")
                nb_dev, clash = geometry(X, m)
                nb_t, clash_t = geometry(y[None], m)
                row = dict(true_rmsd=true_r, gen_rmsd=float(gen_r.mean()), gen_rmsd_sd=float(gen_r.std()),
                           es=es, es_stay=true_r, es_iso=es_iso, mob_corr=corr, nb_dev=nb_dev,
                           clash=clash, nb_dev_true=nb_t, clash_true=clash_t,
                           gen_to_true=float(rmsd(X, y[None], m[None]).mean()))
                if A is not None:
                    row.update(ang_metrics(A, r0.astype(np.float64), r1.astype(np.float64)))
                if net.mob is not None:  # předpovězená pohyblivost: střední velikost skoku a korelace po reziduích
                    with torch.no_grad():
                        sp = torch.exp(net.mob(b["x0"][:1].to(dev), b["types"][:1].to(dev), b["a9"][:1].to(dev),
                                               b["mask"][:1].to(dev), b["lag"][:1].to(dev)))[0].cpu().numpy()
                    row["mob_pred_rmsd"] = float(np.sqrt(3 * (sp ** 2).mean()))
                    row["mob_pred_corr"] = float(np.corrcoef(mob_t, sp)[0, 1]) if mob_t.std() > 0 and sp.std() > 0 \
                        else float("nan")
                rows.append(row)
            keys = sorted({k for r in rows for k in r})
            mean = {k: round(float(np.nanmean([r.get(k, np.nan) for r in rows])), 4) for k in keys} if rows else {}
            res[f"{lag:g}ns"] = dict(n=len(rows), mean=mean)
            print(f"{name} Δ {lag:g} ns ({len(rows)} dvojic): skok skutečný {mean.get('true_rmsd')} / model "
                  f"{mean.get('gen_rmsd')} ± {mean.get('gen_rmsd_sd')} nm; energy score model {mean.get('es')}, "
                  f"bez pohybu {mean.get('es_stay')}, izotropní {mean.get('es_iso')}; korelace pohyblivosti "
                  f"{mean.get('mob_corr')}; Cα–Cα odchylka {mean.get('nb_dev')} (ATLAS {mean.get('nb_dev_true')}), "
                  f"srážky {mean.get('clash')} (ATLAS {mean.get('clash_true')})"
                  + (f"; pohyblivost předpověď {mean.get('mob_pred_rmsd')} nm, korelace {mean.get('mob_pred_corr')}"
                     if "mob_pred_rmsd" in mean else ""), flush=True)
            if "chi_acc" in mean:
                print(f"    torze: stav páteře shoda model {mean['bb_acc']} / bez pohybu {mean['bb_acc_stay']}, změny "
                      f"skutečné {mean['bb_chg_true']} / model {mean['bb_chg_model']}, trefa nového {mean['bb_chg_hit']}; "
                      f"rotamer χ1 shoda {mean['chi_acc']} / {mean['chi_acc_stay']}, změny {mean['chi_chg_true']} / "
                      f"{mean['chi_chg_model']}, trefa {mean['chi_chg_hit']}; chyba φψ {mean['phipsi_err_deg']}° / "
                      f"{mean['phipsi_err_stay_deg']}°, χ1 {mean['chi_err_deg']}° / {mean['chi_err_stay_deg']}°",
                      flush=True)
        report["proteins"][name] = res
    if args.out:
        Path(args.out).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print("Uloženo", args.out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--atlas", default="public/atlas/cache")
    t.add_argument("--train-lists", nargs="*")
    t.add_argument("--val-lists", nargs="*")
    t.add_argument("--names", nargs="*", help="místo seznamů")
    t.add_argument("--max-len", type=int, default=None)
    t.add_argument("--reps", type=int, nargs="*", default=None, help="jen tyto repliky (např. 1)")
    t.add_argument("--t-max-ns", type=float, default=None, help="jen snímky do tohoto času (vlastní krátká MD)")
    t.add_argument("--init", default=None, help="doučit existující model (adresář)")
    t.add_argument("--lags-ns", type=float, nargs="+", default=[1, 2, 5, 10])
    t.add_argument("--sigma", type=float, default=0.1)
    t.add_argument("--tors", action=argparse.BooleanOptionalAction, default=True, help="losovat i φ, ψ, χ1")
    t.add_argument("--sigma-a", type=float, default=0.2, help="šum startu torzí (rad)")
    t.add_argument("--ang-weight", type=float, default=0.05, help="váha ztráty torzí (rad²) proti Cα (nm²)")
    t.add_argument("--rot", action=argparse.BooleanOptionalAction, default=True, help="χ1 jako rotamer (klasifikace)")
    t.add_argument("--rot-weight", type=float, default=0.02, help="váha křížové entropie rotameru")
    t.add_argument("--prior", choices=["iso", "mob"], default="iso", help="start toku: x0 + σ·ε, nebo x0 + k·s_i·ε (MobNet)")
    t.add_argument("--mob-k", type=float, default=1.0, help="--prior mob: násobek s_i ve startu toku")
    t.add_argument("--mob-weight", type=float, default=0.05, help="váha NLL pohyblivosti")
    t.add_argument("--warm", default=None, help="nový model (např. --prior mob) s převzatými vahami modelu v1–v3")
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--edge", type=int, default=64)
    t.add_argument("--layers", type=int, default=6)
    t.add_argument("--steps", type=int, default=20000)
    t.add_argument("--time-budget-min", type=float, default=None, help="počet kroků podle změřené rychlosti (strop --steps)")
    t.add_argument("--batch", type=int, default=16)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--val-batches", type=int, default=8)
    t.add_argument("--log-every", type=int, default=500)
    t.add_argument("--sample-steps", type=int, default=20)
    t.add_argument("--device", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="model_j28")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--atlas", default="public/atlas/cache")
    e.add_argument("--names", nargs="*")
    e.add_argument("--lists", nargs="*")
    e.add_argument("--reps", type=int, nargs="*", default=None)
    e.add_argument("--lags-ns", type=float, nargs="+", default=[1, 10])
    e.add_argument("--n-pairs", type=int, default=40)
    e.add_argument("--n-samples", type=int, default=8)
    e.add_argument("--sample-steps", type=int, default=None)
    e.add_argument("--scale", type=float, default=1.0, help="posun x0 + λ·(x̂ − x0)")
    e.add_argument("--device", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
