"""Model Q: kolektivní posun Cα za τ jako cíl skoku v polohách (hybrid.py --pos-jump model).

Torzní skok P1 celkový tvar proteinu neposune a náhodný posun podél módů ANM ho posune jiným směrem než MD
(krok 17). Model Q se učí z ATLAS rozdělení posunu Cα Δ = Cα(t + τ) − Cα(t), kde Cα(t + τ) je přeložené
na Cα(t) (bez pohybu celku). Rozdělení je normální s nízkým rankem:
    Δ ~ N(μ, U Uᵀ + diag σ²),  U = K kolektivních módů (3L × K),
takže vzorek pohne mnoha rezidui naráz koordinovaně. μ, módy i σ předpovídá transformer nad rezidui
v lokálních rámcích reziduí (N, CA, C) a otočí je do globálních souřadnic, výsledek je tedy nezávislý
na natočení proteinu. Vstupy: typ rezidua, φ/ψ/χ1, log τ, volitelně předchozí posun Cα(t) − Cα(t − τ)
(--hist-disp, nese směr pomalého driftu); dvojice reziduí: vzdálenost Cα (RBF), směr v rámci rezidua i
a odstup v sekvenci (bias pozornosti).

    python model_q.py train --atlas public/atlas/cache --train-lists public/atlas/list_atlas_train_*.json \
        --val-lists public/atlas/list_atlas_val_*.json --out model_q1
    python model_q.py eval --model model_q1 --atlas public/atlas/cache \
        --lists public/atlas/list_atlas_test_*.json --out q1_eval_test.json

--proj M (Model Q2): věrohodnost jen průmětu posunu na M nejměkčích módů ANM ze struktury v čase t (jako hybrid
--pos-proj M). Rychlý tepelný šum mimo ně se neučí a kapacita jde do pomalé, fyzikálně dosažitelné části;
v hybridu pak --pos-proj se stejným M.
"""
import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import scipy.linalg
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_c import AA

NTYPE = len(AA) + 1


def read_lists(files):
    names = []
    for f in files or []:
        names += [n for n in json.loads(Path(f).read_text()) if n not in names]
    return names


def load_atlas(root, names):
    """Repliky jako slovníky s mmap poli ang (T, L, 3) a bb (T, L, 3, 3); chybějící proteiny přeskočí."""
    out = []
    for name in names:
        d = Path(root) / name
        if not (d / "meta.json").exists():
            print(f"  {name}: v cache není, přeskakuji", flush=True)
            continue
        meta = json.loads((d / "meta.json").read_text())
        for i, _ in enumerate(meta["reps"], 1):
            out.append(dict(name=f"{name}_R{i}", prot=name, types=np.array(meta["types"], np.int64),
                            dt=float(meta["dt_ps"]), ang=np.load(d / f"R{i}_ang.npy", mmap_mode="r"),
                            bb=np.load(d / f"R{i}_bb.npy", mmap_mode="r")))
    return out


def kabsch_onto(Y, X):
    """Y, X (B, n, 3) → Y optimálně přeložené na X (po snímcích)."""
    yc, xc = Y.mean(1, keepdims=True), X.mean(1, keepdims=True)
    H = np.einsum("bni,bnj->bij", Y - yc, X - xc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(Y), 3))
    D[:, 2] = d
    R = np.einsum("bij,bj,bjk->bik", U, D, Vt)
    return np.einsum("bni,bij->bnj", Y - yc, R) + xc


def anm_basis(X, n_modes, cutoff=1.5, k_seq=10.0):
    """Nejměkčí módy elastické sítě Cα jako hybrid.anm_modes (pružiny do cutoff nm, sousedé v řetězci k_seq× tužší):
    X (L, 3) → ortonormální V (3L, n_modes) bez 6 tuhých pohybů."""
    L = len(X)
    d = X[None, :, :] - X[:, None, :]
    r2 = (d ** 2).sum(-1)
    on = (r2 < cutoff ** 2) & ~np.eye(L, dtype=bool)
    w = np.where(on, 1.0 / np.where(on, r2, 1.0), 0.0)
    i = np.arange(L - 1)
    w[i, i + 1] *= k_seq
    w[i + 1, i] *= k_seq
    B = -w[..., None, None] * d[..., :, None] * d[..., None, :]
    B[np.arange(L), np.arange(L)] = -B.sum(1)
    return scipy.linalg.eigh(B.transpose(0, 2, 1, 3).reshape(3 * L, 3 * L), subset_by_index=[0, n_modes + 5])[1][:, 6:]


def bases(bb, proj):
    """Módy ANM pro každý snímek bb (b, L, 3, 3) → (b, 3L, proj) float32, nebo None při proj = 0."""
    if not proj:
        return None
    return np.stack([anm_basis(x, proj) for x in np.asarray(bb)[:, :, 1].astype(np.float64)]).astype(np.float32)


def sample_rep(p, b, lags_ps, rng, hist_disp):
    """b vzorků z repliky: types (L,), bb(t) (b, L, 3, 3), úhly (b, L, 3), lag (b,), Δ (b, L, 3), předchozí posun."""
    T = p["bb"].shape[0]
    lags = [g for g in lags_ps if abs(g / p["dt"] - round(g / p["dt"])) < 1e-6 and 2 * g / p["dt"] < T]
    lag = rng.choice(lags, size=b).astype(np.float64)
    lf = np.round(lag / p["dt"]).astype(np.int64)
    t = rng.integers(lf if hist_disp else 0, T - lf)
    f32 = lambda x: np.asarray(x, np.float32)
    bb = f32(p["bb"][t])
    X = bb[:, :, 1].astype(np.float64)
    D = kabsch_onto(f32(p["bb"][t + lf])[:, :, 1].astype(np.float64), X) - X
    prev = (X - kabsch_onto(f32(p["bb"][t - lf])[:, :, 1].astype(np.float64), X)) if hist_disp else None
    return p["types"], bb, f32(p["ang"][t]), lag, D.astype(np.float32), None if prev is None else prev.astype(np.float32)


def frames(bb):
    """bb (..., L, 3, 3) N, CA, C → rámce reziduí R (..., L, 3, 3) se sloupci e1, e2, e3 a Cα (..., L, 3)."""
    n, ca, c = bb[..., 0, :], bb[..., 1, :], bb[..., 2, :]
    e1 = F.normalize(c - ca, dim=-1)
    u = n - ca
    e2 = F.normalize(u - (u * e1).sum(-1, keepdim=True) * e1, dim=-1)
    return torch.stack([e1, e2, torch.cross(e1, e2, dim=-1)], -1), ca


class Layer(nn.Module):
    def __init__(self, d, heads):
        super().__init__()
        self.h, self.dh = heads, d // heads
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.o = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.ff = nn.Sequential(nn.Linear(d, 4 * d), nn.GELU(), nn.Linear(4 * d, d))

    def forward(self, x, bias):
        B, L, d = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, L, 3, self.h, self.dh).permute(2, 0, 3, 1, 4)
        a = torch.softmax(q @ k.transpose(-1, -2) / math.sqrt(self.dh) + bias, -1)
        x = x + self.o((a @ v).transpose(1, 2).reshape(B, L, d))
        return x + self.ff(self.ln2(x))


class NetQ(nn.Module):
    def __init__(self, d=128, layers=4, heads=8, K=16, n_rbf=16, max_sep=32, hist_disp=True, scale=0.1):
        super().__init__()
        self.cfg = dict(d=d, layers=layers, heads=heads, K=K, n_rbf=n_rbf, max_sep=max_sep, hist_disp=hist_disp,
                        scale=scale)
        self.K, self.max_sep, self.hist_disp, self.scale = K, max_sep, hist_disp, scale
        self.register_buffer("mu_rbf", torch.linspace(0.3, 2.0, n_rbf))
        self.emb = nn.Embedding(NTYPE, d)
        self.inp = nn.Linear(6 + 3 + 1 + (3 if hist_disp else 0), d)
        self.sep = nn.Embedding(2 * max_sep + 2, 32)
        self.pair = nn.Sequential(nn.Linear(n_rbf + 3 + 32, 64), nn.GELU(), nn.Linear(64, heads * layers))
        self.layers = nn.ModuleList([Layer(d, heads) for _ in range(layers)])
        self.ln = nn.LayerNorm(d)
        self.head = nn.Linear(d, 3 + 3 * K + 1)
        with torch.no_grad():  # malý start; módy ne nulové (gradient podle U je v U = 0 nulový)
            self.head.weight.mul_(0.1)
            self.head.bias.zero_()

    def forward(self, types, bb, ang, loglag, prev=None):
        """types (B, L), bb (B, L, 3, 3), ang (B, L, 3), loglag (B,), prev (B, L, 3) → μ (B, 3L), U (B, 3L, K), σ² (B, 3L)."""
        B, L = types.shape
        R, x = frames(bb)
        ok = ~torch.isnan(ang)
        a = torch.where(ok, ang, torch.zeros_like(ang))
        f = [torch.sin(a), torch.cos(a), ok.float(), loglag[:, None, None].expand(B, L, 1)]
        if self.hist_disp:
            prev = torch.zeros_like(x) if prev is None else prev
            f.append(torch.einsum("blji,blj->bli", R, prev) / self.scale)
        h = self.emb(types) + self.inp(torch.cat(f, -1))
        dx = x[:, None, :, :] - x[:, :, None, :]  # (B, i, j, 3) = x_j − x_i
        dist = dx.norm(dim=-1).clamp(min=1e-6)
        rbf = torch.exp(-((dist[..., None] - self.mu_rbf) / 0.15) ** 2)
        dirl = torch.einsum("bimk,bijm->bijk", R, dx / dist[..., None])  # směr k j v rámci i
        idx = torch.arange(L, device=types.device)
        sep = (idx[None, :] - idx[:, None]).clamp(-self.max_sep - 1, self.max_sep) + self.max_sep + 1
        pf = torch.cat([rbf, dirl, self.sep(sep)[None].expand(B, L, L, 32)], -1)
        bias = self.pair(pf).permute(0, 3, 1, 2)  # (B, heads · layers, L, L)
        nh = bias.shape[1] // len(self.layers)
        for i, lay in enumerate(self.layers):
            h = lay(h, bias[:, i * nh:(i + 1) * nh])
        o = self.head(self.ln(h))
        mu = torch.einsum("blij,blj->bli", R, o[..., :3]) * self.scale
        V = o[..., 3:3 + 3 * self.K].view(B, L, self.K, 3)
        U = torch.einsum("blij,blkj->blik", R, V) * self.scale / math.sqrt(self.K)
        var = (F.softplus(o[..., -1]) * self.scale) ** 2 + 1e-6
        return mu.reshape(B, 3 * L), U.reshape(B, 3 * L, self.K), var[..., None].expand(B, L, 3).reshape(B, 3 * L)


def lowrank_logp(mu, U, var, x):
    """log N(x; μ, U Uᵀ + diag var) po vzorcích přes Woodburyho identitu. Malá matice K × K se rozkládá na CPU
    (Choleského rozklad na MPS nemusí být k dispozici), převod mezi zařízeními je derivovatelný."""
    r = x - mu
    Ud = U / var[..., None]
    C = torch.eye(U.shape[-1], device=U.device, dtype=U.dtype) + U.transpose(-1, -2) @ Ud
    y = (Ud.transpose(-1, -2) @ r[..., None])[..., 0]
    Lc = torch.linalg.cholesky(C.cpu())
    w = torch.cholesky_solve(y.cpu()[..., None], Lc)[..., 0]
    logdet = torch.log(var).sum(-1) + 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1).to(U.device)
    quad = (r * r / var).sum(-1) - (y.cpu() * w).sum(-1).to(U.device)
    return -0.5 * (quad + logdet + x.shape[-1] * math.log(2 * math.pi))


def proj_logp(mu, U, var, x, V):
    """log N(Vᵀx; Vᵀμ, VᵀU UᵀV + Vᵀ diag(var) V): rozdělení promítnuté na módy V (B, 3L, M); Cholesky M × M na CPU."""
    Vt = V.transpose(-1, -2)
    A = Vt @ U
    C = A @ A.transpose(-1, -2) + Vt @ (V * var[..., None])
    r = (Vt @ (x - mu)[..., None])[..., 0].cpu()
    Lc = torch.linalg.cholesky(C.cpu())
    w = torch.cholesky_solve(r[..., None], Lc)[..., 0]
    logdet = 2 * torch.log(torch.diagonal(Lc, dim1=-2, dim2=-1)).sum(-1)
    return (-0.5 * ((r * w).sum(-1) + logdet + V.shape[-1] * math.log(2 * math.pi))).to(U.device)


def nll(net, dev, types, bb, ang, lag, D, prev, V=None):
    """Záporná log-věrohodnost posunu v bitech na reziduum (průměr přes vzorky); V = jen průmět na módy ANM."""
    b, L = D.shape[:2]
    T = lambda x: torch.as_tensor(np.ascontiguousarray(x), device=dev)
    mu, U, var = net(T(np.broadcast_to(types, (b, L))), T(bb), T(ang), T(np.log(lag).astype(np.float32)),
                     None if prev is None else T(prev))
    x = T(D.reshape(b, 3 * L))
    return -(lowrank_logp(mu, U, var, x) if V is None else proj_logp(mu, U, var, x, T(V))) / (L * math.log(2))


def iso_bits(D, s2, V=None):
    """Izotropní normální rozdělení N(0, s² I) v bitech na reziduum (s V v souřadnicích módů)."""
    L = D.shape[1]
    y = D.reshape(len(D), -1) if V is None else np.einsum("bnm,bn->bm", V, D.reshape(len(D), -1))
    return (0.5 * (y ** 2).sum(1) / s2 + 0.5 * y.shape[1] * np.log(2 * np.pi * s2)) / (L * np.log(2))


def load_model_q(path, dev):
    ck = torch.load(Path(path) / "model.pt", map_location=dev, weights_only=False)
    net = NetQ(**ck["cfg"]).to(dev)
    net.load_state_dict(ck["state"])
    net.eval()
    return net, ck


@torch.no_grad()
def sample_q(net, dev, types, bb, ang, lag_ps, prev, rng, diag=1.0):
    """Jeden vzorek posunu Cα (L, 3) v nm pro hybrid; bb (L, 3, 3), ang (L, 3), prev (L, 3) nebo None;
    diag = násobek nezávislého šumu σε (0 = jen μ + U z)."""
    T = lambda x: torch.as_tensor(np.asarray(x, np.float32)[None], device=dev)
    mu, U, var = net(torch.as_tensor(np.asarray(types)[None], device=dev), T(bb), T(ang),
                     torch.tensor([math.log(lag_ps)], device=dev), None if prev is None else T(prev))
    z = torch.as_tensor(rng.standard_normal(U.shape[-1]), dtype=U.dtype, device=dev)
    e = torch.as_tensor(rng.standard_normal(var.shape[-1]), dtype=U.dtype, device=dev)
    d = mu[0] + U[0] @ z + diag * var[0].sqrt() * e
    return d.view(-1, 3).cpu().numpy().astype(np.float64)


def evaluate_set(net, dev, reps, args, rng, n_per_rep):
    """Bity na reziduum modelu a izotropní referencí (s² z trénovacích dat) + RMSD skutečných a vzorkovaných posunů."""
    out = {}
    proj = getattr(args, "proj", 0)
    net.eval()
    with torch.no_grad():
        for p in reps:
            types, bb, ang, lag, D, prev = sample_rep(p, n_per_rep, args.lags, rng, net.hist_disp)
            V = bases(bb, proj)
            b = nll(net, dev, types, bb, ang, lag, D, prev, V).cpu().numpy()
            mu, U, var = net(*[torch.as_tensor(np.ascontiguousarray(x), device=dev) for x in
                               (np.broadcast_to(types, D.shape[:2]), bb, ang, np.log(lag).astype(np.float32))],
                             None if prev is None else torch.as_tensor(prev, device=dev))
            z = torch.randn(U.shape[0], U.shape[-1], 1, device=dev)
            S = (mu + (U @ z)[..., 0] + var.sqrt() * torch.randn_like(var)).view(D.shape).cpu().numpy()
            if V is not None:  # skutečný posun i vzorek jen v módech ANM
                P = lambda Y: np.einsum("bnm,bkm,bk->bn", V, V, Y.reshape(len(Y), -1)).reshape(Y.shape)
                D, S = P(D), P(S)
            X = bb[:, :, 1].astype(np.float64)
            rs = np.sqrt(((kabsch_onto(X + S, X) - X) ** 2).sum(-1).mean(-1))
            r = out.setdefault(p["prot"], dict(model=[], iso=[], rmsd_true=[], rmsd_sample=[]))
            r["model"] += b.tolist()
            r["iso"] += iso_bits(D, args.iso_s2, V).tolist()
            r["rmsd_true"] += np.sqrt((D ** 2).sum(-1).mean(-1)).tolist()
            r["rmsd_sample"] += rs.tolist()
    net.train()
    return {k: {m: float(np.mean(v)) for m, v in r.items()} for k, r in out.items()}


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    tr = load_atlas(args.atlas, read_lists(args.train_lists))
    va = load_atlas(args.atlas, read_lists(args.val_lists))
    print(f"trénink {len(tr)} replik, validace {len(va)}, zařízení {dev}", flush=True)
    # izotropní reference: střední kvadrát posunu na souřadnici z trénovacích dat (s --proj v souřadnicích módů)
    smp = [sample_rep(tr[i], 8, args.lags, rng, False) for i in rng.choice(len(tr), size=min(64, len(tr)))]
    args.iso_s2 = float(np.mean([(s[4] ** 2).mean() if not args.proj else
                                 (np.einsum("bnm,bn->bm", bases(s[1], args.proj), s[4].reshape(len(s[4]), -1)) ** 2).mean()
                                 for s in smp]))
    net = NetQ(d=args.hidden, layers=args.layers, heads=args.heads, K=args.K, hist_disp=args.hist_disp).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=args.steps, pct_start=0.05)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    t0, log, best = time.time(), [], None
    for step in range(1, args.steps + 1):
        loss = 0.0
        for i in rng.choice(len(tr), size=min(args.prot_per_batch, len(tr)), replace=False):
            types, bb, ang, lag, D, prev = sample_rep(tr[i], args.per_prot, args.lags, rng, args.hist_disp)
            V = bases(bb, args.proj)
            loss = loss + nll(net, dev, types, bb, ang, lag, D, prev, V).mean() / args.prot_per_batch
        opt.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), 1.0)
        opt.step()
        sched.step()
        if step % args.eval_every == 0 or step == args.steps:
            ev = evaluate_set(net, dev, va, args, np.random.default_rng(123), args.val_per_rep)
            vm = float(np.mean([v["model"] for v in ev.values()]))
            vi = float(np.mean([v["iso"] for v in ev.values()]))
            log.append(dict(step=step, train_bits=loss.item(), val_bits=vm, val_iso_bits=vi, s=round(time.time() - t0)))
            print(f"krok {step}: trénink {loss.item():.3f}, validace {vm:.3f} bitů/rez. (izotropní {vi:.3f}), "
                  f"{time.time() - t0:.0f} s", flush=True)
            if best is None or vm < best:
                best = vm
                torch.save(dict(cfg=net.cfg, state=net.state_dict(), args=vars(args), log=log), out / "model.pt")
    (out / "train_log.json").write_text(json.dumps(dict(args=vars(args), log=log, best_val_bits=best), indent=1))
    print(f"Hotovo, nejlepší validace {best:.3f} bitů/rez., uloženo {out}", flush=True)


def evaluate(args):
    dev = torch.device(args.device or "cpu")
    net, ck = load_model_q(args.model, dev)
    args.iso_s2 = ck["args"]["iso_s2"]
    args.lags = args.lags or ck["args"]["lags"]
    args.proj = ck["args"].get("proj", 0)
    reps = load_atlas(args.atlas, read_lists(args.lists))
    ev = evaluate_set(net, dev, reps, args, np.random.default_rng(args.seed), args.n_per_rep)
    for k, v in ev.items():
        print(f"  {k}: model {v['model']:.3f}, izotropní {v['iso']:.3f} bitů/rez.; RMSD posunu skutečný "
              f"{v['rmsd_true']:.3f}, vzorek {v['rmsd_sample']:.3f} nm", flush=True)
    m = {key: float(np.mean([v[key] for v in ev.values()])) for key in next(iter(ev.values()))}
    print(f"Průměr: model {m['model']:.3f}, izotropní {m['iso']:.3f} bitů/rez.; RMSD skutečný {m['rmsd_true']:.3f}, "
          f"vzorek {m['rmsd_sample']:.3f} nm")
    if args.out:
        Path(args.out).write_text(json.dumps(dict(model=args.model, mean=m, per_protein=ev), indent=1))


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--atlas", required=True)
    t.add_argument("--train-lists", nargs="+", required=True)
    t.add_argument("--val-lists", nargs="+", required=True)
    t.add_argument("--lags", type=float, nargs="+", default=[100.0])
    t.add_argument("--hist-disp", action=argparse.BooleanOptionalAction, default=True)
    t.add_argument("--hidden", type=int, default=128)
    t.add_argument("--layers", type=int, default=4)
    t.add_argument("--heads", type=int, default=8)
    t.add_argument("--K", type=int, default=16, help="počet kolektivních módů (rank kovariance)")
    t.add_argument("--proj", type=int, default=0,
                   help="učit jen průmět posunu na tolik nejměkčích módů ANM (cutoff 1,5 nm, sousedé 10×); 0 = celý posun")
    t.add_argument("--steps", type=int, default=4000)
    t.add_argument("--lr", type=float, default=3e-4)
    t.add_argument("--prot-per-batch", type=int, default=8)
    t.add_argument("--per-prot", type=int, default=8)
    t.add_argument("--eval-every", type=int, default=250)
    t.add_argument("--val-per-rep", type=int, default=32)
    t.add_argument("--device", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--atlas", required=True)
    e.add_argument("--lists", nargs="+", required=True)
    e.add_argument("--lags", type=float, nargs="+", default=None)
    e.add_argument("--n-per-rep", type=int, default=128)
    e.add_argument("--device", default=None)
    e.add_argument("--seed", type=int, default=0)
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
