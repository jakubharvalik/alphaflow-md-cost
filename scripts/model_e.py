"""Model E: Model D + prostorové okolí (rezidua blízko v prostoru, ale mimo okno n±r v sekvenci).

Pro reziduum n v čase t se z poloh páteře (N, CA, C) vyberou až --max-nb nejbližších reziduí podle
vzdálenosti Cα do --cutoff nm s |m − n| > r. Každý soused nese: typ, své torze v čase t, vzdálenost
Cα–Cα a Cβ–Cβ (virtuální Cβ z páteře, RBF), směr k sousedovi a směr jeho Cβ v lokální soustavě
rezidua n, sekvenční odstup. Sousedé se sečtou (DeepSets) a přičtou do první vrstvy těla Model D.
Projekce je na začátku nulová, takže Model E na startu dává přesně totéž co výchozí Model D
a doladění učí jen opravu z prostorového kontextu. Bez poloh (replay MDGen) je příspěvek nulový.

    python model_e.py train --base model_d_ft300d --systems ala5 kgpwe agwek kwage sgvwd dwlgk kt10 \
        --replay-dir public/mdgen_tors --out model_e1
    python model_e.py eval --model model_e1 --base model_d_ft300d --systems eagwk cln025 cln025f
"""
import argparse
import json
import time
from pathlib import Path

import mdtraj as md
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_c import AA, NB, bin_to_angle, load_dir, tensors
from model_d import UNDEF, NetD, bins_or_undef, left_feat, make_ctx, mlp, sample_batch
from outcond import load_model, residue_types, torsion_quads, torsions

N_RBF = 16
N_SEP = 8
D_NB = 2 * N_RBF + 3 + 3 + 9 + N_SEP + 1


def rbf(d, cutoff):
    c = np.linspace(0.3, cutoff, N_RBF, dtype=np.float32)
    w = (cutoff - 0.3) / (N_RBF - 1)
    return np.exp(-((d[..., None] - c) / w) ** 2)


def virtual_cb(N, CA, C):
    """Ideální Cβ z páteře; konstanty jsou pro Å, proto × 10 u vektorového součinu (polohy v nm)."""
    b, c = CA - N, C - CA
    a = 10 * np.cross(b, c)
    return -0.58273431 * a + 0.56802827 * b - 0.54067466 * c + CA


def local_frame(N, CA, C):
    """Řádky = osy lokální soustavy rezidua (x podél CA→C, y v rovině N-CA-C)."""
    x = C - CA
    x /= np.linalg.norm(x, axis=-1, keepdims=True)
    y = N - CA
    y -= (y * x).sum(-1, keepdims=True) * x
    y /= np.linalg.norm(y, axis=-1, keepdims=True)
    return np.stack([x, y, np.cross(x, y)], -2)


def ang9(a):
    m = ~np.isnan(a)
    a0 = np.where(m, a, 0.0)
    return np.concatenate([np.sin(a0) * m, np.cos(a0) * m, m.astype(np.float32)], -1)


def spatial_feats(bb, types, ang, n, r, max_nb, cutoff):
    """bb (B, L, 3, 3) polohy N, CA, C v nm; types (B, L); ang (B, L, 3) v čase t; n (B,).
    Vrací typy sousedů (B, M), rysy (B, M, D_NB) a masku (B, M)."""
    bb = bb.astype(np.float32)
    B, L = types.shape
    rows = np.arange(B)
    N, CA, C = bb[:, :, 0], bb[:, :, 1], bb[:, :, 2]
    CB = virtual_cb(N, CA, C)
    R = local_frame(N[rows, n], CA[rows, n], C[rows, n])  # (B, 3, 3)
    d = np.linalg.norm(CA - CA[rows, n][:, None], axis=-1)
    sep = np.abs(np.arange(L)[None] - n[:, None])
    valid = (sep > r) & (d < cutoff) & (types < len(AA))
    M = min(max_nb, L)
    order = np.argsort(np.where(valid, d, np.inf), axis=1)[:, :M]
    take = lambda x: np.take_along_axis(x, order.reshape(B, M, *([1] * (x.ndim - 2))), 1)
    mask = take(valid)
    dm, sm = take(d), take(sep)
    ca_m, cb_m = take(CA), take(CB)
    d_cb = np.linalg.norm(cb_m - CB[rows, n][:, None], axis=-1)
    u = (ca_m - CA[rows, n][:, None]) / np.maximum(dm[..., None], 1e-6)
    v = cb_m - ca_m
    v /= np.maximum(np.linalg.norm(v, axis=-1, keepdims=True), 1e-6)
    to_loc = lambda w: np.einsum("bij,bmj->bmi", R, w)
    sep_oh = np.eye(N_SEP, dtype=np.float32)[np.clip(sm - r - 1, 0, N_SEP - 1)]
    side = (np.take_along_axis(np.broadcast_to(np.arange(L), (B, L)), order, 1) > n[:, None])[..., None]
    f = np.concatenate([rbf(dm, cutoff), rbf(d_cb, cutoff), to_loc(u), to_loc(v), ang9(take(ang)), sep_oh,
                        side.astype(np.float32)], -1) * mask[..., None]
    if M < max_nb:  # krátké peptidy: doplnit na pevný počet sousedů
        pad = max_nb - M
        f = np.pad(f, ((0, 0), (0, pad), (0, 0)))
        mask = np.pad(mask, ((0, 0), (0, pad)))
        order = np.pad(order, ((0, 0), (0, pad)))
    t_nb = np.where(mask, np.take_along_axis(types, np.minimum(order, L - 1), 1), len(AA))
    return t_nb, f.astype(np.float32), mask.astype(np.float32)


class NetE(NetD):
    def __init__(self, r=1, hist=3, ar=True, emb=16, hidden=1024, layers=4, nb_hidden=128, max_nb=8,
                 cutoff=1.2):
        super().__init__(r=r, hist=hist, ar=ar, emb=emb, hidden=hidden, layers=layers)
        self.max_nb, self.cutoff = max_nb, cutoff
        self.nb_emb = nn.Embedding(len(AA) + 1, emb)
        self.nb_mlp = mlp(D_NB + emb, nb_hidden, 2)
        self.nb_proj = nn.Linear(nb_hidden, hidden, bias=False)
        nn.init.zeros_(self.nb_proj.weight)

    def forward(self, types_w, feats_w, pos_w, loglag, left_new, nb=None):
        x = torch.cat([self.emb(types_w), feats_w, pos_w], -1).flatten(1)
        parts = [x, loglag[:, None]] + ([left_new] if self.ar else [])
        h = self.body[0](torch.cat(parts, -1))
        if nb is not None:
            t_nb, f_nb, m_nb = nb
            e = self.nb_mlp(torch.cat([self.nb_emb(t_nb), f_nb], -1)) * m_nb[..., None]
            h = h + self.nb_proj(e.sum(1))
        return self.body[1:](h)


def load_model_e(model_dir, dev):
    summ = json.loads((Path(model_dir) / "model_d_summary.json").read_text())
    cfg = summ["config"]
    if not summ.get("spatial"):
        return load_model(model_dir, dev)
    sp = summ["spatial"]
    net = NetE(r=cfg["r"], hist=cfg["hist"], ar=cfg["ar"], hidden=cfg["hidden"], layers=cfg["layers"],
               nb_hidden=sp["nb_hidden"], max_nb=sp["max_nb"], cutoff=sp["cutoff"]).to(dev)
    net.load_state_dict(torch.load(Path(model_dir) / "model_d.pt", map_location=dev))
    net.eval()
    return net, dict(cfg, spatial=sp)


def backbone_index(top):
    out = []
    for res in top.residues:
        names = {a.name: a.index for a in res.atoms}
        out.append([names["N"], names["CA"], names["C"]])
    return np.array(out)


def load_sys_e(sysdir, reps=None):
    sysdir = Path(sysdir)
    top = md.load(str(sysdir / "solute.pdb")).topology
    Q, bbi, types = torsion_quads(top), backbone_index(top), residue_types(top)
    files = [f for f in sorted((sysdir / "runs").glob("rep_*.npz")) if ".tmp" not in f.name]
    files = [f for i, f in enumerate(files) if reps is None or i in reps]
    out = []
    for f in files:
        p = np.load(f)["pos"]
        bb = p[:, bbi]
        bb = (bb - bb.mean((1, 2), keepdims=True)).astype(np.float16)  # posun těžiště, ať stačí float16
        out.append(dict(name=f"{sysdir.name}_{f.stem}", types=types, ang=torsions(p, Q), bb=bb))
    return out


def sample_batch_e(peps, B, lags, hist, rng):
    """Jako model_d.sample_batch, navíc polohy páteře v čase t."""
    pi = rng.integers(len(peps), size=B)
    lag = rng.choice(lags, size=B)
    L = peps[0]["types"].shape[0]
    n = rng.integers(L, size=B)
    types = np.stack([peps[p]["types"] for p in pi])
    frames, nxt, bb = [[] for _ in range(hist)], [], []
    for p, g in zip(pi, lag):
        a = peps[p]["ang"]
        t = rng.integers((hist - 1) * g, a.shape[0] - g)
        for k in range(hist):
            frames[k].append(a[t - k * g])
        nxt.append(a[t + g])
        bb.append(peps[p]["bb"][t])
    f32 = lambda x: np.stack(x).astype(np.float32)
    return types, [f32(f) for f in frames], f32(nxt), n, lag, f32(bb)


def loss_and_ll_e(net, dev, types, frames, nxt, n, lag, bb, pred_chi):
    """Jako model_d.loss_and_ll; bb = None → bez prostorového kontextu (stejné jako Model D)."""
    t_w, f, pos = make_ctx(types, frames, n, net.r)
    lf = left_feat(nxt, n)
    rows = np.arange(len(n))
    (phi_b, hphi), (psi_b, hpsi), (chi_b, hchi) = (bins_or_undef(nxt[rows, n, k]) for k in range(3))
    t_w, f, pos, ll_, lf, phi_b, psi_b, chi_b, hphi, hpsi, hchi = tensors(
        dev, t_w, f, pos, np.log(lag).astype(np.float32), lf, phi_b, psi_b, chi_b, hphi, hpsi, hchi)
    nb = None
    if bb is not None and isinstance(net, NetE):
        nb = tensors(dev, *spatial_feats(bb, types, frames[0], n, net.r, net.max_nb, net.cutoff))
    h = net(t_w, f, pos, ll_, lf, nb) if isinstance(net, NetE) else net(t_w, f, pos, ll_, lf)
    nll = lambda logits, b, m: -F.log_softmax(logits, -1).gather(1, b.clamp(max=NB - 1)[:, None])[:, 0] * m
    n_phi = nll(net.head_phi(h), phi_b, hphi)
    n_psi = nll(net.psi_logits(h, phi_b), psi_b, hpsi)
    n_chi = nll(net.chi_logits(h, phi_b, psi_b), chi_b, hchi) if pred_chi else torch.zeros_like(n_phi)
    hchi = hchi if pred_chi else torch.zeros_like(hchi)
    loss = (n_phi.sum() + n_psi.sum() + n_chi.sum()) / (hphi.sum() + hpsi.sum() + hchi.sum())
    return loss, ((n_phi + n_psi) / np.log(2)), hphi & hpsi, (n_chi / np.log(2)), hchi


@torch.no_grad()
def step_e(net, dev, types, frames, bb, lag, pred_chi, rng):
    """Jeden skok o lag ps pro R trajektorií: types (L,), frames [t, t−τ, …] (R, L, 3), bb (R, L, 3, 3)."""
    R, L = frames[0].shape[:2]
    types = np.broadcast_to(types, (R, L)).copy()
    now = frames[0]
    new = np.full_like(now, np.nan)
    for n in range(L):
        nn_ = np.full(R, n)
        t_w, f, pos = make_ctx(types, frames, nn_, net.r)
        t_w, f, pos, ll_, lf = tensors(dev, t_w, f, pos, np.full(R, np.log(lag), np.float32), left_feat(new, nn_))
        if isinstance(net, NetE):
            nb = tensors(dev, *spatial_feats(bb, types, now, nn_, net.r, net.max_nb, net.cutoff))
            h = net(t_w, f, pos, ll_, lf, nb)
        else:
            h = net(t_w, f, pos, ll_, lf)
        has = ~np.isnan(now[:, n])
        pb = torch.multinomial(F.softmax(net.head_phi(h), -1), 1)[:, 0].cpu().numpy()
        pb = np.where(has[:, 0], pb, UNDEF)
        pb_t = torch.as_tensor(pb, device=dev)
        qb = torch.multinomial(F.softmax(net.psi_logits(h, pb_t), -1), 1)[:, 0].cpu().numpy()
        qb = np.where(has[:, 1], qb, UNDEF)
        new[:, n, 0] = np.where(has[:, 0], bin_to_angle(np.minimum(pb, NB - 1), rng), np.nan)
        new[:, n, 1] = np.where(has[:, 1], bin_to_angle(np.minimum(qb, NB - 1), rng), np.nan)
        if pred_chi:
            cb = torch.multinomial(F.softmax(net.chi_logits(h, pb_t, torch.as_tensor(qb, device=dev)), -1),
                                   1)[:, 0].cpu().numpy()
            new[:, n, 2] = np.where(has[:, 2], bin_to_angle(cb, rng), np.nan)
        else:
            new[:, n, 2] = now[:, n, 2]
    return new


def by_len(peps):
    return {L: [p for p in peps if len(p["types"]) == L] for L in sorted({len(p["types"]) for p in peps})}


def train(args):
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    base, cfg = load_model(args.base, dev)
    net = NetE(r=cfg["r"], hist=cfg["hist"], ar=cfg["ar"], hidden=cfg["hidden"], layers=cfg["layers"],
               nb_hidden=args.nb_hidden, max_nb=args.max_nb, cutoff=args.cutoff).to(dev)
    missing = net.load_state_dict(base.state_dict(), strict=False)
    assert all(k.startswith("nb_") for k in missing.missing_keys) and not missing.unexpected_keys
    del base
    train_p, val_p = [], []
    for s in args.systems:
        peps = load_sys_e(s)
        v = args.n_val_reps
        k = min(args.n_train_reps, len(peps) - v)
        train_p += peps[:k]
        val_p += peps[k:k + v]
    replay = load_dir(args.replay_dir, args.replay_n) if args.replay_dir else None
    tg, vg = by_len(train_p), by_len(val_p)
    print(f"Model E z {args.base}: {len(train_p)} replik ({ {L: len(g) for L, g in tg.items()} }), validace "
          f"{len(val_p)}, zařízení {dev}, sousedé ≤ {args.max_nb} do {args.cutoff} nm"
          + (f", replay {len(replay)} MDGen" if replay else ""), flush=True)

    def val_bits():
        out = []
        with torch.no_grad():
            for g in vg.values():
                r = np.random.default_rng(123)  # stejné přechody při každé validaci
                _, b, m, _, _ = loss_and_ll_e(net, dev, *sample_batch_e(g, 4096, args.lags, cfg["hist"], r),
                                              cfg["pred_chi"])
                out.append(b[m])
        return float(torch.cat(out).mean())

    # nulová projekce → na startu stejné bity jako výchozí model
    log = [dict(step=0, val_bits_phipsi=val_bits())]
    print(f"krok      0  val bity φψ {log[0]['val_bits_phipsi']:.3f}", flush=True)
    params = [{"params": [p for k, p in net.named_parameters() if k.startswith("nb_")], "lr": args.lr_nb},
              {"params": [p for k, p in net.named_parameters() if not k.startswith("nb_")], "lr": args.lr}]
    opt = torch.optim.AdamW(params, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    for step in range(1, args.steps + 1):
        loss = 0.0
        for g in tg.values():
            w = len(g) / len(train_p)
            lg = loss_and_ll_e(net, dev, *sample_batch_e(g, max(256, round(args.batch * w)), args.lags,
                                                         cfg["hist"], rng), cfg["pred_chi"])[0]
            loss = loss + w * lg
        if replay:
            types, frames, nxt, n, lag = sample_batch(replay, args.batch, args.replay_lags, cfg["hist"], rng)
            loss = loss + args.replay_w * loss_and_ll_e(net, dev, types, frames, nxt, n, lag, None,
                                                        cfg["pred_chi"])[0]
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            log.append(dict(step=step, train_loss=loss.item(), val_bits_phipsi=val_bits()))
            print(f"krok {step:6d}  loss {loss.item():.3f}  val bity φψ {log[-1]['val_bits_phipsi']:.3f}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    torch.save(net.state_dict(), out / "model_d.pt")
    (out / "model_d_summary.json").write_text(json.dumps(dict(
        config=cfg, spatial=dict(nb_hidden=args.nb_hidden, max_nb=args.max_nb, cutoff=args.cutoff),
        base_model=args.base, systems=args.systems, n_train_reps=args.n_train_reps, lags=args.lags,
        replay_dir=args.replay_dir, replay_n=args.replay_n if args.replay_dir else 0, replay_lags=args.replay_lags,
        replay_w=args.replay_w, steps=args.steps, lr=args.lr, lr_nb=args.lr_nb,
        train_s=round(time.time() - t0, 1), log=log), indent=2))
    print("Uloženo", out / "model_d.pt")


def evaluate(args):
    """Bity φψ (a χ1) na stejných přechodech pro výchozí model a Model E, po reziduích a lazích."""
    dev = torch.device(args.device or "cpu")
    nets = {"base": load_model_e(args.base, dev), "E": load_model_e(args.model, dev)}
    res = {}
    for s in args.systems:
        peps = load_sys_e(s)
        if not peps:
            continue
        names = [str(r) for r in md.load(str(Path(s) / "solute.pdb")).topology.residues]
        L = len(peps[0]["types"])
        res[Path(s).name] = {}
        for lag in args.lags:
            rng = np.random.default_rng(7)
            per = {}
            for n in range(L):
                # stejné přechody pro oba modely: pevně dané reziduum n
                batch = list(sample_batch_e(peps, args.n_eval, [lag], nets["base"][1]["hist"], rng))
                batch[3] = np.full(args.n_eval, n)
                row = {}
                # E_bez = tentýž Model E s vypnutým prostorovým vstupem (kolik dělá samotné okolí)
                runs = [(k, net, cfg, batch) for k, (net, cfg) in nets.items()]
                runs.append(("E_bez", *nets["E"], batch[:5] + [None]))
                for key, net, cfg, bt in runs:
                    with torch.no_grad():
                        _, b, m, bc, mc = loss_and_ll_e(net, dev, *bt, cfg["pred_chi"])
                    if m.any():
                        row[f"bits_{key}"] = float(b[m].mean())
                    if mc.any():
                        row[f"chi_{key}"] = float(bc[mc].mean())
                if "bits_base" in row:
                    row["d_bits"] = row["bits_E"] - row["bits_base"]
                    row["d_bits_spatial"] = row["bits_E"] - row["bits_E_bez"]
                if "chi_base" in row:
                    row["d_chi"] = row["chi_E"] - row["chi_base"]
                per[names[n]] = row
            res[Path(s).name][f"lag{lag}"] = per
            dd = [v["d_bits"] for v in per.values() if "d_bits" in v]
            print(f"{Path(s).name:8s} τ {lag:4d} ps: Δbity φψ (E − výchozí) "
                  + " ".join(f"{k}:{v['d_bits']:+.2f}" for k, v in per.items() if "d_bits" in v)
                  + f" | průměr {np.mean(dd):+.3f}; z toho prostorový vstup "
                  + " ".join(f"{k}:{v['d_bits_spatial']:+.2f}" for k, v in per.items() if "d_bits" in v), flush=True)
    out = Path(args.out or f"{Path(args.model).name}_eval.json")
    out.write_text(json.dumps(res, indent=1))
    print("Uloženo", out)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    t = sub.add_parser("train")
    t.add_argument("--base", default="model_d_ft300d")
    t.add_argument("--systems", nargs="+", required=True)
    t.add_argument("--n-train-reps", type=int, default=50)
    t.add_argument("--n-val-reps", type=int, default=10)
    t.add_argument("--lags", type=int, nargs="+", default=[1, 10, 30, 50, 100])
    t.add_argument("--replay-dir", default=None)
    t.add_argument("--replay-n", type=int, default=300)
    t.add_argument("--replay-lags", type=int, nargs="+", default=[1, 10, 100])
    t.add_argument("--replay-w", type=float, default=1.0)
    t.add_argument("--max-nb", type=int, default=8)
    t.add_argument("--cutoff", type=float, default=1.2)
    t.add_argument("--nb-hidden", type=int, default=128)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=2048)
    t.add_argument("--lr", type=float, default=5e-5)
    t.add_argument("--lr-nb", type=float, default=5e-4)
    t.add_argument("--device", default=None)
    t.add_argument("--seed", type=int, default=0)
    t.add_argument("--out", default="model_e1")
    e = sub.add_parser("eval")
    e.add_argument("--model", required=True)
    e.add_argument("--base", default="model_d_ft300d")
    e.add_argument("--systems", nargs="+", required=True)
    e.add_argument("--lags", type=int, nargs="+", default=[10, 30, 100])
    e.add_argument("--n-eval", type=int, default=4000)
    e.add_argument("--device", default=None)
    e.add_argument("--out", default=None)
    args = ap.parse_args()
    train(args) if args.cmd == "train" else evaluate(args)


if __name__ == "__main__":
    main()
