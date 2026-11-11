"""Model D: rozšíření Model C o pohyb postranních řetězců a vazbu mezi sousedy v jednom kroku.

Oproti Model C (vše se dá vypnout pro ablaci):
  - --pred-chi: předpovídá i χ1, autoregresivně p(φ') p(ψ'|φ') p(χ1'|φ',ψ'); bez toho drží χ1 pevně,
  - --hist K: historie K snímků (t, t−τ, …, t−(K−1)τ), Model C má K = 2,
  - --ar: rezidua se v jednom kroku vzorkují postupně zleva a reziduum n vidí už nový stav n−1
    (sdružené rozdělení sousedů místo nezávislého vzorkování),
  - větší síť (--hidden, --layers).
Nové metriky: delší rollout i pro τ = 1 ps, JS rozdělení χ1, rychlost přechodů α↔β v ψ vs. MD,
nativní tabulka i v 36 binech (spravedlivější srovnání v bitech).

    python model_d.py --train public/mdgen_tors --test public/mdgen_tors_test --pred-chi --ar --hist 3 --out model_d_full
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_c import AA, NB, PAD, ang_feats, bin_to_angle, baselines_ll, js, load_dir, rama_hist, tensors, \
    to_bin, type_marginals

UNDEF = NB  # úhel není definován (φ na N-konci, ψ na C-konci, χ1 u Gly/Ala)
PSI_ALPHA = (-2.094, 0.873)  # ψ v (−120°, 50°) = α, jinak β/PPII


def mlp(d_in, hidden, layers):
    mods, d = [], d_in
    for _ in range(layers):
        mods += [nn.Linear(d, hidden), nn.SiLU()]
        d = hidden
    return nn.Sequential(*mods)


class NetD(nn.Module):
    def __init__(self, r=1, hist=3, ar=True, emb=16, hidden=1024, layers=4):
        super().__init__()
        self.r, self.hist, self.ar = r, hist, ar
        w = 2 * r + 1
        self.emb = nn.Embedding(len(AA) + 1, emb)
        d_in = w * (emb + 9 * hist + 2) + 1 + (9 if ar else 0)
        self.body = mlp(d_in, hidden, layers)
        self.head_phi = nn.Linear(hidden, NB)
        self.phi_emb = nn.Embedding(NB + 1, 32)
        self.psi_emb = nn.Embedding(NB + 1, 32)
        self.head_psi = nn.Sequential(nn.Linear(hidden + 32, hidden), nn.SiLU(), nn.Linear(hidden, NB))
        self.head_chi = nn.Sequential(nn.Linear(hidden + 64, hidden), nn.SiLU(), nn.Linear(hidden, NB))

    def forward(self, types_w, feats_w, pos_w, loglag, left_new):
        x = torch.cat([self.emb(types_w), feats_w, pos_w], -1).flatten(1)
        parts = [x, loglag[:, None]] + ([left_new] if self.ar else [])
        return self.body(torch.cat(parts, -1))

    def psi_logits(self, h, phi_b):
        return self.head_psi(torch.cat([h, self.phi_emb(phi_b)], -1))

    def chi_logits(self, h, phi_b, psi_b):
        return self.head_chi(torch.cat([h, self.phi_emb(phi_b), self.psi_emb(psi_b)], -1))


def make_ctx(types, frames, n, r):
    """types (B, L), frames = [úhly v t, t−τ, …] tvaru (B, L, 3), n (B,) → okno n±r."""
    B, L = types.shape
    idx = n[:, None] + np.arange(-r, r + 1)[None, :]
    inside = (idx >= 0) & (idx < L)
    idc = np.clip(idx, 0, L - 1)
    rows = np.arange(B)[:, None]
    t_w = np.where(inside, types[rows, idc], PAD)
    f = np.concatenate([ang_feats(fr[rows, idc]) for fr in frames], -1) * inside[..., None]
    pos = np.stack([(idx == 0), (idx == L - 1)], -1).astype(np.float32) * inside[..., None]
    return t_w, f.astype(np.float32), pos


def left_feat(new, n):
    """Nový stav levého souseda n−1 (nuly pro n = 0)."""
    a = new[np.arange(len(n)), np.maximum(n - 1, 0)]
    return (ang_feats(a) * (n > 0)[:, None]).astype(np.float32)


def bins_or_undef(a):
    ok = ~np.isnan(a)
    return np.where(ok, to_bin(np.nan_to_num(a)), UNDEF), ok


def sample_batch(peps, B, lags, hist, rng):
    pi = rng.integers(len(peps), size=B)
    lag = rng.choice(lags, size=B)
    L = peps[0]["types"].shape[0]
    n = rng.integers(L, size=B)
    types = np.stack([peps[p]["types"] for p in pi])
    frames = [[] for _ in range(hist)]
    nxt = []
    for p, g in zip(pi, lag):
        a = peps[p]["ang"]
        t = rng.integers((hist - 1) * g, a.shape[0] - g)
        for k in range(hist):
            frames[k].append(a[t - k * g])
        nxt.append(a[t + g])
    return types, [np.stack(f) for f in frames], np.stack(nxt), n, lag


def loss_and_ll(net, dev, types, frames, nxt, n, lag, pred_chi):
    t_w, f, pos = make_ctx(types, frames, n, net.r)
    lf = left_feat(nxt, n)
    rows = np.arange(len(n))
    (phi_b, hphi), (psi_b, hpsi), (chi_b, hchi) = (bins_or_undef(nxt[rows, n, k]) for k in range(3))
    t_w, f, pos, ll_, lf, phi_b, psi_b, chi_b, hphi, hpsi, hchi = tensors(
        dev, t_w, f, pos, np.log(lag).astype(np.float32), lf, phi_b, psi_b, chi_b, hphi, hpsi, hchi)
    h = net(t_w, f, pos, ll_, lf)
    nll = lambda logits, b, m: -F.log_softmax(logits, -1).gather(1, b.clamp(max=NB - 1)[:, None])[:, 0] * m
    n_phi = nll(net.head_phi(h), phi_b, hphi)
    n_psi = nll(net.psi_logits(h, phi_b), psi_b, hpsi)
    n_chi = nll(net.chi_logits(h, phi_b, psi_b), chi_b, hchi) if pred_chi else torch.zeros_like(n_phi)
    hchi = hchi if pred_chi else torch.zeros_like(hchi)
    loss = (n_phi.sum() + n_psi.sum() + n_chi.sum()) / (hphi.sum() + hpsi.sum() + hchi.sum())
    both = hphi & hpsi
    bits_pp = ((n_phi + n_psi) / np.log(2))[both]
    bits_chi = (n_chi / np.log(2))[hchi]
    return loss, bits_pp, bits_chi


@torch.no_grad()
def rollout(net, dev, pep_list, lag, n_steps, n_traj, pred_chi, rng):
    """Rollout všech peptidů najednou; vrací (P, n_traj, n_steps, L, 3) úhly φ, ψ, χ1."""
    L = pep_list[0]["types"].shape[0]
    P, H = len(pep_list), net.hist
    R = P * n_traj
    types = np.repeat(np.stack([p["types"] for p in pep_list]), n_traj, 0)
    frames = [[] for _ in range(H)]
    for p in pep_list:
        ts = rng.integers((H - 1) * lag, p["ang"].shape[0], size=n_traj)
        for k in range(H):
            frames[k].append(p["ang"][ts - k * lag])
    frames = [np.concatenate(f) for f in frames]
    out = np.empty((R, n_steps, L, 3), np.float32)
    order = range(L)  # při --ar zleva doprava
    for s in range(n_steps):
        now = frames[0]
        out[:, s] = now
        new = np.full_like(now, np.nan)
        for n in order:
            nn_ = np.full(R, n)
            t_w, f, pos = make_ctx(types, frames, nn_, net.r)
            t_w, f, pos, ll_, lf = tensors(dev, t_w, f, pos, np.full(R, np.log(lag), np.float32),
                                           left_feat(new, nn_))
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
        frames = [new] + frames[:-1]
    return out.reshape(P, n_traj, n_steps, L, 3)


def chi_hist(x):
    x = x[~np.isnan(x)]
    return np.histogram(x, bins=NB, range=(-np.pi, np.pi))[0] + 1e-9


def ab_rate(psi, dt_ps):
    """Přechody α↔β v ψ za ns; psi tvaru (n_traj, T) nebo (T,)."""
    psi = np.atleast_2d(psi)
    st = (psi > PSI_ALPHA[0]) & (psi < PSI_ALPHA[1])
    flips = np.sum(st[:, 1:] != st[:, :-1])
    return flips / (psi.shape[0] * (psi.shape[1] - 1) * dt_ps / 1000)


def native36_ll(test_pep, lag, idx_t, n):
    """Nativní tabulka přímo v 36×36 binech z první poloviny testovací MD."""
    a = test_pep["ang"]
    half = a.shape[0] // 2
    x0, x1 = a[:half - lag, n, :2], a[lag:half, n, :2]
    ok = ~(np.isnan(x0).any(1) | np.isnan(x1).any(1))
    k0 = to_bin(x0[ok, 0]) * NB + to_bin(x0[ok, 1])
    k1 = to_bin(x1[ok, 0]) * NB + to_bin(x1[ok, 1])
    K = NB * NB
    m = (np.bincount(k1, minlength=K) + 1.0) / (len(k1) + K)
    cur, nxt = a[idx_t, n, :2], a[idx_t + lag, n, :2]
    kc = to_bin(cur[:, 0]) * NB + to_bin(cur[:, 1])
    kn = to_bin(nxt[:, 0]) * NB + to_bin(nxt[:, 1])
    # řádky jen pro potřebné výchozí biny (plná matice 1296² je zbytečná)
    out = np.empty(len(kc))
    for k in np.unique(kc):
        sel = kc == k
        c = np.bincount(k1[k0 == k], minlength=K).astype(float)
        out[sel] = -np.log2(((c + 5 * m) / (c.sum() + 5))[kn[sel]])
    return float(out.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 10, 100], help="v snímcích (1 ps)")
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--hist", type=int, default=3)
    ap.add_argument("--ar", action="store_true")
    ap.add_argument("--pred-chi", action="store_true")
    ap.add_argument("--hidden", type=int, default=1024)
    ap.add_argument("--layers", type=int, default=4)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-ns", type=float, default=20.0)
    ap.add_argument("--rollout-traj", type=int, default=8)
    ap.add_argument("--max-rollout-steps", type=int, default=20000)
    ap.add_argument("--out", default="model_d_out")
    ap.add_argument("--device", default=None)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    torch.manual_seed(args.seed)
    dev = torch.device(args.device or ("mps" if torch.backends.mps.is_available() else "cpu"))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    train_all = load_dir(args.train, args.max_train)
    test = load_dir(args.test, args.max_test)
    train_names = {p["name"] for p in train_all}
    test = [p for p in test if p["name"] not in train_names]
    n_val = max(1, len(train_all) // 10)
    val, train = train_all[:n_val], train_all[n_val:]
    cfg = dict(r=args.r, hist=args.hist, ar=args.ar, pred_chi=args.pred_chi, hidden=args.hidden,
               layers=args.layers, steps=args.steps)
    print(f"Zařízení {dev}; trénink {len(train)}, validace {len(val)}, test {len(test)} (nevidět); {cfg}")

    net = NetD(r=args.r, hist=args.hist, ar=args.ar, hidden=args.hidden, layers=args.layers).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    log = []
    for step in range(1, args.steps + 1):
        loss, _, _ = loss_and_ll(net, dev, *sample_batch(train, args.batch, args.lags, args.hist, rng),
                                 args.pred_chi)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 1000 == 0 or step == args.steps:
            with torch.no_grad():
                _, vpp, vchi = loss_and_ll(net, dev, *sample_batch(val, 8192, args.lags, args.hist, rng),
                                           args.pred_chi)
            vc = float(vchi.mean()) if len(vchi) else float("nan")
            log.append(dict(step=step, train_loss=loss.item(), val_bits_phipsi=float(vpp.mean()), val_bits_chi=vc))
            print(f"krok {step:6d}  loss {loss.item():.3f}  val bity φψ {float(vpp.mean()):.3f}  χ1 {vc:.3f}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    torch.save(net.state_dict(), out / "model_d.pt")

    res = dict(config=cfg, train=len(train), val=len(val), test=len(test),
               train_s=round(time.time() - t0, 1), log=log, lags=dict())
    net.eval()
    erng = np.random.default_rng(args.seed + 1)  # stejné testovací body pro všechny varianty
    L = test[0]["types"].shape[0]
    inner = [n for n in range(L) if not np.isnan(test[0]["ang"][0, n, :2]).any()]
    margs = type_marginals(train)
    for lag in args.lags:
        b_pp, b_chi, b_marg, b_n18, b_n36 = [], [], [], [], []
        for p in test:
            T = p["ang"].shape[0]
            idx_t = erng.integers(T // 2 + (args.hist - 1) * lag, T - lag, size=2000)
            for n in range(L):
                types = np.repeat(p["types"][None], len(idx_t), 0)
                frames = [p["ang"][idx_t - k * lag] for k in range(args.hist)]
                with torch.no_grad():
                    _, pp, ch = loss_and_ll(net, dev, types, frames, p["ang"][idx_t + lag],
                                            np.full(len(idx_t), n), np.full(len(idx_t), lag), args.pred_chi)
                if len(ch):
                    b_chi.append(float(ch.mean()))
                if n in inner:
                    b_pp.append(float(pp.mean()))
                    bm, bn = baselines_ll(margs, p, lag, idx_t, n)
                    b_marg.append(bm)
                    b_n18.append(bn)
                    b_n36.append(native36_ll(p, lag, idx_t, n))
        n_steps = max(2, min(args.max_rollout_steps, int(args.rollout_ns * 1000 / lag)))
        t1 = time.time()
        traj = rollout(net, dev, test, lag, n_steps, args.rollout_traj, args.pred_chi, erng)
        t_roll = time.time() - t1
        js_m, js_n, jc_m, jc_n, rate_lr = [], [], [], [], []
        for i, p in enumerate(test):
            a = p["ang"]
            T = a.shape[0]
            for n in range(L):
                if n in inner:
                    js_m.append(js(rama_hist(traj[i, :, :, n, 0].ravel(), traj[i, :, :, n, 1].ravel()),
                                   rama_hist(a[:, n, 0], a[:, n, 1])))
                    js_n.append(js(rama_hist(a[:T // 2, n, 0], a[:T // 2, n, 1]),
                                   rama_hist(a[T // 2:, n, 0], a[T // 2:, n, 1])))
                    r_md = ab_rate(a[::lag, n, 1], lag)
                    if r_md > 0:
                        rate_lr.append(np.log2(max(ab_rate(traj[i, :, :, n, 1], lag), 1e-3) / r_md))
                if not np.isnan(a[0, n, 2]):
                    jc_m.append(js(chi_hist(traj[i, :, :, n, 2]), chi_hist(a[:, n, 2])))
                    jc_n.append(js(chi_hist(a[:T // 2, n, 2]), chi_hist(a[T // 2:, n, 2])))
        rate_lr = np.array(rate_lr)
        r_ = dict(bits_phipsi_model=float(np.mean(b_pp)), bits_marginal_by_type=float(np.mean(b_marg)),
                  bits_native18=float(np.mean(b_n18)), bits_native36=float(np.mean(b_n36)),
                  bits_chi_model=float(np.mean(b_chi)) if b_chi else None,
                  JS_rama_median=float(np.median(js_m)), JS_rama_MD_halves_median=float(np.median(js_n)),
                  rama_within_2x_noise=float(np.mean(np.array(js_m) <= 2 * np.array(js_n))),
                  JS_chi_median=float(np.median(jc_m)), JS_chi_MD_halves_median=float(np.median(jc_n)),
                  chi_within_2x_noise=float(np.mean(np.array(jc_m) <= 2 * np.array(jc_n))),
                  ab_rate_log2_ratio_median=float(np.median(rate_lr)),
                  ab_rate_within_2x=float(np.mean(np.abs(rate_lr) <= 1)),
                  rollout_ns_per_traj=n_steps * lag / 1000, rollout_s=round(t_roll, 1))
        res["lags"][f"{lag}ps"] = r_
        print(f"τ={lag:>3} ps: bity φψ model {r_['bits_phipsi_model']:.2f} | marginála {r_['bits_marginal_by_type']:.2f} "
              f"| nativní 18 {r_['bits_native18']:.2f} / 36 {r_['bits_native36']:.2f} | χ1 {r_['bits_chi_model']}\n"
              f"        rollout {r_['rollout_ns_per_traj']:.0f} ns: JS Rama {r_['JS_rama_median']:.4f} "
              f"(šum {r_['JS_rama_MD_halves_median']:.4f}, v 2× {100 * r_['rama_within_2x_noise']:.0f} %); "
              f"JS χ1 {r_['JS_chi_median']:.4f} (šum {r_['JS_chi_MD_halves_median']:.4f}, "
              f"v 2× {100 * r_['chi_within_2x_noise']:.0f} %); rychlost α↔β log2 poměr "
              f"{r_['ab_rate_log2_ratio_median']:+.2f}, do 2× {100 * r_['ab_rate_within_2x']:.0f} % | {t_roll:.0f} s",
              flush=True)
    (out / "model_d_summary.json").write_text(json.dumps(res, indent=2))
    print("Uloženo", out / "model_d_summary.json")


if __name__ == "__main__":
    main()
