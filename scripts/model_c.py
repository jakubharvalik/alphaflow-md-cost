"""Model C: neuronový propagátor torzních úhlů, přenositelný mezi sekvencemi.

Pro reziduum n předpovídá rozdělení (φ, ψ) v čase t+τ z:
  - typů reziduí v okně n±r (embedding) a polohy v řetězci,
  - úhlů φ, ψ, χ1 v okně v čase t a t−τ (historie kvůli nemarkovskosti),
  - délky kroku τ (jeden model pro 1, 10 i 100 ps).
Výstup: p(φ' | kontext) · p(ψ' | φ', kontext), biny 10° (36 tříd na úhel).

Trénink na peptidech z --train (npz z fetch_mdgen.py), test na NEVIDĚNÝCH sekvencích z --test:
  1. log2-věrohodnost přechodů (bity na reziduum) vs. základní modely:
     marginála podle typu (bez dynamiky) a „nativní“ tabulka z první poloviny testovací MD,
  2. rollout („simulace simulace“) a JS divergence Ramachandranových rozdělení vs. MD,
     s šumem MD = JS mezi první a druhou polovinou MD.

    python model_c.py --train public/mdgen_tors --test public/mdgen_tors_test --steps 20000
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

AA = ["ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
      "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL"]
ALIAS = {"HID": "HIS", "HIE": "HIS", "HIP": "HIS", "CYX": "CYS"}
PAD = len(AA)  # „žádné reziduum“ mimo řetězec
NB = 36
NO_PHI = NB  # φ není definováno (N-konec)


def load_dir(d, max_n=None):
    files = sorted(Path(d).glob("*.npz"))
    files = [f for f in files if not f.name.endswith(".tmp.npz")][:max_n]
    peps = []
    for f in files:
        z = np.load(f)
        seq = [ALIAS.get(s, s) for s in z["seq"].tolist()]
        types = np.array([AA.index(s) if s in AA else PAD for s in seq], np.int64)
        ang = np.stack([z["phi"], z["psi"], z["chi1"]], -1).astype(np.float32)  # (T, L, 3)
        peps.append(dict(name=f.stem, types=types, ang=ang))
    return peps


def to_bin(a):
    return np.clip(((a + np.pi) / (2 * np.pi) * NB).astype(np.int64), 0, NB - 1)


def bin_to_angle(b, rng):
    return (-np.pi + (b + rng.random(b.shape)) * 2 * np.pi / NB).astype(np.float32)


def ang_feats(a):
    """(…, 3) úhly s NaN → (…, 9): sin, cos, maska."""
    m = ~np.isnan(a)
    a0 = np.where(m, a, 0.0)
    return np.concatenate([np.sin(a0) * m, np.cos(a0) * m, m.astype(np.float32)], -1)


class Net(nn.Module):
    def __init__(self, r=1, emb=16, hidden=512):
        super().__init__()
        self.r = r
        w = 2 * r + 1
        self.emb = nn.Embedding(len(AA) + 1, emb)
        d_in = w * (emb + 9 + 9 + 2) + 1
        self.body = nn.Sequential(nn.Linear(d_in, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU(),
                                  nn.Linear(hidden, hidden), nn.SiLU())
        self.head_phi = nn.Linear(hidden, NB)
        self.phi_emb = nn.Embedding(NB + 1, 32)
        self.head_psi = nn.Sequential(nn.Linear(hidden + 32, hidden), nn.SiLU(), nn.Linear(hidden, NB))

    def forward(self, types_w, feat_now_w, feat_prev_w, pos_w, loglag):
        x = torch.cat([self.emb(types_w), feat_now_w, feat_prev_w, pos_w], -1)
        x = torch.cat([x.flatten(1), loglag[:, None]], -1)
        return self.body(x)

    def psi_logits(self, h, phi_bin):
        return self.head_psi(torch.cat([h, self.phi_emb(phi_bin)], -1))


def make_context(types, ang_now, ang_prev, n, r):
    """types (B, L), ang_* (B, L, 3), n (B,) → okno n±r jako numpy pole."""
    B, L = types.shape
    offs = np.arange(-r, r + 1)
    idx = n[:, None] + offs[None, :]
    inside = (idx >= 0) & (idx < L)
    idc = np.clip(idx, 0, L - 1)
    rows = np.arange(B)[:, None]
    t_w = np.where(inside, types[rows, idc], PAD)
    f_now = ang_feats(ang_now[rows, idc]) * inside[..., None]
    f_prev = ang_feats(ang_prev[rows, idc]) * inside[..., None]
    pos = np.stack([(idx == 0), (idx == L - 1)], -1).astype(np.float32) * inside[..., None]
    return t_w, f_now.astype(np.float32), f_prev.astype(np.float32), pos


def sample_batch(peps, B, lags, rng):
    pi = rng.integers(len(peps), size=B)
    lag = rng.choice(lags, size=B)
    L = peps[0]["types"].shape[0]
    n = rng.integers(L, size=B)
    types = np.stack([peps[p]["types"] for p in pi])
    now, prev, nxt = [], [], []
    for p, g in zip(pi, lag):
        T = peps[p]["ang"].shape[0]
        t = rng.integers(g, T - g)
        a = peps[p]["ang"]
        prev.append(a[t - g]); now.append(a[t]); nxt.append(a[t + g])
    return pi, types, np.stack(now), np.stack(prev), np.stack(nxt), n, lag


def tensors(dev, *arrs):
    return [torch.as_tensor(a, device=dev) for a in arrs]


def loss_and_ll(net, dev, types, now, prev, nxt, n, lag):
    t_w, f_now, f_prev, pos = make_context(types, now, prev, n, net.r)
    phi_t = nxt[np.arange(len(n)), n, 0]
    psi_t = nxt[np.arange(len(n)), n, 1]
    has_phi, has_psi = ~np.isnan(phi_t), ~np.isnan(psi_t)
    phi_b = np.where(has_phi, to_bin(np.nan_to_num(phi_t)), NO_PHI)
    psi_b = to_bin(np.nan_to_num(psi_t))
    t_w, f_now, f_prev, pos, ll_, phi_b_t, psi_b_t, hphi, hpsi = tensors(
        dev, t_w, f_now, f_prev, pos, np.log(lag).astype(np.float32), phi_b, psi_b, has_phi, has_psi)
    h = net(t_w, f_now, f_prev, pos, ll_)
    lp_phi = F.log_softmax(net.head_phi(h), -1)
    lp_psi = F.log_softmax(net.psi_logits(h, phi_b_t), -1)
    nll_phi = -lp_phi.gather(1, phi_b_t.clamp(max=NB - 1)[:, None])[:, 0] * hphi
    nll_psi = -lp_psi.gather(1, psi_b_t[:, None])[:, 0] * hpsi
    loss = (nll_phi.sum() + nll_psi.sum()) / (hphi.sum() + hpsi.sum())
    joint = (nll_phi + nll_psi) / np.log(2)  # bity na reziduum
    both = hphi & hpsi
    return loss, joint[both]


@torch.no_grad()
def rollout(net, dev, pep_list, lag, n_steps, n_traj, rng):
    """Synchronní rollout všech peptidů najednou. Vrací (P, n_traj, n_steps, L, 2) úhly φ, ψ."""
    L = pep_list[0]["types"].shape[0]
    P = len(pep_list)
    types = np.repeat(np.stack([p["types"] for p in pep_list]), n_traj, 0)  # (P*n_traj, L)
    starts = []
    for p in pep_list:
        ts = rng.integers(lag, p["ang"].shape[0], size=n_traj)
        starts.append(np.stack([p["ang"][ts - lag], p["ang"][ts]], 1))
    st = np.concatenate(starts)  # (P*n_traj, 2, L, 3)
    prev, now = st[:, 0].copy(), st[:, 1].copy()
    chi = now[..., 2].copy()  # χ1 se nepredikuje, drží se poslední známá hodnota
    out = np.empty((P * n_traj, n_steps, L, 2), np.float32)
    R = P * n_traj
    for s in range(n_steps):
        out[:, s] = now[..., :2]
        new = now.copy()
        for n in range(L):
            nn_ = np.full(R, n)
            t_w, f_now, f_prev, pos = make_context(types, now, prev, nn_, net.r)
            t_w, f_now, f_prev, pos, ll_ = tensors(dev, t_w, f_now, f_prev, pos,
                                                   np.full(R, np.log(lag), np.float32))
            h = net(t_w, f_now, f_prev, pos, ll_)
            has_phi = ~np.isnan(now[:, n, 0])
            pb = torch.multinomial(F.softmax(net.head_phi(h), -1), 1)[:, 0].cpu().numpy()
            pb = np.where(has_phi, pb, NO_PHI)
            qb = torch.multinomial(F.softmax(net.psi_logits(h, torch.as_tensor(pb, device=dev)), -1),
                                   1)[:, 0].cpu().numpy()
            new[:, n, 0] = np.where(has_phi, bin_to_angle(np.minimum(pb, NB - 1), rng), np.nan)
            has_psi = ~np.isnan(now[:, n, 1])
            new[:, n, 1] = np.where(has_psi, bin_to_angle(qb, rng), np.nan)
        new[..., 2] = chi
        prev, now = now, new
    return out.reshape(P, n_traj, n_steps, L, 2)


def rama_hist(phi, psi, nb=18):
    m = ~(np.isnan(phi) | np.isnan(psi))
    h, _, _ = np.histogram2d(phi[m], psi[m], bins=nb, range=[[-np.pi, np.pi], [-np.pi, np.pi]])
    return h.ravel() + 1e-9


def js(h1, h2):
    p, q = h1 / h1.sum(), h2 / h2.sum()
    m = 0.5 * (p + q)
    kl = lambda a, b: float(np.sum(a * np.log2(a / b)))
    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def type_marginals(train):
    """Marginála (φ, ψ) podle typu rezidua z tréninku, 36×36 binů."""
    hist = np.ones((len(AA) + 1, NB * NB))
    for p in train:
        for m, typ in enumerate(p["types"]):
            x = p["ang"][::10, m, :2]
            ok = ~np.isnan(x).any(1)
            np.add.at(hist[typ], to_bin(x[ok, 0]) * NB + to_bin(x[ok, 1]), 1)
    return hist / hist.sum(1, keepdims=True)


def baselines_ll(margs, test_pep, lag, idx_t, n):
    """Bity na reziduum pro marginálu podle typu a nativní tabulku (první polovina testovací MD)."""
    a = test_pep["ang"]
    T = a.shape[0]
    marg = margs[test_pep["types"][n]]
    # nativní tabulka 18×18 z první poloviny, rozprostřená na 36×36
    half = T // 2
    x0 = a[:half - lag, n, :2]
    x1 = a[lag:half, n, :2]
    ok = ~(np.isnan(x0).any(1) | np.isnan(x1).any(1))
    b18 = lambda v: np.clip(((v + np.pi) / (2 * np.pi) * 18).astype(int), 0, 17)
    k0 = b18(x0[ok, 0]) * 18 + b18(x0[ok, 1])
    k1 = b18(x1[ok, 0]) * 18 + b18(x1[ok, 1])
    m18 = (np.bincount(k1, minlength=324) + 1.0) / (len(k1) + 324)
    C = np.zeros((324, 324))
    np.add.at(C, (k0, k1), 1)
    tab = (C + 5 * m18) / (C.sum(1, keepdims=True) + 5)
    cur, nxt = a[idx_t, n, :2], a[idx_t + lag, n, :2]
    j = to_bin(nxt[:, 0]) * NB + to_bin(nxt[:, 1])
    ll_marg = -np.log2(marg[j])
    ll_nat = -np.log2(tab[b18(cur[:, 0]) * 18 + b18(cur[:, 1]), b18(nxt[:, 0]) * 18 + b18(nxt[:, 1])] / 4)
    return float(ll_marg.mean()), float(ll_nat.mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--max-train", type=int, default=None)
    ap.add_argument("--max-test", type=int, default=None)
    ap.add_argument("--lags", type=int, nargs="+", default=[1, 10, 100], help="v snímcích (1 ps)")
    ap.add_argument("--r", type=int, default=1)
    ap.add_argument("--steps", type=int, default=20000)
    ap.add_argument("--batch", type=int, default=2048)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--rollout-ns", type=float, default=20.0)
    ap.add_argument("--rollout-traj", type=int, default=8)
    ap.add_argument("--max-rollout-steps", type=int, default=5000)
    ap.add_argument("--out", default="model_c_out")
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
    print(f"Zařízení {dev}; trénink {len(train)} peptidů, validace {len(val)}, test {len(test)} (nevidět)")

    net = Net(r=args.r).to(dev)
    opt = torch.optim.AdamW(net.parameters(), lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, args.steps)
    t0 = time.time()
    log = []
    for step in range(1, args.steps + 1):
        _, *b = sample_batch(train, args.batch, args.lags, rng)
        loss, _ = loss_and_ll(net, dev, *b)
        opt.zero_grad()
        loss.backward()
        opt.step()
        sched.step()
        if step % 500 == 0 or step == args.steps:
            with torch.no_grad():
                _, *vb = sample_batch(val, 8192, args.lags, rng)
                _, vll = loss_and_ll(net, dev, *vb)
            log.append(dict(step=step, train_loss=loss.item(), val_bits=float(vll.mean())))
            print(f"krok {step:6d}  loss {loss.item():.3f}  val bity/reziduum {float(vll.mean()):.3f}  "
                  f"{time.time() - t0:.0f} s", flush=True)
    torch.save(net.state_dict(), out / "model_c.pt")

    res = dict(train=len(train), val=len(val), test=len(test), r=args.r, steps=args.steps,
               train_s=round(time.time() - t0, 1), log=log, lags=dict())
    net.eval()
    L = test[0]["types"].shape[0]
    inner = [n for n in range(L) if not np.isnan(test[0]["ang"][0, n, :2]).any()]
    margs = type_marginals(train)
    for lag in args.lags:
        bits_model, bits_marg, bits_nat = [], [], []
        for p in test:
            T = p["ang"].shape[0]
            idx_t = rng.integers(T // 2 + lag, T - lag, size=2000)  # druhá polovina = test
            for n in inner:
                with torch.no_grad():
                    types = np.repeat(p["types"][None], len(idx_t), 0)
                    _, ll = loss_and_ll(net, dev, types, p["ang"][idx_t], p["ang"][idx_t - lag],
                                        p["ang"][idx_t + lag], np.full(len(idx_t), n), np.full(len(idx_t), lag))
                bits_model.append(float(ll.mean()))
                bm, bn = baselines_ll(margs, p, lag, idx_t, n)
                bits_marg.append(bm)
                bits_nat.append(bn)
        n_steps = max(2, min(args.max_rollout_steps, int(args.rollout_ns * 1000 / lag)))
        t1 = time.time()
        traj = rollout(net, dev, test, lag, n_steps, args.rollout_traj, rng)
        t_roll = time.time() - t1
        js_model, js_noise = [], []
        for i, p in enumerate(test):
            T = p["ang"].shape[0]
            for n in inner:
                h_md = rama_hist(p["ang"][:, n, 0], p["ang"][:, n, 1])
                h_gen = rama_hist(traj[i, :, :, n, 0].ravel(), traj[i, :, :, n, 1].ravel())
                h1 = rama_hist(p["ang"][:T // 2, n, 0], p["ang"][:T // 2, n, 1])
                h2 = rama_hist(p["ang"][T // 2:, n, 0], p["ang"][T // 2:, n, 1])
                js_model.append(js(h_gen, h_md))
                js_noise.append(js(h1, h2))
        r_ = dict(bits_model=float(np.mean(bits_model)), bits_marginal_by_type=float(np.mean(bits_marg)),
                  bits_native_table=float(np.mean(bits_nat)),
                  JS_rollout_median=float(np.median(js_model)), JS_rollout_mean=float(np.mean(js_model)),
                  JS_MD_halves_median=float(np.median(js_noise)),
                  frac_within_2x_noise=float(np.mean(np.array(js_model) <= 2 * np.array(js_noise))),
                  rollout_ns_per_traj=n_steps * lag / 1000, rollout_s=round(t_roll, 1))
        res["lags"][f"{lag}ps"] = r_
        print(f"τ={lag:>3} ps: bity/reziduum model {r_['bits_model']:.2f} | marginála {r_['bits_marginal_by_type']:.2f} "
              f"| nativní tabulka {r_['bits_native_table']:.2f}; JS rollout medián {r_['JS_rollout_median']:.4f} "
              f"vs šum MD {r_['JS_MD_halves_median']:.4f}; v 2× šumu {100 * r_['frac_within_2x_noise']:.0f} % "
              f"| rollout {t_roll:.0f} s", flush=True)
    (out / "model_c_summary.json").write_text(json.dumps(res, indent=2))
    print("Uloženo", out / "model_c_summary.json")


if __name__ == "__main__":
    main()
