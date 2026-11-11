"""v13 (Gemini 38j/38k, sel38): --ens-select ed = K cílů pro --ens-strat se vybere hladovou minimalizací energy
distance k celému souboru v prvních --ens-ed-pcs složkách PCA jádra souboru (rovné váhy), první bod náhodně z 10
nejbližších medoidu (los 2000 + seed), u seed 0 přímo medoid (= výběr ověřený offline v sel38); offline na 7jfl_C
TICA o ~0,05 lepší než PC1 a s menším rozptylem. Různé starty dávají skoro disjunktní, stejně dobré výběry.
Výchozí --ens-select pc1 = beze změny proti v12.
v11 (test C podle Gemini 38d): --ens-strat K = stratifikovaný výběr cílů: vzorky souboru se rozdělí podle PC1
(PCA jádra souboru) na K stejně početných skupin, z každé jeden náhodný vzorek (los 2000 + seed); při každém skoku
se z dosud nepoužitých vybere ten nejbližší současnému stavu x (RMSD jádra), tj. hladová cesta souborem s kratšími
skoky. Po vyčerpání skupin náhodný vzorek jako dřív. Záznam skoku: ens_strat_group.
v10 (samostudium 38 / test C, 7. 11. 2026 po B na 7jfl_C): --ens-tors sample = torze φ, ψ, χ1 řízení z téhož
vylosovaného vzorku y souboru jako Cα (jen s --ens-noise ou; s --ens-lam 1 --ens-gamma 1 --ens-ends drift je cíl celý
vzorek y: Cα i torze konzistentní, „roj startů ze souboru“ postupně v jedné krabici, MD 1 ns pak opraví lokální strukturu).
Orákulum 27b ukázalo, že samotné Cα bez konzistentních torzí nestačí (PCA 0,58 proti 0,28). Záznam skoku: ens_idx.
v9 (samostudium 38, 7. 11. 2026, oracle38 = oracle36b + --mode ens podle oponentury Gemini 38b): skok s driftem ke
středu vygenerovaného souboru (ESMFlow-MD / AlphaFlow-MD, Jing et al. 2024, soubor PDB s více modely, --ens-pdb).
Střed μ = průměr --ens-m náhodných vzorků souboru (Cα, přeložené podle jádra bez --ens-trim reziduí na konci; los podle
--seed). Při skoku se μ (a celý soubor jednou rotací) přeloží na současné Cα x podle jádra a cíl Cα je
  x + λ·(μ − x) [RMS posunu jádra nejvýš --ens-cap nm] + šum,
šum: model = vzorek modelu skoku (jump28) minus průměr --ens-k-noise vzorků × sqrt(n/(n − 1)) × --ens-noise-scale
(drift modelu se nahradí driftem ke středu souboru, rozptyl zůstane z modelu); ou = γ·sqrt(λ(2 − λ))·(y − μ), y náhodný
vzorek souboru (přechod Ornstein-Uhlenbeck, Gemini 38b); none = bez šumu. Torze φ, ψ, χ1 vždy z modelu skoku
(--tors model, s --bb-consensus jako v režimu model), torze souboru se nepoužijí (šroubovice AF2 −72°/−33°, diag38).
--ens-once: kontrola B0, jediný skok do μ (λ = 1, bez šumu a stropu) hned v čase 0, pak už jen MD (celkem --total-ns).
Záznam skoku: ens_dist_mu (RMS jádra x → μ), ens_shift, ens_capped, ens_noise (nm).
Podle Gemini 38c: průměr μ má na volných koncích zkrácené vzdálenosti, proto drift ke středu jen na jádru; konce
(--ens-trim reziduí na každé straně) berou vzorek modelu skoku celý (--ens-ends model). Kontrola B0 skáče do medoidu
32 vzorků (nejmenší součet RMSD jádra k ostatním; skutečná struktura), ne do průměru (--ens-once-target medoid).
v8 (krok 36c, oracle36b = oracle36 + podle oponentury Gemini 36b): --local-reps omezí databázi lokálního orákula
na vybrané repliky (např. 2,3 = bez R1, hodnocení proti vynechané R1), --local-tries: když řízení na vylosovaný snímek
selže, losuje se znovu ze zbylých sousedů (bez vracení), nejvýš tolik pokusů. Záznam local_tries (repliky databáze jsou v args).

v7 (krok 36, oracle36 = oracle35 + --mode local podle oponentury Gemini 35d): lokální orákulum bez přibíjení k R1.
Režim same bere cíl vždy z repliky --rep v čase t + skok bez ohledu na to, kde hybrid je (horní mez). --mode local
napodobí dokonalý podmíněný model p(x za --jump-ns | x): ze všech snímků ATLAS (všechny repliky, jen snímky, které mají
--jump-ns budoucnost) najde --local-k nejbližších současnému stavu z různých průchodů (vybrané snímky téže repliky
od sebe aspoň --local-sep-ps; sousední snímky po 10 ps jsou jeden průchod) a jeden vylosuje s vahou
exp(−(d² − d_min²)/2σ²) (σ = medián vzdáleností těch K, nebo --local-sigma); cíl = snímek téže repliky o --jump-ns dál
(Cα a s --tors atlas i φ, ψ, χ1), Cα cíle přeložené na současné Cα podle jádra (konce cíle jádrem neotáčí).
Vzdálenost d: Cα přeložené na krystal (snímek 0 R1) podle jádra bez --local-trim reziduí na každém konci, vážený RMSD
s vahou 1 na jádru a --local-end-w na koncích. Záznam: local_d_match, local_d_nn, local_sigma, local_rep, local_t_match_ps,
local_repeat (vybraný snímek do ±3 ns od dřívějšího výběru v téže replice; nic se nevylučuje, jen se zaznamená),
local_n_reps (z kolika replik je K sousedů), local_span_ns (rozpětí jejich časů).

v6 (krok 35, oracle35 = oracle34 + výběr cíle podle oponentury Gemini 34c): dva levné zdroje směru bez tréninku.
--tabu-k K (s --mode anm): vylosuje K náhodných cílů ANM (každý přesně --kick-rmsd) a vybere ten v nejméně
navštívené oblasti: nejmenší hustota sum_m exp(−d_m²/2σ²) / M přes snímky vlastní trajektorie po --tabu-stride-ps
(d_m = RMSD Cα po přeložení, σ = --tabu-sigma); směr „pryč od prozkoumaného“. Nejmenší RMSD k navštíveným nestačí:
všechny cíle mají od současného stavu přesně --kick-rmsd, takže by rozhodoval jen současný stav. --filt-k K (s --mode model): vylosuje z modelu K cílů najednou, z těch s velikostí v --filt-j
(nm) vybere ten s největším podílem posunu v --filt-modes nejměkčích módech ANM (--filt-cutoff, --anm-kseq); když
žádný velikostí nesedí, vezme ten nejblíž rozmezí. --bb-consensus pak počítá shodu stavu páteře přes všech K vzorků.
Ostatní beze změny jako oracle34 (kontrolní body ckpt34.*).

v5 (krok 34): přerušení a pokračování. SIGTERM běh nezabije hned: v nejbližším bezpečném místě (po 1 ps MD, ne během
řízení) uloží kontrolní bod do --out (ckpt34.pkl + ckpt34_state.xml: stav OpenMM s polohami a rychlostmi, záznamy,
stav losování) a rozpracovaný výstup, pak skončí. Stejný příkaz se stejným --out pak pokračuje od kontrolního bodu;
po dokončení se kontrolní bod smaže. Řízení trvá desítky sekund, proto dát procesu po SIGTERM aspoň 3 minuty.

v4 (krok 34, oracle34 = oracle30 + --mode anm): kontrola „postrčení“. Cíl skoku není z modelu, ale náhodný posun
současných Cα podél --anm-modes nejměkčích módů elastické sítě (ANM z hybrid.anm_displacement, koeficienty ~ N(0, 1/λ),
pružiny do --anm-cutoff nm, sousedé v řetězci --anm-kseq× tužší), sousední Cα vrácené na délky vazeb a celé přeškálované
tak, aby RMSD cíle od současných Cα (po přeložení) bylo přesně --kick-rmsd nm. Řídí se jen Cα (--tors none), torze
dotáhne MD. --kick-rmsd 0 = nulový skok: stejná smyčka, řízení 100 ps drží Cα na místě (vlastní stejně drahá MD).
Rychlosti se po skoku nelosují znovu (jako ve všech režimech), během řízení je přizpůsobuje termostat.

Krok 28 (oracle28 = oracle27 + --mode model): hybrid s modelem velkého skoku. V režimu model je cíl skoku vylosovaný
modelem jump28 (Cα o --jump-ns dál podle současných Cα, torzí a sekvence) místo snímku ATLAS. --tors none: řídí se jen Cα,
torze dotáhne MD; --tors model: řídí se i torze φ, ψ, χ1 vylosované modelem (model s torzemi). Doporučeno řízení
--steer-ps 100 (orákulum 33.7). Ostatní režimy beze změny jako oracle27:

v3 (krok 30, dokument 33.12): model skoku zná velikost pohybu jen přibližně a doučení na krátké MD se přeučí, pro hybrid
ale rozhoduje hlavně velikost skoku. --calib-first-ns C: první úsek MD trvá C ns a z něj se velikost skoku kalibruje
pro tento protein bez tréninku: dvojice snímků vlastní MD vzdálené --calib-lag-ns (od --calib-skip-ns, po 100 ps),
skutečný posun Cα proti posunu, který pro stejné snímky vylosuje model; λ = skutečný / model (oříznuté --calib-clip)
násobí --jump-scale u všech skoků. --bb-consensus f: model po doučení mění stav páteře příliš často (6ono_C: rezidua
mimo ATLAS). Vylosuje se --bb-samples cílů; změnu stavu páteře (α, β, aL) u rezidua cíl ponechá, jen když ji má aspoň
podíl f všech vzorků, jinak řízení drží φ, ψ rezidua na současných hodnotách.

Krok 27b: orákulum velkého skoku (dokument oddíly 32.4 a 33). MD úseky střídá se skokem o --jump-ns, jehož cíl
je skutečný snímek ATLAS: Cα (a volitelně torze φ, ψ, χ1) se řízenou MD (hybrid.steer, polohové restrainty Cα
a torzní restrainty, voda se přizpůsobuje) přenesou do systému ve vodě, pak pokračuje volná MD.

  --mode same : cíl = replika --rep v čase t + skok (správná budoucnost, „dokonalý model“),
  --mode other: cíl = náhodný snímek jiné repliky (správné rozdělení bez paměti),
  --tors atlas: řídit i torze k ATLAS; none: jen Cα (jako budoucí model Cα, torze dotáhne MD).

Výstup ve formátu hybrid.py (torsions.npz, pos.npz, hybrid_summary.json), takže na něj jdou atlas_compare,
pca17, rmsd17 a contacts23 (adresář hyb27_<protein>_<varianta>_100ns_s<seed>). Navíc oracle27_jumps.npz
a v souhrnu údaje o každém skoku: velikost (RMSD Cα po přeložení), odchylka od cíle po řízení, energie a
po 10, 100 a 1000 ps MD vzdálenost od cíle a od stavu před skokem (vrací se MD ke stavu před skokem?).
hybrid.py se nemění.

    python oracle27.py --sys p2_7jfl_C --atlas-dir public/atlas/cache --mode same --tors atlas \
        --total-ns 100 --md-ns 1 --jump-ns 10 --platform OpenCL --seed 0 --out hyb27_7jfl_C_orasame_100ns_s0
"""
import argparse
import json
import signal
import time
from pathlib import Path

import numpy as np
import openmm.unit as u

from hybrid import add_ca_restraints, add_restraints, rotation_sets, steer
from outcond import load_system, torsions
from run_replicas import load, make_sim


def kabsch_onto(P, Q):
    """P přeložené (rotace + posun) na Q, obojí (N, 3)."""
    pc, qc = P.mean(0), Q.mean(0)
    H = (P - pc).T @ (Q - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    return (P - pc) @ R.T + qc


def rmsd_fit(P, Q):
    return float(np.sqrt(((kabsch_onto(P, Q) - Q) ** 2).sum(1).mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sys", required=True)
    ap.add_argument("--atlas-dir", default="public/atlas/cache")
    ap.add_argument("--rep", type=int, default=1)
    ap.add_argument("--mode", choices=["same", "other", "model", "anm", "local", "ens"], default="same")
    ap.add_argument("--ens-pdb", default=None, help="--mode ens: PDB vygenerovaného souboru (více modelů)")
    ap.add_argument("--ens-m", type=int, default=32, help="--mode ens: počet náhodných vzorků pro střed (0 = všechny)")
    ap.add_argument("--ens-trim", type=int, default=4, help="--mode ens: konce mimo jádro (přeložení, strop)")
    ap.add_argument("--ens-lam", type=float, default=0.5, help="--mode ens: λ v x + λ(μ − x)")
    ap.add_argument("--ens-cap", type=float, default=0.25, help="--mode ens: strop RMS posunu jádra (nm), 0 = bez")
    ap.add_argument("--ens-noise", choices=["model", "ou", "none"], default="model")
    ap.add_argument("--ens-k-noise", type=int, default=8, help="--ens-noise model: vzorků modelu pro odečtení průměru")
    ap.add_argument("--ens-noise-scale", type=float, default=1.0)
    ap.add_argument("--ens-gamma", type=float, default=1.0, help="--ens-noise ou: γ")
    ap.add_argument("--ens-once", action="store_true", help="kontrola B0: jediný skok do μ v čase 0, pak jen MD")
    ap.add_argument("--ens-ends", choices=["model", "drift"], default="model",
                    help="konce mimo jádro: model = vzorek modelu skoku celý, drift = jako jádro")
    ap.add_argument("--ens-once-target", choices=["medoid", "mean"], default="medoid", help="cíl kontroly B0")
    ap.add_argument("--ens-strat", type=int, default=0,
                    help="--ens-noise ou: K skupin podle PC1 souboru, cíl = nejbližší nepoužitý zástupce skupiny (0 = náhodně)")
    ap.add_argument("--ens-select", choices=["pc1", "ed"], default="pc1",
                    help="--ens-strat: výběr cílů pc1 = skupiny podle PC1, ed = hladová energy distance (v13)")
    ap.add_argument("--ens-ed-pcs", type=int, default=10, help="--ens-select ed: počet PC souboru")
    ap.add_argument("--ens-tors", choices=["model", "sample"], default="model",
                    help="torze řízení: model = model skoku, sample = vylosovaný vzorek souboru (jen --ens-noise ou)")
    ap.add_argument("--local-k", type=int, default=15, help="--mode local: počet nejbližších snímků ATLAS k losování")
    ap.add_argument("--local-trim", type=int, default=4, help="--mode local: konce (rezidua na každé straně) mimo jádro")
    ap.add_argument("--local-end-w", type=float, default=0.2, help="--mode local: váha konců ve vzdálenosti")
    ap.add_argument("--local-sigma", type=float, default=0.0, help="--mode local: šířka vah (nm); 0 = medián K vzdáleností")
    ap.add_argument("--local-reps", default="", help="--mode local: repliky databáze, např. 2,3 (prázdné = všechny)")
    ap.add_argument("--local-tries", type=int, default=3, help="--mode local: pokusy při selhání řízení (bez vracení)")
    ap.add_argument("--local-sep-ps", type=float, default=500.0,
                    help="--mode local: vybraní sousedé téže repliky od sebe aspoň tolik ps (různé průchody)")
    ap.add_argument("--kick-rmsd", type=float, default=0.27, help="--mode anm: velikost náhodného skoku (RMSD Cα, nm); 0 = nulový")
    ap.add_argument("--anm-cutoff", type=float, default=1.5, help="--mode anm: dosah pružin ANM mezi Cα (nm)")
    ap.add_argument("--anm-modes", type=int, default=20, help="--mode anm: počet nejměkčích módů")
    ap.add_argument("--anm-kseq", type=float, default=10.0, help="--mode anm: tuhost sousedů v řetězci (× ostatní)")
    ap.add_argument("--tabu-k", type=int, default=0, help="--mode anm: z K náhodných cílů vybrat nejvzdálenější od "
                    "navštívených stavů; 0 = vypnuto")
    ap.add_argument("--tabu-stride-ps", type=float, default=20.0, help="--tabu-k: navštívené snímky po tolika ps")
    ap.add_argument("--tabu-sigma", type=float, default=0.2, help="--tabu-k: šířka jádra hustoty navštívených stavů (nm)")
    ap.add_argument("--filt-k", type=int, default=0, help="--mode model: vylosovat K cílů a vybrat podle --filt-j "
                    "a podílu v měkkých módech ANM; 0 = vypnuto")
    ap.add_argument("--filt-j", type=float, nargs=2, default=[0.22, 0.27], help="--filt-k: povolená velikost skoku (nm)")
    ap.add_argument("--filt-modes", type=int, default=10, help="--filt-k: počet nejměkčích módů ANM")
    ap.add_argument("--filt-cutoff", type=float, default=1.3, help="--filt-k: dosah pružin ANM (nm)")
    ap.add_argument("--jump-model", default=None, help="--mode model: adresář modelu jump28")
    ap.add_argument("--jump-device", default="cpu")
    ap.add_argument("--jump-steps", type=int, default=None, help="kroky vzorkování (výchozí z modelu)")
    ap.add_argument("--jump-scale", type=float, default=1.0, help="--mode model: posun cíle x0 + λ·(x̂ − x0)")
    ap.add_argument("--max-tries", type=int, default=1,
                    help="--mode model: když řízení na vylosovaný cíl selže, vylosovat nový cíl (nejvýš tolikrát)")
    ap.add_argument("--calib-first-ns", type=float, default=0.0,
                    help="--mode model: první úsek MD (ns), z něj kalibrace velikosti skoku; 0 = bez kalibrace")
    ap.add_argument("--calib-lag-ns", type=float, default=1.0, help="odstup dvojic pro kalibraci (ns)")
    ap.add_argument("--calib-skip-ns", type=float, default=0.5, help="kalibrace od tohoto času (ns)")
    ap.add_argument("--calib-samples", type=int, default=4, help="vzorků modelu na snímek při kalibraci")
    ap.add_argument("--calib-clip", type=float, nargs=2, default=[0.5, 3.0], help="meze λ")
    ap.add_argument("--bb-consensus", type=float, default=0.0,
                    help="--tors model: změna stavu páteře v cíli jen se shodou aspoň tohoto podílu vzorků; 0 = vypnuto")
    ap.add_argument("--bb-samples", type=int, default=8, help="vzorků cíle pro --bb-consensus")
    ap.add_argument("--tors", choices=["atlas", "none", "model"], default="atlas",
                    help="model: torze φ, ψ, χ1 vylosované modelem skoku (jen s --mode model a modelem s torzemi)")
    ap.add_argument("--total-ns", type=float, default=100.0)
    ap.add_argument("--md-ns", type=float, default=1.0)
    ap.add_argument("--jump-ns", type=float, default=10.0)
    ap.add_argument("--steer-ps", type=float, default=20.0)
    ap.add_argument("--steer-stages", type=int, default=50)
    ap.add_argument("--k-pos", type=float, default=5000.0, help="kJ/mol/nm²")
    ap.add_argument("--k-rest", type=float, default=1000.0, help="kJ/mol/rad²")
    ap.add_argument("--min-other-ns", type=float, default=1.0, help="--mode other: cíl až od tohoto času repliky")
    ap.add_argument("--platform", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    rng = np.random.default_rng(args.seed)
    if args.mode == "ens" and not args.ens_pdb:
        raise SystemExit("--mode ens potřebuje --ens-pdb")
    if args.mode in ("model", "ens"):
        if not args.jump_model:
            raise SystemExit("--mode model/ens potřebuje --jump-model")
        if args.tors == "atlas":
            raise SystemExit("--mode model jen s --tors none nebo model")
        import torch
        from jump28 import load_jump
        jdev = torch.device(args.jump_device)
        jnet, jck = load_jump(args.jump_model, jdev)
        if args.tors == "model" and not jnet.tors:
            raise SystemExit("--tors model potřebuje model skoku s torzemi")
    elif args.tors == "model":
        raise SystemExit("--tors model jen s --mode model nebo ens")
    if args.mode == "local" and args.tors == "model":
        raise SystemExit("--mode local jen s --tors atlas nebo none")
    if args.mode == "anm" and args.tors != "none":
        raise SystemExit("--mode anm jen s --tors none")
    if args.tabu_k > 0 and args.mode != "anm":
        raise SystemExit("--tabu-k jen s --mode anm")
    if args.ens_once and args.mode != "ens":
        raise SystemExit("--ens-once jen s --mode ens")
    if args.ens_strat > 0 and (args.mode != "ens" or args.ens_noise != "ou" or args.ens_once):
        raise SystemExit("--ens-strat jen s --mode ens --ens-noise ou, bez --ens-once")
    if args.ens_select == "ed" and args.ens_strat <= 0:
        raise SystemExit("--ens-select ed jen s --ens-strat K > 0")
    if args.ens_tors == "sample" and (args.mode != "ens" or args.ens_noise != "ou" or args.tors != "model" or args.ens_once):
        raise SystemExit("--ens-tors sample jen s --mode ens --ens-noise ou --tors model, bez --ens-once")
    if args.filt_k > 0 and args.mode != "model":
        raise SystemExit("--filt-k jen s --mode model")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    name = Path(args.sys).name.removeprefix("p2_")
    cache = Path(args.atlas_dir) / name
    meta_a = json.loads((cache / "meta.json").read_text())
    dt_a = float(meta_a["dt_ps"])
    reps = list(range(1, len(meta_a["reps"]) + 1))
    A_ang = {r: np.load(cache / f"R{r}_ang.npy", mmap_mode="r") for r in reps}
    A_ca = {r: np.load(cache / f"R{r}_bb.npy", mmap_mode="r") for r in reps}  # (T, L, 3, 3) N, Cα, C v nm
    if args.mode == "local":
        db = [int(x) for x in args.local_reps.split(",") if x.strip()] or reps
        if not set(db) <= set(reps):
            raise SystemExit(f"--local-reps {args.local_reps}: ATLAS má repliky {reps}")
        local = local_setup(A_ca, db, dt_a, args)
        local["db"] = db
    else:
        local = None

    S = load_system(args.sys)
    Q = S["Q"]
    meta, system, state0, omm_top = load(args.sys)
    n_sol = meta["n_solute"]
    sets = rotation_sets(omm_top, n_sol, Q)
    rest, rest_idx = add_restraints(system, Q, sets)
    ca = np.array([a.index for a in S["top"].atoms if a.name == "CA"])
    posres = add_ca_restraints(system, ca) + (ca,)
    L = len(ca)
    if A_ang[args.rep].shape[1] != L or Q.shape[0] != L:
        raise SystemExit(f"Nesedí počet reziduí: systém {L}/{Q.shape[0]}, ATLAS {A_ang[args.rep].shape[1]}")
    ens = None
    if args.mode == "ens":
        import mdtraj as md
        tr_e = md.load(args.ens_pdb)
        Y = tr_e.xyz[:, tr_e.topology.select("name CA")].astype(np.float64)
        if Y.shape[1] != L:
            raise SystemExit(f"Soubor {args.ens_pdb}: {Y.shape[1]} Cα, systém {L}")
        ang_e = None
        if args.ens_tors == "sample":  # φ, ψ, χ1 vzorků po reziduích (NaN kde úhel není), jako torsion_quads/af38
            ang_e = np.full((tr_e.n_frames, L, 3), np.nan, np.float64)
            for k, fn in enumerate((md.compute_phi, md.compute_psi, md.compute_chi1)):
                idx, val = fn(tr_e)
                for jq, quad in enumerate(idx):
                    ang_e[:, tr_e.topology.atom(int(quad[1])).residue.index, k] = val[:, jq]
            print(f"  torze souboru: {int(np.isfinite(ang_e[0]).sum())} úhlů na vzorek", flush=True)
        ens = ens_setup(Y, args, ang=ang_e)
    sim = make_sim(meta, system, state0, omm_top, seed=1, platform_name=args.platform)
    spp = int(round(1000 / meta["dt_fs"]))
    kT = meta["temperature_K"] * u.kelvin
    sim.context.setState(state0)
    sim.context.setVelocitiesToTemperature(kT, 7000 + args.seed)
    sim.context.setParameter("k_rest", 0.0)
    sim.context.setParameter("k_pos", 0.0)
    steer_args = argparse.Namespace(steer_ps=args.steer_ps, steer_stages=args.steer_stages, k_pos=args.k_pos,
                                    k_rest=args.k_rest)
    pos = lambda: sim.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(u.nanometer)
    print(f"{args.sys}: {meta['n_atoms']} atomů, {L} reziduí, platforma {sim.context.getPlatform().getName()}; "
          f"orákulum {args.mode}, torze {args.tors}; MD {args.md_ns:g} ns + skok {args.jump_ns:g} ns do "
          f"{args.total_ns:g} ns; řízení {args.steer_ps:g} ps / {args.steer_stages} stupňů, k_pos {args.k_pos:g}, "
          f"k_rest {args.k_rest:g}", flush=True)

    X0 = pos()[:n_sol]
    times, angs, poss = [0.0], [torsions(X0, Q)], [X0.astype(np.float32)]
    jumps, pre_l, tgt_l, aft_l = [], [], [], []
    wall = dict(md=0.0, steer=0.0)
    total_ps, md_ps, jump_ps = args.total_ns * 1000, args.md_ns * 1000, args.jump_ns * 1000
    probe = {10.0, 100.0, 1000.0}
    t, t_start, open_jump = 0.0, time.time(), None
    extra = {}
    resume = load_ckpt(out, sim)
    if resume is not None:
        times, angs, poss = resume["times"], resume["angs"], resume["poss"]
        jumps, pre_l, tgt_l, aft_l = resume["jumps"], resume["pre_l"], resume["tgt_l"], resume["aft_l"]
        wall, extra, jump_ps = resume["wall"], resume["extra"], resume["jump_ps"]
        rng.bit_generator.state = resume["rng"]
        t_start = time.time() - resume["elapsed"]
        print(f"  pokračuji z kontrolního bodu v {resume['t']:g} ps (konec úseku MD {resume['seg_end']:g} ps, "
              f"skoků {len(pre_l)}, dosavadní čas {resume['elapsed']:.0f} s)", flush=True)

    class Stop(Exception):
        pass

    stop_req = [False]

    def on_term(signum, frame):
        if not stop_req[0]:
            print("  SIGTERM: končím v nejbližším bezpečném místě (po 1 ps MD) s kontrolním bodem", flush=True)
        stop_req[0] = True

    signal.signal(signal.SIGTERM, on_term)
    partial = False
    try:
        t = run_loop(locals())
    except Stop:
        partial = True
        print(f"  SIGTERM v {times[-1]:g} ps: ukládám rozpracovaný běh (hybrid_summary_partial.json)", flush=True)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)
    save(out, args, times, angs, poss, pre_l, tgt_l, aft_l, jumps, wall, t_start, partial, extra)
    if not partial:
        for f in ("ckpt34.pkl", "ckpt34_state.xml"):
            (out / f).unlink(missing_ok=True)


def save_ckpt(out, sim, d):
    """Kontrolní bod: stav OpenMM (polohy, rychlosti, box) v XML a ostatní v pickle; zápis přes dočasné soubory."""
    import pickle
    from openmm import XmlSerializer
    st = sim.context.getState(getPositions=True, getVelocities=True, getParameters=True)
    (out / "ckpt34_state.xml.tmp").write_text(XmlSerializer.serialize(st))
    with open(out / "ckpt34.pkl.tmp", "wb") as fh:
        pickle.dump(d, fh, protocol=pickle.HIGHEST_PROTOCOL)
    (out / "ckpt34_state.xml.tmp").replace(out / "ckpt34_state.xml")
    (out / "ckpt34.pkl.tmp").replace(out / "ckpt34.pkl")


def load_ckpt(out, sim):
    """Načte kontrolní bod z --out a nastaví stav OpenMM; None, když není."""
    import pickle
    from openmm import XmlSerializer
    ck, xs = out / "ckpt34.pkl", out / "ckpt34_state.xml"
    if not (ck.exists() and xs.exists()):
        return None
    with open(ck, "rb") as fh:
        d = pickle.load(fh)
    sim.context.setState(XmlSerializer.deserialize(xs.read_text()))
    sim.context.setParameter("k_rest", 0.0)
    sim.context.setParameter("k_pos", 0.0)
    return d


def fix_ca_bonds(x, ref, iters=30, max_ref=0.45):
    """Sousední Cα v x vrátí na délky z ref (současná struktura), mimo přerušení řetězce (ref > max_ref nm).
    Stejné jako jump28.fix_ca_bonds (kopie, aby režim anm nepotřeboval torch)."""
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


def anm_kick(c0, rng, kick, cutoff, n_modes, kseq):
    """Náhodný cíl Cα: posun podél nejměkčích módů ANM, vazby Cα–Cα na délkách z c0, přesně RMSD kick po přeložení."""
    c0 = np.asarray(c0, np.float64)
    if kick <= 0:
        return c0.copy(), dict(kick_scale=0.0)
    from hybrid import anm_displacement
    d = anm_displacement(c0, rng, cutoff, n_modes, 1.0, kseq)
    s = kick / max(float(np.sqrt((d ** 2).sum(1).mean())), 1e-9)
    for _ in range(6):
        r = rmsd_fit(fix_ca_bonds(c0 + s * d, c0), c0)
        s *= kick / max(r, 1e-9)
    tgt = kabsch_onto(fix_ca_bonds(c0 + s * d, c0), c0)
    disp = np.sqrt(((tgt - c0) ** 2).sum(1))
    nb = np.linalg.norm(tgt[1:] - tgt[:-1], axis=-1) - np.linalg.norm(c0[1:] - c0[:-1], axis=-1)
    return tgt, dict(kick_scale=round(float(s), 4), kick_max_res_nm=round(float(disp.max()), 3),
                     kick_argmax_res=int(disp.argmax()), bond_dev_max_nm=round(float(np.abs(nb).max()), 4))


def rmsd_many(T, V):
    """RMSD po optimálním přeložení (Kabsch) cíle T (L, 3) ke každému snímku V (M, L, 3), vektorově."""
    T = np.asarray(T, np.float64) - np.mean(T, 0)
    V = np.asarray(V, np.float64)
    V = V - V.mean(1, keepdims=True)
    H = np.einsum("mli,lj->mij", V, T)
    U, S, Wt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Wt))
    S[:, 2] *= d
    e = (V ** 2).sum((1, 2)) + (T ** 2).sum() - 2 * S.sum(1)
    return np.sqrt(np.maximum(e, 0.0) / T.shape[0])


def visited_ca(times, poss, ca, stride_ps):
    """Cα navštívených stavů: snímky vlastní trajektorie po stride_ps (vždy i poslední)."""
    T = np.asarray(times)
    idx = np.where(np.isclose(np.mod(T, stride_ps), 0.0) | np.isclose(np.mod(T, stride_ps), stride_ps))[0]
    idx = np.unique(np.append(idx, len(T) - 1))
    return np.stack([poss[i][ca] for i in idx]).astype(np.float64)


def tabu_kick(c0, rng, args, vis):
    """--tabu-k náhodných cílů ANM, vybrán ten v nejméně navštívené oblasti (nejmenší jádrová hustota snímků)."""
    cands = [anm_kick(c0, rng, args.kick_rmsd, args.anm_cutoff, args.anm_modes, args.anm_kseq) for _ in range(args.tabu_k)]
    D = np.stack([rmsd_many(tg, vis) for tg, _ in cands])  # (K, M)
    dens = np.exp(-D ** 2 / (2 * args.tabu_sigma ** 2)).mean(1)
    k = int(np.argmin(dens))
    tgt, info = cands[k]
    info = dict(info, tabu_density=round(float(dens[k]), 4), tabu_density_med=round(float(np.median(dens)), 4),
                tabu_density_max=round(float(dens.max()), 4), tabu_mean_rmsd=round(float(D[k].mean()), 4),
                tabu_mean_rmsd_med=round(float(np.median(D.mean(1))), 4), tabu_n_visited=int(len(vis)))
    return tgt, info


def align_w(X, ref, sel):
    """X (N, L, 3) → přeložené na ref (L, 3) podle atomů sel (Kabsch po snímcích)."""
    Xc = X - X[:, sel].mean(1, keepdims=True)
    Rc = ref - ref[sel].mean(0)
    H = np.einsum("kni,nj->kij", Xc[:, sel], Rc[sel])
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(U @ Vt))
    D = np.ones((len(X), 3))
    D[:, 2] = d
    R = np.einsum("kij,kj,kjl->kil", U, D, Vt)
    return np.einsum("kni,kij->knj", Xc, R) + ref[sel].mean(0)


def ed_greedy(Z, M, first=None):
    """Hladový výběr M řádků Z (N, p) minimalizující energy distance k celému Z (rovné váhy 1/M); první bod first."""
    D = np.sqrt(((Z[:, None] - Z[None]) ** 2).sum(-1))
    a = D.mean(1)
    S, sumS = [], np.zeros(len(Z))
    for m in range(1, M + 1):
        if m == 1 and first is not None:
            i = int(first)
        else:
            base_a = a[S].sum() if S else 0.0
            base_d = D[np.ix_(S, S)].sum() if S else 0.0
            cand = 2.0 / m * (base_a + a) - (base_d + 2.0 * sumS) / m ** 2
            cand[S] = np.inf
            i = int(np.argmin(cand))
        S.append(i)
        sumS += D[i]
    return S


def ens_setup(Y, args, ang=None):
    """Vygenerovaný soubor (N, L, 3) nm → podmnožina --ens-m vzorků přeložená podle jádra na svůj průměr, střed μ.
    ang (N, L, 3) torze vzorků (--ens-tors sample) se vyberou stejně."""
    L = Y.shape[1]
    core = np.arange(args.ens_trim, L - args.ens_trim)
    n_all = len(Y)
    if 0 < args.ens_m < n_all:
        sel = np.random.default_rng(1000 + args.seed).choice(n_all, args.ens_m, replace=False)
        Y = Y[sel]
        ang = ang[sel] if ang is not None else None
    Ya = align_w(Y, Y[0], core)
    for _ in range(3):
        Ya = align_w(Ya, Ya.mean(0), core)
    mu = Ya.mean(0)
    spread = float(np.sqrt(((Ya[:, core] - mu[core][None]) ** 2).sum(-1).mean()))
    Dm = np.sqrt(((Ya[:, None, core] - Ya[None, :, core]) ** 2).sum(-1).mean(-1))
    im = int(np.argmin(Dm.sum(1)))
    med_mu = float(np.sqrt(((Ya[im, core] - mu[core]) ** 2).sum(-1).mean()))
    print(f"  soubor: {len(Ya)} z {n_all} vzorků, jádro {core[0] + 1}–{core[-1] + 1}, rozptyl kolem středu (RMS jádra) "
          f"{spread:.3f} nm, medoid {im} ({med_mu:.3f} nm od středu); λ {args.ens_lam:g}, strop {args.ens_cap:g} nm, "
          f"šum {args.ens_noise}, konce {args.ens_ends}"
          + (f", kontrola B0 (jediný skok v čase 0 do {args.ens_once_target})" if args.ens_once else ""), flush=True)
    strat, grp = None, None
    if getattr(args, "ens_strat", 0) > 0:  # skupiny podle PC1 jádra souboru, z každé jeden náhodný zástupce
        Z = (Ya[:, core] - mu[core][None]).reshape(len(Ya), -1)
        if getattr(args, "ens_select", "pc1") == "ed":  # v13: hladová energy distance v prvních PC souboru
            if args.ens_strat > len(Ya):
                raise SystemExit(f"--ens-strat {args.ens_strat} > {len(Ya)} vzorků souboru")
            _, Sv, Vt = np.linalg.svd(Z, full_matrices=False)
            p = min(args.ens_ed_pcs, len(Sv))
            Zp = Z @ Vt[:p].T
            Dp = np.sqrt(((Zp[:, None] - Zp[None]) ** 2).sum(-1))
            near = np.argsort(Dp[int(np.argmin(Dp.sum(1)))])[:10]
            r2 = np.random.default_rng(2000 + args.seed)
            strat = ed_greedy(Zp, args.ens_strat, first=None if args.seed == 0 else int(r2.choice(near)))
            print(f"  výběr ED: {args.ens_strat} cílů, {p} PC souboru ({float((Sv[:p] ** 2).sum() / (Sv ** 2).sum()):.2f} "
                  f"rozptylu), zástupci {strat}", flush=True)
        else:
            pc1 = np.linalg.svd(Z, full_matrices=False)[2][0]
            sc = Z @ pc1
            groups = np.array_split(np.argsort(sc), args.ens_strat)
            r2 = np.random.default_rng(2000 + args.seed)
            strat = [int(r2.choice(gi)) for gi in groups]
            print(f"  stratifikace: {args.ens_strat} skupin podle PC1 souboru ({len(groups[0])}–{len(groups[-1])} vzorků), "
                  f"zástupci {strat}", flush=True)
        grp = {i: k for k, i in enumerate(strat)}
    return dict(Y=Ya, mu=mu, core=core, spread=spread, med=Ya[im], med_i=im, ang=ang, strat=strat, strat_grp=grp)


def ens_target(E, c0, tg, kk, rng, args, once, used=()):
    """Cíl Cα skoku režimu ens (před vrácením délek vazeb) a záznam; tg (n, L, 3) vzorky modelu přeložené na c0."""
    core = E["core"]
    pc, qc = E["mu"][core].mean(0), c0[core].mean(0)
    H = (E["mu"][core] - pc).T @ (c0[core] - qc)
    U, _, Vt = np.linalg.svd(H)
    d = np.sign(np.linalg.det(Vt.T @ U.T))
    R = Vt.T @ np.diag([1.0, 1.0, d]) @ U.T
    move = lambda X: (X - pc) @ R.T + qc
    mu = move(E["mu"])
    rc = lambda D: float(np.sqrt((D[core] ** 2).sum(-1).mean()))
    lam = 1.0 if once else args.ens_lam
    if once and args.ens_once_target == "medoid":  # medoid přeložený na c0 podle vlastního jádra
        M = E["med"]
        pm = M[core].mean(0)
        Hm = (M[core] - pm).T @ (c0[core] - qc)
        Um, _, Vtm = np.linalg.svd(Hm)
        dm = np.sign(np.linalg.det(Vtm.T @ Um.T))
        Rm = Vtm.T @ np.diag([1.0, 1.0, dm]) @ Um.T
        mu = (M - pm) @ Rm.T + qc
    sh = lam * (mu - c0)
    s0 = rc(sh)
    capped = bool(not once and args.ens_cap > 0 and s0 > args.ens_cap)
    if capped:
        sh = sh * (args.ens_cap / s0)
    if once or args.ens_noise == "none":
        nz = np.zeros_like(c0)
    elif args.ens_noise == "model":
        n = len(tg)
        nz = (tg[kk] - tg.mean(0)) * (np.sqrt(n / (n - 1)) if n > 1 else 0.0) * args.ens_noise_scale
    else:
        left = [i for i in (E.get("strat") or []) if i not in used]
        if left:  # stratifikace: nejbližší nepoužitý zástupce skupiny (RMSD jádra po přeložení na x)
            Ya = align_w(E["Y"][left], c0, core)
            dl = np.sqrt(((Ya[:, core] - c0[core][None]) ** 2).sum(-1).mean(-1))
            jy = left[int(np.argmin(dl))]
        else:
            jy = int(rng.integers(len(E["Y"])))
        y = move(E["Y"][jy])
        nz = args.ens_gamma * np.sqrt(lam * (2 - lam)) * (y - mu)
    info = dict(ens_dist_mu=round(rc(mu - c0), 4), ens_shift=round(rc(sh), 4), ens_capped=capped,
                ens_noise=round(rc(nz), 4), ens_once=bool(once))
    if not once and args.ens_noise == "ou":
        info["ens_idx"] = jy
        if E.get("strat_grp") is not None:
            info["ens_strat_group"] = E["strat_grp"].get(jy)
    tgt = c0 + sh + nz
    if not once and args.ens_ends == "model":  # konce: vzorek modelu skoku (průměr souboru má konce smrštěné)
        ends = np.ones(len(c0), bool)
        ends[core] = False
        tgt[ends] = tg[kk][ends]
    return tgt, info


def local_setup(A_ca, reps, dt_a, args):
    """Kandidáti lokálního orákula: všechny snímky ATLAS s --jump-ns budoucností, Cα přeložené na krystal podle jádra."""
    ref = np.asarray(A_ca[min(A_ca)][0][:, 1], np.float64)  # krystal = snímek 0 první repliky (i když není v databázi)
    L = len(ref)
    core = np.arange(args.local_trim, L - args.local_trim)
    w = np.full(L, args.local_end_w)
    w[core] = 1.0
    jf = int(round(args.jump_ns * 1000 / dt_a))
    X, rr, ff = [], [], []
    for r in reps:
        n = A_ca[r].shape[0] - jf
        if n <= 0:
            continue
        X.append(align_w(np.asarray(A_ca[r][:n, :, 1], np.float64), ref, core))
        rr.append(np.full(n, r))
        ff.append(np.arange(n))
    X = np.concatenate(X)
    print(f"  lokální orákulum: {len(X)} kandidátních snímků ATLAS, jádro rezidua {core[0] + 1}–{core[-1] + 1}, "
          f"váha konců {args.local_end_w:g}, K {args.local_k}, repliky databáze {list(reps)}", flush=True)
    return dict(ref=ref, core=core, w=w / w.sum(), X=X, rep=np.concatenate(rr), frame=np.concatenate(ff), jf=jf)


def local_pick(loc, c0, rng, args, jump_ps, dt_a, jumps):
    """Vylosuje snímek ATLAS z --local-k nejbližších současnému stavu; vrací (replika, snímek cíle, záznam)."""
    x = align_w(np.asarray(c0, np.float64)[None], loc["ref"], loc["core"])[0]
    d = np.sqrt((((loc["X"] - x[None]) ** 2).sum(-1) * loc["w"][None]).sum(1))
    m = min(len(d), 20000)
    pool = np.argpartition(d, m - 1)[:m]
    pool = pool[np.argsort(d[pool])]
    sep = args.local_sep_ps / dt_a
    nn = []
    for j in pool:  # nejbližší snímky z různých průchodů: téže replice aspoň --local-sep-ps od už vybraných
        if all(loc["rep"][j] != loc["rep"][q] or abs(int(loc["frame"][j]) - int(loc["frame"][q])) >= sep for q in nn):
            nn.append(j)
            if len(nn) == args.local_k:
                break
    nn = np.array(nn)
    sig = args.local_sigma if args.local_sigma > 0 else max(float(np.median(d[nn])), 1e-4)
    p = np.exp(-(d[nn] ** 2 - d[nn[0]] ** 2) / (2 * sig ** 2))
    p = p / p.sum()
    ex = np.sqrt(np.maximum(d[nn] ** 2 - d[nn[0]] ** 2, 0.0))  # pro srovnání: σ podle Gemini 36 = medián přebytku
    sa = max(float(np.median(ex)), 1e-4)
    pa = np.exp(-ex ** 2 / (2 * sa ** 2))
    pa = pa / pa.sum()
    i = int(rng.choice(nn, p=p))
    r, f0, info = local_info_for(loc, i, nn, d, dt_a, jumps)
    info.update(local_d_nn=round(float(d[nn[0]]), 4),
                local_d_k=round(float(d[nn[-1]]), 4), local_sigma=round(sig, 4),
                local_n_reps=int(len(set(loc["rep"][nn].tolist()))),
                local_span_ns=round(float(np.ptp(loc["frame"][nn]) * dt_a / 1000), 2),
                local_neff=round(float(1 / (p ** 2).sum()), 2), local_p_max=round(float(p.max()), 3),
                local_neff_alt=round(float(1 / (pa ** 2).sum()), 2), local_tries=1)
    return r, f0 + loc["jf"], info, dict(nn=nn, p=p, d=d[nn], tried=[i])


def local_info_for(loc, i, nn, d, dt_a, jumps):
    """Replika, snímek a záznam pro vybraného souseda i (d = vzdálenosti všech kandidátů)."""
    r, f0 = int(loc["rep"][i]), int(loc["frame"][i])
    win = int(round(3000 / dt_a))
    rep_hit = any(j.get("local_rep") == r and abs(j.get("local_t_match_ps", -1e9) / dt_a - f0) <= win for j in jumps)
    return r, f0, dict(local_d_match=round(float(d[i]), 4), local_rep=r, local_t_match_ps=f0 * dt_a,
                       local_rank=int(np.where(nn == i)[0][0]), local_repeat=bool(rep_hit))


def local_redraw(loc, cand, rng, dt_a, jumps, info):
    """Po selhání řízení: další soused ze zbylých (bez vracení, stejné váhy); None, když žádný nezbývá."""
    nn, p = cand["nn"], cand["p"]
    m = ~np.isin(nn, cand["tried"])
    if not m.any():
        return None
    q = p * m
    i = int(rng.choice(nn, p=q / q.sum()))
    cand["tried"].append(i)
    dd = np.full(int(max(nn)) + 1, np.nan)
    dd[nn] = cand["d"]
    r, f0, upd = local_info_for(loc, i, nn, dd, dt_a, jumps)
    info.update(upd, local_tries=len(cand["tried"]))
    return r, f0 + loc["jf"]


def anm_modes_local(Xca, cutoff, n_modes, k_seq=1.0):
    """Nejměkčí módy ANM (kopie hybrid.anm_modes): pružiny mezi Cα do cutoff nm, sousedé v řetězci k_seq× tužší;
    vlastní čísla (n,) a ortonormální módy (3N, n) bez 6 tuhých pohybů."""
    N = len(Xca)
    Hm = np.zeros((3 * N, 3 * N))
    for i in range(N):
        d = Xca[i + 1:] - Xca[i]
        r2 = (d ** 2).sum(1)
        for jj in np.where(r2 < cutoff ** 2)[0]:
            j = i + 1 + jj
            b = -np.outer(d[jj], d[jj]) / r2[jj] * (k_seq if j == i + 1 else 1.0)
            Hm[3 * i:3 * i + 3, 3 * j:3 * j + 3] = b
            Hm[3 * j:3 * j + 3, 3 * i:3 * i + 3] = b
            Hm[3 * i:3 * i + 3, 3 * i:3 * i + 3] -= b
            Hm[3 * j:3 * j + 3, 3 * j:3 * j + 3] -= b
    lam, V = np.linalg.eigh(Hm)
    keep = np.where(lam > 1e-6 * lam.max())[0][:n_modes]
    return lam[keep], V[:, keep]


def anm_fraction(c0, tgts, cutoff, n_modes, kseq):
    """Podíl kvadrátu posunu (cíl přeložený na c0 − c0) v n_modes nejměkčích módech ANM struktury c0."""
    _, V = anm_modes_local(np.asarray(c0, np.float64), cutoff, n_modes, kseq)
    out = []
    for tg in tgts:
        d = (kabsch_onto(np.asarray(tg, np.float64), c0) - c0).reshape(-1)
        out.append(float(((V.T @ d) ** 2).sum() / max((d ** 2).sum(), 1e-12)))
    return np.array(out)


def calibrate(args, times, poss, angs, ca, types, jnet, jck, jdev, rng):
    """λ = skutečný posun Cα vlastní MD o --calib-lag-ns / posun vylosovaný modelem ze stejných snímků."""
    from jump28 import sample_jump
    T = np.asarray(times)
    lag = args.calib_lag_ns * 1000
    starts = np.arange(args.calib_skip_ns * 1000, T[-1] - lag + 1e-6, 100.0)
    act, mod = [], []
    for s0 in starts:
        i, j = int(np.argmin(np.abs(T - s0))), int(np.argmin(np.abs(T - (s0 + lag))))
        c0, c1 = poss[i][ca].astype(np.float64), poss[j][ca].astype(np.float64)
        act.append(rmsd_fit(c1, c0))
        xs, _ = sample_jump(jnet, jck, c0, types, angs[i], args.calib_lag_ns, jdev, n=args.calib_samples,
                            steps=args.jump_steps, seed=int(rng.integers(1 << 30)))
        mod += [rmsd_fit(x, c0) for x in xs]
    if not act:
        print(f"  kalibrace: první úsek je kratší než {args.calib_skip_ns:g} + {args.calib_lag_ns:g} ns, λ = 1",
              flush=True)
        return 1.0, dict(lam=1.0, n_pairs=0)
    raw = float(np.mean(act) / max(np.mean(mod), 1e-6))
    lam = float(np.clip(raw, *args.calib_clip))
    return lam, dict(lam=round(lam, 4), lam_raw=round(raw, 4), actual=round(float(np.mean(act)), 4),
                     actual_sd=round(float(np.std(act)), 4), model=round(float(np.mean(mod)), 4),
                     n_pairs=len(act), lag_ns=args.calib_lag_ns)


def run_loop(env):
    """Hlavní smyčka MD + skoky (proměnné z main přes env, seznamy se mění na místě); vrací čas v ps."""
    g = env
    args, sim, pos, n_sol, ca, Q, spp, rng = (g[k] for k in ("args", "sim", "pos", "n_sol", "ca", "Q", "spp", "rng"))
    times, angs, poss, jumps, pre_l, tgt_l, aft_l, wall = (g[k] for k in (
        "times", "angs", "poss", "jumps", "pre_l", "tgt_l", "aft_l", "wall"))
    total_ps, md_ps, jump_ps, probe = g["total_ps"], g["md_ps"], g["jump_ps"], g["probe"]
    A_ang, A_ca, dt_a, reps, meta_a = g["A_ang"], g["A_ca"], g["dt_a"], g["reps"], g["meta_a"]
    rest, rest_idx, steer_args, posres = g["rest"], g["rest_idx"], g["steer_args"], g["posres"]
    jnet, jck, jdev = g.get("jnet"), g.get("jck"), g.get("jdev")
    extra = g["extra"]
    if args.mode in ("model", "ens"):
        from jump28 import bb_state, fix_ca_bonds, sample_jump
    scale = args.jump_scale
    need_calib = args.mode == "model" and args.calib_first_ns > 0
    n_bb = args.bb_samples if args.mode in ("model", "ens") and args.tors == "model" and args.bb_consensus > 0 else 1
    if args.mode == "ens" and args.ens_noise == "model":
        n_bb = max(n_bb, args.ens_k_noise)
    if args.mode == "model" and args.filt_k > 0:
        n_bb = max(n_bb, args.filt_k)
    t, open_jump, seg_resume = 0.0, None, None
    R = g.get("resume")
    if R is not None:
        t, seg_resume, scale, need_calib = R["t"], R["seg_end"], R["scale"], R["need_calib"]
        open_jump = jumps[R["open_jump_idx"]] if R["open_jump_idx"] is not None else None
    stop_req, Stop, out, t_start = g["stop_req"], g["Stop"], g["out"], g["t_start"]

    def stop_here(seg_end):
        save_ckpt(out, sim, dict(t=t, seg_end=seg_end, scale=scale, need_calib=need_calib, jump_ps=jump_ps,
                                 open_jump_idx=(jumps.index(open_jump) if open_jump is not None else None),
                                 rng=rng.bit_generator.state, times=times, angs=angs, poss=poss, jumps=jumps,
                                 pre_l=pre_l, tgt_l=tgt_l, aft_l=aft_l, wall=wall, extra=extra,
                                 elapsed=time.time() - t_start))
        print(f"  kontrolní bod v {t:g} ps uložen (ckpt34.pkl), stejný příkaz pokračuje", flush=True)
        raise Stop()

    while t < total_ps - 1e-9:
        t0 = time.time()
        if seg_resume is not None:
            seg_end, seg_resume = seg_resume, None
        else:
            seg_end = min(t + (args.calib_first_ns * 1000 if need_calib and t == 0.0 else md_ps), total_ps)
            if args.mode == "ens" and args.ens_once and not any(j["ok"] for j in jumps) and t == 0.0:
                seg_end = t  # kontrola B0: skok hned v čase 0
        while t < seg_end - 1e-9:
            sim.step(spp)
            t += 1.0
            X = pos()[:n_sol]
            if not np.isfinite(X).all():
                raise SystemExit(f"NaN v MD v {t:g} ps, končím (uloženo nebude)")
            times.append(t)
            angs.append(torsions(X, Q))
            poss.append(X.astype(np.float32))
            if open_jump is not None:
                since = t - open_jump["t_after_ps"]
                if any(abs(since - p) < 1e-6 for p in probe):
                    c = X[ca]
                    open_jump[f"to_target_{since:g}ps"] = round(rmsd_fit(c, tgt_l[-1]), 4)
                    open_jump[f"to_pre_{since:g}ps"] = round(rmsd_fit(c, pre_l[-1]), 4)
            if stop_req[0]:
                wall["md"] += time.time() - t0
                stop_here(seg_end)
        wall["md"] += time.time() - t0
        if stop_req[0]:
            stop_here(t)
        if need_calib:
            tc = time.time()
            lam, info = calibrate(args, times, poss, angs, ca, np.array(meta_a["types"]), jnet, jck, jdev, rng)
            scale = args.jump_scale * lam
            extra["calib"] = dict(info, jump_scale=round(scale, 4), wall_s=round(time.time() - tc, 1))
            wall["calib"] = round(time.time() - tc, 1)
            print(f"  kalibrace z vlastní MD ({info.get('n_pairs')} dvojic po {args.calib_lag_ns:g} ns): skutečný posun "
                  f"{info.get('actual')} nm, model {info.get('model')} nm → λ {info.get('lam')} (bez ořezu "
                  f"{info.get('lam_raw')}), škála skoku {scale:.3f}", flush=True)
            need_calib = False
        if t + jump_ps > total_ps + 1e-9:
            continue
        # skok na snímek ATLAS
        t0 = time.time()
        if args.mode == "same":
            r = args.rep
            f = int(round((t + jump_ps) / dt_a))
            if f >= A_ang[r].shape[0]:
                print(f"  {t:g} ps: cíl {f} mimo repliku R{r}, dál jen MD", flush=True)
                jump_ps = total_ps + 1.0  # žádné další skoky
                continue
        elif args.mode == "other":
            r = int(rng.choice([x for x in reps if x != args.rep]))
            f = int(rng.integers(int(args.min_other_ns * 1000 / dt_a), A_ang[r].shape[0]))
        else:
            r, f = None, None
        saved = sim.context.getState(getPositions=True, getVelocities=True)
        Xs = saved.getPositions(asNumpy=True).value_in_unit(u.nanometer)[:n_sol]
        c0 = Xs[ca]
        local_info = None
        if args.mode == "local":
            r, f, local_info, local_cand = local_pick(g["local"], c0, rng, args, jump_ps, dt_a, jumps)
        tries = args.max_tries if args.mode in ("model", "anm", "ens") else (args.local_tries if args.mode == "local" else 1)
        ens_info = None
        kick_info = None
        fails = []
        for attempt in range(tries):
            if args.mode == "local" and attempt > 0:  # další soused ze zbylých
                rf = local_redraw(g["local"], local_cand, rng, dt_a, jumps, local_info)
                if rf is None:
                    break
                r, f = rf
            if args.mode in ("model", "ens"):  # Cα vylosované modelem skoku, přeložené na současné Cα
                ts = time.time()
                a_cur = torsions(Xs, Q)
                tg, ta = sample_jump(jnet, jck, c0.astype(np.float64), np.array(meta_a["types"]), a_cur,
                                     args.jump_ns, jdev, n=n_bb, steps=args.jump_steps, seed=int(rng.integers(1 << 30)),
                                     scale=scale)
                kk, filt_info = 0, None
                if args.filt_k > 0:  # výběr z K vzorků: velikost v --filt-j, pak největší podíl v měkkých módech ANM
                    fx = [fix_ca_bonds(x, c0) for x in tg]
                    Js = np.array([rmsd_fit(c0, x) for x in fx])
                    fr = anm_fraction(c0, fx, args.filt_cutoff, args.filt_modes, args.anm_kseq)
                    lo, hi = args.filt_j
                    okj = (Js >= lo) & (Js <= hi)
                    kk = int(np.argmax(np.where(okj, fr, -1.0))) if okj.any() else int(
                        np.argmin(np.maximum(lo - Js, 0) + np.maximum(Js - hi, 0)))
                    filt_info = dict(filt_n_ok=int(okj.sum()), filt_J=round(float(Js[kk]), 4),
                                     filt_frac=round(float(fr[kk]), 4), filt_frac_med=round(float(np.median(fr)), 4),
                                     filt_J_med=round(float(np.median(Js)), 4))
                tgt = fix_ca_bonds(tg[kk], c0)
                if args.mode == "ens":  # drift ke středu vygenerovaného souboru + šum (Gemini 38b)
                    tg_e, ens_info = ens_target(g["ens"], c0, tg, kk, rng, args,
                                                once=args.ens_once and not any(j["ok"] for j in jumps),
                                                used={j["ens_idx"] for j in jumps if "ens_idx" in j}  # i zamítnuté
                                                | {x["ens_idx"] for x in fails if "ens_idx" in x})  # a pokusy teď
                    tgt = fix_ca_bonds(tg_e, c0)
                tors_m = ta[kk].copy() if ta is not None and args.tors == "model" else None
                bb_info = None
                if tors_m is not None and n_bb > 1 and args.bb_consensus > 0:  # změna stavu páteře jen se shodou vzorků
                    s_cur, S_all = bb_state(a_cur), bb_state(ta)
                    chg = (S_all[kk] != s_cur) & (S_all[kk] >= 0) & (s_cur >= 0)
                    agree = (S_all == S_all[kk][None]).mean(0)
                    guard = chg & (agree < args.bb_consensus)
                    tors_m[guard, :2] = np.nan  # řízení drží současné φ, ψ
                    bb_info = dict(bb_changes=int(chg.sum()), bb_guarded=int(guard.sum()))
                if args.mode == "ens" and args.ens_tors == "sample" and ens_info is not None and "ens_idx" in ens_info:
                    tors_m = np.array(g["ens"]["ang"][ens_info["ens_idx"]], np.float64)  # torze téhož vzorku jako Cα
                    bb_info = None
                wall["sample"] = wall.get("sample", 0.0) + time.time() - ts
            elif args.mode == "anm" and args.tabu_k > 0:  # z --tabu-k náhodných cílů ten nejdál od navštívených stavů
                ts = time.time()
                tgt, kick_info = tabu_kick(c0, rng, args, visited_ca(times, poss, ca, args.tabu_stride_ps))
                wall["sample"] = wall.get("sample", 0.0) + time.time() - ts
            elif args.mode == "anm":  # náhodný posun podél měkkých módů ANM o přesně --kick-rmsd
                tgt, kick_info = anm_kick(c0, rng, args.kick_rmsd, args.anm_cutoff, args.anm_modes, args.anm_kseq)
            else:
                tc = np.asarray(A_ca[r][f][:, 1], np.float64)
                if args.mode == "local":  # přeložení podle jádra, aby konce cíle neotočily jádrem
                    cs = g["local"]["core"]
                    tgt = align_w(tc[None], c0, cs)[0]
                else:
                    tgt = kabsch_onto(tc, c0)  # Cα cíle přeložené na současné Cα
            tors_t = np.asarray(A_ang[r][f], np.float64) if args.tors == "atlas" else (
                tors_m if args.mode in ("model", "ens") else None)
            ok, e = steer(sim, saved, Q, rest, rest_idx, tors_t, steer_args, spp, n_sol, tgt, posres)
            if ok:
                break
            D = np.linalg.norm(tgt[:, None] - tgt[None], axis=-1)
            iu = np.triu_indices(len(tgt), 3)
            fails.append(dict(jump_rmsd=round(rmsd_fit(c0, tgt), 4), min_ca_nb=round(float(D[iu].min()), 3)))
            if ens_info is not None and "ens_idx" in ens_info:  # v12: neúspěšný vzorek už znovu nevybírat
                fails[-1]["ens_idx"] = ens_info["ens_idx"]
            sim.context.setState(saved)
            if attempt + 1 < tries:
                print(f"  {t:g} ps: řízení na vylosovaný cíl selhalo (velikost {fails[-1]['jump_rmsd']:.3f} nm, "
                      f"nejbližší Cα |i−j|≥3 {fails[-1]['min_ca_nb']:.3f} nm), losuji znovu", flush=True)
        wall["steer"] += time.time() - t0
        rec = dict(t_before_ps=t, target_rep=r, target_frame=f, target_ps=f * dt_a if f is not None else None,
                   jump_rmsd=round(rmsd_fit(c0, tgt), 4),
                   ok=bool(ok), E_after=round(float(e), 1) if ok else None)
        if fails:
            rec["failed_tries"] = fails
        if kick_info is not None:
            rec.update(kick_info)
        if local_info is not None:
            rec.update(local_info)
        if ens_info is not None:
            rec.update(ens_info)
        if args.mode == "model":
            rec["jump_scale"] = round(scale, 4)
            if bb_info is not None:
                rec.update(bb_info)
            if filt_info is not None:
                rec.update(filt_info)
        if args.mode == "ens" and bb_info is not None:
            rec.update(bb_info)
        if not ok:
            sim.context.setState(saved)
            print(f"  {t:g} ps: řízení na " + (f"R{r}/{f}" if f is not None else "cíl") + " selhalo, skok zamítnut", flush=True)
            rec["rejected"] = True
            jumps.append(rec)
            open_jump = None
            continue
        t += jump_ps
        if args.mode == "ens" and args.ens_once:
            jump_ps = total_ps + 1.0  # kontrola B0: další skoky už ne
        X = pos()[:n_sol]
        a = torsions(X, Q)
        rec["t_after_ps"] = t
        rec["ca_dev_after_steer"] = round(rmsd_fit(X[ca], tgt), 4)
        if tors_t is not None:
            d = np.abs(np.arctan2(np.sin(a - tors_t), np.cos(a - tors_t)))
            rec["tors_dev_deg"] = round(float(np.degrees(np.nanmean(d[:, :2]))), 2)
            rec["chi_dev_deg"] = round(float(np.degrees(np.nanmean(d[:, 2]))), 2)
        times.append(t)
        angs.append(a)
        poss.append(X.astype(np.float32))
        pre_l.append(c0.copy())
        tgt_l.append(tgt.copy())
        aft_l.append(X[ca].copy())
        jumps.append(rec)
        open_jump = rec
        where_ = (f"na R{r} {f * dt_a / 1000:g} ns" + (f" (nejbližší snímek {rec['local_t_match_ps'] / 1000:g} ns, "
                  f"d {rec['local_d_match']:.3f}, nejbližší {rec['local_d_nn']:.3f}, {args.local_k}. {rec['local_d_k']:.3f} nm, "
                  f"z {rec['local_n_reps']} replik v {rec['local_span_ns']:g} ns, neff {rec['local_neff']:g} / alt {rec['local_neff_alt']:g})"
                  if "local_d_match" in rec else "")
                  ) if f is not None else (
            "na náhodný cíl ANM" if args.mode == "anm" else ("na cíl modelu" if args.mode == "model" else
            f"ke středu souboru (od středu {rec['ens_dist_mu']:.3f}, posun {rec['ens_shift']:.3f}"
            + (" strop" if rec["ens_capped"] else "") + f", šum {rec['ens_noise']:.3f} nm)"))
        print(f"  skok {len(pre_l)}: {rec['t_before_ps']:g} → {t:g} ps {where_}, velikost "
              f"{rec['jump_rmsd']:.3f} nm, po řízení od cíle {rec['ca_dev_after_steer']:.3f} nm"
              + (f", torze {rec['tors_dev_deg']}°" if tors_t is not None else "")
              + (f", změny páteře {rec['bb_changes']} (drženo {rec['bb_guarded']})" if "bb_changes" in rec else "")
              + (f", tabu hustota {rec['tabu_density']:.3f} (medián {rec['tabu_density_med']:.3f}), střední RMSD k navštíveným "
                 f"{rec['tabu_mean_rmsd']:.3f} (medián {rec['tabu_mean_rmsd_med']:.3f})" if "tabu_density" in rec else "")
              + (f", filtr {rec['filt_n_ok']}/{args.filt_k} ve velikosti, podíl ANM {rec['filt_frac']:.2f} (medián "
                 f"{rec['filt_frac_med']:.2f})" if "filt_n_ok" in rec else "")
              + f", E {rec['E_after']}", flush=True)
    return t


def save(out, args, times, angs, poss, pre_l, tgt_l, aft_l, jumps, wall, t_start, partial=False, extra=None):
    """Uloží trajektorii a souhrn; rozpracovaný běh (SIGTERM) bez komprese a do hybrid_summary_partial.json,
    aby ho skripty nebraly jako hotový."""
    tot = time.time() - t_start
    n = min(len(times), len(angs), len(poss))
    T = np.array(times[:n])
    sv = np.savez if partial else np.savez_compressed
    sv(out / "torsions.npz", times=T, angs=np.stack(angs[:n])[None])
    sv(out / "pos.npz", times=T, pos=np.stack(poss[:n])[None])
    sv(out / "oracle27_jumps.npz", pre=np.array(pre_l), target=np.array(tgt_l), after=np.array(aft_l))
    sim_ns = T[-1] / 1000
    ok_j = [j for j in jumps if j["ok"]]
    res = dict(args=vars(args), oracle27=args.mode in ("same", "other"), oracle28=True, oracle34=True, oracle35=True, oracle36=True, oracle36b=True, oracle38=True, partial=partial, wall_s=round(tot, 1),
               simulated_ns=sim_ns,
               effective_ns_per_day=sim_ns / tot * 86400, wall_breakdown_s={k: round(v, 1) for k, v in wall.items()},
               jumps=len(ok_j), rejected=len(jumps) - len(ok_j), jump_records=jumps,
               mean=dict({k: round(float(np.mean([j[k] for j in ok_j if k in j])), 4)
                          for k in ["jump_rmsd", "ca_dev_after_steer", "to_target_10ps", "to_pre_10ps",
                                    "to_target_100ps", "to_pre_100ps", "to_target_1000ps", "to_pre_1000ps"]
                          if any(k in j for j in ok_j)}))
    res.update(extra or {})
    (out / ("hybrid_summary_partial.json" if partial else "hybrid_summary.json")).write_text(
        json.dumps(res, indent=2, ensure_ascii=False))
    print(f"Celkem {tot:.0f} s za {sim_ns:g} ns = {res['effective_ns_per_day']:.0f} ns/den (MD {wall['md']:.0f} s, "
          f"řízení {wall['steer']:.0f} s); skoky {res['jumps']}, zamítnuto {res['rejected']}; průměry {res['mean']}",
          flush=True)
    print("Uloženo", out)


if __name__ == "__main__":
    main()
