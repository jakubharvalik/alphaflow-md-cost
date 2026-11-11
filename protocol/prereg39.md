# Linie 39 (preprint): kritéria zapsaná předem

Zapsáno 9. 11. 2026 14:20 CET, **před stažením nových dat a před jakýmkoli výpočtem linie 39**. Návrh kritérií: Gemini 38x (oponentura_gemini_38x.md), Jakubova volba 9. 11. 14:13 „Preprint + 24 proteinů na MetaCentru“. Případné změny po oponentuře Gemini 39a se zapíší níže s časem, vždy před výsledky fáze B.

## Otázka preprintu

Dá soubor AI generátoru (AlphaFlow-MD, Jing et al. 2024) + krátká MD (S18) soubor konformací blízký ATLAS (3 × 100 ns) levněji než obyčejná MD z krystalu na stejné GPU, když se poctivě započte cena generátoru? Očekávaný (negativní) závěr z linie 38 na 6 proteinech: bez ceny generátoru medián ~1,7×, s ní ~1× až 0,16×. Preprint má ověřit, zda závěr platí obecně, a ukázat, jak volba účtování mění verdikt.

## Data

- Proteiny: testovací část ATLAS rozdělení AlphaFlow/MDGen (`splits/atlas_test.csv`, 82 řetězců, SHA256 c5029649db0b7ff6…). Generátor je při tréninku neviděl.
- Reference: ATLAS, 3 × 100 ns, CHARMM36m + TIP3P, snímky po 10 ps, prvních 100 ps vynechat.
- Vzorky AI: zveřejněné soubory AlphaFlow (huggingface.co/bjing-mit/alphaflow, samples), 250 struktur na protein. Fáze A: AF-MD, AF-MD+templates, ESMFlow-MD (co jde stáhnout). Fáze B: AF-MD+templates (`alphaflow_md_templates_base_202402`, stejně jako linie 38).

## Fáze A: rozbor AF bez MD (všech 82)

- `af38.py` beze změny (SHA 1834af14…), n = 250, 20 losů ATLAS, biny 30 (a 15 jako kontrola), trim 4, TICA lag 5 ns, 10 dim.
- Každý protein, který se podaří stáhnout a převést. Neúspěchy (stažení, převod, nesouhlas délky AF a ATLAS) se hlásí s důvodem, nic se tiše nevypouští.
- Výstup: rozdělení JS jádra PCA a TICA, podíl mimo ATLAS, Rama/χ1, korelace RMSF pro AF proti ATLAS a proti „replika celá, 250“ a „replika do 10,9 ns, 250“; závislost na délce proteinu.
- **Dobrý AF** = JS TICA jádra (AF-MD+templates, n = 250, biny 30) < 0,45.

## Fáze B: S18 na GPU MetaCentra

- **Výběr:** všech 25 testovacích proteinů s délkou seqres ≤ 120 (žádný další výběr, místo „24 stratifikovaných“ z karty, aby výběr nezávisel na výsledcích fáze A): 7lp1_A, 7jfl_C, 6okd_C, 6ro6_A, 6l34_A, 6pce_B, 6odd_B, 6jv8_A, 6ono_C, 6in7_A, 7a66_B, 6h86_A, 7bwf_B, 6d7y_A, 6ypi_A, 6tly_A, 7s86_A, 7n0j_E, 6pxz_B, 6p5h_A, 6tgk_C, 6vjg_A, 6yhu_B, 7k7p_B, 6uof_A. Vyřazení jen technické (stavba p2 nebo boxu selže, nestandardní rezidua, přerušená páteř, chybí vzorky AF), s důvodem. Box, který selže (NaN), se hlásí; S18 platí s ≥ 16 boxy.
- **Protokol:** `s18.py` beze změny (SHA ec3ed937…): 18 cílů ED z 250 vzorků (výběr jen ze souboru AF, ATLAS se nepoužije), box CHARMM36m + TIP3P, 0,15 M NaCl, HMR, dt 4 fs, minimalizace, 80 ps s restrainty (k 1000), 520 ps volné MD; systém p2_<N> z ATLAS PDB přes `build_system.py` (SHA 1022f4cd…). Platforma CUDA, GPU s compute capability ≥ 7.0.
- **Seedy:** primární analýza seed 0 na všech 25. Sekundární seed 1 (stejné cíle, jiné rychlosti a šum) na všech 25 po doběhnutí primárních, pořadí podle délky; hlásí se, na kolika doběhl. Výsledky 6 proteinů z Macu (linie 38) jen jako kontrola reprodukovatelnosti.
- **Hodnocení:** `sub38.py` (SHA 95df1583…) okno 50–520 ps, jádro PCA a TICA; leave-one-out proti sjednocení dvou ostatních replik ATLAS (pro každou repliku zvlášť, průměr přes 3); `rel38.py` Rama a χ1; srovnávací řádky „MD z krystalu 10,9 ns“ a „celá 100ns replika“.

## Účtování ceny (vše na stejné GPU, voleno ve prospěch S18)

- **MD z krystalu:** ekvivalentní čas t* ze `sub38` (nejkratší t, kdy průměr replik ATLAS od krystalu dosáhne JS běhu S18 při stejném n), cena = t* / v_p2 + jedna ekvilibrace, kde v_p2 je rychlost MD boxu p2_<N> naměřená ve stejné úloze (50 ps). Když křivka hodnoty S18 nedosáhne do 100 ns, t* ≥ 100 ns: počítá se 100 ns jako dolní mez (cenzurování v neprospěch S18, oprava Z2) a zrychlení se hlásí jako „≥“.
- **S18:** naměřený čas GPU 18 boxů (minimalizace + ekvilibrace + MD). Stavba boxů na CPU se do primárního účtování nepočítá (zvýhodní S18), sekundárně se uvede sériově se stavbou.
- **Generátor:** zveřejněný čas AlphaFlow na A100 (plná verze 38 s na vzorek, destilovaná 3,8 s; referenční délku ověřím v publikaci) přepočtený na délku L faktorem (L / L_ref)², A4000 počítána stejně rychlá jako A100 (obojí zvýhodní S18). Primárně plná verze a 250 vzorků (tak S18 vzorky skutečně vybíral), sekundárně destilovaná a 250, destilovaná a 50, a nula (vzorky už existují).
- **Zrychlení** = cena MD z krystalu / (cena S18 + cena generátoru), zvlášť pro jádro a TICA.

## Kritéria (rozhodnutí ano/ne)

1. **(i) Cena [nahrazeno změnou Z3 níže]:** při primárním účtování (plný generátor, 250 vzorků) dá S18 zrychlení ≥ 1,5× v jádru i TICA na **méně než 50 %** proteinů. Splněno = negativní závěr linie 38 platí obecně. Totéž se uvede pro všechna sekundární účtování; rozhodující je jen primární.
2. **(ii) Přínos pokrytí na dobrém AF:** na proteinech s dobrým AF je S18 v leave-one-out blíž dvěma nezávislým replikám než celá 100ns replika (JS jádra i TICA menší) aspoň v **75 %** případů a zároveň má Rama ≤ 0,08 a χ1 ≤ 0,08 (rel38, okno 50–520). Když dobrý AF má méně než 4 proteiny, (ii) se nehodnotí a uvede se jen popisně.
3. **(iii) Kontrola selhání:** na proteinech se špatným AF (JS TICA ≥ 0,45) se uvede podíl, kde S18 prohraje s MD z krystalu 10,9 ns (očekávám většinu), jen popisně.

## Co se do preprintu nedostane bez dalšího souhlasu Jakuba

Instalace AlphaFlow/ESMFlow pro přímé měření ceny generátoru (cena jen z publikace). Nic placeného. Data ATLAS jsou CC-BY-NC, do preprintu jen odvozené metriky.

## Změny po zapsání

**9. 11. 2026 14:28 CET, po oponentuře Gemini 39a (oponentura_gemini_39a.md), před prohlédnutím jakýchkoli nových dat** (zkušební úloha 25395144 běží, výsledky jsem neviděl):

- **Z1 cena generátoru:** 38 s / 3,8 s na vzorek je podle AlphaFlow (příl. B.3) průměr přes všech 82 testovacích proteinů na A100, proto c(L) = T_pub · L^p / průměr_82(L^p). Primárně p = 2 (pro L ≤ 120 nejpříznivější pro S18), citlivost p = 1,5 (odhad z poměru PDB/ATLAS v tab. 3), p = 1 a p = 0 (neškálováno). MDGen uvádí 70 s na vzorek AlphaFlow, hlásí se jako horní varianta (rese39.md).
- **Z2 cenzurování:** t* nedosažené do 100 ns je dolní mez (v neprospěch S18), viz oprava výše.
- **Z3 kritérium (i) nahrazeno:** dvě spolu-primární účtování, (i-a) plný AlphaFlow, 250 vzorků (skutečně použitý) a (i-b) destilovaný, 50 vzorků (předpoklad stejné kvality jako S18 z 250, ve prospěch S18). Pro každé: **S18 se nevyplácí, když horní mez jednostranného 95% bootstrap intervalu mediánu zrychlení (10 000 výběrů přes proteiny, cenzurované hodnoty jako dolní mez) je < 1,5× v jádru i TICA.** Medián a podíl proteinů ≥ 1,5× se jen hlásí. Primární p = 2; když verdikt závisí na p, uvede se to a cenu generátoru je nutné změřit (instalace AlphaFlow jen se souhlasem Jakuba).
- **Z4 kritérium (ii):** primární práh beze změny; sekundární „dobrý AF“ = JS_TICA(AF, 250) ≤ průměr JS_TICA(ATLAS replika celá, 250 rovnoměrně) + 0,10 (obojí af38). Leave-one-out se shodným n: nový `eval39.py` podvzorkuje 100ns repliku rovnoměrně na počet snímků S18 v okně 50–520 (sub38 to nedělá), průměr přes 3 repliky.
- **Z5 seed 1:** jiný výběr cílů i rychlostí: ED-18 z náhodné poloviny (125 z 250) vzorků AF (nová volba `--pool-half-seed 1` v s18.py, seed 0 a jeho výsledky beze změny), MD seed 1.
- **Z6 doplnění:** když z 25 technicky projde < 20, nebo má dobrý AF (primárně) < 8, doplní se testovací proteiny s L > 120 v pořadí rostoucí délky, dokud podmínka neplatí, nejvýš 10.
- **Z7 kontroly (popisné, nerozhodují):** (a) ATLAS 3 × 3,6 ns z krystalu sloučené (10,8 ns) proti sjednocení; (b) K18 na 7jfl_C, 6jv8_A, 6ro6_A: 18 boxů po 0,6 ns z krystalu stejným protokolem (s18.py, 18 kopií struktury ATLAS, --targets 0..17, seed 0); (c) párové ΔJS(S18 − AF) při n = 250 v jádru, TICA, Rama, χ1; (d) bod zvratu: nejvyšší čas generátoru na vzorek (pro 50 i 250 vzorků), při kterém zrychlení dosáhne 1,5×; (e) ED-18 z prvních 50 vzorků AF (bez MD).

**9. 11. 2026 15:46 CET (v dotazu gem39b omylem uvedeno 15:50), změna Z8: naměřená cena generátoru** (souhlas Jakuba 14:34 „Ano, AlphaFlow i ESMFlow“ s instalací na MetaCentrum). Zapsáno před instalací a před jakýmkoli měřením generátoru i před výsledky fáze B (fáze A běží, výsledky fáze A cenu neovlivní).

- **Proč:** Z1 až Z3 berou cenu z publikace (A100, průměr přes 82 proteinů) a přepočet na délku závisí na neznámém p; podle simulace na smyšlených datech (sum39) verdikt (i) na p závisí. Z3 pro ten případ předepisuje měření.
- **Instalace:** autorský kód github.com/bjing2016/alphaflow (MIT, commit se zapíše), prostředí přesně podle README (python 3.9, torch 1.12.1+cu113, openfold@103d037, CUDA 11.8). Jediné úpravy kódu: predict.py bez `weights_only=False` (torch 1.12 argument nezná) a dočasné soubory dotazu MSA v $TMPDIR; obě se zapíší jako git diff.
- **Modely:** primárně AlphaFlow-MD+Templates plný (vzorky, ze kterých S18 vybíral; účtování i-a) a destilovaný (i-b). Vedlejší: 12vrstvé verze téhož (plná, destilovaná) a ESMFlow-MD+Templates (plná, destilovaná), jen popisně.
- **Proteiny:** sada B = všech 25 proteinů fáze B; sada K (kalibrace délky) = 7jrq_A 131, 6kty_A 196, 7c45_A 302, 6p5x_B 457, 5znj_A 567, 6sms_A 724 (kvantily 35–100 % délek 82 testovacích proteinů). Šablona: sada B struktura ATLAS (jako u autorů), sada K první zveřejněný vzorek AF-MD+templates (čas nezávisí na obsahu šablony, jen na délce). MSA ze serveru ColabFold skriptem autorů.
- **Měření:** `predict.py --runtime_json` (čas `model.inference` na vzorek včetně převodu výsledku na CPU, tedy po synchronizaci GPU), dávka 1, výchozí nastavení (10 kroků, destilované `--noisy_first --no_diffusion`). Vzorků: sada B plné 6 a destilované 11, sada K 4 a 6. První vzorek každého proteinu se zahodí (zahřátí), cena na vzorek t_gen(N) = medián ostatních. Načtení modelu a příprava vstupů (wall celého volání) a čas dotazu MSA se hlásí zvlášť a do primární ceny se nepočítají (stejně jako stavba boxů S18 na CPU, ve prospěch S18).
- **Stejná GPU:** v téže úloze se na téže GPU změří rychlost MD (bench39) systémů p2_<N> (kotva p2_7jfl_C z bench3 a všechny už postavené systémy fáze B). Cena generátoru se převede na čas GPU, na které běžel S18 daného proteinu: C_gen(N) = n · t_gen(N) · v_gen(N) / v_B(N), kde v_gen a v_B jsou ns/den téhož systému p2_<N> na GPU měření generátoru a na GPU úlohy B (když p2_<N> na GPU generátoru změřen není, poměr GPU z kotvy p2_7jfl_C). Ekvivalentně se hlásí jako „MD-ns“ = n · t_gen · v_gen / 86 400.
- **Rozhodující účtování pro kritérium (i)** (nahrazuje c(L) ze Z1, pravidlo Z3 beze změny): (i-a) n = 250 a naměřený plný model, (i-b) n = 50 a naměřený destilovaný model. Publikovaná cena c(L) s p = 2, 1,5, 1, 0 a 70 s (MDGen) se uvede jako citlivost a ke srovnání s publikací.
- **Kontrola publikovaného čísla:** fit t_gen(L) = a + b · L^p na sadách B + K (zvlášť pro každý model), z něj odhad průměru přes 82 testovacích délek na naší GPU a poměr k 38 s / 3,8 s (A100). Popisně, nerozhoduje.
- **Kontrola instalace:** RMSD Cα našich vzorků plného modelu sady B proti šabloně, srovnání s 250 zveřejněnými vzorky téhož proteinu (medián poměru). Když je poměr mimo 0,5–2, instalace se považuje za vadnou a měření se nepoužije (pak platí Z1–Z3 s publikovanou cenou a verdikt se hlásí jako podmíněný p).
- **Selhání:** nedostatek paměti GPU nebo jiná chyba u proteinu se hlásí s důvodem. Když měření plného modelu sady B neproběhne aspoň na 20 z 25 proteinů, (i) se vyhodnotí podle Z1–Z3 a uvede se proč. Chybějící proteiny sady B se doplní fitem z ostatních (označeno).

**9. 11. 2026 15:56 CET, změna Z8a po oponentuře Gemini 39b** (oponentura_gemini_39b.md), stále před instalací a jakýmkoli měřením generátoru a před výsledky fáze B:

- **Kontrola instalace (nahrazuje pásmo 0,5–2):** pro každý protein sady B průměrné RMSD Cα ke šabloně našich vzorků plného modelu a 250 zveřejněných; instalace platí, když medián poměru (naše / zveřejněné) přes 25 proteinů leží v 0,75–1,33 a Spearmanova korelace obou průměrů napříč proteiny je ≥ 0,70. Jinak se měření nepoužije (platí Z1–Z3).
- **Fit:** t_gen(L) = a + b · L^p s a ≥ 0, b ≥ 0. U každého proteinu se uvede počet sekvencí MSA.
- **GPU:** gen39 se žádá na architekturu 8.x od 8.6 (`gpu_cap=sm_86`, typicky RTX A4000 jako bench3). Citlivost: (i) jen na proteinech, jejichž úloha B běžela na stejném modelu GPU jako gen39 (bez přepočtu v_gen / v_B).
- **Omezení:** dávka > 1 se nezkouší (autoři měří s dávkou 1); kdyby čas na vzorek snížila, je cena generátoru nadhodnocená, tedy v neprospěch S18.

**9. 11. 2026 21:50 CET, oprava implementace (ne změna kritérií ani metody)**, po výsledcích fáze B se zveřejněnou cenou, před měřením generátoru:

- **Rychlost MD na GPU úlohy B:** prereg39 předepisuje převod t* na GPU sekundy rychlostí MD systému p2_<N> na GPU úlohy S18. Úloha b39.pbs ale měřila bench_p2_<N> jen v první úloze daného proteinu; druhý seed a K18 mohly běžet na jiné GPU a převzaly cizí rychlost. Bench měřila ve všech 25 případech úloha S18_s0, takže seed 0 je správně. Oprava pro seed 1 a K18 (22:20, sum39 `--v-src xfer`): když úloha běžela na stejném modelu GPU jako S18_s0, platí bench; jinak v_B = bench × (propustnost MD boxů této úlohy) / (propustnost MD boxů úlohy S18_s0), propustnost = ns/den × průměrný počet atomů boxu (boxy S18 mají jiný počet atomů než p2, proto normalizace). Původní výpočet (`--v-src bench`) se uvede pro srovnání.
- **Citlivost definice t* (post hoc, NEREGISTROVÁNO, nerozhoduje):** primární t* zůstává podle prereg (první bod mřížky, kde průměr replik při shodném n dosáhne JS běhu). Protože mřížka začíná 0,5 ns (~40 snímků), kde je JS na šumovém dnu, uvede se navíc t* jako bod, od kterého replika zůstává ≤ běh („last“), a t* bez shodného n proti sjednocení a v leave-one-out.

**10. 11. 2026 14:41 CET, změna Z9: S18 z 50 vzorků destilovaného modelu (S18d50), a změna Z9b: měření na A100.** Jakub na kartě 14:26 CET zvolil „Test destil. S18 + A100 + psát“ (doporučení Gemini 39i). Zapsáno po konečném účtování Z8 a po oponenturách Gemini 39j a 39k (oponentura_gemini_39j.md, oponentura_gemini_39k.md), před generováním destilovaných vzorků, před jakýmkoli během S18d50 a před měřením na A100. Jediná nová data před zápisem: kontrola kolizí atomů ve vzorcích gen39b a fáze B (clash39.py) a účetní mřížka z hotových dat (grid39.py), obojí bez výsledků S18d50.

- **Proč:** (i-b) v Z8 spojilo cenu destilovaného modelu (50 vzorků) s trajektoriemi S18 vybranými z 250 vzorků plného modelu (předpoklad stejné kvality, ve prospěch S18). Z9 změří S18 přímo z 50 destilovaných vzorků. (i-a) a všechny výsledky Z8 platí beze změny.
- **Vzorky (gen39d.pbs, SHA e9fe1470…):** AlphaFlow-MD+Templates destilovaný 202402 (`--noisy_first --no_diffusion`, ostatní výchozí), 50 vzorků na protein, všech 25 proteinů sady B, vstupy beze změny z gen39b (gen39_in.csv, MSA, šablona = struktura ATLAS s ID řetězce A, predict.py se zapsanými úpravami). predict.py nemá seed. Soubor platí jen s 50 modely a počtem Cα = délka seqres v každém modelu. GPU libovolná 8.x; čas z gen39d jen popisně.
- **Filtr kolizí (s39d.py, SHA 60521e2c…; s18.py beze změny):** vzorky s dvojicí těžkých atomů blíž než 0,5 Å (fyzikálně nemožná geometrie; ve vzorcích gen39b destilované 18,2 %, plné 0,5–0,7 %) se před výběrem ED-18 vynechají; zbude-li méně než 25 vzorků, vezme se 25 vzorků s největší nejmenší vzdáleností. Cena generátoru dál počítá všech 50 vygenerovaných vzorků. Počet vynechaných se hlásí. U S18_s0 by stejný filtr vynechal 4 ze 450 cílů (S18_s0 se nemění).
- **Běh S18d50_s0 (b39d.pbs, SHA 2cf76109…, = b39.pbs s jiným zdrojem vzorků, TAG a filtrem):** ED-18 z vzorků po filtru, seed 0 (jeden seed, Gemini 39j). Rozbory sub38, eval39, rel38, af38 (af38 se zveřejněnými plnými vzorky AF_MD_templ jako u S18_s0 a navíc AF_dist50). Bench p2 se neopakuje; v_B jako u seedu 1 (sum39 `--v-src xfer` vůči úloze S18_s0).
- **Selhání boxů (Gemini 39k, změna pravidla ≥ 16 boxů pro Z9):** protein, jehož běh skončí s více než 2 selhanými boxy (nekonečná energie nebo NaN; s18 zapíše jen hybrid_summary_partial.json), je fyzikální selhání metody: v Z9-i se počítá jako zrychlení 0× v jádru i TICA, v Z9-ii jako neúspěch, v Z9-iii chybí a hlásí se. Technický výpadek (úloha bez souhrnu i bez partial) se spustí znovu. Počet selhaných boxů se hlásí u každého proteinu.
- **Z9-i (primární, nahrazuje předpoklad stejné kvality v i-b):** zrychlení S18d50_s0 v účtování „Z8 i-b“ (n = 50, t_gen destilovaného modelu z gen39b na A40, v_gen z téže úlohy), t* „first“ (prereg). **Nevyplácí se:** horní mez jednostranného 95% bootstrap CI mediánu (10 000 výběrů přes proteiny) < 1,5× v jádru i TICA (pravidlo Z3 beze změny). **Vyplácí se v daném prostoru** (jádro, TICA zvlášť): medián ≥ 1,5× a dolní mez jednostranného 95% CI > 1,0×. Jinak nerozhodnuto. Nerozhodnuto také, když platných a fyzikálně selhaných proteinů je dohromady < 20. Citlivost (nerozhoduje): t* „last“.
- **Z9-ii (sekundární):** pevná kohorta 15 proteinů s dobrým AF podle fáze A (plný model, JS_TICA(AF, 250) < 0,45; stejná jako u S18_s0). Splněno, když S18d50 porazí v LOO TICA 100ns repliku (eval39, shodné n) u ≥ 75 % a má Rama ≤ 0,08 i χ1 ≤ 0,08 (rel38, okno 50–520 ps) u ≥ 75 %. Rozklad LOO TICA / LOO jádro / Rama / χ1 / průnik všech čtyř se hlásí pro S18d50 i S18_s0 (15/11/14/13/9 z 15).
- **Z9-iii (sekundární):** párový ΔJS = JS(S18d50_s0) − JS(S18_s0) přes proteiny s oběma během; destilace kvalitu S18 nezhorší, když medián ΔJS ≤ +0,03 v jádru i TICA. Primárně JS všech snímků okna 50–520 ps proti sjednocení replik (sub38 js_union, n ≈ 8 460 podle počtu boxů), sekundárně af38 (běh 250 rovnoměrně). Měřítko šumu: týž rozdíl S18_s1 − S18_s0 (popisně).
- **Z9b (A100, citlivost, nerozhoduje):** gen39e.pbs = gen39b.pbs upravená patch39l.py (SHA 904a5214…) na A100 (cluster zia.cerit-sc.cz, 4× A100 40 GB, ověřeno pbsnodes 10. 11.), jen afT_base a afT_dist (sady B a K, stejný počet vzorků) + bench39 na téže A100 (zahřátí, kotva, všech 25 p2, kotva na konci). Když GPU není A100, úloha končí. (i-a) a (i-b) obou seedů a Z9-i se přepočtou s t_gen a v_gen z A100 a nahradí citlivost „÷ 2,57“. Primární účtování zůstává A40 (Z8). Fit a poměr k 38 s / 3,8 s z publikace se hlásí.
- **Vyhodnocení:** sum39.py `--root l39 --tag S18d50_s0 --gen gen39b_sum.json --v-src xfer --tstar first`, pak cmp39d.py (SHA b6cb9de9…). Výsledek Z9 se do preprintu uvede vždy, ať vyjde jakkoli; kolize atomů v destilovaných vzorcích se uvedou jako samostatný výsledek.

## Z10 — opravený přepočet po externím posudku (zapsáno 11. 11. 2026 16:00 CET, PŘED výpočtem)

Důvod: externí posudek (11. 11.) a kontrola kódu ukázaly tři vady primárního vyhodnocení, které zvýhodňují MD: (1) t* porovnává okno repliky se sjednocením všech tří replik, které ji obsahuje; (2) sub38 nedrží stejné n na celé mřížce (100 ns: 9 991 snímků repliky proti 8 460 S18); (4) eval39 fituje PCA/TICA a hrany histogramu i z vynechané repliky a burn-in 100 ps neuplatní při stavbě referencí. Dále (6) cena nezahrnuje čas selhaných boxů. Pravidla podle Gemini 39t, upravená. Původní předem daná vyhodnocení (Z3, Z8, Z9) zůstávají beze změny a v textu se uvádějí; Z10 je opravený přepočet uvedený vedle nich, ne jejich náhrada.

- **Data:** existující běhy S18_s0, S18_s1, S18d50_s0, K18 a ATLAS 3 × 100 ns pro 25 proteinů fáze B. Žádné nové simulace.
- **Foldy k = 1, 2, 3:** reference R_-k = dvě zbylé repliky, snímky od 100 ps. Burn-in 100 ps platí všude stejně (reference, replika, kontroly).
- **Projekce bez úniku:** zarovnání Cα jádra (trim 4), PCA (prvních 10 složek) a TICA (lag 5 ns počítaný jen uvnitř každé souvislé trajektorie, nikdy přes šev; 10 PC → 2 TIC) se fitují jen na R_-k. Hrany 2D histogramu 30 × 30 z R_-k s okrajem 25 % jako dosud; body mimo hrany se přiřadí do krajních binů. JS v bitech (log2), neodmocněná.
- **Větev MD:** prefix vynechané repliky R_k od 0,1 ns do t na mřížce sub38. **Větev S18:** platné boxy, okno 50–520 ps.
- **Stejné n:** na každém bodě mřížky n(t) = min(|R_k[0,1; t]|, n_S18); obě větve se podvzorkují na n(t) (i v bodě 100 ns): replika rovnoměrným krokem v čase s náhodným posunem, S18 stejným počtem snímků z každého boxu. 20 opakování, JS se průměruje.
- **t*:** JS_MD(t) a JS_S18 se průměrují přes 3 foldy; t* = první t, kdy průměrná JS_MD(t) ≤ průměrná JS_S18, lineární interpolace v čase mezi body mřížky jako dosud; citlivost t* „last“; cenzura na 100 ns jako dosud.
- **Cena:** jako v sum39 (v_B xfer, generátor z gen39b na A40) a navíc GPU čas selhaných boxů a pokusů, pokud je v logu; kde v logu není, horní odhad = čas úspěšného boxu stejného proteinu a hlásí se obě varianty.
- **Rozhodovací pravidla:** stejné prahy jako Z3 a Z9-i (nevyplácí se: horní mez jednostranného 95% CI mediánu < 1,5×; vyplácí se: medián ≥ 1,5× a dolní mez > 1,0×; jinak nerozhodnuto), bootstrap 10 000 přes proteiny. Výsledek se označí jako opravený přepočet po externím posudku.
- **Podskupiny:** všech 25 proteinů a 19 proteinů bez šesti pilotních z linie 38 (7jfl_C, 6ro6_A, 6okd_C, 6ono_C, 6jv8_A, 6pce_B); obě se hlásí.
- **Citlivost (nerozhoduje):** 15 × 15 binů; energy distance na týchž 2D projekcích (bez binování).
- **LOO (exploratorní 23/25):** přepočítá se s projekcí a hranami fitovanými po foldech a se stejným n.
- **Kontrola stejného rozpočtu MD bez in-sample reference:** pro fold k reference = samotná R_k (projekce z R_k), kontrola = prvních 5,4 ns každé ze dvou zbylých replik spojených, S18 proti téže R_k; hlásí se počet proteinů, kde je S18 blíž, a medián rozdílu JS s párovým bootstrapem. Původní Z7a (3 × 3,6 ns) zůstává uvedena.
- **Vyhodnocení:** nové skripty (sub40.py, eval40.py) se před spuštěním zapíšou se SHA256 do tohoto logu; výsledky se uvedou vždy, ať vyjdou jakkoli.

**Z10 — implementace (zapsáno 11. 11. 2026 ~17:10 CET, před spuštěním na skutečných datech; testováno jen na umělých datech).** Skripty (SHA256):
- `l39/sub40.py` 27f6042d50513bda6a1edae76a06d17e579cd04b86e8e0c03481b294be8e5805 (t*, LOO, (iii), kontrola 2 × 5,4 ns; po foldech; vypisuje i RMSD Cα jádra od krystalu jako kontrolu mapování atomů)
- `l39/patch40.py` c3cc232243a340d4383e558b118ba3140e4729554fd43915a7619ae1797ac566 → `l39/sum40.py` 9a5a325cd9ca1b4c63b42f1e73fb292669d0162d056f59a343a5c9fa60200bfe (= sum39 539504c3… + patch40: cena, rozhodnutí)
- `l39/fail40.py` 57d950d9ebe3c68569cd5a828fa7b093db513b6ec2566857dd70fd598cdfefcc (čas selhaných pokusů z logů b39)
- `metacentrum/jobs/ana40.pbs` 9a4b8557a2ed962c3fdedb1ee40b4b64b8b212e6f45ac6696e23b2a4b58a288b (jen CPU, 25 proteinů)

Upřesnění, která Z10 výslovně neurčovalo (stanovena teď, před výpočtem):
- Projekce PCA/TICA počítá sub40 stejnými vzorci jako core36.Space (ověřeno na umělých datech: při fitu na všech třech replikách shodné mu, |W| i histogram).
- Uvnitř boxu S18 se snímky vybírají také rovnoměrným krokem s náhodným posunem; zbytek n(t) mod počet boxů dostanou náhodné boxy po jednom snímku.
- Energy distance: V-statistika, reference = R_-k podvzorkovaná na 1 500 bodů, každá větev nejvýš 1 500 bodů z téhož výběru se shodným n, 5 losů z 20 (výpočetní cena); jen citlivost.
- Kontrola 2 × 5,4 ns: snímky 100–5 400 ps obou zbylých replik (stejná konvence jako Z7a 100–3 600 ps); S18 podvzorkován na stejný počet, 20 losů; jen 30 binů.
- Čas selhaných pokusů: „log“ = razítko SELHAL minus předchozí razítko (zahrnuje i stavbu boxu na CPU, tedy horní odhad pro dané selhání); „upper“ = počet selhání × průměrný GPU čas úspěšného boxu (min + ekv + MD) téhož běhu. Počet selhání = max(počet v hybrid_summary, počet v logu). Hlásí se obě, primárně „log“.
- Fyzikální selhání (jen hybrid_summary_partial.json, tj. S18d50 u 7jfl_C a 6in7_A) = 0× ve všech účtováních a neúspěch v (ii), jako Z9.
- Pravidlo Z9 „nerozhodnuto při < 20 proteinech“ se v Z10 neuplatňuje (podskupina 19 by byla nerozhodnutá automaticky); počet proteinů se hlásí u každého výsledku.
- Meze rozhodnutí (i) jsou 5. a 95. percentil bootstrapového mediánu (jednostranné 95% meze, tj. dvoustranný 90% interval), párový rozdíl u kontroly 2 × 5,4 ns s 2,5. a 97,5. percentilem.

**Po Z10 — post hoc, exploratorní (zapsáno 11. 11. 2026 ~22:05 CET, PO výsledcích Z10; nerozhoduje o žádném verdiktu).**
- `l39/diag40.py` 33e693fac072eff889450f671f68060f44598f3dafaace40a56969c372991ad4: rozklad poklesu LOO mezi eval39 a sub40 (varianty A–D báze/hrany), 8 proteinů s lokálními daty, S18d50_s0 (7jfl_C z partial, jen diagnostika). Výsledek: změnu dělá jen báze (A ≈ B, C ≈ D ≤ 0,005); vada 4 v LOO zvýhodňovala S18. Preprint SI Tabulka S9.
- `l39/cmp40.py` 0c67cff5d4e261fa641addd3fceedd17407133658fa8024c8a7ee9c306a96bb6: jen tabulky po proteinech (původní → Z10), žádný nový výpočet metrik. Preprint SI Tabulky S6–S7, Obr. 5.
- Poznámka k implementaci: `eval40.py` uvedený v pravidlech Z10 nevznikl; LOO, (iii) i kontrolu 2 × 5,4 ns počítá přímo `sub40.py` (zapsáno v implementaci ~17:10 CET).
