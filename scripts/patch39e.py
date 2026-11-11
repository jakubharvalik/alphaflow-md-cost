p='inst39.pbs'
s=open(p,encoding='utf-8').read()
a=s.index("if [ ! -f $ENV/.af39_ok ]; then\n")
b=s.index("P=$ENV/bin/python\nexport LD_LIBRARY_PATH=$ENV/lib:${LD_LIBRARY_PATH:-}\ncd $G || exit 5\n")
old=s[a:b]
clr=old[old.index("  cat > $SCRATCHDIR/clrx39.py <<'CLREOF'\n"):old.index("CLREOF\n  $P $SCRATCHDIR/clrx39.py")+len("CLREOF\n")]
pipblk=old[old.index("  $PIP numpy==1.21.2 pandas==1.5.3 || exit 3\n"):old.index("  # 25403724: na uzlech")]
condablk=old[old.index("  # 25403459: mdtraj 1.9.9 (formát TNG"):old.index("  # 25400154: mdtraj 1.9.9 se stavěl ze zdroje")]
exp=old[old.index("  # 25400154: mdtraj 1.9.9 se stavěl ze zdroje"):old.index("  $PIP numpy==1.21.2")]
exp=exp.replace('export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib"  # zlib a další knihovny prostředí i při stavbě rozšíření',
                'export LDFLAGS="-L$ENV/lib -Wl,-rpath,$ENV/lib -Wl,-z,noexecstack"  # knihovny prostředí; 39e: nové .so bez spustitelného zásobníku')
indent=lambda t: "".join(("  "+l if l.strip() else l) for l in t.splitlines(True))
new=("if [ ! -f $ENV/.af39_ok ]; then\n"
"  t0=$(date +%s)\n"
"  # 39e: dvě značky; .af39_base_ok = conda + pip základ (torch, mdtraj) prošel importem, pak se při opakování staví jen openfold\n"
"  if [ -f $ENV/.af39_base_ok ]; then\n"
"    echo \"$(date '+%T') základ prostředí af39 hotový dřív (torch, mdtraj), pokračuji stavbou openfold\"\n"
"  else\n"
"    [ -d $ENV ] && { echo \"$(date '+%T') neúplné prostředí z dřívějška, mažu $ENV\"; rm -rf $ENV; }\n"
+indent(condablk)+
"  fi\n"
+exp+
"  # 25403724: na uzlech s glibc >= 2.41 (Debian 13) odmítne dlopen knihovny s příznakem spustitelného zásobníku\n"
"  # („libtorch_cpu.so: cannot enable executable stack“, torch 1.12.1). Příznak PF_X v PT_GNU_STACK je u nich zbytečný\n"
"  # (objekty bez .note.GNU-stack), proto se smaže, totéž co execstack -c / patchelf --clear-execstack. Jen knihovny s RWE.\n"
+clr+
"  if [ ! -f $ENV/.af39_base_ok ]; then\n"
+indent(pipblk)+
"    $P $SCRATCHDIR/clrx39.py $ENV/lib || exit 4\n"
"    $P -c \"import torch, numpy, mdtraj; print('import torch', torch.__version__, 'OK')\" || { echo \"import torch/mdtraj selhal\"; exit 4; }\n"
"    touch $ENV/.af39_base_ok\n"
"    echo \"$(date '+%T') základ prostředí af39 hotový za $(( $(date +%s) - t0 )) s\"\n"
"  fi\n"
"  $PIP --no-build-isolation 'openfold @ git+https://github.com/aqlaboratory/openfold.git@103d037' || { echo \"openfold selhal\"; exit 4; }\n"
"  $P $SCRATCHDIR/clrx39.py $ENV/lib || exit 4  # i nově postavené attn_core_inplace_cuda (objekty z nvcc)\n"
+old[old.index("  LD_LIBRARY_PATH=$ENV/lib $P -c \"import torch, openfold"):])
s=s[:a]+new+s[b:]
o='for f in $G/msa/*/a3m/*.a3m; do printf "%s %s\\n" "$(basename $f .a3m)" "$(grep -c \'^>\' $f)"; done > $G/msa_counts.txt\necho "MSA hotových $(wc -l < $G/msa_counts.txt) z $NM (jméno, počet sekvencí):"; tr \'\\n\' \' \' < $G/msa_counts.txt; echo\n'
assert s.count(o)==1, "msa"
n=o+'# 39e: neúplné nebo prázdné MSA = chyba (gen39 navazuje afterok a predict.py by na sadě B spadl celý); opakování dotáhne chybějící\nnok=$(awk \'$2>0\' $G/msa_counts.txt | wc -l)\n[ $nok -ge $NM ] || { echo "MSA neúplné: $nok z $NM neprázdných"; exit 7; }\n'
s=s.replace(o,n)
o="#    architektur OpenFold (mezi nimi sm_70, sm_75, sm_80; kód sm_80 běží i na 8.6 a 8.9). Značka hotového prostředí .af39_ok.\n"
n="#    architektur OpenFold (sm_37, 52, 61, 70, 80; kód sm_80 běží i na 8.6 a 8.9, sm_70 na 7.5). Značky: .af39_base_ok (základ), .af39_ok (vše).\n"
assert s.count(o)==1, "hlavicka"
s=s.replace(o,n)
open(p,'w',encoding='utf-8').write(s)
print("patch OK")
