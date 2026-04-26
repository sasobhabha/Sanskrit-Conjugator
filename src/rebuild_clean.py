#!/usr/bin/env python
"""Rebuild dataset using ONLY core verb forms (no modifications).

The Heritage CSV has a 'modification' column: empty = base verb,
'caus' = causative, 'desid' = desiderative, 'intens' = intensive, etc.
We want ONLY the base verb forms for accurate dhatu conjugations.
"""

import csv
import json
import os
from collections import defaultdict

# SLP1 to IAST
SLP1_TO_IAST = {
    'a':'a','A':'ā','i':'i','I':'ī','u':'u','U':'ū','f':'ṛ','F':'ṝ','x':'ḷ','X':'ḹ',
    'e':'e','E':'ai','o':'o','O':'au','M':'ṃ','H':'ḥ',
    'k':'k','K':'kh','g':'g','G':'gh','N':'ṅ','c':'c','C':'ch','j':'j','J':'jh','Y':'ñ',
    'w':'ṭ','W':'ṭh','q':'ḍ','Q':'ḍh','R':'ṇ','t':'t','T':'th','d':'d','D':'dh','n':'n',
    'p':'p','P':'ph','b':'b','B':'bh','m':'m','y':'y','r':'r','l':'l','v':'v',
    'S':'ś','z':'ṣ','s':'s','h':'h',
}

def slp1_to_iast(text):
    result = []; i = 0
    while i < len(text):
        matched = False
        for length in [2,1]:
            if i+length <= len(text):
                sub = text[i:i+length]
                if sub in SLP1_TO_IAST:
                    result.append(SLP1_TO_IAST[sub]); i += length; matched = True; break
        if not matched: result.append(text[i]); i += 1
    return ''.join(result)

MODE_TO_LAKARA = {
    'pres':'lata','ipft':'lan','impv':'lot','opt':'vid','perf':'lit',
    'sfut':'lrt','pfut':'lut','cond':'lrn','ben':'ashirlinga','aor':'lunj',
    'inj':'injunctive','ipfct':'ipfct'
}
PERSON_MAP = {'1':'uttama','2':'madhyama','3':'prathama'}
NUMBER_MAP = {'s':'ekavachana','d':'dvivachana','p':'bahuvachana'}

def parse(csv_path):
    conjugations = defaultdict(lambda: defaultdict(dict))
    pairs = []
    total = 0
    kept = 0
    with open(csv_path,'r',encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 50000 == 0: print(f"  {total} lines...")

            # ONLY keep rows with NO modification (base verb forms)
            modification = (row.get('modification') or '').strip()
            if modification:
                continue

            mode = row['mode']
            lakara = MODE_TO_LAKARA.get(mode)
            if not lakara: continue

            form_slp = row['form']; root_slp = row['root']; vclass = row['class']
            person = row['person']; number = row['number']; voice = row['voice']

            person_name = PERSON_MAP.get(person); number_name = NUMBER_MAP.get(number)
            if not person_name or not number_name: continue
            pn_key = f"{person_name}_{number_name}"

            form = slp1_to_iast(form_slp); root = slp1_to_iast(root_slp)
            verb_key = f"{root}|{vclass}|{voice}"
            conjugations[verb_key][lakara][pn_key] = form
            pairs.append({"source": f"{root}|{lakara}|{pn_key}|{voice}", "target": form,
                          "root": root, "class": vclass, "lakara": lakara,
                          "person": person_name, "number": number_name, "voice": voice})
            kept += 1

    print(f"Parsed {total} lines, kept {kept} base forms")
    print(f"Unique verb roots: {len(conjugations)}")
    print(f"Training pairs: {len(pairs)}")

    full_records = []
    for verb_key, lakaras in conjugations.items():
        root, vclass, voice = verb_key.split('|')
        full_records.append({"root": root, "class": vclass, "voice": voice,
                             "conjugations": lakaras})
    return full_records, pairs

def main():
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/sanskrit_heritage_roots.csv'
    full_out = 'data/real_conjugations_full.json'
    pairs_out = 'data/real_training_pairs.json'

    print("Parsing Heritage data (base forms only)...")
    full_recs, pairs = parse(csv_path)

    with open(full_out,'w',encoding='utf-8') as f: json.dump(full_recs, f, ensure_ascii=False, indent=2)
    with open(pairs_out,'w',encoding='utf-8') as f: json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved to {full_out} and {pairs_out}")

    stats = defaultdict(int)
    for p in pairs: stats[p['lakara']] += 1
    print("\nLakara distribution:")
    for lak,count in sorted(stats.items()):
        print(f"  {lak}: {count}")

    # Show a few sample roots
    if full_recs:
        print("\nSample roots:")
        for r in full_recs[:5]:
            print(f"  {r['root']} (cls {r['class']}, {r['voice']}): {list(r['conjugations'].keys())}")

if __name__ == "__main__":
    main()
