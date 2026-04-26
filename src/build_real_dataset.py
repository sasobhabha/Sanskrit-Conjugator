#!/usr/bin/env python
"""Build training dataset from authentic Sanskrit Heritage verb data.

This script parses roots.csv which contains ~134k verb forms derived from
Panini's grammar (SL_roots.xml). Data source: https://github.com/sanskrit/data

Each line: form,root,class,person,number,mode,voice,modification

Modes to Lakara (our 10):
  pres  → lata   (Present)
  ipft  → lan    (Imperfect)
  impv  → lot    (Imperative)
  opt   → vid    (Optative)
  perf  → lit    (Perfect)
  sfut  → lrt    (Simple Future)
  pfut  → lut    (Periphrastic/Past Future)
  cond  → lrn    (Conditional)
  ben   → ashirlinga (Benedictive)
  aor   → aorist (not in our 10, but we'll keep as extra)

We also have inj and ipfct - rare; we'll treat as extras or discard.
"""

import csv
import json
import os
from collections import defaultdict
from typing import Dict, List, Tuple

# SLP1 to IAST transliteration (complete)
SLP1_TO_IAST = {
    # Vowels
    'a': 'a', 'A': 'ā', 'i': 'i', 'I': 'ī',
    'u': 'u', 'U': 'ū', 'f': 'ṛ', 'F': 'ṝ',
    'x': 'ḷ', 'X': 'ḹ',
    'e': 'e', 'E': 'ai', 'o': 'o', 'O': 'au',
    # Diacritics
    'M': 'ṃ', 'H': 'ḥ',
    # Consonants
    'k': 'k',  'K': 'kh', 'g': 'g',  'G': 'gh', 'N': 'ṅ',
    'c': 'c',  'C': 'ch', 'j': 'j',  'J': 'jh', 'Y': 'ñ',
    'w': 'ṭ',  'W': 'ṭh', 'q': 'ḍ',  'Q': 'ḍh', 'R': 'ṇ',
    't': 't',  'T': 'th', 'd': 'd',  'D': 'dh', 'n': 'n',
    'p': 'p',  'P': 'ph', 'b': 'b',  'B': 'bh', 'm': 'm',
    'y': 'y',  'r': 'r',  'l': 'l',  'v': 'v',
    'S': 'ś',  'z': 'ṣ',  's': 's',  'h': 'h',
}

def slp1_to_iast(text: str) -> str:
    result = []
    i = 0
    while i < len(text):
        matched = False
        for length in [2, 1]:
            if i + length <= len(text):
                sub = text[i:i+length]
                if sub in SLP1_TO_IAST:
                    result.append(SLP1_TO_IAST[sub])
                    i += length
                    matched = True
                    break
        if not matched:
            result.append(text[i])
            i += 1
    return ''.join(result)

# Mode to Lakara code
MODE_TO_LAKARA = {
    'pres': 'lata',
    'ipft': 'lan',
    'impv': 'lot',
    'opt': 'vid',
    'perf': 'lit',
    'sfut': 'lrt',
    'pfut': 'lut',
    'cond': 'lrn',
    'ben': 'ashirlinga',
    'aor': 'lunj',   # aorist (extra)
    'inj': 'injunctive',  # extra rare
    'ipfct': 'ipfct',  # extra rare
}

PERSON_MAP = {'1': 'uttama', '2': 'madhyama', '3': 'prathama'}
NUMBER_MAP = {'s': 'ekavachana', 'd': 'dvivachana', 'p': 'bahuvachana'}

def parse(csv_path: str) -> Tuple[List[Dict], List[Dict]]:
    """Returns (full_conjugation_records, training_pairs)."""
    # nested: root_key -> {lakara -> {pn_key -> form}}
    conjugations = defaultdict(lambda: defaultdict(dict))
    pairs = []

    total = 0
    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            total += 1
            if total % 10000 == 0:
                print(f"  Processed {total} lines...")

            form_slp = row['form']
            root_slp = row['root']
            vclass = row['class']
            person = row['person']
            number = row['number']
            mode = row['mode']
            voice = row['voice']
            modification = row.get('modification', '').strip()

            # Skip derived forms (causative, desiderative, intensive, etc.)
            # Keep only the base verb forms
            if modification:
                continue

            # Map mode to lakara; skip unknown
            lakara = MODE_TO_LAKARA.get(mode)
            if not lakara:
                continue

            # Convert to IAST
            form = slp1_to_iast(form_slp)
            root = slp1_to_iast(root_slp)

            person_name = PERSON_MAP.get(person)
            number_name = NUMBER_MAP.get(number)
            if not person_name or not number_name:
                continue
            pn_key = f"{person_name}_{number_name}"

            verb_key = f"{root}|{vclass}|{voice}"
            conjugations[verb_key][lakara][pn_key] = form

            src = f"{root}|{lakara}|{pn_key}|{voice}"
            pairs.append({
                "source": src,
                "target": form,
                "root": root,
                "class": vclass,
                "lakara": lakara,
                "person": person_name,
                "number": number_name,
                "voice": voice,
            })

    print(f"Parsed {total} lines")
    print(f"Unique verb roots: {len(conjugations)}")
    print(f"Training pairs: {len(pairs)}")

    # Build list of full conjugation records
    full_records = []
    for verb_key, lakaras in conjugations.items():
        root, vclass, voice = verb_key.split('|')
        full_records.append({
            "root": root,
            "class": vclass,
            "voice": voice,
            "conjugations": lakaras,
        })

    return full_records, pairs

def main():
    os.makedirs('data', exist_ok=True)
    csv_path = 'data/sanskrit_heritage_roots.csv'
    full_out = 'data/real_conjugations_full.json'
    pairs_out = 'data/real_training_pairs.json'

    print("Parsing authentic Sanskrit Heritage verb data...")
    full_recs, pairs = parse(csv_path)

    print(f"\nSaving full conjugations to {full_out}")
    with open(full_out, 'w', encoding='utf-8') as f:
        json.dump(full_recs, f, ensure_ascii=False, indent=2)

    print(f"Saving training pairs to {pairs_out}")
    with open(pairs_out, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)

    # Show stats per lakara
    stats = defaultdict(int)
    for p in pairs:
        stats[p['lakara']] += 1
    print("\nLakara distribution:")
    for lak, count in sorted(stats.items()):
        print(f"  {lak}: {count}")

    # Sample
    if full_recs:
        print("\nSample conjugation (first root):")
        r = full_recs[0]
        print(f"  Root: {r['root']} (class {r['class']}, {r['voice']})")
        for lk, forms in r['conjugations'].items():
            print(f"  {lk}: {forms}")
            break

if __name__ == "__main__":
    main()
