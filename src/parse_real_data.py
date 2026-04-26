#!/usr/bin/env python
"""Parse real Sanskrit dhatu conjugation data from Sanskrit Heritage site.

The roots.csv file contains authentic verb forms derived from Panini's grammar.
Source: https://github.com/sanskrit/data (sanskrit-heritage-site)

CSV columns: form,root,class,person,number,mode,voice,modification

Mode mapping (to our 10 lakaras):
  pres      → lata   (Present tense)
  ipft      → lan    (Imperfect past)
  impv      → lot    (Imperative)
  opt       → vid    (Optative)
  pfut      → lit    (Perfect/Perfect future)
  sfut      → lut    (Simple future)
  ipfct     → lrt    (Aorist/simple past)
  cond      → lrn    (Conditional)
  bh        → ling   (Potential - "bhaviṣya" or "bh" endings)
  ash       → ashirlinga (Benedictive)
  perf      → lit    (Perfect - alternative encoding)
  aorist    → lrt    (Aorist)

Note: The heritage data uses SLP1 encoding for Sanskrit. We'll convert to IAST.
"""

import csv
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

# SLP1 to IAST transliteration mapping
SLP1_TO_IAST = {
    'a': 'a', 'A': 'ā', 'i': 'i', 'I': 'ī', 'u': 'u', 'U': 'ū',
    'R': 'ṛ', 'RR': 'ṝ', 'lR': 'ḷ', 'lRR': 'ḹ',
    'e': 'e', 'E': 'ai', 'o': 'o', 'O': 'au',
    'M': 'ṃ', 'H': 'ḥ',
    'k': 'k', 'kh': 'kh', 'g': 'g', 'gh': 'gh', 'G': 'ṅ',
    'c': 'c', 'ch': 'ch', 'j': 'j', 'jh': 'jh', 'J': 'ñ',
    'w': 'ṭ', 'W': 'ṭh', 'q': 'ḍ', 'Q': 'ḍh', 'R': 'ṇ',
    't': 't', 'th': 'th', 'd': 'd', 'dh': 'dh', 'n': 'n',
    'p': 'p', 'ph': 'ph', 'b': 'b', 'bh': 'bh', 'm': 'm',
    'y': 'y', 'r': 'r', 'l': 'l', 'v': 'v',
    'S': 'ś', 'z': 'ṣ', 's': 's', 'h': 'h',
}

def slp1_to_iast(text: str) -> str:
    """Convert SLP1 encoding to IAST transliteration."""
    # Multi-char first
    result = []
    i = 0
    while i < len(text):
        matched = False
        for length in [2, 1]:
            if i + length <= len(text):
                substr = text[i:i+length]
                if substr in SLP1_TO_IAST:
                    result.append(SLP1_TO_IAST[substr])
                    i += length
                    matched = True
                    break
        if not matched:
            result.append(text[i])  # Keep unknown chars
            i += 1
    return ''.join(result)

# Mode to Lakara mapping
MODE_TO_LAKARA = {
    'pres': 'lata',
    'ipft': 'lan',      # Imperfect (laṅ)
    'impv': 'lot',      # Imperative (loṭ)
    'opt': 'vid',       # Optative (vidhi)
    'pfut': 'lit',      # Perfect future/perfect
    'sfut': 'lut',      # Simple future (luṭ)
    'ipfct': 'lrt',     # Imperfective aorist (lṛṭ)
    'cond': 'lrn',      # Conditional (lṛṅ)
    'bh': 'ling',       # Potential/bhaviṣya (ling)
    'ash': 'ashirlinga', # Benedictive (āśīrliṅ)
    'perf': 'lit',      # Perfect alt
    'aorist': 'lrt',    # Aorist alt
}

# Person/number mapping
PERSON_MAP = {
    '1': 'uttama',
    '2': 'madhyama',
    '3': 'prathama',
}
NUMBER_MAP = {
    's': 'ekavachana',
    'd': 'dvivachana',
    'p': 'bahuvachana',
}

def parse_roots_csv(csv_path: str) -> Tuple[Dict, List[Dict]]:
    """Parse roots.csv into full conjugations and training pairs."""

    full_conjugations = defaultdict(lambda: defaultdict(dict))
    training_pairs = []

    with open(csv_path, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f)
        for row in reader:
            form_slp = row['form']
            root_slp = row['root']
            verb_class = row['class']
            person = row['person']
            number = row['number']
            mode = row['mode']
            voice = row['voice']
            # modification = row.get('modification', '')

            # Convert from SLP1 to IAST
            form = slp1_to_iast(form_slp)
            root = slp1_to_iast(root_slp)

            # Map mode to lakara
            lakara = MODE_TO_LAKARA.get(mode)
            if not lakara:
                # Skip unknown modes for now (could add more mappings)
                continue

            # Person number key
            person_name = PERSON_MAP.get(person)
            number_name = NUMBER_MAP.get(number)
            if not person_name or not number_name:
                continue
            pn_key = f"{person_name}_{number_name}"

            # Store in conjugation table
            # Use root+class+voice as key to group by verb
            verb_key = f"{root}|{verb_class}|{voice}"
            if lakara not in full_conjugations[verb_key]:
                full_conjugations[verb_key][lakara] = {}
            full_conjugations[verb_key][lakara][pn_key] = form

            # Create training pair (source → target)
            # source: root|lakara|pn_key
            source = f"{root}|{lakara}|{pn_key}"
            training_pairs.append({
                "source": source,
                "target": form,
                "root": root,
                "class": verb_class,
                "lakara": lakara,
                "person": person_name,
                "number": number_name,
                "voice": voice,
            })

    return full_conjugations, training_pairs

def build_full_json(conjugations: Dict) -> List[Dict]:
    """Convert nested conjugations dict to list of records."""
    results = []
    for verb_key, lakaras in conjugations.items():
        root, verb_class, voice = verb_key.split('|')
        entry = {
            "root": root,
            "class": verb_class,
            "voice": voice,
            "conjugations": lakaras,
        }
        results.append(entry)
    return results

def main():
    input_csv = "data/sanskrit_heritage_roots.csv"
    output_full = "data/real_conjugations_full.json"
    output_pairs = "data/real_training_pairs.json"

    print(f"Reading {input_csv} ...")
    full_conv, pairs = parse_roots_csv(input_csv)

    print(f"Parsed {len(full_conv)} unique verb roots")
    print(f"Generated {len(pairs)} training pairs")

    # Save full conjugation tables
    full_list = build_full_json(full_conv)
    with open(output_full, 'w', encoding='utf-8') as f:
        json.dump(full_list, f, ensure_ascii=False, indent=2)
    print(f"Saved full conjugations to {output_full}")

    # Save training pairs
    with open(output_pairs, 'w', encoding='utf-8') as f:
        json.dump(pairs, f, ensure_ascii=False, indent=2)
    print(f"Saved training pairs to {output_pairs}")

    # Print sample
    if full_list:
        sample = full_list[0]
        print("\nSample entry:")
        print(f"  Root: {sample['root']} (class {sample['class']}, {sample['voice']})")
        for lakara, forms in sample['conjugations'].items():
            print(f"  {lakara}: {forms}")
            break

if __name__ == "__main__":
    main()
