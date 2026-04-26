#!/usr/bin/env python
"""Quick verification that real data lookup works correctly."""

import json
from src.data_generator import iast_to_dev_safe

# Load real conjugation database
with open('data/real_conjugations_full.json') as f:
    REAL_DB = json.load(f)

def lookup_conjugations(dhatu: str, voice: str = 'para'):
    """Look up dhatu in real Heritage dataset."""
    matches = [e for e in REAL_DB if e['root'] == dhatu.lower() and e['voice'] == voice]
    if not matches:
        matches = [e for e in REAL_DB if e['root'] == dhatu.lower()]
    if not matches:
        return None
    # Return first match
    entry = matches[0]
    return {
        "dhatu_devanagari": iast_to_dev_safe(entry['root']),
        "dhatu_iast": entry['root'],
        "pada": entry['voice'],
        "meaning": "",
        "conjugations": entry['conjugations'],
    }

# Test gam
result = lookup_conjugations('gam')
if result:
    print("gam (to go) — from REAL DATA:")
    for lakara in ['lata', 'lan', 'lot', 'vid']:
        if lakara in result['conjugations']:
            print(f"\n{lakara}:")
            for pn, form in result['conjugations'][lakara].items():
                print(f"  {pn}: {form}")
else:
    print("gam not found in database!")

# Test kri (kṛ)
result2 = lookup_conjugations('kf')  # SLP1 for kṛ
if result2:
    print("\n\nkṛ (to do) — from REAL DATA:")
    for lakara in ['lata', 'lan', 'lot']:
        if lakara in result2['conjugations']:
            print(f"\n{lakara}:")
            for pn, form in sorted(result2['conjugations'][lakara].items())[:3]:
                print(f"  {pn}: {form}")
