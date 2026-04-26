#!/usr/bin/env python
"""Real Sanskrit verb lookup using IAST-rooted Heritage dataset."""

import json
import os
from typing import Dict, Optional

HERITAGE_DB_PATH = "data/real_conjugations_full.json"
_db = None

def load_db():
    global _db
    if _db is None:
        if os.path.exists(HERITAGE_DB_PATH):
            with open(HERITAGE_DB_PATH, 'r', encoding='utf-8') as f:
                _db = json.load(f)
        else:
            _db = []
    return _db

def lookup_verb(dhatu: str, voice: str = 'para') -> Optional[Dict]:
    """Look up a verb root in the IAST-based Heritage dataset.

    Args:
        dhatu: Verb root in IAST (e.g., 'gam', 'kṛ', 'bhū')
        voice: 'para' or 'atma'

    Returns:
        Dict with conjugations or None.
    """
    db = load_db()
    # Normalize: lowercase, strip spaces
    key = dhatu.lower().strip()

    # Direct match
    matches = [e for e in db if e['root'].lower() == key]

    # Alias map: map common IAST spellings to possible root keys in DB
    aliases = {
        'kṛ': ['kṛ#1', 'kṛ#2'],
        'kri': ['kṛ#1', 'kṛ#2'],
        'kr': ['kṛ#1'],
        'bhū': ['bhū#1'],
        'bhu': ['bhū#1'],
        'bh': ['bhū#1'],
        'bhav': ['bhū#1'],
        'as': ['as#1', 'as#2'],
        'i': ['i'],
        'gam': ['gam'],
        'gacch': ['gam'],
        'vad': ['vad'],
        'dhā': ['dhā#1', 'dhā#2', 'dhāv#1'],
        'dha': ['dhā#1'],
        'da': ['dhā#1'],
        'pā': ['pā#1', 'pā#2'],
        'pa': ['pā#1'],
        'kṣi': ['kṣi', 'kṣip'],
        'kṣ': ['kṣi'],
        'yaj': ['yaj'],
        'vid': ['vid'],
        'jñā': ['jñā'],
        'jna': ['jñā'],
        'sukh': ['sukh'],
        'sthā': ['sthā'],
        'sri': ['śri'],  # maybe
        'ruh': ['ruh'],
        'pat': ['pat'],
        'bandh': ['bandh'],
        'ji': ['ji'],
        'dṛś': ['dṛś'],
        'śru': ['śru'],
        'lī': ['lī'],
        'pū': ['pū'],
        'stu': ['stu'],
        'dā': ['dā'],
        'mṛ': ['mṛ'],
        'yā': ['yā'],
        'sṛ': ['sṛ'],
        'vah': ['vah'],
        'nah': ['nah'],
        'seh': ['seh'],
    }

    if not matches and key in aliases:
        for alt in aliases[key]:
            matches = [e for e in db if e['root'].lower() == alt.lower()]
            if matches:
                break

    if not matches:
        return None

    entry = next((m for m in matches if m.get('voice') == voice), matches[0])
    return {
        "root": entry['root'],
        "class": entry.get('class', ''),
        "pada": entry.get('voice', voice),
        "conjugations": entry['conjugations'],
    }

def get_all_forms(entry: Dict) -> Dict:
    return entry.get('conjugations', {})

def get_lakara_forms(entry: Dict, lakara: str) -> Dict:
    return entry.get('conjugations', {}).get(lakara, {})

def list_available_roots() -> list:
    db = load_db()
    return sorted(set(e['root'] for e in db))

if __name__ == "__main__":
    test_roots = ['gam', 'kṛ', 'kri', 'bhū', 'vad', 'as', 'i', 'dhā', 'pā', 'kṣi']
    for root in test_roots:
        result = lookup_verb(root)
        if result:
            lakaras = sorted(result['conjugations'].keys())
            print(f"{root} → {result['root']} (cls {result['class']}, {result['pada']}): {', '.join(lakaras[:6])}")
        else:
            print(f"{root}: NOT FOUND")
