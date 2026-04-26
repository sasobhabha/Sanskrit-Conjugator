"""
Sanskrit Verb Conjugation Data Generator
Based on Panini's grammar rules from Ashtadhyayi
"""

import random
import json
import os
import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
from enum import Enum

class Lakara(Enum):
    """The 10 lakaras (tenses/moods) in Sanskrit"""
    LAT = "lata"      # Present/Future indicative
    LIT = "lit"       # Perfect
    LRT = "lrt"       # Simple past (aorist)
    LUT = "lut"       # Past future
    LRN = "lrn"       # Conditional
    LAN = "lan"       # Imperfect
    LING = "ling"     # Potential
    LOT = "lot"       # Imperative
    VID = "vid"       # Optative
    ASHIRLINGA = "ashirlinga"  # Benedictive

class Purusha(Enum):
    """Three persons in Sanskrit"""
    PRATHAMA = "prathama"   # 3rd person (he/she/it)
    MADHYAMA = "madhyama"   # 2nd person (you)
    UTTAMA = "uttama"       # 1st person (I/we)

class Vachana(Enum):
    """Numbers"""
    EKAVACHANA = "ekavachana"     # Singular
    DVAIVACHANA = "dvivachana"    # Dual
    BAHUVACHANA = "bahuvachana"   # Plural

@dataclass
class VerbStem:
    """Represents a Sanskrit verb stem with its properties"""
    stem: str
    pada: str  # 'parasmaipada' or 'atmanepada'
    class_code: str  # 10 classes (1-10) from Panini
    meaning: str
    transliteration: str

# Classical Sanskrit verb classes (dhatu classification from Panini)
VERB_CLASSES = {
    "1": {"ending": "ati", "thematic": True, "name": "bhave (to be)"},
    "2": {"ending": "ati", "thematic": True, "name": "upagrahe (to enjoy)"},
    "3": {"ending": "ati", "thematic": False, "name": "krishne (to do)"},
    "4": {"ending": "ati", "thematic": False, "name": "sampradane (to give)"},
    "5": {"ending": "ati", "thematic": False, "name": "adhane (to notice)"},
    "6": {"ending": "ati", "thematic": False, "name": "vibhaktau (to divide)"},
    "7": {"ending": "ati", "thematic": False, "name": "rudhake (to obstruct)"},
    "8": {"ending": "ati", "thematic": False, "name": "curadi (to steal)"},
    "9": {"ending": "ati", "thematic": True, "name": "krida ayam (to play)"},
    "10": {"ending": "ati", "thematic": True, "name": "bhavaya (to become)"},
}

# Define known verb stems (dhatus) from traditional Sanskrit grammar
# These are the root forms used in Ashtadhyayi

# The parsing engine for developing strong stems are unreliable to not develop at all.
# Kept for later.

class ConjugationEngine:
    """Generates Sanskrit verb conjugations based on Panini's grammar rules

    Reference: Panini's Ashtadhyayi, Siddhanta Kaumudi
    """

    # Base endings from traditional grammar for present system (lat/lat)
    LAT_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "ti",
            Vachana.DVAIVACHANA: "taḥ",
            Vachana.BAHUVACHANA: "anti",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "si",
            Vachana.DVAIVACHANA: "thaḥ",
            Vachana.BAHUVACHANA: "tha",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "mi",
            Vachana.DVAIVACHANA: "vaḥ",
            Vachana.BAHUVACHANA: "maḥ",
        }
    }

    # Perfect (lit) endings - traditional, Oldperfect
    LIT_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "va",
            Vachana.DVAIVACHANA: "viva",
            Vachana.BAHUVACHANA: "re",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "tha",
            Vachana.DVAIVACHANA: "vivathas",
            Vachana.BAHUVACHANA: "re",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "a",
            Vachana.DVAIVACHANA: "iva",
            Vachana.BAHUVACHANA: "ma",
        }
    }

    # Aorist (lrt) - simple past
    LRT_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "t",
            Vachana.DVAIVACHANA: "tāṃ",
            Vachana.BAHUVACHANA: "an",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "thāḥ",
            Vachana.DVAIVACHANA: "thās",
            Vachana.BAHUVACHANA: "dhvam",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "i",
            Vachana.DVAIVACHANA: "iva",
            Vachana.BAHUVACHANA: "ima",
        }
    }

    # S Future (lut) - periphrastic future, past-based
    LUT_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "iṣyati",
            Vachana.DVAIVACHANA: "iṣyataḥ",
            Vachana.BAHUVACHANA: "iṣyanti",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "iṣyasi",
            Vachana.DVAIVACHANA: "iṣyathaḥ",
            Vachana.BAHUVACHANA: "iṣyatha",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "iṣyāmi",
            Vachana.DVAIVACHANA: "iṣyāvaḥ",
            Vachana.BAHUVACHANA: "iṣyāmaḥ",
        }
    }

    # Conditional (lrn) - future-perfect, periphrastic conditional type
    LRN_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "iṣyati",
            Vachana.DVAIVACHANA: "iṣyataḥ",
            Vachana.BAHUVACHANA: "iṣyanti",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "iṣyasi",
            Vachana.DVAIVACHANA: "iṣyathaḥ",
            Vachana.BAHUVACHANA: "iṣyatha",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "iṣyāmi",
            Vachana.DVAIVACHANA: "iṣyāvaḥ",
            Vachana.BAHUVACHANA: "iṣyāmaḥ",
        }
    }

    # Imperfect (lan) - secondary simple past, following augment "a"
    LAN_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "t",
            Vachana.DVAIVACHANA: "tāṃ",
            Vachana.BAHUVACHANA: "n",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "thāḥ",
            Vachana.DVAIVACHANA: "thās",
            Vachana.BAHUVACHANA: "dhvam",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "i",
            Vachana.DVAIVACHANA: "iva",
            Vachana.BAHUVACHANA: "ima",
        }
    }

    # Benedictive/Potential: added ing to lit form bened true
    LING_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "yāt",
            Vachana.DVAIVACHANA: "yātām",
            Vachana.BAHUVACHANA: "yāyuḥ",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "yās",
            Vachana.DVAIVACHANA: "yāstām",
            Vachana.BAHUVACHANA: "yāsta",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "yāsam",
            Vachana.DVAIVACHANA: "yāsvah",
            Vachana.BAHUVACHANA: "yāsmah",
        }
    }

    # Imperative (lot) - order forms!
    LOT_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "tu",
            Vachana.DVAIVACHANA: "tām",
            Vachana.BAHUVACHANA: "ntu",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "hi",
            Vachana.DVAIVACHANA: "tām",
            Vachana.BAHUVACHANA: "ta",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "āni",
            Vachana.DVAIVACHANA: "āva",
            Vachana.BAHUVACHANA: "āma",
        }
    }

    # Optative (vid) - request forms
    VID_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "yāt",
            Vachana.DVAIVACHANA: "yātām",
            Vachana.BAHUVACHANA: "yāyuḥ",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "yās",
            Vachana.DVAIVACHANA: "yāstām",
            Vachana.BAHUVACHANA: "yāsta",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "yāsam",
            Vachana.DVAIVACHANA: "yāsvah",
            Vachana.BAHUVACHANA: "yāsmah",
        }
    }

    # Benedictive (ashirlinga) - pray/curse
    ASHIRLINGA_ENDINGS = {
        Purusha.PRATHAMA: {
            Vachana.EKAVACHANA: "sāṭ",
            Vachana.DVAIVACHANA: "sāṭām",
            Vachana.BAHUVACHANA: "sāṭām",
        },
        Purusha.MADHYAMA: {
            Vachana.EKAVACHANA: "sās",
            Vachana.DVAIVACHANA: "sāṭām",
            Vachana.BAHUVACHANA: "sāṭa",
        },
        Purusha.UTTAMA: {
            Vachana.EKAVACHANA: "sāsam",
            Vachana.DVAIVACHANA: "sāve",
            Vachana.BAHUVACHANA: "sāme",
        }
    }

    def __init__(self):
        self.lakara_endings = {
            Lakara.LAT: self.LAT_ENDINGS,
            Lakara.LIT: self.LIT_ENDINGS,
            Lakara.LRT: self.LRT_ENDINGS,
            Lakara.LUT: self.LUT_ENDINGS,
            Lakara.LRN: self.LRN_ENDINGS,
            Lakara.LAN: self.LAN_ENDINGS,
            Lakara.LING: self.LING_ENDINGS,
            Lakara.LOT: self.LOT_ENDINGS,
            Lakara.VID: self.VID_ENDINGS,
            Lakara.ASHIRLINGA: self.ASHIRLINGA_ENDINGS,
        }

    def get_ending(self, lakara: Lakara, purusha: Purusha,
                   vachana: Vachana) -> str:
        """Get the appropriate ending for given person/number/lakara"""
        endings = self.lakara_endings[lakara]
        return endings[purusha][vachana]

class SanskritNormalizer:
    """Handles Sanskrit sandhi and phonetics

    Uses rules from Panini's Ashtadhyayi (sUtras 6.1-6.4)
    Reference: https://sa.wikisource.org/wiki/अष्टाध्यायी
    """

    # Basic IAST mapping for Devanagari letters
    DEVANAGARI_TO_IAST = {
        'अ': 'a', 'आ': 'ā', 'इ': 'i', 'ई': 'ī',
        'उ': 'u', 'ऊ': 'ū', 'ऋ': 'ṛ', 'ॠ': 'ṝ',
        'ऌ': 'ḷ', 'ॡ': 'ḹ', 'ए': 'e', 'ऐ': 'ai',
        'ओ': 'o', 'औ': 'au', 'ं': 'ṃ', 'ः': 'ḥ',
        'क': 'k', 'ख': 'kh', 'ग': 'g', 'घ': 'gh',
        'ङ': 'ṅ', 'च': 'c', 'छ': 'ch', 'ज': 'j',
        'झ': 'jh', 'ञ': 'ñ', 'ट': 'ṭ', 'ठ': 'ṭh',
        'ड': 'ḍ', 'ढ': 'ḍh', 'ण': 'ṇ', 'त': 't',
        'थ': 'th', 'द': 'd', 'ध': 'dh', 'न': 'n',
        'प': 'p', 'फ': 'ph', 'ब': 'b', 'भ': 'bh',
        'म': 'm', 'य': 'y', 'र': 'r', 'ल': 'l',
        'व': 'v', 'श': 'ś', 'ष': 'ṣ', 'स': 's',
        'ह': 'h',
    }

    IAST_TO_DEVANAGARI = {v: k for k, v in DEVANAGARI_TO_IAST.items()}

    @staticmethod
    def iast_to_devanagari(text: str) -> str:
        """Convert IAST transliteration to Devanagari script"""
        result = []
        i = 0
        while i < len(text):
            # Check for diacritics and long vowels
            if i + 1 < len(text):
                combined = text[i:i+2]
                if combined in SanskritNormalizer.IAST_TO_DEVANAGARI:
                    result.append(combined)
                    i += 2
                    continue

            # Single character
            if text[i] in SanskritNormalizer.IAST_TO_DEVANAGARI:
                result.append(text[i])

            i += 1

        return ''.join(result)

    @staticmethod
    def devanagari_to_iast(text: str) -> str:
        """Convert Devanagari to IAST"""
        result = []
        for char in text:
            if char in SanskritNormalizer.DEVANAGARI_TO_IAST:
                result.append(SanskritNormalizer.DEVANAGARI_TO_IAST[char])
        return ''.join(result)

    @staticmethod
    def apply_sandhi(stem: str, ending: str) -> str:
        """Apply basic sandhi rules as per Ashtadhyayi"""

        # Internal and external sandhi rules
        result = stem + ending

        # Rule: Final 'a' + 'i' -> 'e' (guNa)
        result = result.replace("ai", "e")

        # Rule: Final 'a' + 'u' -> 'o' (guNa)
        result = result.replace("au", "o")

        # Rule: a + a = ā (vrddhi) - rarely in conjugation
        # more rules following akanksha, adesha, svara

        return result

    @staticmethod
    def get_stem_variations(dhatu: str) -> List[str]:
        """Generate valid stem variations for root dhatu (verb root)"""

        dhatu = dhatu.lower()

        variations = [
            dhatu,
        ]

        # Add thematic vowel for class 1 verbs
        if dhatu.endswith(("k", "c", "ṭ", "t", "p", "g", "j", "ḍ", "d", "b")):
            # Add 'a' as connecting vowel
            variations.append(dhatu + "a")

        # Stem forms (guna, vrddhi)
        # . TODO: proper rules for dhatu modification according to class
        return variations

def build_class1_conjugations(dhatu: str) -> Dict[str, str]:
    """Build complete conjugations for a Class 1 (bhave) verb"""

    # Create stem with connecting vowel
    stem = dhatu
    if not dhatu.endswith('a'):
        stem = dhatu + "a"

    results = {}
    engine = ConjugationEngine()
    normalizer = SanskritNormalizer()

    for lakara in Lakara:
        for purusha in Purusha:
            for vachana in Vachana:
                ending = engine.get_ending(lakara, purusha, vachana)

                # Apply bandha-sandhi variants
                conjugated = normalizer.apply_sandhi(stem, ending)

                key = f"{lakara.value}_{purusha.value}_{vachana.value}"
                results[key] = conjugated

    return results

def iast_to_dev_safe(iast: str) -> str:
    """Convert IAST to Devanagari safely"""
    mapping = {
        'a': 'अ', 'ā': 'आ', 'i': 'इ', 'ī': 'ई',
        'u': 'उ', 'ū': 'ऊ', 'ṛ': 'ऋ', 'ṛṛ': 'ॠ',
        'ḷ': 'ऌ', 'ḹ': 'ॡ', 'e': 'ए', 'ai': 'ऐ',
        'o': 'ओ', 'au': 'औ', 'ṃ': 'ं', 'ḥ': 'ः',
        'k': 'क', 'kh': 'ख', 'g': 'ग', 'gh': 'घ', 'ṅ': 'ङ',
        'c': 'च', 'ch': 'छ', 'j': 'ज', 'jh': 'झ', 'ñ': 'ञ',
        'ṭ': 'ट', 'ṭh': 'ठ', 'ḍ': 'ड', 'ḍh': 'ढ', 'ṇ': 'ण',
        't': 'त', 'th': 'थ', 'd': 'द', 'dh': 'ध', 'n': 'न',
        'p': 'प', 'ph': 'फ', 'b': 'ब', 'bh': 'भ', 'm': 'म',
        'y': 'य', 'r': 'र', 'l': 'ल', 'v': 'व', 'ś': 'श',
        'ṣ': 'ष', 's': 'स', 'h': 'ह',
    }

    # Sort by length descending to match longer sequences
    sorted_keys = sorted(mapping.keys(), key=len, reverse=True)

    result = []
    i = 0
    while i < len(iast):
        matched = False
        for key in sorted_keys:
            if iast[i:i+len(key)] == key:
                result.append(mapping[key])
                i += len(key)
                matched = True
                break
        if not matched:
            result.append(iast[i])
            i += 1

    return ''.join(result)

def build_full_conjugations(dhatu: str, pada: str = "parasmaipada",
                          meaning: str = "", translit: str = "") -> Dict:
    """Build full conjugation table for any dhatu

    First tries to look up real data from Sanskrit Heritage dataset.
    Falls back to rule-based generation if not found.
    """

    # Try real data first
    real_data_path = "data/real_conjugations_full.json"
    if os.path.exists(real_data_path):
        try:
            with open(real_data_path, 'r', encoding='utf-8') as f:
                real_data = json.load(f)

            # Normalize dhatu (strip diacritics, lowercase)
            dhatu_clean = dhatu.lower().strip()

            # Find matching entries
            matches = [e for e in real_data if e['root'] == dhatu_clean]

            if matches:
                # Prefer para (parasmaipada) voice, or first match
                entry = next((m for m in matches if m['voice'] == 'para'), matches[0])
                return {
                    "dhatu_devanagari": iast_to_dev_safe(dhatu_clean),
                    "dhatu_iast": dhatu_clean,
                    "pada": entry.get('voice', pada),
                    "meaning": meaning or "",
                    "transliteration": translit or dhatu_clean,
                    "conjugations": entry['conjugations'],
                }
        except Exception as e:
            pass  # Fall through to rule-based

    # Fallback: rule-based generation
    engine = ConjugationEngine()
    normalizer = SanskritNormalizer()

    stem = dhatu
    if pada == "parasmaipada":
        if dhatu.endswith(('k', 'c', 'ṭ', 't', 'p')):
            stem = dhatu + "ay"
        elif dhatu.endswith(('g', 'j', 'ḍ', 'd', 'b')):
            stem = dhatu[:-1] + "ay" if dhatu else dhatu
        elif dhatu:
            if dhatu[-1] in 'aāīūṛ':
                stem = dhatu
            else:
                stem = dhatu + "a"

    results = {}
    for lakara in Lakara:
        lakara_results = {}
        for purusha in Purusha:
            for vachana in Vachana:
                ending = engine.get_ending(lakara, purusha, vachana)
                form = normalizer.apply_sandhi(stem, ending)
                key = f"{purusha.value}_{vachana.value}"
                lakara_results[key] = form
        results[lakara.value] = lakara_results

    return {
        "dhatu_devanagari": iast_to_dev_safe(dhatu),
        "dhatu_iast": dhatu,
        "pada": pada,
        "meaning": meaning,
        "transliteration": translit or dhatu,
        "conjugations": results,
    }

def generate_dataset(num_verbs: int = 500) -> List[Dict]:
    """Generate training dataset with various verb forms

    Uses classic dhatus from Panini's Dhatupatha (traditional list of 2000+ roots)

    """

    # Known dhatus from Dhatupatha (selected common ones)
    base_dhatus_iast = [
        "gacch", "as", "kri", "bhu", "vad", "ish", "dha", "pa",
        "bhav", "sukh", "yaj", "vid", "i", "ag", "prap", "stu",
        "bandh", "mrit", "vad", "gam", "bhaj", "dhyai", "drś",
        "kṣam", "labh", "muc", "nind", "pat", "raṇ", "sad",
        "tan", "vas", "viś", "yā", "ṣṭhiv", "sṛj", "han",
        "jñā", "dā", "pā", "śru", "ci", "śi", "sthā", "smi",
        "vṛdh", "śri", "ruh", "l-i-kh", "pāṭh", "bodh",
        "vah", "sev", "bhū", "kṛ", "sthā", "bhṛ", "smit",
        "bādh", "cyu", "vad", "svap", "cal", "gaṇ", "tṝ",
        "pṝ", "vah", "pṛ", "spṛś", "kṛṣ", "yabh", "muc",
        "ku", "jīv", "śī", "vṛdh", "pracch", "vyāhṛ", "kīrt",
        "pūj", "bhikṣ", "śraddhā", "prī", "śaṃs", "glai",
        "śmaśru", "śuc", "vibhū", "āp", "ādhā", "abhyas",
        "cint", "pracal", "udbhid", "niṣad", "prajñā", "śri",
    ]

    dataset = []

    for dhatu in base_dhatus_iast[:num_verbs]:
        try:
            conjugate = build_full_conjugations(
                dhatu,
                pada="parasmaipada",
                meaning="",
                translit=dhatu
            )
            dataset.append(conjugate)
        except Exception as e:
            print(f"Error generating {dhatu}: {e}")
            continue

    return dataset

def save_dataset(filename: str, data: List[Dict]):
    """Save training dataset to file"""
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def create_training_pairs(data: List[Dict]) -> List[Dict]:
    """Convert conjugation tables into training pairs (source, target)

    Source format: verb_stem|lakara_code|person_number
    Target format: conjugated_form
    Example: "gacch|lata|prathama_ekavachana" -> "gacchati"
    """

    pairs = []

    for item in data:
        dhatu = item["dhatu_iast"]
        for lakara, forms in item["conjugations"].items():
            for person_number, form in forms.items():
                source = f"{dhatu}|{lakara}|{person_number}"
                pairs.append({
                    "source": source,
                    "target": form,
                })

    return pairs

if __name__ == "__main__":
    import os

    print("Generating Sanskrit verb conjugation dataset...")
    print("Using classical rules from Panini's Ashtadhyayi")
    print("=" * 60)

    dataset = generate_dataset(200)

    os.makedirs("data", exist_ok=True)

    # Save full conjugation tables
    save_dataset("data/verb_conjugations_full.json", dataset)

    # Save source-target pairs for training
    training_pairs = create_training_pairs(dataset)
    save_dataset("data/training_pairs.json", training_pairs)

    print(f"Generated {len(dataset)} verb conjugation tables")
    print(f"Generated {len(training_pairs)} training pairs (source->target)")
    print("Saved to data/verb_conjugations_full.json")
    print("Saved to data/training_pairs.json")
    print()
    print("Sample conjugation for 'gacch':")
    sample = build_full_conjugations("gacch")
    for lakara, forms in sample["conjugations"].items():
        print(f"  {lakara}:")
        for pn, form in forms.items():
            print(f"    {pn}: {form}")
