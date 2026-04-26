# Interactive CLI for Sanskrit Verb Conjugation

import argparse
import sys
import torch
from pathlib import Path
from typing import Dict

from .model import CharacterTokenizer, load_model, SanskritVerbConjugator
from .real_lookup import lookup_verb
from .data_generator import iast_to_dev_safe

class SanskritConjugator:
    """Main application class for Sanskrit verb conjugation"""

    def __init__(self, model_path: str = "models/verb_conjugator_best.pt"):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.tokenizer = CharacterTokenizer(max_length=64)
        self.model = None

        model_file = Path(model_path)
        if model_file.exists():
            print(f"Loading model from {model_path}...")
            self.model = load_model(model_path, self.device)
            self.model.eval()
            print("Model loaded successfully")
        else:
            print(f"Model not found at {model_path}")
            print("Will use rule-based conjugation")
            self.model = None

    def conjugate(self, verb_stem: str, method: str = "rule") -> Dict:
        """Conjugate verb using neural network or rule-based fallback"""

        if method == "model" and self.model is not None:
            return self._conjugate_with_model(verb_stem)
        else:
            return self._conjugate_rule_based(verb_stem)

    def _conjugate_with_model(self, verb_stem: str) -> Dict:
        """Use neural model to predict all conjugations"""

        verb_stem = verb_stem.lower().strip()

        # Prepare source strings for all 90 combinations (10 lakaras × 9 person/number)
        sources = []
        keys = []

        for lakara in Lakara:
            for purusha in Purusha:
                for vachana in Vachana:
                    person_number = f"{purusha.value}_{vachana.value}"
                    src_str = f"{verb_stem}|{lakara.value}|{person_number}"
                    sources.append(src_str)
                    keys.append((lakara.value, person_number))

        # Batch encode
        src_tensor = self.tokenizer.batch_encode(sources).to(self.device)

        # Generate predictions
        with torch.no_grad():
            preds = self.model.predict(src_tensor, max_len=32)  # [num_combos, seq_len]

        # Decode predictions
        conjugations = {}
        for (lakara_val, person_num), tokens in zip(keys, preds):
            tokens_list = tokens.cpu().numpy().tolist()
            form = self.tokenizer.decode(tokens_list)

            if lakara_val not in conjugations:
                conjugations[lakara_val] = {}
            conjugations[lakara_val][person_num] = form

        return {
            "dhatu_devanagari": iast_to_dev_safe(verb_stem),
            "dhatu_iast": verb_stem,
            "pada": "parasmaipada",   # Trained only on parasmipada
            "meaning": "",
            "conjugations": conjugations,
        }

    def _conjugate_rule_based(self, verb_stem: str) -> Dict:
        """Use authentic Sanskrit Heritage data lookup, fallback to rule-based."""
        # Try real database first
        entry = lookup_verb(verb_stem, voice='para')
        if entry:
            return {
                "dhatu_devanagari": iast_to_dev_safe(entry['root']),
                "dhatu_iast": entry['root'],
                "pada": entry['pada'],
                "meaning": "",
                "conjugations": entry['conjugations'],
            }
        # Fallback to approximate Panini rule engine
        from .data_generator import build_full_conjugations
        return build_full_conjugations(verb_stem)

    def print_conjugations(self, conjugations: Dict):
        """Pretty-print conjugation table"""

        dhatu_dev = conjugations["dhatu_devanagari"]
        dhatu_iast = conjugations["dhatu_iast"]
        pada = conjugations["pada"]
        meaning = conjugations["meaning"]

        print("\n" + "="*60)
        print(f" Sanskrit Verb: {dhatu_dev} ({dhatu_iast})")
        if meaning:
            print(f" Meaning: {meaning}")
        print(f" Pada: {pada}")
        print("="*60)
        print()

        # Lakara names in IAST for clarity
        lakara_names = {
            "lata": "लट् (Present/Future)",
            "lit": "लिट् (Perfect)",
            "lrt": "लृट् (Simple Past)",
            "lut": "लुट् (Past Future)",
            "lrn": "लृङ् (Conditional)",
            "lan": "लङ् (Imperfect)",
            "ling": "लिङ्ग् (Potential)",
            "lot": "लोट् (Imperative)",
            "vid": "विद् (Optative)",
            "ashirlinga": "आशीर्लिङ्ग् (Benedictive)",
        }

        person_names = {
            "prathama_ekavachana": "3rd Person Singular",
            "prathama_dvivachana": "3rd Person Dual",
            "prathama_bahuvachana": "3rd Person Plural",
            "madhyama_ekavachana": "2nd Person Singular",
            "madhyama_dvivachana": "2nd Person Dual",
            "madhyama_bahuvachana": "2nd Person Plural",
            "uttama_ekavachana": "1st Person Singular",
            "uttama_dvivachana": "1st Person Dual",
            "uttama_bahuvachana": "1st Person Plural",
        }

        for lakara, forms in conjugations["conjugations"].items():
            print(f"\n{lakara_names.get(lakara, lakara)}:")
            print("-" * 40)

            # Print as table
            print(f"{'Person/Number':<30} {'Form (Devanagari)':<25} {'Form (IAST)':<25}")
            print("-" * 80)

            for pn, form in forms.items():
                dev_form = iast_to_dev_safe(form)
                print(f"{person_names.get(pn, pn):<30} {dev_form:<25} {form:<25}")

        print()

def main():
    parser = argparse.ArgumentParser(
        description="Interactive Sanskrit Verb Conjugator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s gacchati        # Conjugate 'gacch' (to go)
  %(prog)s kri             # Conjugate 'kri' (to do)
  %(prog)s vad             # Conjugate 'vad' (to speak)
        """
    )

    parser.add_argument(
        "verb",
        nargs="?",
        help="Sanskrit verb stem (dhatu) in IAST transliteration"
    )
    parser.add_argument(
        "--method",
        choices=["model", "rule"],
        default="rule",
        help="Conjugation method: neural model or rule-based grammar"
    )
    parser.add_argument(
        "--model",
        default="models/verb_conjugator_best.pt",
        help="Path to trained model checkpoint"
    )

    args = parser.parse_args()

    # Initialize conjugator
    conjugator = SanskritConjugator(args.model)

    # Interactive mode if no verb provided
    if not args.verb:
        print("Sanskrit Verb Conjugator - Interactive Mode")
        print("Type 'exit' or 'quit' to end")
        print("Type 'help' for list of sample verbs")
        print()

        sample_verbs = ["gacch", "kri", "bhu", "vad", "pa", "as", "i", "dha"]

        while True:
            try:
                verb = input("Enter verb stem (dhatu): ").strip().lower()

                if verb in ['exit', 'quit']:
                    print("Namaskar!")
                    break
                elif verb == 'help':
                    print("\nSample verb stems (dhatus):")
                    for v in sample_verbs:
                        print(f"  {v}")
                    print()
                    continue
                elif not verb:
                    continue

                conjugations = conjugator.conjugate(verb, args.method)
                conjugator.print_conjugations(conjugations)

            except KeyboardInterrupt:
                print("\n\nNamaskar!")
                break
            except Exception as e:
                print(f"Error: {e}")
                continue
    else:
        # Single verb conjugation
        verb = args.verb.lower().strip()
        conjugations = conjugator.conjugate(verb, args.method)
        conjugator.print_conjugations(conjugations)

if __name__ == "__main__":
    main()
