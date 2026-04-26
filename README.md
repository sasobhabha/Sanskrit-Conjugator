# Sanskrit Verb Conjugator Neural Network

An interactive neural network for conjugating Sanskrit verbs across all 10 lakaras (tenses/moods) using sequence-to-sequence deep learning.

## Features

- Conjugates any Sanskrit verb in all 10 lakaras
- Supports IAST transliteration input
- Neural network trained on classical Sanskrit grammar sources (Panini's Ashtadhyayi)
- Interactive CLI and Web API
- Rule-based fallback for instant use

## The 10 Lakaras

| Code | Name (Devanagari) | Name (English) |
|------|-------------------|----------------|
| lata | लट् | Present/Future |
| lit | लिट् | Perfect |
| lrt | लृट् | Simple Past (Aorist) |
| lut | लुट् | Past Future |
| lrn | लृङ् | Conditional |
| lan | लङ् | Imperfect |
| ling | लिङ्ग् | Potential |
| lot | लोट् | Imperative |
| vid | विद् | Optative |
| ashirlinga | आशीर्लिङ्ग् | Benedictive |

## Installation

```bash
cd "/Users/Shashwath/Sanskrit Conjugator"
pip install -r requirements.txt
```

## Usage

### Command Line Interface

Run the interactive CLI:

```bash
python -m src.cli
```

Type a verb stem (dhatu) in IAST transliteration to see all conjugations, e.g.:

```
Enter verb stem (dhatu): gacch
```

Or conjugate a specific verb directly:

```bash
python -m src.cli gacch
# or
python -m src.cli vr --method rule   # use rule-based grammar
```

### Web API

Start the FastAPI server:

```bash
uvicorn src.api:app --reload
```

API will be available at `http://localhost:8000`

Endpoints:
- `GET /` – API information
- `GET /health` – Health check
- `GET /lakaras` – List all 10 lakaras with descriptions
- `GET /lakaras/{code}` – Details of a specific lakara
- `POST /conjugate` or `GET /conjugate/{verb_stem}` – Conjugate a verb

Example:
```bash
curl "http://localhost:8000/conjugate/gacch"
```

### Training the Model

The neural network is trained on a synthetic dataset generated using classical Paninian grammar rules from official Sanskrit sources (Ashtadhyayi, Siddhanta Kaumudi, Dhatupatha).

Generate data and train:

```bash
# Generate training data (if not present)
python src/data_generator.py

# Train the model
python src/train.py --epochs 30 --embed-dim 128 --hidden-dim 256
```

Model checkpoints are saved in the `models/` directory. Once trained, the CLI and API will automatically use the neural model (if `--method model` is specified in CLI, or by default in API).

## Architecture

- Tokenizer: Character-level for Sanskrit IAST characters
- Encoder: Bidirectional LSTM
- Decoder: LSTM with Bahdanau attention
- Conditional generation: Input `verb|lakara|person_number` → Output conjugated form
- Total parameters: ~2 million (configurable)

## Data Sources

Training data is derived from authoritative Sanskrit grammatical sources:

- **Panini's Ashtadhyayi** – the foundational grammar (c. 4th century BCE)
- **Siddhanta Kaumudi** – Bhattoji Dikshita's systematic exposition (16th century)
- **Dhatupatha** – traditional list of 2000+ Sanskrit verb roots with conjugation patterns

The data generator implements Paninian rules for all 10 lakaras, producing correct classical Sanskrit forms.

## Project Structure

```
├── src/
│   ├── model.py          # Neural network definition
│   ├── data_generator.py # Sanskrit grammar engine
│   ├── train.py          # Training script
│   ├── cli.py            # Command-line interface
│   └── api.py            # FastAPI REST service
├── data/
│   ├── verb_conjugations_full.json   # Full conjugation tables
│   └── training_pairs.json           # Source-target pairs for training
├── models/              # Saved model checkpoints (after training)
├── requirements.txt
└── README.md
```

## License

MIT – Feel free to use and extend.
# Sanskrit-Conjugator
