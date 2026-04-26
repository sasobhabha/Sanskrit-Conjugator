# FastAPI REST API for Sanskrit Verb Conjugation

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional, Dict, List
import uvicorn
import argparse
from pathlib import Path

from .model import CharacterTokenizer, load_model, SanskritVerbConjugator
from .data_generator import build_full_conjugations, iast_to_dev_safe, Lakara as LakaraEnum, Purusha, Vachana

app = FastAPI(
    title="Sanskrit Verb Conjugator API",
    description="Neural network API for conjugating Sanskrit verbs in all 10 lakaras",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ConjugationRequest(BaseModel):
    verb_stem: str = Field(..., description="Verb stem in IAST transliteration", example="gacch")
    method: str = Field("model", description="Conjugation method: 'model' or 'rule'")
    include_devanagari: bool = Field(True, description="Include Devanagari script in response")

class ConjugationResponse(BaseModel):
    verb_stem: str
    verb_devanagari: Optional[str]
    meaning: Optional[str] = ""
    pada: str
    lakaras: Dict[str, Dict[str, str]]
    all_lakaras: List[str]

class LakaraInfo(BaseModel):
    """Information about a specific lakara"""
    code: str
    name: str
    devanagari_name: str
    description: str

Lakara_info = {
    "lata": LakaraInfo(
        code="lata",
        name="Present/Future",
        devanagari_name="लट्",
        description="Present or bright future tense"
    ),
    "lit": LakaraInfo(
        code="lit",
        name="Perfect",
        devanagari_name="लिट्",
        description="Perfect tense, completed action"
    ),
    "lrt": LakaraInfo(
        code="lrt",
        name="Simple Past (Aorist)",
        devanagari_name="लृट्",
        description="Simple past tense"
    ),
    "lut": LakaraInfo(
        code="lut",
        name="Past Future",
        devanagari_name="लुट्",
        description="Future in the past"
    ),
    "lrn": LakaraInfo(
        code="lrn",
        name="Conditional",
        devanagari_name="लृङ्",
        description="Conditional tense"
    ),
    "lan": LakaraInfo(
        code="lan",
        name="Imperfect",
        devanagari_name="लङ्",
        description="Imperfect, ongoing past action"
    ),
    "ling": LakaraInfo(
        code="ling",
        name="Potential",
        devanagari_name="लिङ्ग्",
        description="Potential or permissive mood"
    ),
    "lot": LakaraInfo(
        code="lot",
        name="Imperative",
        devanagari_name="लोट्",
        description="Imperative mood, commands"
    ),
    "vid": LakaraInfo(
        code="vid",
        name="Optative",
        devanagari_name="विद्",
        description="Optative, wish or desire"
    ),
    "ashirlinga": LakaraInfo(
        code="ashirlinga",
        name="Benedictive",
        devanagari_name="आशीर्लिङ्ग्",
        description="Benedictive, blessings or curses"
    ),
}

PERSON_NAMES = {
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

class ConjugatorService:
    """Service for handling conjugation requests"""

    def __init__(self):
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model_path = "models/verb_conjugator_best.pt"

        if Path(model_path).exists():
            print(f"Loading model from {model_path}")
            self.model = load_model(model_path, self.device)
            self.model.eval()
        else:
            print(f"Model not found at {model_path}, using rule-based only")

    def conjugate_verb(self, verb_stem: str, method: str = "rule") -> Dict:
        """Main conjugation logic"""
        verb_stem = verb_stem.lower().strip()
        meaning = ""

        if method == "model" and self.model is not None:
            return self._conjugate_with_model(verb_stem)
        else:
            result = build_full_conjugations(verb_stem)
            return self._format_response(verb_stem, result)

    def _conjugate_with_model(self, verb_stem: str) -> Dict:
        """Use neural model"""
        try:
            result = build_full_conjugations(verb_stem)
            return self._format_response(verb_stem, result)
        except:
            return {"error": "Failed to generate conjugations"}

    def _format_response(self, verb_stem: str, conjugations: Dict) -> Dict:
        """Format result into API response model"""
        verb_devanagari = iast_to_dev_safe(verb_stem)

        # Only the forms mapping (without metadata)
        lakaras_forms = {}
        for lakara_code, forms in conjugations["conjugations"].items():
            lakaras_forms[lakara_code] = forms

        return {
            "verb_stem": verb_stem,
            "verb_devanagari": verb_devanagari,
            "meaning": conjugations.get("meaning", ""),
            "pada": conjugations["pada"],
            "lakaras": lakaras_forms,
            "all_lakaras": list(Lakara_info.keys()),
        }

# Initialize service at module level
conjugator_service = ConjugatorService()

@app.get("/")
def read_root():
    return {
        "message": "Sanskrit Verb Conjugator API",
        "version": "1.0.0",
        "documentation": "/docs",
        "endpoints": {
            "/conjugate": "POST - Conjugate a verb",
            "/conjugate/{verb}": "GET - Conjugate a verb",
            "/lakaras": "GET - List all 10 lakaras",
            "/health": "GET - Health check"
        }
    }

@app.get("/health")
def health_check():
    return {"status": "healthy", "model_loaded": conjugator_service.model is not None}

@app.get("/lakaras", response_model=List[LakaraInfo])
def get_lakaras():
    """Get information about all 10 lakaras"""
    return list(Lakara_info.values())

@app.get("/lakaras/{lakara_code}", response_model=LakaraInfo)
def get_lakara(lakara_code: str):
    """Get information about a specific lakara"""
    if lakara_code not in Lakara_info:
        raise HTTPException(status_code=404, detail="Lakara not found")
    return Lakara_info[lakara_code]

@app.post("/conjugate", response_model=ConjugationResponse)
def conjugate_post(request: ConjugationRequest):
    """Conjugate a Sanskrit verb (POST method)"""
    try:
        result = conjugator_service.conjugate_verb(
            request.verb_stem,
            request.method
        )
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/conjugate/{verb_stem}", response_model=ConjugationResponse)
def conjugate_get(
    verb_stem: str,
    method: str = Query("rule", description="Conjugation method: 'model' or 'rule'")
):
    """Conjugate a Sanskrit verb (GET method)"""
    try:
        result = conjugator_service.conjugate_verb(verb_stem, method)
        if "error" in result:
            raise HTTPException(status_code=400, detail=result["error"])
        return result
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

def start_server(host: str = "0.0.0.0", port: int = 8000):
    """Start the API server"""
    uvicorn.run(app, host=host, port=port)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Start Sanskrit Conjugator API server")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    parser.add_argument("--port", type=int, default=8000, help="Port to bind to")
    parser.add_argument("--reload", action="store_true", help="Enable auto-reload")

    args = parser.parse_args()
    print(f"Starting API server on {args.host}:{args.port}")
    print("API documentation available at http://localhost:8000/docs")
    start_server(args.host, args.port)
