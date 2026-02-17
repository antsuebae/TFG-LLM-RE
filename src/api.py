"""
API REST para Analisis de Requisitos de Software
TFG: Evaluacion de LLMs en Ingenieria de Requisitos
"""

from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from enum import Enum
from pathlib import Path
import os
import ollama
import re
import json

# Cargar variables de entorno desde .env
from dotenv import load_dotenv
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)

app = FastAPI(
    title="RE-LLM API",
    description="API para analisis de requisitos de software usando LLMs",
    version="1.0.0"
)

# CORS para permitir acceso desde frontends
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ============== MODELOS DE DATOS ==============

class ModelType(str, Enum):
    QWEN = "qwen2.5-coder:7b-instruct-q5_K_M"
    LLAMA = "llama3.1:8b-instruct-q4_K_M"
    NIM_LLAMA_70B = "meta/llama-3.1-70b-instruct"
    NIM_LLAMA_8B = "meta/llama-3.1-8b-instruct"
    NIM_MISTRAL = "mistralai/mistral-7b-instruct-v0.3"

class ClassifyRequest(BaseModel):
    text: str = Field(..., description="Texto del requisito a clasificar")
    model: Optional[ModelType] = Field(default=ModelType.QWEN, description="Modelo a usar")

class ClassifyResponse(BaseModel):
    classification: Literal["F", "NF", "UNKNOWN"]
    confidence: float
    justification: str
    model_used: str

class DefectType(str, Enum):
    AMBIGUOUS = "ambiguous"
    INCOMPLETE = "incomplete"
    INCONSISTENT = "inconsistent"
    NON_VERIFIABLE = "non_verifiable"

class Defect(BaseModel):
    type: DefectType
    description: str
    severity: Literal["low", "medium", "high"]
    suggestion: Optional[str] = None

class AnalyzeRequest(BaseModel):
    text: str = Field(..., description="Texto del requisito a analizar")
    model: Optional[ModelType] = Field(default=ModelType.QWEN)

class AnalyzeResponse(BaseModel):
    original_text: str
    classification: Literal["F", "NF", "UNKNOWN"]
    defects: List[Defect]
    quality_score: float
    improved_version: Optional[str] = None
    model_used: str

class ElicitRequest(BaseModel):
    context: str = Field(..., description="Descripcion del sistema o funcionalidad")
    existing_requirements: Optional[List[str]] = Field(default=[], description="Requisitos ya identificados")
    model: Optional[ModelType] = Field(default=ModelType.QWEN)

class ElicitResponse(BaseModel):
    suggested_requirements: List[str]
    clarifying_questions: List[str]
    missing_areas: List[str]
    model_used: str

class ValidateRequest(BaseModel):
    requirements: List[str] = Field(..., description="Lista de requisitos a validar")
    model: Optional[ModelType] = Field(default=ModelType.QWEN)

class Inconsistency(BaseModel):
    req1_index: int
    req2_index: int
    req1_text: str
    req2_text: str
    conflict_description: str

class ValidateResponse(BaseModel):
    is_consistent: bool
    inconsistencies: List[Inconsistency]
    duplicates: List[tuple]
    coverage_gaps: List[str]
    model_used: str


# ============== FUNCIONES AUXILIARES ==============

def call_llm(prompt: str, model: ModelType) -> str:
    """Llama al modelo LLM seleccionado."""
    # Modelos NVIDIA NIM
    if model.value.startswith("meta/") or model.value.startswith("mistralai/"):
        api_key = os.getenv("NVIDIA_API_KEY")
        if not api_key:
            raise HTTPException(
                status_code=400,
                detail="NVIDIA_API_KEY no configurada. Obtenla en https://build.nvidia.com y configura: export NVIDIA_API_KEY=tu_key"
            )
        try:
            from openai import OpenAI
            client = OpenAI(base_url="https://integrate.api.nvidia.com/v1", api_key=api_key)
            response = client.chat.completions.create(
                model=model.value,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.3,
                max_tokens=1024,
            )
            return response.choices[0].message.content
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error NVIDIA NIM: {str(e)}")

    # Modelos locales (Ollama)
    try:
        response = ollama.chat(
            model=model.value,
            messages=[{"role": "user", "content": prompt}],
            options={"temperature": 0.3, "num_predict": 1024}
        )
        return response["message"]["content"]
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error al llamar al modelo: {str(e)}")


def parse_json_response(response: str) -> dict:
    """Intenta extraer JSON de la respuesta del modelo."""
    # Buscar JSON en la respuesta
    json_match = re.search(r'\{[\s\S]*\}', response)
    if json_match:
        try:
            return json.loads(json_match.group())
        except json.JSONDecodeError:
            pass
    return {}


# ============== ENDPOINTS ==============

@app.get("/")
async def root():
    """Endpoint raiz con informacion de la API."""
    return {
        "name": "RE-LLM API",
        "version": "1.0.0",
        "endpoints": {
            "/classify": "Clasificar requisito como F/NF",
            "/analyze": "Analizar defectos en requisito",
            "/elicit": "Ayudar en elicitacion de requisitos",
            "/validate": "Validar consistencia entre requisitos",
            "/analyze-document": "Analizar documento completo de requisitos"
        },
        "models": {
            "local_ollama": ["qwen2.5-coder:7b-instruct-q5_K_M", "llama3.1:8b-instruct-q4_K_M"],
            "nvidia_nim": ["meta/llama-3.1-70b-instruct", "meta/llama-3.1-8b-instruct", "mistralai/mistral-7b-instruct-v0.3"]
        },
        "nvidia_nim_configured": bool(os.getenv("NVIDIA_API_KEY"))
    }


@app.get("/health")
async def health_check():
    """Verificar estado de la API y modelos."""
    models_status = {}

    # Verificar modelos locales (Ollama)
    for model in [ModelType.QWEN, ModelType.LLAMA]:
        try:
            ollama.chat(model=model.value, messages=[{"role": "user", "content": "test"}],
                       options={"num_predict": 1})
            models_status[model.value] = "available"
        except:
            models_status[model.value] = "unavailable"

    # Verificar NVIDIA NIM
    nim_status = "not_configured"
    if os.getenv("NVIDIA_API_KEY"):
        nim_status = "configured"

    return {
        "status": "healthy",
        "models_local": models_status,
        "nvidia_nim": nim_status
    }


@app.post("/classify", response_model=ClassifyResponse)
async def classify_requirement(request: ClassifyRequest):
    """
    Clasifica un requisito como Funcional (F) o No Funcional (NF).

    - **F (Funcional)**: Describe QUE debe hacer el sistema
    - **NF (No Funcional)**: Describe COMO debe comportarse el sistema
    """
    prompt = f"""Clasifica el siguiente requisito de software.

Requisito: "{request.text}"

Responde en formato JSON exactamente asi:
{{
    "classification": "F" o "NF",
    "confidence": numero entre 0 y 1,
    "justification": "explicacion breve"
}}

Donde:
- F = Funcional (describe una funcion o capacidad del sistema)
- NF = No Funcional (describe rendimiento, seguridad, usabilidad, etc.)

JSON:"""

    response = call_llm(prompt, request.model)
    parsed = parse_json_response(response)

    classification = parsed.get("classification", "UNKNOWN")
    if classification not in ["F", "NF"]:
        # Intentar extraer de texto
        if "NF" in response.upper() or "NO FUNCIONAL" in response.upper():
            classification = "NF"
        elif "FUNCIONAL" in response.upper() or response.strip().upper().startswith("F"):
            classification = "F"
        else:
            classification = "UNKNOWN"

    return ClassifyResponse(
        classification=classification,
        confidence=parsed.get("confidence", 0.5),
        justification=parsed.get("justification", response[:200]),
        model_used=request.model.value
    )


@app.post("/analyze", response_model=AnalyzeResponse)
async def analyze_requirement(request: AnalyzeRequest):
    """
    Analiza un requisito para detectar defectos de calidad.

    Detecta:
    - **Ambiguedad**: Terminos vagos, pronombres sin referencia
    - **Incompletitud**: Falta informacion necesaria
    - **No verificabilidad**: No se puede probar objetivamente
    """
    prompt = f"""Analiza el siguiente requisito de software para detectar defectos de calidad.

Requisito: "{request.text}"

Responde en formato JSON:
{{
    "classification": "F" o "NF",
    "defects": [
        {{
            "type": "ambiguous|incomplete|inconsistent|non_verifiable",
            "description": "descripcion del defecto",
            "severity": "low|medium|high",
            "suggestion": "como mejorarlo"
        }}
    ],
    "quality_score": numero de 0 a 1 (1 = sin defectos),
    "improved_version": "version mejorada del requisito si tiene defectos"
}}

Criterios:
- ambiguous: terminos vagos como "rapido", "facil", "adecuado", pronombres ambiguos
- incomplete: falta manejo de errores, condiciones limite, criterios de aceptacion
- non_verifiable: no se puede medir objetivamente
- inconsistent: contradice informacion implicita

JSON:"""

    response = call_llm(prompt, request.model)
    parsed = parse_json_response(response)

    defects = []
    for d in parsed.get("defects", []):
        try:
            defects.append(Defect(
                type=d.get("type", "ambiguous"),
                description=d.get("description", ""),
                severity=d.get("severity", "medium"),
                suggestion=d.get("suggestion")
            ))
        except:
            continue

    return AnalyzeResponse(
        original_text=request.text,
        classification=parsed.get("classification", "UNKNOWN"),
        defects=defects,
        quality_score=parsed.get("quality_score", 0.5),
        improved_version=parsed.get("improved_version"),
        model_used=request.model.value
    )


@app.post("/elicit", response_model=ElicitResponse)
async def elicit_requirements(request: ElicitRequest):
    """
    Ayuda en la elicitacion de requisitos generando sugerencias y preguntas.

    A partir de una descripcion del sistema, sugiere:
    - Requisitos que podrian faltar
    - Preguntas clarificadoras para el stakeholder
    - Areas no cubiertas
    """
    existing = "\n".join([f"- {r}" for r in request.existing_requirements]) if request.existing_requirements else "Ninguno"

    prompt = f"""Eres un ingeniero de requisitos experto. Ayuda a elicitar requisitos para el siguiente sistema.

Descripcion del sistema:
{request.context}

Requisitos ya identificados:
{existing}

Responde en formato JSON:
{{
    "suggested_requirements": [
        "requisito sugerido 1",
        "requisito sugerido 2"
    ],
    "clarifying_questions": [
        "pregunta para el stakeholder 1",
        "pregunta para el stakeholder 2"
    ],
    "missing_areas": [
        "area no cubierta 1 (ej: seguridad, rendimiento, accesibilidad)"
    ]
}}

Considera:
- Requisitos funcionales y no funcionales
- Casos de error y excepciones
- Requisitos de seguridad, rendimiento, usabilidad
- Integraciones con otros sistemas

JSON:"""

    response = call_llm(prompt, request.model)
    parsed = parse_json_response(response)

    return ElicitResponse(
        suggested_requirements=parsed.get("suggested_requirements", []),
        clarifying_questions=parsed.get("clarifying_questions", []),
        missing_areas=parsed.get("missing_areas", []),
        model_used=request.model.value
    )


@app.post("/validate", response_model=ValidateResponse)
async def validate_requirements(request: ValidateRequest):
    """
    Valida consistencia entre un conjunto de requisitos.

    Detecta:
    - Inconsistencias entre requisitos
    - Requisitos duplicados
    - Gaps de cobertura
    """
    reqs_text = "\n".join([f"{i+1}. {r}" for i, r in enumerate(request.requirements)])

    prompt = f"""Analiza la consistencia del siguiente conjunto de requisitos.

Requisitos:
{reqs_text}

Responde en formato JSON:
{{
    "is_consistent": true o false,
    "inconsistencies": [
        {{
            "req1_index": indice del primer requisito (empezando en 0),
            "req2_index": indice del segundo requisito,
            "conflict_description": "descripcion del conflicto"
        }}
    ],
    "duplicates": [[indice1, indice2], ...],
    "coverage_gaps": [
        "area importante no cubierta"
    ]
}}

Busca:
- Requisitos que se contradigan entre si
- Requisitos duplicados o muy similares
- Areas tipicas no cubiertas (seguridad, errores, rendimiento)

JSON:"""

    response = call_llm(prompt, request.model)
    parsed = parse_json_response(response)

    inconsistencies = []
    for inc in parsed.get("inconsistencies", []):
        try:
            idx1 = inc.get("req1_index", 0)
            idx2 = inc.get("req2_index", 0)
            inconsistencies.append(Inconsistency(
                req1_index=idx1,
                req2_index=idx2,
                req1_text=request.requirements[idx1] if idx1 < len(request.requirements) else "",
                req2_text=request.requirements[idx2] if idx2 < len(request.requirements) else "",
                conflict_description=inc.get("conflict_description", "")
            ))
        except:
            continue

    return ValidateResponse(
        is_consistent=parsed.get("is_consistent", True),
        inconsistencies=inconsistencies,
        duplicates=parsed.get("duplicates", []),
        coverage_gaps=parsed.get("coverage_gaps", []),
        model_used=request.model.value
    )


@app.post("/analyze-document")
async def analyze_document(file: UploadFile = File(...), model: ModelType = ModelType.QWEN):
    """
    Analiza un documento completo de requisitos.

    Acepta archivos .txt o .md con requisitos (uno por linea o separados por lineas vacias).
    """
    if not file.filename.endswith(('.txt', '.md')):
        raise HTTPException(status_code=400, detail="Solo se aceptan archivos .txt o .md")

    content = await file.read()
    text = content.decode('utf-8')

    # Separar requisitos (por lineas no vacias)
    requirements = [line.strip() for line in text.split('\n') if line.strip() and not line.startswith('#')]

    if not requirements:
        raise HTTPException(status_code=400, detail="No se encontraron requisitos en el documento")

    results = []
    for req in requirements[:20]:  # Limitar a 20 para evitar timeouts
        analysis = await analyze_requirement(AnalyzeRequest(text=req, model=model))
        results.append(analysis.model_dump())

    # Resumen
    total_defects = sum(len(r['defects']) for r in results)
    avg_quality = sum(r['quality_score'] for r in results) / len(results)

    return {
        "filename": file.filename,
        "total_requirements": len(requirements),
        "analyzed": len(results),
        "summary": {
            "total_defects_found": total_defects,
            "average_quality_score": round(avg_quality, 2),
            "requirements_with_defects": sum(1 for r in results if r['defects'])
        },
        "details": results
    }


# ============== MAIN ==============

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
