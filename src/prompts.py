"""
Patrones de prompt para tareas de ingenieria de requisitos.

5 tareas: classification (A1), ambiguity (A2), completeness (A3),
          inconsistency (V1), testability (V2)
5 estrategias: question_refinement, cognitive_verifier, persona_context,
               few_shot, chain_of_thought
"""

import re
import logging

logger = logging.getLogger(__name__)

# ============================================================
# PROMPTS: dict[task][strategy] -> template string
# ============================================================

PROMPTS = {
    # ── A1: Clasificacion F / NF ──────────────────────────────
    "classification": {
        "question_refinement": (
            'Clasifica el siguiente requisito de software como F (Funcional) o NF (No Funcional).\n\n'
            'Si la clasificacion no es clara, reformula mentalmente la pregunta para entender mejor el requisito antes de clasificar.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde SOLO con una de estas opciones exactas:\n'
            '- F (si es un requisito funcional - describe QUE debe hacer el sistema)\n'
            '- NF (si es un requisito no funcional - describe COMO debe comportarse el sistema)\n\n'
            'Clasificacion:'
        ),
        "cognitive_verifier": (
            'Analiza el siguiente requisito de software paso a paso:\n\n'
            'Requisito: "{requirement}"\n\n'
            'Pasos de analisis:\n'
            '1. Identifica el sujeto principal del requisito\n'
            '2. Determina si describe una FUNCION (que hace) o una CUALIDAD (como lo hace)\n'
            '3. Considera: Las funciones son acciones concretas, las cualidades son caracteristicas del sistema\n\n'
            'Basandote en tu analisis, clasifica como:\n'
            '- F (Funcional): describe una funcion o capacidad especifica\n'
            '- NF (No Funcional): describe rendimiento, seguridad, usabilidad, u otra cualidad\n\n'
            'Clasificacion (responde solo F o NF):'
        ),
        "persona_context": (
            'Eres un ingeniero de requisitos senior con 10 anos de experiencia en IEEE 830 e ISO 29148.\n\n'
            'Tu tarea es clasificar requisitos de software en dos categorias:\n'
            '- F (Funcional): Requisitos que describen QUE debe hacer el sistema (funciones, caracteristicas, capacidades)\n'
            '- NF (No Funcional): Requisitos que describen COMO debe ser el sistema (rendimiento, seguridad, usabilidad, mantenibilidad, etc.)\n\n'
            'Requisito a clasificar: "{requirement}"\n\n'
            'Como experto, proporciona tu clasificacion (responde unicamente F o NF):'
        ),
        "few_shot": (
            'Clasifica el requisito como F (Funcional) o NF (No Funcional).\n\n'
            'Ejemplos:\n'
            '- "The system shall allow users to login" -> F\n'
            '- "The system shall respond within 2 seconds" -> NF\n'
            '- "The system shall generate monthly reports" -> F\n'
            '- "The system shall be available 99.9% of the time" -> NF\n'
            '- "The system shall send email notifications" -> F\n\n'
            'Requisito: "{requirement}"\n'
            'Clasificacion:'
        ),
        "chain_of_thought": (
            'Clasifica el requisito como F (Funcional) o NF (No Funcional).\n'
            'Piensa paso a paso:\n'
            '1. Identifica el sujeto del requisito\n'
            '2. Determina si describe una ACCION/FUNCION o una CUALIDAD/RESTRICCION\n'
            '3. Las funciones son F, las cualidades son NF\n'
            'Muestra tu razonamiento y luego da la clasificacion final como una sola letra: F o NF.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Razonamiento:'
        ),
    },

    # ── A2: Deteccion de ambiguedad ───────────────────────────
    "ambiguity": {
        "question_refinement": (
            'Analiza si el siguiente requisito de software es ambiguo.\n\n'
            'Si no estas seguro, reformula mentalmente el requisito desde distintas perspectivas para detectar posibles interpretaciones multiples.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde en este formato exacto:\n'
            'AMBIGUOUS: YES o NO\n'
            'TYPE: vague_term / pronoun / quantifier / none\n'
            'WORDS: lista de palabras ambiguas separadas por coma, o "none"'
        ),
        "cognitive_verifier": (
            'Analiza el siguiente requisito paso a paso para detectar ambiguedad:\n\n'
            'Requisito: "{requirement}"\n\n'
            '1. Busca terminos vagos (rapido, facil, adecuado, eficiente, amigable)\n'
            '2. Busca pronombres sin referencia clara (it, they, this, that)\n'
            '3. Busca cuantificadores imprecisos (some, many, few, several)\n'
            '4. Determina si el requisito puede interpretarse de mas de una forma\n\n'
            'Responde en este formato exacto:\n'
            'AMBIGUOUS: YES o NO\n'
            'TYPE: vague_term / pronoun / quantifier / none\n'
            'WORDS: lista de palabras ambiguas separadas por coma, o "none"'
        ),
        "persona_context": (
            'Eres un experto en calidad de requisitos con conocimiento en IEEE 830 y deteccion de requirements smells.\n\n'
            'Analiza si el siguiente requisito contiene ambiguedades que impidan una unica interpretacion:\n\n'
            'Requisito: "{requirement}"\n\n'
            'Tipos de ambiguedad a detectar:\n'
            '- vague_term: terminos subjetivos o imprecisos\n'
            '- pronoun: pronombres con referencia poco clara\n'
            '- quantifier: cantidades no especificadas\n\n'
            'Responde en este formato exacto:\n'
            'AMBIGUOUS: YES o NO\n'
            'TYPE: vague_term / pronoun / quantifier / none\n'
            'WORDS: lista de palabras ambiguas separadas por coma, o "none"'
        ),
        "few_shot": (
            'Determina si el requisito es ambiguo.\n\n'
            'Ejemplos:\n'
            '- "The system should be fast" -> AMBIGUOUS: YES, TYPE: vague_term, WORDS: fast\n'
            '- "The system shall respond within 200ms" -> AMBIGUOUS: NO, TYPE: none, WORDS: none\n'
            '- "It should handle the data properly" -> AMBIGUOUS: YES, TYPE: pronoun, WORDS: it, properly\n'
            '- "The system shall encrypt all passwords using AES-256" -> AMBIGUOUS: NO, TYPE: none, WORDS: none\n\n'
            'Requisito: "{requirement}"\n\n'
            'AMBIGUOUS:'
        ),
        "chain_of_thought": (
            'Determina si el requisito es ambiguo. Piensa paso a paso:\n'
            '1. Lee el requisito y busca terminos que puedan tener multiples interpretaciones\n'
            '2. Comprueba si hay pronombres sin referencia clara\n'
            '3. Comprueba si hay cantidades imprecisas\n'
            '4. Decide si un desarrollador podria interpretarlo de formas distintas\n\n'
            'Requisito: "{requirement}"\n\n'
            'Razonamiento:\n'
            '(Tras razonar, responde en este formato)\n'
            'AMBIGUOUS: YES o NO\n'
            'TYPE: vague_term / pronoun / quantifier / none\n'
            'WORDS: palabras ambiguas o "none"'
        ),
    },

    # ── A3: Completitud ───────────────────────────────────────
    "completeness": {
        "question_refinement": (
            'Analiza si el siguiente requisito de software esta completo.\n\n'
            'Si no estas seguro, piensa que preguntas haria un desarrollador al leer este requisito.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde en este formato exacto:\n'
            'COMPLETE: YES o NO\n'
            'MISSING: lista de elementos faltantes separados por coma (error_handling, boundary_conditions, acceptance_criteria, preconditions), o "none"'
        ),
        "cognitive_verifier": (
            'Analiza paso a paso si el siguiente requisito esta completo:\n\n'
            'Requisito: "{requirement}"\n\n'
            '1. Tiene manejo de errores definido?\n'
            '2. Especifica condiciones limite?\n'
            '3. Tiene criterios de aceptacion claros?\n'
            '4. Define precondiciones necesarias?\n\n'
            'Responde en este formato exacto:\n'
            'COMPLETE: YES o NO\n'
            'MISSING: lista de elementos faltantes separados por coma (error_handling, boundary_conditions, acceptance_criteria, preconditions), o "none"'
        ),
        "persona_context": (
            'Eres un analista de requisitos senior especializado en evaluar completitud segun IEEE 830.\n\n'
            'Un requisito completo debe especificar: manejo de errores, condiciones limite, criterios de aceptacion y precondiciones.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde en este formato exacto:\n'
            'COMPLETE: YES o NO\n'
            'MISSING: lista de elementos faltantes separados por coma (error_handling, boundary_conditions, acceptance_criteria, preconditions), o "none"'
        ),
        "few_shot": (
            'Determina si el requisito esta completo.\n\n'
            'Ejemplos:\n'
            '- "The system shall allow users to login with username and password. Invalid credentials shall display an error. Max 3 attempts before lockout." -> COMPLETE: YES, MISSING: none\n'
            '- "The system shall process payments" -> COMPLETE: NO, MISSING: error_handling, boundary_conditions, acceptance_criteria\n'
            '- "Users can upload files up to 10MB in PDF format. Unsupported formats show error message." -> COMPLETE: NO, MISSING: boundary_conditions, preconditions\n\n'
            'Requisito: "{requirement}"\n\n'
            'COMPLETE:'
        ),
        "chain_of_thought": (
            'Determina si el requisito esta completo. Piensa paso a paso:\n'
            '1. Describe manejo de errores? (que pasa si algo falla)\n'
            '2. Define condiciones limite? (limites, rangos, maximos)\n'
            '3. Tiene criterios de aceptacion? (como saber si esta correcto)\n'
            '4. Especifica precondiciones? (que debe cumplirse antes)\n\n'
            'Requisito: "{requirement}"\n\n'
            'Razonamiento:\n'
            '(Tras razonar, responde en este formato)\n'
            'COMPLETE: YES o NO\n'
            'MISSING: elementos faltantes o "none"'
        ),
    },

    # ── V1: Deteccion de inconsistencias (pares) ──────────────
    "inconsistency": {
        "question_refinement": (
            'Analiza si los siguientes dos requisitos son inconsistentes entre si.\n\n'
            'Si no estas seguro, reformula cada requisito para entender sus implicaciones.\n\n'
            'Requisito A: "{requirement_a}"\n'
            'Requisito B: "{requirement_b}"\n\n'
            'Responde en este formato exacto:\n'
            'INCONSISTENT: YES o NO\n'
            'DESCRIPTION: breve descripcion del conflicto, o "none"'
        ),
        "cognitive_verifier": (
            'Analiza paso a paso si estos dos requisitos son inconsistentes:\n\n'
            'Requisito A: "{requirement_a}"\n'
            'Requisito B: "{requirement_b}"\n\n'
            '1. Que dice exactamente cada requisito?\n'
            '2. Pueden coexistir ambos en el mismo sistema?\n'
            '3. Se contradicen directa o indirectamente?\n\n'
            'Responde en este formato exacto:\n'
            'INCONSISTENT: YES o NO\n'
            'DESCRIPTION: breve descripcion del conflicto, o "none"'
        ),
        "persona_context": (
            'Eres un ingeniero de requisitos senior experto en validacion de consistencia.\n\n'
            'Determina si los siguientes requisitos se contradicen:\n\n'
            'Requisito A: "{requirement_a}"\n'
            'Requisito B: "{requirement_b}"\n\n'
            'Responde en este formato exacto:\n'
            'INCONSISTENT: YES o NO\n'
            'DESCRIPTION: breve descripcion del conflicto, o "none"'
        ),
        "few_shot": (
            'Determina si los dos requisitos son inconsistentes.\n\n'
            'Ejemplos:\n'
            '- A: "The system shall store data locally" / B: "All data must be stored in the cloud" -> INCONSISTENT: YES, DESCRIPTION: Conflict between local and cloud storage\n'
            '- A: "Users can login with email" / B: "Users can also login with phone number" -> INCONSISTENT: NO, DESCRIPTION: none\n'
            '- A: "Response time under 1 second" / B: "System shall validate against 5 external APIs before responding" -> INCONSISTENT: YES, DESCRIPTION: Multiple API calls likely exceed 1s response time\n\n'
            'Requisito A: "{requirement_a}"\n'
            'Requisito B: "{requirement_b}"\n\n'
            'INCONSISTENT:'
        ),
        "chain_of_thought": (
            'Determina si los dos requisitos son inconsistentes. Piensa paso a paso:\n'
            '1. Que exige el requisito A?\n'
            '2. Que exige el requisito B?\n'
            '3. Pueden cumplirse ambos simultaneamente?\n'
            '4. Hay alguna contradiccion directa o implicita?\n\n'
            'Requisito A: "{requirement_a}"\n'
            'Requisito B: "{requirement_b}"\n\n'
            'Razonamiento:\n'
            '(Tras razonar, responde en este formato)\n'
            'INCONSISTENT: YES o NO\n'
            'DESCRIPTION: descripcion del conflicto o "none"'
        ),
    },

    # ── V2: Testabilidad ──────────────────────────────────────
    "testability": {
        "question_refinement": (
            'Analiza si el siguiente requisito es testable (verificable objetivamente).\n\n'
            'Si no estas seguro, piensa si un tester podria disena un caso de prueba concreto.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde en este formato exacto:\n'
            'TESTABLE: YES o NO\n'
            'REASON: measurable / vague / subjective'
        ),
        "cognitive_verifier": (
            'Analiza paso a paso si el siguiente requisito es testable:\n\n'
            'Requisito: "{requirement}"\n\n'
            '1. Contiene metricas o valores concretos?\n'
            '2. Se puede disenar un caso de prueba objetivo?\n'
            '3. Hay terminos subjetivos que impiden la verificacion?\n\n'
            'Responde en este formato exacto:\n'
            'TESTABLE: YES o NO\n'
            'REASON: measurable / vague / subjective'
        ),
        "persona_context": (
            'Eres un ingeniero de QA senior con experiencia en verificacion de requisitos segun IEEE 830.\n\n'
            'Un requisito testable debe poder verificarse mediante una prueba objetiva con resultado pass/fail.\n\n'
            'Requisito: "{requirement}"\n\n'
            'Responde en este formato exacto:\n'
            'TESTABLE: YES o NO\n'
            'REASON: measurable / vague / subjective'
        ),
        "few_shot": (
            'Determina si el requisito es testable.\n\n'
            'Ejemplos:\n'
            '- "The system shall respond within 2 seconds" -> TESTABLE: YES, REASON: measurable\n'
            '- "The system should be user-friendly" -> TESTABLE: NO, REASON: subjective\n'
            '- "The system shall support many users" -> TESTABLE: NO, REASON: vague\n'
            '- "The login page shall load in under 500ms on 4G networks" -> TESTABLE: YES, REASON: measurable\n\n'
            'Requisito: "{requirement}"\n\n'
            'TESTABLE:'
        ),
        "chain_of_thought": (
            'Determina si el requisito es testable. Piensa paso a paso:\n'
            '1. Identifica que se debe verificar\n'
            '2. Hay metricas concretas o valores medibles?\n'
            '3. Un tester podria disenar un caso de prueba con resultado pass/fail?\n'
            '4. Hay terminos subjetivos que impidan la verificacion?\n\n'
            'Requisito: "{requirement}"\n\n'
            'Razonamiento:\n'
            '(Tras razonar, responde en este formato)\n'
            'TESTABLE: YES o NO\n'
            'REASON: measurable / vague / subjective'
        ),
    },
}

# ── Prompt de reescritura de requisitos ────────────────────
REWRITE_PROMPT = (
    'Eres un ingeniero de requisitos senior. Tu tarea es reescribir un requisito de software '
    'para corregir los problemas de calidad detectados.\n\n'
    'Requisito original: "{requirement}"\n\n'
    'Problemas detectados:\n{problems}\n\n'
    'Instrucciones:\n'
    '- Reescribe el requisito corrigiendo TODOS los problemas listados\n'
    '- Mantén la intencion original del requisito\n'
    '- Hazlo especifico, medible, completo y testable\n'
    '- Si es ambiguo, reemplaza terminos vagos por valores concretos\n'
    '- Si es incompleto, anade los elementos faltantes\n'
    '- Si no es testable, anade criterios medibles\n'
    '- Responde SOLO con el requisito reescrito, sin explicaciones adicionales\n\n'
    'Requisito corregido:'
)


# All valid task names
TASK_NAMES = list(PROMPTS.keys())

# All valid strategy names
STRATEGY_NAMES = ["question_refinement", "cognitive_verifier", "persona_context",
                  "few_shot", "chain_of_thought"]


def build_prompt(task: str, strategy: str, **kwargs) -> str:
    """
    Construye el prompt segun la tarea y estrategia.

    Args:
        task: classification, ambiguity, completeness, inconsistency, testability
        strategy: question_refinement, cognitive_verifier, persona_context, few_shot, chain_of_thought
        **kwargs: Variables del template (requirement, requirement_a, requirement_b)

    Returns:
        Prompt formateado
    """
    if task not in PROMPTS:
        raise ValueError(f"Tarea desconocida: {task}. Opciones: {TASK_NAMES}")
    if strategy not in PROMPTS[task]:
        raise ValueError(f"Estrategia desconocida: {strategy}. Opciones: {STRATEGY_NAMES}")

    return PROMPTS[task][strategy].format(**kwargs)


# ============================================================
# PARSERS: one per task
# ============================================================

def parse_classification(response: str) -> str:
    """Extrae F o NF de la respuesta. Returns F, NF, or INVALID."""
    response = response.strip()
    upper = response.upper()

    # Exact match
    if upper in ("F", "NF"):
        return upper

    # Look for final answer patterns like "Clasificacion: F" or "**NF**"
    # Search from end of response (more likely to be the final answer)
    lines = upper.split('\n')
    for line in reversed(lines):
        line = line.strip()
        # Match patterns: "F", "NF", "Clasificacion: F", "**F**", "Answer: NF"
        m = re.search(r'(?:clasificacion|classification|answer|respuesta|final)\s*[:\-]?\s*\*{0,2}(NF|F)\b', line)
        if m:
            return m.group(1)
        # Standalone F or NF (possibly bold)
        m = re.match(r'^\*{0,2}(NF|F)\*{0,2}\.?$', line)
        if m:
            return m.group(1)

    # Fallback heuristics
    if "NO FUNCIONAL" in upper or "NON-FUNCTIONAL" in upper or "NON FUNCTIONAL" in upper:
        return "NF"
    # Check NF before F to avoid false positive on "NF" containing "F"
    if re.search(r'\bNF\b', upper):
        return "NF"
    if re.search(r'\bFUNCIONAL\b', upper) or re.search(r'\bFUNCTIONAL\b', upper):
        if "NO" not in upper and "NON" not in upper:
            return "F"
    if re.search(r'\bF\b', upper) and not re.search(r'\bNF\b', upper):
        return "F"

    return "INVALID"


def _detect_yes_no(response: str, keyword: str) -> bool | None:
    """Detect YES/NO in response, handling both 'KEYWORD: YES' and standalone 'YES' at start."""
    upper = response.upper().strip()
    # Try explicit pattern: KEYWORD: YES/NO
    m = re.search(rf'{keyword}\s*[:\-]\s*(YES|NO|SI|SÍ)\b', upper)
    if m:
        return m.group(1) in ("YES", "SI", "SÍ")
    # Fallback: response starts with YES/NO (when prompt ends with "KEYWORD:")
    m = re.match(r'^\s*(YES|NO|SI|SÍ)\b', upper)
    if m:
        return m.group(1) in ("YES", "SI", "SÍ")
    return None


def parse_ambiguity(response: str) -> dict:
    """Parse ambiguity detection response.

    Returns:
        dict with keys: is_ambiguous (bool), ambiguity_type (str), ambiguous_words (list)
    """
    upper = response.upper()
    result = {"is_ambiguous": False, "ambiguity_type": "none", "ambiguous_words": []}

    # Parse AMBIGUOUS field (or standalone YES/NO)
    detected = _detect_yes_no(response, "AMBIGUOUS")
    if detected is not None:
        result["is_ambiguous"] = detected

    # Parse TYPE field
    m = re.search(r'TYPE\s*[:\-]\s*(VAGUE_TERM|PRONOUN|QUANTIFIER|NONE)', upper)
    if m:
        result["ambiguity_type"] = m.group(1).lower()

    # Parse WORDS field
    m = re.search(r'WORDS\s*[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        words_str = m.group(1).strip()
        if words_str.lower() != "none":
            result["ambiguous_words"] = [w.strip() for w in words_str.split(',') if w.strip()]

    return result


def parse_completeness(response: str) -> dict:
    """Parse completeness analysis response.

    Returns:
        dict with keys: is_complete (bool), missing_elements (list)
    """
    result = {"is_complete": False, "missing_elements": []}

    # Parse COMPLETE field (or standalone YES/NO)
    detected = _detect_yes_no(response, "COMPLETE")
    if detected is not None:
        result["is_complete"] = detected

    # Parse MISSING field - try single-line format first
    m = re.search(r'MISSING\s*[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        missing_str = m.group(1).strip()
        if missing_str.lower() not in ("none", "n/a", ""):
            elements = [e.strip().strip('-').strip() for e in missing_str.split(',')]
            result["missing_elements"] = [e for e in elements if e and e.lower() != "none"]

    # Fallback: collect bullet points after MISSING
    if not result["missing_elements"]:
        m = re.search(r'MISSING\s*[:\-]\s*\n((?:\s*[-*]\s*.+\n?)+)', response, re.IGNORECASE)
        if m:
            bullets = re.findall(r'[-*]\s*(.+)', m.group(1))
            result["missing_elements"] = [b.strip().rstrip('.') for b in bullets if b.strip()]

    return result


def parse_inconsistency(response: str) -> dict:
    """Parse inconsistency detection response.

    Returns:
        dict with keys: is_inconsistent (bool), description (str)
    """
    result = {"is_inconsistent": False, "description": "none"}

    # Parse INCONSISTENT field (or standalone YES/NO)
    detected = _detect_yes_no(response, "INCONSISTENT")
    if detected is not None:
        result["is_inconsistent"] = detected

    m = re.search(r'DESCRIPTION\s*[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        desc = m.group(1).strip()
        if desc.lower() not in ("none", "n/a", ""):
            result["description"] = desc

    return result


def parse_testability(response: str) -> dict:
    """Parse testability analysis response.

    Returns:
        dict with keys: is_testable (bool), reason (str)
    """
    upper = response.upper()
    result = {"is_testable": False, "reason": "vague"}

    # Parse TESTABLE field (or standalone YES/NO)
    detected = _detect_yes_no(response, "TESTABLE")
    if detected is not None:
        result["is_testable"] = detected

    m = re.search(r'REASON\s*[:\-]\s*(MEASURABLE|VAGUE|SUBJECTIVE)', upper)
    if m:
        result["reason"] = m.group(1).lower()

    return result


# Map task names to their parsers
PARSERS = {
    "classification": parse_classification,
    "ambiguity": parse_ambiguity,
    "completeness": parse_completeness,
    "inconsistency": parse_inconsistency,
    "testability": parse_testability,
}


def parse_response(task: str, response: str):
    """
    Parse model response for a given task.

    Args:
        task: Task name (classification, ambiguity, etc.)
        response: Raw model response string

    Returns:
        Parsed result (type depends on task)
    """
    if task not in PARSERS:
        raise ValueError(f"No parser for task: {task}. Options: {list(PARSERS.keys())}")
    return PARSERS[task](response)
