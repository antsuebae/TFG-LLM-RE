"""
Modulo de deteccion de advertencias de calidad para requisitos analizados por LLMs.

Detecta patrones problematicos comunes:
- Alucinaciones en reescrituras
- Cambios vacios en correcciones
- Clasificaciones F/NF sospechosas
- Requisitos fuente demasiado cortos (codigos)
- Mezcla de idiomas
- Sobreingenieria en correcciones
"""

import re
import pandas as pd

# ── Stopwords simples para calculo de similitud ──────────────
_STOPWORDS_ES = {
    'el', 'la', 'los', 'las', 'un', 'una', 'unos', 'unas', 'de', 'del',
    'en', 'y', 'o', 'a', 'al', 'por', 'para', 'con', 'sin', 'que', 'se',
    'es', 'no', 'lo', 'su', 'como', 'mas', 'pero', 'si', 'ser', 'este',
    'esta', 'estos', 'estas', 'todo', 'toda', 'todos', 'todas',
}

_STOPWORDS_EN = {
    'the', 'a', 'an', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
    'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
    'should', 'may', 'might', 'can', 'shall', 'to', 'of', 'in', 'for',
    'on', 'with', 'at', 'by', 'from', 'as', 'into', 'through', 'during',
    'before', 'after', 'and', 'but', 'or', 'not', 'no', 'if', 'then',
    'than', 'that', 'this', 'it', 'its',
}

_STOPWORDS = _STOPWORDS_ES | _STOPWORDS_EN

# ── Palabras clave para deteccion ────────────────────────────
_NF_KEYWORDS = {
    'rendimiento', 'disponibilidad', 'latencia', 'escalabilidad', 'seguridad',
    'fiabilidad', 'usabilidad', 'mantenibilidad', 'portabilidad', 'eficiencia',
    'performance', 'availability', 'latency', 'scalability', 'security',
    'reliability', 'usability', 'maintainability', 'portability', 'efficiency',
    'vram', 'memoria', 'memory', 'cpu', 'gpu', 'throughput', 'uptime',
    'response time', 'tiempo de respuesta', 'backup', 'recovery', 'cifrado',
    'encryption', 'concurrent', 'concurrente', 'bandwidth', 'ancho de banda',
}

_F_VERBS = {
    'permitir', 'generar', 'calcular', 'mostrar', 'registrar', 'enviar',
    'recibir', 'crear', 'eliminar', 'modificar', 'buscar', 'exportar',
    'importar', 'notificar', 'validar', 'autenticar', 'autorizar',
    'allow', 'generate', 'calculate', 'display', 'show', 'register',
    'send', 'receive', 'create', 'delete', 'modify', 'search', 'export',
    'import', 'notify', 'validate', 'authenticate', 'authorize', 'login',
    'logout', 'upload', 'download',
}

_ENGLISH_MARKERS = {'shall', 'must', 'should', 'the system', 'the application'}
_SPANISH_MARKERS = {'debe', 'el sistema', 'la aplicacion', 'debera', 'permitira'}

_TECH_KEYWORDS = {
    'kubernetes', 'docker', 'microservic', 'kafka', 'rabbitmq', 'redis',
    'mongodb', 'postgresql', 'elasticsearch', 'nginx', 'graphql', 'grpc',
    'blockchain', 'machine learning', 'deep learning', 'neural network',
    'api gateway', 'load balancer', 'cdn', 'ci/cd', 'terraform', 'ansible',
    'react', 'angular', 'vue', 'spring boot', 'django', 'flask',
}


def _tokenize(text: str) -> set[str]:
    """Extrae palabras significativas (sin stopwords) de un texto."""
    words = set(re.findall(r'[a-zA-ZáéíóúñüÁÉÍÓÚÑÜ]{3,}', text.lower()))
    return words - _STOPWORDS


def _jaccard_similarity(text_a: str, text_b: str) -> float:
    """Calcula similitud Jaccard entre dos textos (sin stopwords)."""
    tokens_a = _tokenize(text_a)
    tokens_b = _tokenize(text_b)
    if not tokens_a or not tokens_b:
        return 0.0
    intersection = tokens_a & tokens_b
    union = tokens_a | tokens_b
    return len(intersection) / len(union) if union else 0.0


# ── Detectores ───────────────────────────────────────────────

def detect_hallucinations(corrections: list[dict]) -> list[dict]:
    """Detecta correcciones donde el texto reescrito diverge mucho del original.

    Si el texto corregido comparte < 30% de palabras significativas con el
    original, se marca como posible alucinacion.
    """
    warnings = []
    affected = []
    threshold = 0.30

    for i, c in enumerate(corrections):
        if not c.get('changes_made'):
            continue
        original = c.get('original', '')
        corrected = c.get('corrected', '')
        if not original or not corrected:
            continue

        sim = _jaccard_similarity(original, corrected)
        if sim < threshold:
            affected.append(i + 1)

    if affected:
        warnings.append({
            'level': 'warning',
            'category': 'alucinacion',
            'message': (
                f"{len(affected)} correcciones tienen baja similitud con el "
                f"original (posibles alucinaciones del modelo)"
            ),
            'details': [f"REQ {', '.join(str(a) for a in affected[:10])}"
                        + (f"... (+{len(affected)-10} mas)" if len(affected) > 10 else "")],
            'affected_reqs': affected,
        })
    return warnings


def detect_empty_changes(corrections: list[dict]) -> list[dict]:
    """Detecta correcciones con cambios vacios o parentesis vacios."""
    warnings = []
    affected = []
    empty_pattern = re.compile(r'\(\s*\)|\[\s*\]')

    for i, c in enumerate(corrections):
        if not c.get('changes_made'):
            continue
        for change in c['changes_made']:
            if empty_pattern.search(str(change)):
                affected.append(i + 1)
                break
        # Correccion marcada como modificada pero sin cambios reales
        if c.get('changes_made') and c.get('original') == c.get('corrected'):
            if (i + 1) not in affected:
                affected.append(i + 1)

    if affected:
        warnings.append({
            'level': 'warning',
            'category': 'cambios_vacios',
            'message': f"{len(affected)} correcciones tienen cambios vacios \"()\"",
            'details': [f"REQ {', '.join(str(a) for a in affected[:10])}"
                        + (f"... (+{len(affected)-10} mas)" if len(affected) > 10 else "")],
            'affected_reqs': affected,
        })
    return warnings


def detect_classification_issues(results_df: pd.DataFrame) -> list[dict]:
    """Detecta clasificaciones F/NF sospechosas basandose en palabras clave."""
    warnings = []
    if 'classification' not in results_df.columns or 'text' not in results_df.columns:
        return warnings

    nf_as_f = []
    f_as_nf = []

    for i, row in results_df.iterrows():
        text_lower = row['text'].lower()
        classification = str(row.get('classification', '')).upper()

        # NF clasificado como F
        if classification == 'F':
            nf_matches = [kw for kw in _NF_KEYWORDS if kw in text_lower]
            if len(nf_matches) >= 2:
                nf_as_f.append(i + 1)

        # F clasificado como NF
        elif classification == 'NF':
            f_matches = [v for v in _F_VERBS if v in text_lower]
            has_nf_kw = any(kw in text_lower for kw in _NF_KEYWORDS)
            if len(f_matches) >= 2 and not has_nf_kw:
                f_as_nf.append(i + 1)

    if nf_as_f:
        warnings.append({
            'level': 'warning',
            'category': 'clasificacion_nf_como_f',
            'message': (
                f"{len(nf_as_f)} requisitos parecen No Funcionales pero estan "
                f"clasificados como Funcionales"
            ),
            'details': [f"REQ {', '.join(str(a) for a in nf_as_f[:10])}"
                        + (f"... (+{len(nf_as_f)-10} mas)" if len(nf_as_f) > 10 else "")],
            'affected_reqs': nf_as_f,
        })

    if f_as_nf:
        warnings.append({
            'level': 'info',
            'category': 'clasificacion_f_como_nf',
            'message': (
                f"{len(f_as_nf)} requisitos parecen Funcionales pero estan "
                f"clasificados como No Funcionales"
            ),
            'details': [f"REQ {', '.join(str(a) for a in f_as_nf[:10])}"
                        + (f"... (+{len(f_as_nf)-10} mas)" if len(f_as_nf) > 10 else "")],
            'affected_reqs': f_as_nf,
        })

    return warnings


def detect_short_requirements(results_df: pd.DataFrame) -> list[dict]:
    """Detecta requisitos demasiado cortos o que parecen codigos (CU-01, RF-02)."""
    warnings = []
    if 'text' not in results_df.columns:
        return warnings

    code_pattern = re.compile(r'^[A-Z]{1,4}-?\d{1,3}$')
    affected = []

    for i, row in results_df.iterrows():
        text = str(row['text']).strip()
        if len(text) < 15 or code_pattern.match(text):
            affected.append(i + 1)

    if affected:
        warnings.append({
            'level': 'warning',
            'category': 'requisitos_cortos',
            'message': (
                f"{len(affected)} requisitos fuente son codigos cortos o textos "
                f"muy breves que pueden producir alucinaciones al reescribirse"
            ),
            'details': [f"REQ {', '.join(str(a) for a in affected[:10])}"
                        + (f"... (+{len(affected)-10} mas)" if len(affected) > 10 else "")],
            'affected_reqs': affected,
        })
    return warnings


def detect_language_mixing(results_df: pd.DataFrame) -> list[dict]:
    """Detecta requisitos que mezclan ingles y espanol."""
    warnings = []
    if 'text' not in results_df.columns:
        return warnings

    affected = []

    for i, row in results_df.iterrows():
        text_lower = str(row['text']).lower()
        has_english = any(marker in text_lower for marker in _ENGLISH_MARKERS)
        has_spanish = any(marker in text_lower for marker in _SPANISH_MARKERS)
        if has_english and has_spanish:
            affected.append(i + 1)

    if affected:
        warnings.append({
            'level': 'info',
            'category': 'mezcla_idiomas',
            'message': (
                f"{len(affected)} requisitos mezclan ingles y espanol"
            ),
            'details': [f"REQ {', '.join(str(a) for a in affected[:10])}"
                        + (f"... (+{len(affected)-10} mas)" if len(affected) > 10 else "")],
            'affected_reqs': affected,
        })
    return warnings


def detect_overengineering(corrections: list[dict]) -> list[dict]:
    """Detecta correcciones con sobreingenieria (texto >5x mas largo o tecnologias nuevas)."""
    warnings = []
    affected_length = []
    affected_tech = []

    for i, c in enumerate(corrections):
        if not c.get('changes_made'):
            continue
        original = c.get('original', '')
        corrected = c.get('corrected', '')
        if not original:
            continue

        # Ratio de longitud
        ratio = len(corrected) / max(len(original), 1)
        if ratio > 5.0:
            affected_length.append(i + 1)

        # Tecnologias no mencionadas en el original
        original_lower = original.lower()
        corrected_lower = corrected.lower()
        for tech in _TECH_KEYWORDS:
            if tech in corrected_lower and tech not in original_lower:
                if (i + 1) not in affected_tech:
                    affected_tech.append(i + 1)
                break

    if affected_length:
        warnings.append({
            'level': 'info',
            'category': 'sobreingenieria_longitud',
            'message': (
                f"{len(affected_length)} correcciones son >5x mas largas que el "
                f"original (posible sobreingenieria)"
            ),
            'details': [f"REQ {', '.join(str(a) for a in affected_length[:10])}"
                        + (f"... (+{len(affected_length)-10} mas)" if len(affected_length) > 10 else "")],
            'affected_reqs': affected_length,
        })

    if affected_tech:
        warnings.append({
            'level': 'warning',
            'category': 'sobreingenieria_tecnologias',
            'message': (
                f"{len(affected_tech)} correcciones introducen tecnologias/frameworks "
                f"no mencionados en el requisito original"
            ),
            'details': [f"REQ {', '.join(str(a) for a in affected_tech[:10])}"
                        + (f"... (+{len(affected_tech)-10} mas)" if len(affected_tech) > 10 else "")],
            'affected_reqs': affected_tech,
        })

    return warnings


def generate_all_warnings(
    results_df: pd.DataFrame,
    corrections: list[dict] | None = None,
) -> list[dict]:
    """Ejecuta todos los detectores aplicables y devuelve lista unificada de advertencias."""
    all_warnings = []

    # Detectores sobre resultados del pipeline
    if results_df is not None and not results_df.empty:
        all_warnings.extend(detect_classification_issues(results_df))
        all_warnings.extend(detect_short_requirements(results_df))
        all_warnings.extend(detect_language_mixing(results_df))

    # Detectores sobre correcciones
    if corrections:
        all_warnings.extend(detect_hallucinations(corrections))
        all_warnings.extend(detect_empty_changes(corrections))
        all_warnings.extend(detect_overengineering(corrections))

    return all_warnings


def check_input_warnings(text: str) -> list[dict]:
    """Verifica un texto de entrada individual y devuelve advertencias.

    Para usar en paginas individuales (Clasificar, Calidad, Consistencia).
    """
    warnings = []
    text = text.strip()

    # Texto muy corto o codigo
    code_pattern = re.compile(r'^[A-Z]{1,4}-?\d{1,3}$')
    if len(text) < 15 or code_pattern.match(text):
        warnings.append({
            'level': 'warning',
            'category': 'texto_corto',
            'message': (
                "El texto es muy corto o parece un codigo de referencia. "
                "Los modelos tienden a alucinar con entradas breves."
            ),
        })

    # Mezcla de idiomas
    text_lower = text.lower()
    has_english = any(marker in text_lower for marker in _ENGLISH_MARKERS)
    has_spanish = any(marker in text_lower for marker in _SPANISH_MARKERS)
    if has_english and has_spanish:
        warnings.append({
            'level': 'info',
            'category': 'mezcla_idiomas',
            'message': "El requisito parece mezclar ingles y espanol.",
        })

    return warnings
