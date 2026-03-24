"""
GitHub Action: Análisis de Requisitos en PRs
TFG: Optimización de la respuesta de los LLMs en el proceso de Ingeniería de Requisitos

Analiza requisitos escritos en el body de un PR usando NVIDIA NIM API.
Modelo: meta/llama-3.1-70b-instruct (mejor rendimiento experimental)
Estrategias: Few-Shot (ambigüedad, F1=1.000) + Chain-of-Thought (completitud, F1=0.844)
"""

import os
import sys
import re
import json
import time
import urllib.request

# ============== CONFIGURACIÓN ==============

MODEL = "meta/llama-3.1-70b-instruct"
NIM_BASE_URL = "https://integrate.api.nvidia.com/v1"
TEMPERATURE = 0.3
MAX_TOKENS = 512
MAX_REQUIREMENTS = 15
MAX_RETRIES = 3

# Keywords que indican un requisito de software (EN + ES)
REQUIREMENT_KEYWORDS = re.compile(
    r'\b(shall|must|should|will|may|can|'
    r'debe|deberá|debería|puede|tiene que|ha de|'
    r'permitir|garantizar|soportar|asegurar)\b',
    re.IGNORECASE
)

# ============== PROMPTS (de src/prompts.py — sin modificar) ==============

AMBIGUITY_PROMPT = (
    'Determina si el requisito es ambiguo.\n\n'
    'Ejemplos:\n'
    '- "The system should be fast" -> AMBIGUOUS: YES, TYPE: vague_term, WORDS: fast\n'
    '- "The system shall respond within 200ms" -> AMBIGUOUS: NO, TYPE: none, WORDS: none\n'
    '- "It should handle the data properly" -> AMBIGUOUS: YES, TYPE: pronoun, WORDS: it, properly\n'
    '- "The system shall encrypt all passwords using AES-256" -> AMBIGUOUS: NO, TYPE: none, WORDS: none\n\n'
    'Requisito: "{requirement}"\n\n'
    'AMBIGUOUS:'
)

COMPLETENESS_PROMPT = (
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
)

# ============== NIM API ==============


def call_nim(prompt: str, api_key: str) -> str:
    """Llama a NVIDIA NIM usando la interfaz OpenAI-compatible."""
    from openai import OpenAI

    client = OpenAI(base_url=NIM_BASE_URL, api_key=api_key)

    for attempt in range(MAX_RETRIES):
        try:
            response = client.chat.completions.create(
                model=MODEL,
                messages=[{"role": "user", "content": prompt}],
                temperature=TEMPERATURE,
                max_tokens=MAX_TOKENS,
            )
            return response.choices[0].message.content
        except Exception as e:
            if attempt < MAX_RETRIES - 1:
                wait = 2 ** attempt
                print(f"  Retry {attempt + 1}/{MAX_RETRIES} en {wait}s: {e}")
                time.sleep(wait)
            else:
                raise


# ============== PARSERS (de src/prompts.py — lógica idéntica) ==============


def _detect_yes_no(response: str, keyword: str):
    """Detecta YES/NO en respuesta, con patrón KEYWORD: YES/NO o standalone."""
    upper = response.upper().strip()
    m = re.search(rf'{keyword}\s*[:\-]\s*(YES|NO|SI|SÍ)\b', upper)
    if m:
        return m.group(1) in ("YES", "SI", "SÍ")
    m = re.match(r'^\s*(YES|NO|SI|SÍ)\b', upper)
    if m:
        return m.group(1) in ("YES", "SI", "SÍ")
    return None


def parse_ambiguity(response: str) -> dict:
    """Parsea respuesta de detección de ambigüedad."""
    result = {"is_ambiguous": False, "ambiguous_words": []}

    detected = _detect_yes_no(response, "AMBIGUOUS")
    if detected is not None:
        result["is_ambiguous"] = detected

    m = re.search(r'WORDS\s*[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        words_str = m.group(1).strip()
        if words_str.lower() != "none":
            result["ambiguous_words"] = [w.strip() for w in words_str.split(',') if w.strip()]

    return result


def parse_completeness(response: str) -> dict:
    """Parsea respuesta de evaluación de completitud."""
    result = {"is_complete": False, "missing_elements": []}

    detected = _detect_yes_no(response, "COMPLETE")
    if detected is not None:
        result["is_complete"] = detected

    m = re.search(r'MISSING\s*[:\-]\s*(.+?)(?:\n|$)', response, re.IGNORECASE)
    if m:
        missing_str = m.group(1).strip()
        if missing_str.lower() not in ("none", "n/a", ""):
            elements = [e.strip().strip('-').strip() for e in missing_str.split(',')]
            result["missing_elements"] = [e for e in elements if e and e.lower() != "none"]

    if not result["missing_elements"]:
        m = re.search(r'MISSING\s*[:\-]\s*\n((?:\s*[-*]\s*.+\n?)+)', response, re.IGNORECASE)
        if m:
            bullets = re.findall(r'[-*]\s*(.+)', m.group(1))
            result["missing_elements"] = [b.strip().rstrip('.') for b in bullets if b.strip()]

    return result


# ============== EXTRACCIÓN DE REQUISITOS ==============


def extract_requirements(pr_body: str) -> list:
    """Extrae frases que parecen requisitos del body del PR."""
    if not pr_body or not pr_body.strip():
        return []

    requirements = []
    seen = set()

    for line in pr_body.splitlines():
        line = line.strip()

        if not line or len(line) < 15:
            continue
        if line.startswith(('#', '>', '---', '***', '===')):
            continue
        if re.match(r'^-\s*\[[ xX]\]', line):
            continue

        # Limpiar marcadores de lista markdown
        clean = re.sub(r'^[-*+]\s+', '', line)
        clean = re.sub(r'^\d+\.\s+', '', clean)

        if REQUIREMENT_KEYWORDS.search(clean):
            normalized = clean.lower().strip()
            if normalized not in seen:
                seen.add(normalized)
                requirements.append(clean)

    return requirements[:MAX_REQUIREMENTS]


# ============== FORMATO DEL COMENTARIO ==============


def build_comment(results: list) -> str:
    """Construye el comentario markdown para el PR."""
    lines = [
        "## RE Analysis — Requirements Quality Check\n",
        f"> Modelo: `{MODEL}` (NVIDIA NIM)",
        "> Estrategias: Few-Shot (ambigüedad) · Chain-of-Thought (completitud)",
        f"> Basado en resultados experimentales del TFG\n",
        f"Se encontraron **{len(results)} requisito(s)** en este PR.\n",
        "---\n",
    ]

    issues_found = 0

    for i, r in enumerate(results, 1):
        lines.append(f"### Requisito {i}")
        lines.append(f"> {r['text']}\n")

        if r.get("error"):
            lines.append(f"Error en análisis: {r['error']}\n")
            continue

        amb = r["ambiguity"]
        comp = r["completeness"]

        # Ambigüedad
        if amb["is_ambiguous"]:
            words = ", ".join(f"`{w}`" for w in amb["ambiguous_words"]) if amb["ambiguous_words"] else "—"
            amb_str = f"⚠️ Ambiguo | Palabras: {words}"
            issues_found += 1
        else:
            amb_str = "✅ No ambiguo"

        # Completitud
        if comp["is_complete"]:
            comp_str = "✅ Completo"
        else:
            missing = ", ".join(f"`{m}`" for m in comp["missing_elements"]) if comp["missing_elements"] else "—"
            comp_str = f"❌ Incompleto | Falta: {missing}"
            issues_found += 1

        lines.append("| Análisis | Resultado |")
        lines.append("|----------|-----------|")
        lines.append(f"| Ambigüedad | {amb_str} |")
        lines.append(f"| Completitud | {comp_str} |")
        lines.append("")

    # Resumen
    lines.append("---\n")
    if issues_found == 0:
        lines.append("✅ **Todos los requisitos superaron las comprobaciones.**\n")
    else:
        lines.append(f"⚠️ **{issues_found} problema(s) detectado(s).** Revisa las sugerencias arriba.\n")

    lines.append("> ℹ️ Este análisis es **advisory** — no bloquea el merge.")
    lines.append("> Investigación: TFG — Optimización de LLMs en Ingeniería de Requisitos")

    return "\n".join(lines)


# ============== GITHUB API ==============


def post_comment(repo: str, pr_number: str, body: str, token: str) -> None:
    """Posta un comentario en el PR via GitHub REST API."""
    url = f"https://api.github.com/repos/{repo}/issues/{pr_number}/comments"
    data = json.dumps({"body": body}).encode("utf-8")
    req = urllib.request.Request(
        url,
        data=data,
        headers={
            "Authorization": f"Bearer {token}",
            "Accept": "application/vnd.github+json",
            "X-GitHub-Api-Version": "2022-11-28",
            "Content-Type": "application/json",
        },
        method="POST",
    )
    with urllib.request.urlopen(req) as resp:
        if resp.status not in (200, 201):
            raise RuntimeError(f"GitHub API error: {resp.status}")


# ============== MAIN ==============


def main():
    api_key = os.environ.get("NVIDIA_API_KEY", "")
    pr_body = os.environ.get("PR_BODY", "")
    pr_number = os.environ.get("PR_NUMBER", "")
    token = os.environ.get("GITHUB_TOKEN", "")
    repo = os.environ.get("REPO", "")

    if not api_key:
        print("NVIDIA_API_KEY no configurada. Saltando análisis.")
        sys.exit(0)

    if not pr_body or not pr_body.strip():
        print("PR body vacío. Saltando análisis.")
        sys.exit(0)

    requirements = extract_requirements(pr_body)

    if not requirements:
        print("No se encontraron requisitos en el PR body. Saltando comentario.")
        sys.exit(0)

    print(f"Encontrados {len(requirements)} requisito(s). Analizando...")

    results = []
    for i, req in enumerate(requirements, 1):
        print(f"  [{i}/{len(requirements)}] {req[:60]}...")
        result = {"text": req, "error": None}
        try:
            amb_resp = call_nim(AMBIGUITY_PROMPT.format(requirement=req), api_key)
            comp_resp = call_nim(COMPLETENESS_PROMPT.format(requirement=req), api_key)
            result["ambiguity"] = parse_ambiguity(amb_resp)
            result["completeness"] = parse_completeness(comp_resp)
        except Exception as e:
            result["error"] = str(e)
            print(f"  Error API: {e}")
        results.append(result)

    comment = build_comment(results)

    if not token or not pr_number or not repo:
        print("\nNo se puede postear (faltan GITHUB_TOKEN/PR_NUMBER/REPO).")
        print("Contenido del comentario:\n")
        print(comment)
        sys.exit(0)

    try:
        post_comment(repo, pr_number, comment, token)
        print("Comentario posteado correctamente.")
    except Exception as e:
        print(f"Error al postear comentario: {e}")
        print("Contenido del comentario:\n")
        print(comment)
        sys.exit(0)


if __name__ == "__main__":
    main()
