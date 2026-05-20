#!/usr/bin/env python3
"""
Test del contexto del dominio en el pipeline DAG.

Ejecuta el mismo conjunto de requisitos con y sin contexto
y compara si el modelo cambia su juicio de ambiguedad.

Uso:
    cd /home/suero/Escritorio/TFG/proyecto/src
    python test_context_prompt.py
"""

import sys
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv(Path(__file__).parent.parent / ".env")
sys.path.insert(0, str(Path(__file__).parent))
os.chdir(Path(__file__).parent)

from dag import run_dag_pipeline_from_requirements

MODEL = "nim_llama70b"
STRATEGY = "few_shot"

# Requisitos de prueba: mezcla de vagos y precisos
# Los que contienen "fast" / "good" deben cambiar con el contexto
TEST_REQS = [
    # Deberian dejar de ser ambiguos CON contexto
    "The system should be fast.",               # solo "fast" — caso limpio
    "The system should provide good service.",  # solo "good" — caso limpio
    "The application should be fast enough for real-time use.",  # "fast" + otros vagos
    "The system should provide good response times.",
    # Siempre precisos (control: no deben cambiar)
    "The system shall respond within 200 milliseconds.",
    "The system shall encrypt all passwords using AES-256 before storage.",
    # Siempre ambiguos (control: no deben cambiar)
    "The system should be user-friendly.",
    "The component should interact seamlessly with other parts.",
]

CONTEXT_EN = (
    "DOMAIN GLOSSARY (overrides general ambiguity rules): "
    "'fast' = response time under 200 ms (Core Banking SLA, ISO 20022). "
    "'good' = compliant with agreed SLA thresholds defined in Annex A. "
    "These terms are PRECISELY DEFINED. Do NOT flag 'fast' or 'good' as "
    "ambiguous when they appear in requirements."
)


def run_test(label: str, context: str) -> list[dict]:
    print(f"\n{'='*60}")
    print(f"  TEST: {label}")
    if context:
        print(f"  Contexto: {context[:80]}...")
    else:
        print(f"  Contexto: (ninguno)")
    print(f"{'='*60}")

    ctx = run_dag_pipeline_from_requirements(
        requirements=TEST_REQS,
        model_key=MODEL,
        strategy=STRATEGY,
        skip_inconsistency=True,
        doc_name="test_context",
        context_prompt=context,
    )

    rows = ctx.get("analysis_rows", [])
    results = []
    for row in rows:
        results.append({
            "req": row["text"][:60],
            "ambiguous": row.get("is_ambiguous"),
            "words": row.get("ambiguous_words", ""),
            "testable": row.get("is_testable"),
            "score": row.get("quality_score", 0),
        })
        flag = "AMBIG" if row.get("is_ambiguous") else "OK   "
        words = f"  [{row.get('ambiguous_words','')}]" if row.get("is_ambiguous") else ""
        print(f"  [{flag}] {row['text'][:65]}{words}")
    return results


def compare(without: list[dict], with_ctx: list[dict]):
    print(f"\n{'='*60}")
    print("  DIFERENCIAS (sin contexto → con contexto)")
    print(f"{'='*60}")
    changed = 0
    for a, b in zip(without, with_ctx):
        if a["ambiguous"] != b["ambiguous"]:
            direction = "ambiguo → NO ambiguo" if a["ambiguous"] else "NO ambiguo → ambiguo"
            print(f"  CAMBIO: {a['req'][:60]}")
            print(f"          {direction}")
            changed += 1
    if changed == 0:
        print("  Sin cambios — el contexto no afecto al resultado.")
    else:
        print(f"\n  {changed}/{len(without)} requisito(s) cambiaron.")


if __name__ == "__main__":
    print(f"\nModelo: {MODEL}  |  Estrategia: {STRATEGY}")
    print(f"Requisitos de prueba: {len(TEST_REQS)}")

    res_sin = run_test("Sin contexto", context="")
    res_con = run_test("Con contexto (ingles)", context=CONTEXT_EN)
    compare(res_sin, res_con)
