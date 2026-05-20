#!/usr/bin/env python3
"""
Test riguroso del nodo domain_filter.

Pruebas:
  1. Parser unitario (sin LLM) — cubre todos los casos borde
  2. Integración con NIM (requiere NVIDIA_API_KEY)
     2a. Sin contexto → analisis puramente lingüístico
     2b. Con contexto → el nodo domain_filter corrige flags
     2c. Contexto irrelevante → los flags no cambian
     2d. Requisito sin problemas → el nodo lo omite (0 llamadas LLM adicionales)

Uso:
  python test_domain_filter.py               # solo tests de parser
  python test_domain_filter.py --integration # todos los tests
"""
import logging
import sys
import os
import time

sys.path.insert(0, os.path.dirname(__file__))
logging.basicConfig(level=logging.INFO, format='%(levelname)s %(name)s: %(message)s')
logger = logging.getLogger('test_domain_filter')

from prompts import parse_domain_filter_response, DOMAIN_FILTER_PROMPT


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

PASS = '\033[92m✓\033[0m'
FAIL = '\033[91m✗\033[0m'
_failures = []

def check(name: str, condition: bool, detail: str = ''):
    if condition:
        print(f"  {PASS} {name}")
    else:
        print(f"  {FAIL} {name}" + (f": {detail}" if detail else ''))
        _failures.append(name)


# ─────────────────────────────────────────────────────────────────────────────
# Test 1: Parser unit tests
# ─────────────────────────────────────────────────────────────────────────────

def test_parser():
    print("\n=== Test 1: Parser unitario ===")

    orig_all_bad = {'is_ambiguous': True, 'is_complete': False, 'is_testable': False}
    orig_all_ok  = {'is_ambiguous': False, 'is_complete': True,  'is_testable': True}

    # 1a. LLM elimina ambigüedad
    r = parse_domain_filter_response(
        "AMBIGUOUS: NO\nCOMPLETE: NO\nTESTABLE: NO\nREASON: El dominio aclara el término",
        orig_all_bad,
    )
    check("1a. Ambigüedad eliminada",         r['is_ambiguous'] == False)
    check("1a. Otros flags sin cambio",        r['is_complete'] == False and r['is_testable'] == False)
    check("1a. domain_filter_applied = True",  r['domain_filter_applied'])
    check("1a. Reason capturada",              r['domain_filter_reason'] == 'El dominio aclara el término')

    # 1b. LLM corrige los tres flags a la vez
    r = parse_domain_filter_response(
        "AMBIGUOUS: NO\nCOMPLETE: YES\nTESTABLE: YES\nREASON: Todo aclarado",
        orig_all_bad,
    )
    check("1b. Los 3 flags corregidos",   r['is_ambiguous'] == False and r['is_complete'] == True and r['is_testable'] == True)
    check("1b. domain_filter_applied",    r['domain_filter_applied'])

    # 1c. Sin cambios (LLM confirma el análisis previo)
    r = parse_domain_filter_response(
        "AMBIGUOUS: YES\nCOMPLETE: NO\nTESTABLE: NO\nREASON: NONE",
        orig_all_bad,
    )
    check("1c. Sin cambios: applied = False",  not r['domain_filter_applied'])
    check("1c. Flags sin cambio",              r['is_ambiguous'] == True and r['is_complete'] == False)

    # 1d. Respuesta malformada → se mantienen los originales
    r = parse_domain_filter_response("No puedo determinar nada.", orig_all_bad)
    check("1d. Malformada: applied = False",   not r['domain_filter_applied'])
    check("1d. Malformada: flags preservados", r['is_ambiguous'] == True)

    # 1e. Valores en minúsculas / variaciones
    r = parse_domain_filter_response(
        "ambiguous: no\ncomplete: yes\ntestable: yes\nreason: ok",
        orig_all_bad,
    )
    check("1e. Case-insensitive keys",  r['is_ambiguous'] == False and r['is_complete'] == True)

    # 1f. El nodo NO cambia requisitos sin problemas (aplicado externamente en el nodo)
    r = parse_domain_filter_response(
        "AMBIGUOUS: NO\nCOMPLETE: YES\nTESTABLE: YES\nREASON: NONE",
        orig_all_ok,
    )
    check("1f. Req sin problemas: applied = False", not r['domain_filter_applied'])

    # 1g. YES con texto adicional (p. ej. "YES — se confirma")
    r = parse_domain_filter_response(
        "AMBIGUOUS: YES — el término sigue siendo vago\nCOMPLETE: YES\nTESTABLE: YES\nREASON: NONE",
        orig_all_ok,
    )
    check("1g. YES con texto extra",  r['is_ambiguous'] == True)

    print()


# ─────────────────────────────────────────────────────────────────────────────
# Test 2: Integración con NIM
# ─────────────────────────────────────────────────────────────────────────────

def test_integration():
    print("=== Test 2: Integración con NIM ===\n")
    from dag import run_dag_pipeline_from_requirements

    # Requisitos diseñados para ser ambiguos/incompletos sin contexto
    # pero claros con el contexto de dominio proporcionado.
    reqs_ambiguous = [
        "El sistema debe procesar pagos rápidamente",
        "El sistema debe autenticar usuarios de forma segura",
        "El sistema debe mantener el historial de transacciones completo",
        "El sistema debe responder en tiempo real a las solicitudes",
    ]
    # Requisito sin problemas (no debería ser tocado por el filtro)
    reqs_clean = [
        "El sistema debe permitir al usuario introducir su nombre de usuario y contraseña "
        "y verificarlos contra la base de datos en un tiempo máximo de 500 ms, rechazando "
        "el acceso si las credenciales son incorrectas y registrando el intento fallido.",
    ]

    context = (
        "Sistema de pagos online para banco europeo bajo normativa PCI-DSS. "
        "'Rápidamente' significa cumplir el SLA de latencia: máximo 2 segundos por transacción. "
        "'Seguro' implica cifrado TLS 1.3 y autenticación OAuth 2.0 con MFA obligatorio. "
        "El historial completo equivale a los últimos 12 meses por normativa PCI-DSS. "
        "'Tiempo real' en este contexto significa latencia menor a 500 ms."
    )

    common_kwargs = dict(
        model_key='qwen7b',
        strategy='chain_of_thought',
        skip_inconsistency=True,
    )

    # ── 2a. Sin contexto (baseline) ──────────────────────────────────────────
    print("--- 2a. Sin contexto (baseline) ---")
    t0 = time.time()
    r_no_ctx = run_dag_pipeline_from_requirements(
        reqs_ambiguous, context_prompt='', **common_kwargs
    )
    t_no_ctx = time.time() - t0
    df_no = r_no_ctx['results_df']

    n_issues_no_ctx = sum(
        1 for _, row in df_no.iterrows()
        if row.get('is_ambiguous') is True or row.get('is_complete') is False or row.get('is_testable') is False
    )
    avg_score_no_ctx = df_no['quality_score'].mean()
    print(f"  Requisitos con problemas: {n_issues_no_ctx}/{len(reqs_ambiguous)}")
    print(f"  Quality score medio: {avg_score_no_ctx:.1f}")
    print(f"  Tiempo: {t_no_ctx:.1f}s")
    for _, row in df_no.iterrows():
        print(f"  [{row['text'][:55]}]")
        print(f"    amb={row['is_ambiguous']} compl={row['is_complete']} test={row['is_testable']} score={row['quality_score']:.0f}")

    check("2a. Al menos 2 reqs con problemas sin contexto", n_issues_no_ctx >= 2,
          f"got {n_issues_no_ctx}")

    # ── 2b. Con contexto relevante ────────────────────────────────────────────
    print("\n--- 2b. Con contexto relevante ---")
    t0 = time.time()
    r_ctx = run_dag_pipeline_from_requirements(
        reqs_ambiguous, context_prompt=context, **common_kwargs
    )
    t_ctx = time.time() - t0
    df_ctx = r_ctx['results_df']

    n_issues_ctx = sum(
        1 for _, row in df_ctx.iterrows()
        if row.get('is_ambiguous') is True or row.get('is_complete') is False or row.get('is_testable') is False
    )
    avg_score_ctx = df_ctx['quality_score'].mean()
    n_filter_applied = sum(1 for _, row in df_ctx.iterrows() if row.get('domain_filter_reason'))
    print(f"  Requisitos con problemas: {n_issues_ctx}/{len(reqs_ambiguous)}")
    print(f"  Quality score medio: {avg_score_ctx:.1f}")
    print(f"  Filtro aplicado en: {n_filter_applied} requisitos")
    print(f"  Tiempo: {t_ctx:.1f}s")
    for _, row in df_ctx.iterrows():
        print(f"  [{row['text'][:55]}]")
        print(f"    amb={row['is_ambiguous']} compl={row['is_complete']} test={row['is_testable']} score={row['quality_score']:.0f}")
        if row.get('domain_filter_reason'):
            print(f"    → domain_filter: {row['domain_filter_reason']}")

    check("2b. Contexto mejora quality score",
          avg_score_ctx >= avg_score_no_ctx,
          f"sin={avg_score_no_ctx:.1f} con={avg_score_ctx:.1f}")
    check("2b. El filtro se aplica en al menos 1 req", n_filter_applied >= 1,
          f"aplicado en {n_filter_applied}")
    check("2b. Con contexto hay menos problemas que sin él",
          n_issues_ctx <= n_issues_no_ctx,
          f"sin={n_issues_no_ctx} con={n_issues_ctx}")

    # ── 2c. Contexto irrelevante (no debe cambiar los flags) ──────────────────
    print("\n--- 2c. Contexto irrelevante ---")
    bad_context = "Sistema de gestión de biblioteca universitaria para préstamo de libros."
    r_bad = run_dag_pipeline_from_requirements(
        reqs_ambiguous, context_prompt=bad_context, **common_kwargs
    )
    df_bad = r_bad['results_df']
    avg_score_bad = df_bad['quality_score'].mean()
    print(f"  Quality score medio con contexto irrelevante: {avg_score_bad:.1f}")
    print(f"  Quality score sin contexto: {avg_score_no_ctx:.1f}")
    # Con contexto irrelevante, el score debería ser similar al sin contexto
    # (el filtro no debe inventar mejoras que no existen)
    diff_bad = avg_score_bad - avg_score_no_ctx
    check("2c. Contexto irrelevante no dispara mejoras masivas (diff < 30)",
          diff_bad < 30,
          f"diff={diff_bad:.1f}")

    # ── 2d. Requisito sin problemas: el filtro debe omitirlo ─────────────────
    print("\n--- 2d. Requisito sin problemas (el filtro debe omitirlo) ---")
    r_clean = run_dag_pipeline_from_requirements(
        reqs_clean, context_prompt=context, **common_kwargs
    )
    df_clean = r_clean['results_df']
    for _, row in df_clean.iterrows():
        print(f"  [{row['text'][:70]}]")
        print(f"    amb={row['is_ambiguous']} compl={row['is_complete']} test={row['is_testable']} score={row['quality_score']:.0f}")
        print(f"    domain_filter_reason: '{row.get('domain_filter_reason', '')}'")

    # El filtro no debe haber sido llamado (domain_filter_reason vacío o ausente)
    filter_on_clean = any(row.get('domain_filter_reason') for _, row in df_clean.iterrows())
    check("2d. Filtro no invocado en req sin problemas", not filter_on_clean)


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    test_parser()

    if '--integration' in sys.argv:
        test_integration()

    if _failures:
        print(f"\n\033[91m{len(_failures)} test(s) fallaron:\033[0m {', '.join(_failures)}")
        sys.exit(1)
    else:
        print(f"\n\033[92mTodos los tests pasaron.\033[0m")
        sys.exit(0)
