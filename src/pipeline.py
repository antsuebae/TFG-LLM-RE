"""
Pipeline completo de analisis de requisitos.

Lee un documento con requisitos y ejecuta todas las tareas de RE:
1. Clasificacion F/NF
2. Deteccion de ambiguedad
3. Evaluacion de completitud
4. Evaluacion de testabilidad
5. Deteccion de inconsistencias (pares)

Genera un informe completo en HTML, CSV y Markdown.
"""

import argparse
import re
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
from itertools import combinations
import logging
import sys
import json

from tqdm import tqdm
from models import get_model
from prompts import build_prompt, parse_response, REWRITE_PROMPT

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# Modelos disponibles
MODEL_CONFIGS = {
    "qwen7b": {"name": "qwen2.5:7b-instruct-q5_K_M", "type": "ollama", "short_name": "qwen7b"},
    "llama8b": {"name": "llama3.1:8b-instruct-q4_K_M", "type": "ollama", "short_name": "llama8b"},
    "gemma9b": {"name": "gemma2:9b-instruct-q4_K_M", "type": "ollama", "short_name": "gemma9b"},
    "nim_llama70b": {"name": "meta/llama-3.1-70b-instruct", "type": "nvidia_nim", "short_name": "nim_llama70b"},
    "nim_llama8b": {"name": "meta/llama-3.1-8b-instruct", "type": "nvidia_nim", "short_name": "nim_llama8b"},
    "nim_mistral": {"name": "mistralai/mistral-7b-instruct-v0.3", "type": "nvidia_nim", "short_name": "nim_mistral"},
}


def load_requirements(filepath: str) -> list[str]:
    """Carga requisitos desde un archivo txt, md, csv o pdf."""
    path = Path(filepath)

    if path.suffix == '.csv':
        df = pd.read_csv(path)
        for col in ['text', 'requirement', 'requisito', 'description']:
            if col in df.columns:
                return df[col].dropna().tolist()
        text_cols = [c for c in df.columns if c.lower() != 'id']
        if text_cols:
            return df[text_cols[0]].dropna().tolist()
        raise ValueError(f"No se encontro columna de texto en {filepath}")

    if path.suffix == '.pdf':
        return _load_requirements_from_pdf(path)

    # txt o md: una linea por requisito (ignora vacias y comentarios)
    text = path.read_text(encoding='utf-8')
    return _parse_text_lines(text)


def _parse_text_lines(text: str) -> list[str]:
    """Extrae requisitos de texto plano (un requisito por linea)."""
    lines = []
    for line in text.split('\n'):
        line = line.strip()
        if not line or line.startswith('#') or line.startswith('//'):
            continue
        for prefix in ['- ', '* ']:
            if line.startswith(prefix):
                line = line[len(prefix):]
        if line and line[0].isdigit():
            line = re.sub(r'^\d+[\.\)]\s*', '', line)
        if line:
            lines.append(line)
    return lines


def _clean_pdf_artifacts(text: str) -> str:
    """Elimina artefactos tipicos de extraccion PDF."""
    # Eliminar "Page X of Y..." y variantes
    text = re.sub(r'Page\s+\d+\s+of\s+\d+\S*', '', text)
    # Eliminar rutas file:/// hasta el final de la linea o siguiente espacio
    text = re.sub(r'file:///\S+', '', text)
    # Eliminar referencias a proyecto/fecha de exportacion tipo "Proyecto REM-CMS... DD/MM/YYYY"
    text = re.sub(r'Proyecto\s+REM-CMS[^\n]*?\d{2}/\d{2}/\d{4}\S*', '', text)
    # Limpiar espacios multiples resultantes
    text = re.sub(r'  +', ' ', text)
    return text.strip()


def _load_requirements_from_pdf(path: Path) -> list[str]:
    """Extrae requisitos de un PDF con formato REM-CMS o texto libre."""
    try:
        from pypdf import PdfReader
    except ImportError:
        raise ImportError("Instala pypdf: pip install pypdf")

    reader = PdfReader(str(path))
    full_text = ""
    for page in reader.pages:
        full_text += page.extract_text() + "\n"

    # Limpiar artefactos de PDF antes de procesar
    full_text = _clean_pdf_artifacts(full_text)

    # Intentar extraer campos "Descripcion" de tablas REM-CMS
    # Patron: "Descripcion" seguido del texto del requisito
    descriptions = re.findall(
        r'Descripci[oó]n\s+(.+?)(?=\n\s*(?:Datos espec[ií]ficos|Importancia|Comentarios|Urgencia|Estado|Estabilidad|Versi[oó]n|Autores|Fuentes)\b)',
        full_text, re.DOTALL
    )

    if descriptions:
        # Limpiar descripciones extraidas
        requirements = []
        for desc in descriptions:
            clean = re.sub(r'\s+', ' ', desc).strip()
            # Ignorar descripciones cortas o que son indices/metadatos
            if len(clean) < 20:
                continue
            if re.match(r'^(del |Este actor |\d)', clean):
                continue
            requirements.append(clean)
        if requirements:
            logger.info(f"PDF REM-CMS detectado: {len(requirements)} requisitos extraidos")
            return requirements

    # Fallback: tratar como texto libre, linea por linea
    logger.info("PDF sin formato REM-CMS, extrayendo lineas como requisitos")
    return _parse_text_lines(full_text)


def analyze_requirement(model, req: str, strategy: str) -> dict:
    """Ejecuta todas las tareas individuales sobre un requisito."""
    result = {'text': req}

    # A1: Clasificacion
    prompt = build_prompt('classification', strategy, requirement=req)
    resp = model.generate(prompt, max_tokens=512)
    if resp['success']:
        result['classification'] = parse_response('classification', resp['content'])
    else:
        result['classification'] = 'ERROR'
    result['classification_time'] = resp['time_seconds']

    # A2: Ambiguedad
    prompt = build_prompt('ambiguity', strategy, requirement=req)
    resp = model.generate(prompt, max_tokens=512)
    if resp['success']:
        parsed = parse_response('ambiguity', resp['content'])
        result['is_ambiguous'] = parsed['is_ambiguous']
        result['ambiguity_type'] = parsed['ambiguity_type']
        result['ambiguous_words'] = ', '.join(parsed['ambiguous_words'])
    else:
        result['is_ambiguous'] = 'ERROR'
        result['ambiguity_type'] = ''
        result['ambiguous_words'] = ''
    result['ambiguity_time'] = resp['time_seconds']

    # A3: Completitud
    prompt = build_prompt('completeness', strategy, requirement=req)
    resp = model.generate(prompt, max_tokens=512)
    if resp['success']:
        parsed = parse_response('completeness', resp['content'])
        result['is_complete'] = parsed['is_complete']
        result['missing_elements'] = ', '.join(parsed['missing_elements'])
    else:
        result['is_complete'] = 'ERROR'
        result['missing_elements'] = ''
    result['completeness_time'] = resp['time_seconds']

    # V2: Testabilidad
    prompt = build_prompt('testability', strategy, requirement=req)
    resp = model.generate(prompt, max_tokens=512)
    if resp['success']:
        parsed = parse_response('testability', resp['content'])
        result['is_testable'] = parsed['is_testable']
        result['testability_reason'] = parsed['reason']
    else:
        result['is_testable'] = 'ERROR'
        result['testability_reason'] = ''
    result['testability_time'] = resp['time_seconds']

    return result


def analyze_inconsistencies(model, requirements: list[str], strategy: str,
                            max_pairs: int = 50) -> list[dict]:
    """V1: Analiza pares de requisitos buscando inconsistencias."""
    pairs = list(combinations(range(len(requirements)), 2))
    if len(pairs) > max_pairs:
        # Muestrear pares aleatorios
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    results = []
    for i, j in tqdm(pairs, desc="Consistencia", unit="par"):
        prompt = build_prompt('inconsistency', strategy,
                              requirement_a=requirements[i],
                              requirement_b=requirements[j])
        resp = model.generate(prompt, max_tokens=512)
        if resp['success']:
            parsed = parse_response('inconsistency', resp['content'])
            if parsed['is_inconsistent']:
                results.append({
                    'req_a_idx': i + 1,
                    'req_b_idx': j + 1,
                    'req_a': requirements[i],
                    'req_b': requirements[j],
                    'description': parsed['description'],
                })
    return results


def compute_quality_score(row: dict) -> float:
    """Calcula una puntuacion de calidad 0-100 para un requisito."""
    score = 100.0
    if row.get('is_ambiguous') is True:
        score -= 30
    if row.get('is_complete') is False:
        score -= 30
        missing = row.get('missing_elements', '')
        if missing:
            n_missing = len(missing.split(','))
            score -= min(n_missing * 5, 15)
    if row.get('is_testable') is False:
        score -= 25
    return max(score, 0)


def generate_html_report(results_df: pd.DataFrame, inconsistencies: list[dict],
                         model_name: str, strategy: str, output_path: Path):
    """Genera informe HTML completo."""
    total = len(results_df)
    n_functional = len(results_df[results_df['classification'] == 'F'])
    n_nonfunctional = len(results_df[results_df['classification'] == 'NF'])
    n_ambiguous = len(results_df[results_df['is_ambiguous'] == True])
    n_incomplete = len(results_df[results_df['is_complete'] == False])
    n_not_testable = len(results_df[results_df['is_testable'] == False])
    avg_quality = results_df['quality_score'].mean()
    n_inconsistencies = len(inconsistencies)

    # Requisitos con problemas
    problems_df = results_df[
        (results_df['is_ambiguous'] == True) |
        (results_df['is_complete'] == False) |
        (results_df['is_testable'] == False)
    ]

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>Informe de Analisis de Requisitos</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 40px; background: #f5f5f5; color: #333; }}
    h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
    h2 {{ color: #2c3e50; margin-top: 30px; }}
    .summary {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(180px, 1fr)); gap: 15px; margin: 20px 0; }}
    .card {{ background: white; padding: 20px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); text-align: center; }}
    .card .number {{ font-size: 2em; font-weight: bold; }}
    .card .label {{ color: #666; font-size: 0.9em; }}
    .good {{ color: #27ae60; }}
    .warning {{ color: #f39c12; }}
    .bad {{ color: #e74c3c; }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #3498db; color: white; padding: 12px; text-align: left; }}
    td {{ padding: 10px 12px; border-bottom: 1px solid #eee; }}
    tr:hover {{ background: #f0f7ff; }}
    .tag {{ display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 0.85em; font-weight: bold; }}
    .tag-f {{ background: #d4edda; color: #155724; }}
    .tag-nf {{ background: #cce5ff; color: #004085; }}
    .tag-yes {{ background: #f8d7da; color: #721c24; }}
    .tag-no {{ background: #d4edda; color: #155724; }}
    .quality-bar {{ height: 20px; border-radius: 10px; background: #eee; overflow: hidden; }}
    .quality-fill {{ height: 100%; border-radius: 10px; }}
    .meta {{ color: #888; font-size: 0.85em; margin: 5px 0; }}
    .inconsistency {{ background: #fff3cd; padding: 15px; border-radius: 8px; margin: 10px 0; border-left: 4px solid #ffc107; }}
</style>
</head>
<body>

<h1>Informe de Analisis de Requisitos</h1>
<p class="meta">Modelo: <strong>{model_name}</strong> | Estrategia: <strong>{strategy}</strong> | Fecha: {datetime.now():%Y-%m-%d %H:%M}</p>

<div class="summary">
    <div class="card"><div class="number">{total}</div><div class="label">Requisitos</div></div>
    <div class="card"><div class="number">{n_functional}</div><div class="label">Funcionales</div></div>
    <div class="card"><div class="number">{n_nonfunctional}</div><div class="label">No Funcionales</div></div>
    <div class="card"><div class="number {'bad' if n_ambiguous > total*0.3 else 'warning' if n_ambiguous > 0 else 'good'}">{n_ambiguous}</div><div class="label">Ambiguos</div></div>
    <div class="card"><div class="number {'bad' if n_incomplete > total*0.3 else 'warning' if n_incomplete > 0 else 'good'}">{n_incomplete}</div><div class="label">Incompletos</div></div>
    <div class="card"><div class="number {'bad' if n_not_testable > total*0.3 else 'warning' if n_not_testable > 0 else 'good'}">{n_not_testable}</div><div class="label">No Testables</div></div>
    <div class="card"><div class="number {'good' if avg_quality >= 70 else 'warning' if avg_quality >= 40 else 'bad'}">{avg_quality:.0f}%</div><div class="label">Calidad Media</div></div>
    <div class="card"><div class="number {'bad' if n_inconsistencies > 0 else 'good'}">{n_inconsistencies}</div><div class="label">Inconsistencias</div></div>
</div>

<h2>Detalle por Requisito</h2>
<table>
<tr><th>#</th><th>Requisito</th><th>Tipo</th><th>Ambiguo</th><th>Completo</th><th>Testable</th><th>Calidad</th></tr>
"""

    for idx, row in results_df.iterrows():
        cls_tag = f'<span class="tag tag-f">F</span>' if row['classification'] == 'F' else f'<span class="tag tag-nf">NF</span>'
        amb_tag = f'<span class="tag tag-yes">Si</span>' if row['is_ambiguous'] else f'<span class="tag tag-no">No</span>'
        comp_tag = f'<span class="tag tag-no">No</span>' if not row['is_complete'] else f'<span class="tag tag-no" style="background:#d4edda;color:#155724">Si</span>'
        test_tag = f'<span class="tag tag-no">No</span>' if not row['is_testable'] else f'<span class="tag tag-no" style="background:#d4edda;color:#155724">Si</span>'

        q = row['quality_score']
        color = '#27ae60' if q >= 70 else '#f39c12' if q >= 40 else '#e74c3c'
        quality_bar = f'<div class="quality-bar"><div class="quality-fill" style="width:{q}%;background:{color}"></div></div>'

        text = row['text'][:120] + ('...' if len(str(row['text'])) > 120 else '')
        html += f'<tr><td>{idx+1}</td><td>{text}</td><td>{cls_tag}</td><td>{amb_tag}</td><td>{comp_tag}</td><td>{test_tag}</td><td>{quality_bar} {q:.0f}%</td></tr>\n'

    html += '</table>\n'

    # Detalles de problemas
    if not problems_df.empty:
        html += '<h2>Requisitos con Problemas</h2>\n<table>\n'
        html += '<tr><th>#</th><th>Requisito</th><th>Problemas</th><th>Detalles</th></tr>\n'
        for idx, row in problems_df.iterrows():
            problems = []
            details = []
            if row['is_ambiguous']:
                problems.append('Ambiguo')
                if row['ambiguous_words']:
                    details.append(f"Palabras: {row['ambiguous_words']}")
            if not row['is_complete']:
                problems.append('Incompleto')
                if row['missing_elements']:
                    details.append(f"Falta: {row['missing_elements']}")
            if not row['is_testable']:
                problems.append('No testable')
                details.append(f"Razon: {row['testability_reason']}")

            text = str(row['text'])[:100] + ('...' if len(str(row['text'])) > 100 else '')
            html += f'<tr><td>{idx+1}</td><td>{text}</td><td>{", ".join(problems)}</td><td>{"; ".join(details)}</td></tr>\n'
        html += '</table>\n'

    # Inconsistencias
    if inconsistencies:
        html += '<h2>Inconsistencias Detectadas</h2>\n'
        for inc in inconsistencies:
            html += f"""<div class="inconsistency">
<strong>Req #{inc['req_a_idx']}</strong>: {inc['req_a'][:150]}<br>
<strong>Req #{inc['req_b_idx']}</strong>: {inc['req_b'][:150]}<br>
<strong>Conflicto:</strong> {inc['description']}
</div>\n"""

    html += '</body></html>'

    output_path.write_text(html, encoding='utf-8')
    logger.info(f"Informe HTML guardado: {output_path}")


def generate_markdown_report(results_df: pd.DataFrame, inconsistencies: list[dict],
                             model_name: str, strategy: str, output_path: Path):
    """Genera informe en Markdown."""
    total = len(results_df)
    avg_quality = results_df['quality_score'].mean()

    lines = [
        f"# Informe de Analisis de Requisitos\n",
        f"**Modelo:** {model_name} | **Estrategia:** {strategy} | **Fecha:** {datetime.now():%Y-%m-%d %H:%M}\n",
        f"## Resumen\n",
        f"| Metrica | Valor |",
        f"|---------|-------|",
        f"| Total requisitos | {total} |",
        f"| Funcionales (F) | {len(results_df[results_df['classification'] == 'F'])} |",
        f"| No Funcionales (NF) | {len(results_df[results_df['classification'] == 'NF'])} |",
        f"| Ambiguos | {len(results_df[results_df['is_ambiguous'] == True])} |",
        f"| Incompletos | {len(results_df[results_df['is_complete'] == False])} |",
        f"| No testables | {len(results_df[results_df['is_testable'] == False])} |",
        f"| Calidad media | {avg_quality:.0f}% |",
        f"| Inconsistencias | {len(inconsistencies)} |",
        f"\n## Detalle\n",
        f"| # | Requisito | Tipo | Ambiguo | Completo | Testable | Calidad |",
        f"|---|-----------|------|---------|----------|----------|---------|",
    ]

    for idx, row in results_df.iterrows():
        text = str(row['text'])[:80] + ('...' if len(str(row['text'])) > 80 else '')
        amb = 'Si' if row['is_ambiguous'] else 'No'
        comp = 'Si' if row['is_complete'] else 'No'
        test = 'Si' if row['is_testable'] else 'No'
        lines.append(f"| {idx+1} | {text} | {row['classification']} | {amb} | {comp} | {test} | {row['quality_score']:.0f}% |")

    if inconsistencies:
        lines.append(f"\n## Inconsistencias Detectadas\n")
        for inc in inconsistencies:
            lines.append(f"- **Req #{inc['req_a_idx']}** vs **Req #{inc['req_b_idx']}**: {inc['description']}")

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f"Informe Markdown guardado: {output_path}")


def rewrite_requirement(model, original: str, analysis: dict) -> dict:
    """Reescribe un requisito corrigiendo los problemas detectados.

    Args:
        model: Instancia del modelo LLM.
        original: Texto del requisito original.
        analysis: Dict con campos is_ambiguous, ambiguous_words, is_complete,
                  missing_elements, is_testable, testability_reason.

    Returns:
        Dict con: original, corrected, changes_made (list[str]).
    """
    problems = []
    changes = []

    if analysis.get('is_ambiguous'):
        words = analysis.get('ambiguous_words', '')
        problems.append(f"- AMBIGUO: contiene terminos vagos ({words})")
        changes.append(f"Eliminar ambiguedad ({words})")

    if analysis.get('is_complete') is False:
        missing = analysis.get('missing_elements', '')
        problems.append(f"- INCOMPLETO: faltan elementos ({missing})")
        changes.append(f"Anadir elementos faltantes ({missing})")

    if analysis.get('is_testable') is False:
        reason = analysis.get('testability_reason', '')
        problems.append(f"- NO TESTABLE: {reason}")
        changes.append(f"Hacer testable ({reason})")

    if not problems:
        return {'original': original, 'corrected': original, 'changes_made': []}

    prompt = REWRITE_PROMPT.format(
        requirement=original,
        problems='\n'.join(problems)
    )
    resp = model.generate(prompt, max_tokens=1024)

    if resp['success']:
        corrected = resp['content'].strip()
        # Clean up: remove quotes if the model wraps the response
        if corrected.startswith('"') and corrected.endswith('"'):
            corrected = corrected[1:-1]
        # Join all non-empty lines into a single coherent requirement
        lines = [l.strip() for l in corrected.split('\n') if l.strip()]
        corrected = ' '.join(lines) if lines else corrected
        # Validate: if truncated (ends with ':') or too short, flag it
        if corrected.rstrip().endswith(':') or len(corrected) < 20:
            changes.append("AVISO: la correccion puede estar truncada")
    else:
        corrected = original
        changes.append("ERROR: no se pudo reescribir")

    # Clean markdown artifacts from changes descriptions
    changes = [re.sub(r'\*{1,2}\s*', '', c).strip() for c in changes]

    return {
        'original': original,
        'corrected': corrected,
        'changes_made': changes,
    }


def generate_corrected_document(results_df: pd.DataFrame, model,
                                 output_path: Path) -> list[dict]:
    """Genera documento con requisitos corregidos.

    Args:
        results_df: DataFrame con resultados del analisis.
        model: Instancia del modelo LLM.
        output_path: Path para guardar el documento (Markdown).

    Returns:
        Lista de dicts con original, corrected, changes_made por cada requisito.
    """
    corrections = []

    for _, row in tqdm(results_df.iterrows(), total=len(results_df),
                       desc="Reescribiendo", unit="req"):
        has_problems = (
            row.get('is_ambiguous') is True or
            row.get('is_complete') is False or
            row.get('is_testable') is False
        )

        if has_problems:
            result = rewrite_requirement(model, row['text'], row.to_dict())
        else:
            result = {
                'original': row['text'],
                'corrected': row['text'],
                'changes_made': [],
            }
        corrections.append(result)

    # Generate Markdown document
    lines = ["# Documento de Requisitos Corregido\n"]
    n_corrected = sum(1 for c in corrections if c['changes_made'])
    lines.append(f"**{n_corrected}** de **{len(corrections)}** requisitos corregidos.\n")
    lines.append("---\n")

    for i, c in enumerate(corrections, 1):
        if c['changes_made']:
            lines.append(f"### Requisito {i} (corregido)\n")
            lines.append(f"**Original:** ~~{c['original']}~~\n")
            lines.append(f"**Corregido:** {c['corrected']}\n")
            lines.append("**Cambios:**")
            for change in c['changes_made']:
                lines.append(f"- {change}")
            lines.append("")
        else:
            lines.append(f"### Requisito {i} (sin cambios)\n")
            lines.append(f"{c['original']}\n")

    output_path.write_text('\n'.join(lines), encoding='utf-8')
    logger.info(f"Documento corregido guardado: {output_path}")

    return corrections


def generate_srs_document(results_df: pd.DataFrame, corrections: list[dict],
                          inconsistencies: list[dict], metadata: dict) -> str:
    """Genera documento corregido con formato SRS profesional (estilo IEEE 830).

    Args:
        results_df: DataFrame con resultados del analisis.
        corrections: Lista de dicts con original, corrected, changes_made.
        inconsistencies: Lista de inconsistencias detectadas.
        metadata: Dict con doc_name, model, strategy, date.

    Returns:
        String con el documento SRS en formato Markdown.
    """
    total = len(results_df)
    n_corrected = sum(1 for c in corrections if c.get('changes_made'))
    n_functional = len(results_df[results_df['classification'] == 'F'])
    n_nonfunctional = total - n_functional
    n_ambiguous = len(results_df[results_df['is_ambiguous'] == True])
    n_incomplete = len(results_df[results_df['is_complete'] == False])
    n_not_testable = len(results_df[results_df['is_testable'] == False])
    avg_quality = results_df['quality_score'].mean()

    model_label = metadata.get('model', 'N/A')
    strategy_label = metadata.get('strategy', 'N/A')
    doc_name = metadata.get('doc_name', 'documento')
    date = metadata.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))

    lines = []

    # ── Cabecera ──────────────────────────────────────────
    lines.append("# Especificacion de Requisitos de Software (SRS)")
    lines.append("")
    lines.append("## Informacion del documento")
    lines.append(f"- **Documento origen:** {doc_name}")
    lines.append(f"- **Fecha de analisis:** {date}")
    lines.append(f"- **Modelo:** {model_label} | **Estrategia:** {strategy_label}")
    lines.append(f"- **Requisitos analizados:** {total} | **Corregidos:** {n_corrected}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # ── Separar requisitos F y NF ─────────────────────────
    functional_indices = []
    nonfunctional_indices = []
    for i, (_, row) in enumerate(results_df.iterrows()):
        if row['classification'] == 'F':
            functional_indices.append(i)
        else:
            nonfunctional_indices.append(i)

    def _format_requirement_section(indices, prefix, section_title):
        """Genera seccion de requisitos con IDs y estado."""
        section = []
        section.append(f"## {section_title}")
        section.append("")
        if not indices:
            section.append("*No se identificaron requisitos en esta categoria.*")
            section.append("")
            return section

        for counter, idx in enumerate(indices, 1):
            row = results_df.iloc[idx]
            correction = corrections[idx] if idx < len(corrections) else None
            req_id = f"{prefix}-{counter:03d}"

            # Titulo corto: primeras palabras del requisito
            text = correction['corrected'] if correction and correction.get('changes_made') else row['text']
            short_title = text[:80].rstrip('.')
            if len(text) > 80:
                short_title += "..."

            has_changes = correction and correction.get('changes_made')
            status = "Corregido" if has_changes else "Sin cambios"

            section.append(f"### {req_id}: {short_title}")
            section.append(f"- **Descripcion:** {text}")
            section.append(f"- **Estado:** {status}")

            if has_changes:
                section.append("- **Cambios realizados:**")
                for change in correction['changes_made']:
                    section.append(f"  - {change}")
                section.append(f"- **Original:** ~~{correction['original']}~~")

            section.append("")

        return section

    lines.extend(_format_requirement_section(
        functional_indices, "REQ-F", "1. Requisitos Funcionales"))
    lines.extend(_format_requirement_section(
        nonfunctional_indices, "REQ-NF", "2. Requisitos No Funcionales"))

    # ── Resumen de calidad ────────────────────────────────
    lines.append("## 3. Resumen de Calidad")
    lines.append("")
    lines.append("| Metrica | Valor |")
    lines.append("|---------|-------|")
    lines.append(f"| Calidad media | {avg_quality:.0f}% |")
    lines.append(f"| Requisitos funcionales | {n_functional} |")
    lines.append(f"| Requisitos no funcionales | {n_nonfunctional} |")
    lines.append(f"| Requisitos ambiguos | {n_ambiguous} |")
    lines.append(f"| Requisitos incompletos | {n_incomplete} |")
    lines.append(f"| Requisitos no testables | {n_not_testable} |")
    lines.append(f"| Requisitos corregidos | {n_corrected} |")
    lines.append("")

    # ── Inconsistencias ───────────────────────────────────
    lines.append("## 4. Inconsistencias Detectadas")
    lines.append("")
    if inconsistencies:
        for inc in inconsistencies:
            lines.append(f"- **Req #{inc['req_a_idx']}** vs **Req #{inc['req_b_idx']}**: {inc['description']}")
        lines.append("")
    else:
        lines.append("*No se detectaron inconsistencias.*")
        lines.append("")

    # ── Trazabilidad de cambios ───────────────────────────
    lines.append("## 5. Trazabilidad de Cambios")
    lines.append("")
    changes_exist = any(c.get('changes_made') for c in corrections)
    if changes_exist:
        lines.append("| REQ ID | Original | Corregido | Cambios |")
        lines.append("|--------|----------|-----------|---------|")

        f_counter = 0
        nf_counter = 0
        for i, (_, row) in enumerate(results_df.iterrows()):
            if row['classification'] == 'F':
                f_counter += 1
                req_id = f"REQ-F-{f_counter:03d}"
            else:
                nf_counter += 1
                req_id = f"REQ-NF-{nf_counter:03d}"

            correction = corrections[i] if i < len(corrections) else None
            if correction and correction.get('changes_made'):
                original_short = correction['original'][:60].replace('|', '\\|')
                corrected_short = correction['corrected'][:60].replace('|', '\\|')
                changes_str = '; '.join(correction['changes_made']).replace('|', '\\|')
                lines.append(f"| {req_id} | {original_short} | {corrected_short} | {changes_str} |")

        lines.append("")
    else:
        lines.append("*No se realizaron cambios.*")
        lines.append("")

    # ── Pie ────────────────────────────────────────────────
    lines.append("---")
    lines.append(f"*Documento generado automaticamente por RE-LLM el {date}*")

    return '\n'.join(lines)


def generate_srs_html(srs_markdown: str, metadata: dict) -> str:
    """Genera version HTML profesional del documento SRS.

    Args:
        srs_markdown: Contenido SRS en Markdown.
        metadata: Dict con doc_name, model, strategy, date.

    Returns:
        String HTML con estilos profesionales.
    """
    import re as re_mod

    doc_name = metadata.get('doc_name', 'documento')
    date = metadata.get('date', datetime.now().strftime('%Y-%m-%d %H:%M'))

    # Convertir markdown a HTML basico
    content = srs_markdown

    # Headers
    content = re_mod.sub(r'^### (.+)$', r'<h3>\1</h3>', content, flags=re_mod.MULTILINE)
    content = re_mod.sub(r'^## (.+)$', r'<h2>\1</h2>', content, flags=re_mod.MULTILINE)
    content = re_mod.sub(r'^# (.+)$', r'<h1>\1</h1>', content, flags=re_mod.MULTILINE)

    # Bold
    content = re_mod.sub(r'\*\*(.+?)\*\*', r'<strong>\1</strong>', content)
    # Italic
    content = re_mod.sub(r'\*(.+?)\*', r'<em>\1</em>', content)
    # Strikethrough
    content = re_mod.sub(r'~~(.+?)~~', r'<del>\1</del>', content)

    # Tables
    def convert_table(match):
        lines = match.group(0).strip().split('\n')
        html_table = '<table>\n'
        for i, line in enumerate(lines):
            if '---' in line and '|' in line:
                continue  # skip separator
            cells = [c.strip() for c in line.strip('|').split('|')]
            tag = 'th' if i == 0 else 'td'
            html_table += '<tr>' + ''.join(f'<{tag}>{c}</{tag}>' for c in cells) + '</tr>\n'
        html_table += '</table>\n'
        return html_table

    content = re_mod.sub(r'(\|.+\|(?:\n\|.+\|)+)', convert_table, content)

    # List items
    content = re_mod.sub(r'^  - (.+)$', r'<li class="sub">\1</li>', content, flags=re_mod.MULTILINE)
    content = re_mod.sub(r'^- (.+)$', r'<li>\1</li>', content, flags=re_mod.MULTILINE)

    # Horizontal rules
    content = re_mod.sub(r'^---$', r'<hr>', content, flags=re_mod.MULTILINE)

    # Paragraphs for remaining lines
    lines_out = []
    for line in content.split('\n'):
        stripped = line.strip()
        if stripped and not stripped.startswith('<'):
            lines_out.append(f'<p>{stripped}</p>')
        else:
            lines_out.append(line)
    content = '\n'.join(lines_out)

    html = f"""<!DOCTYPE html>
<html lang="es">
<head>
<meta charset="UTF-8">
<title>SRS - {doc_name}</title>
<style>
    body {{ font-family: 'Segoe UI', sans-serif; margin: 40px auto; max-width: 900px; background: #f5f5f5; color: #333; line-height: 1.6; }}
    h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; font-size: 1.8em; }}
    h2 {{ color: #2c5282; margin-top: 30px; border-bottom: 1px solid #bee3f8; padding-bottom: 5px; }}
    h3 {{ color: #2d3748; margin-top: 20px; font-size: 1.05em; }}
    table {{ width: 100%; border-collapse: collapse; margin: 15px 0; background: white; border-radius: 8px; overflow: hidden; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
    th {{ background: #2b6cb0; color: white; padding: 10px 12px; text-align: left; font-size: 0.9em; }}
    td {{ padding: 8px 12px; border-bottom: 1px solid #e2e8f0; font-size: 0.9em; }}
    tr:hover {{ background: #ebf8ff; }}
    li {{ margin: 4px 0; }}
    li.sub {{ margin-left: 20px; font-size: 0.95em; }}
    del {{ color: #e53e3e; background: #fed7d7; padding: 1px 3px; border-radius: 2px; }}
    strong {{ color: #2d3748; }}
    hr {{ border: none; border-top: 2px solid #e2e8f0; margin: 30px 0; }}
    em {{ color: #718096; }}
    p {{ margin: 5px 0; }}
</style>
</head>
<body>
{content}
</body>
</html>"""
    return html


def save_pipeline_results(results_df: pd.DataFrame, inconsistencies: list[dict],
                          model_key: str, strategy: str, doc_name: str,
                          base_dir: Path) -> Path:
    """Guarda resultados del pipeline en un subdirectorio organizado con metadata.json.

    Returns:
        Path del subdirectorio creado.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_name = f"{timestamp}_{model_key}_{strategy}"
    run_dir = base_dir / run_name
    run_dir.mkdir(parents=True, exist_ok=True)

    # CSV
    results_df.to_csv(run_dir / "results.csv", index=False)

    # Informes
    generate_html_report(results_df, inconsistencies, model_key, strategy,
                         run_dir / "informe.html")
    generate_markdown_report(results_df, inconsistencies, model_key, strategy,
                             run_dir / "informe.md")

    # Inconsistencias JSON
    if inconsistencies:
        with open(run_dir / "inconsistencias.json", 'w') as f:
            json.dump(inconsistencies, f, indent=2, ensure_ascii=False)

    # Metadata
    total = len(results_df)
    metadata = {
        "timestamp": datetime.now().isoformat(),
        "document": doc_name,
        "model": model_key,
        "strategy": strategy,
        "n_requirements": total,
        "summary": {
            "functional": int(len(results_df[results_df['classification'] == 'F'])),
            "non_functional": int(len(results_df[results_df['classification'] == 'NF'])),
            "ambiguous": int(len(results_df[results_df['is_ambiguous'] == True])),
            "incomplete": int(len(results_df[results_df['is_complete'] == False])),
            "not_testable": int(len(results_df[results_df['is_testable'] == False])),
            "inconsistencies": len(inconsistencies),
            "avg_quality": round(float(results_df['quality_score'].mean()), 1),
        }
    }
    with open(run_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)

    logger.info(f"Resultados guardados en: {run_dir}")
    return run_dir


def load_pipeline_runs(base_dir: Path) -> list[dict]:
    """Carga metadata de todas las ejecuciones del pipeline.

    Returns:
        Lista de dicts con metadata + path de cada ejecucion.
    """
    runs = []
    if not base_dir.exists():
        return runs
    for meta_file in sorted(base_dir.glob("*/metadata.json"), reverse=True):
        try:
            with open(meta_file) as f:
                meta = json.load(f)
            meta['run_dir'] = str(meta_file.parent)
            meta['run_name'] = meta_file.parent.name
            runs.append(meta)
        except Exception as e:
            logger.warning(f"Error cargando {meta_file}: {e}")
    return runs


def run_pipeline(filepath: str, model_key: str = "llama8b", strategy: str = "few_shot",
                 output_dir: str = None, skip_inconsistency: bool = False,
                 max_pairs: int = 50):
    """Ejecuta el pipeline completo sobre un documento de requisitos."""

    # Cargar requisitos
    requirements = load_requirements(filepath)
    logger.info(f"Cargados {len(requirements)} requisitos desde {filepath}")

    if not requirements:
        logger.error("No se encontraron requisitos en el archivo.")
        return

    # Crear modelo
    model = get_model(MODEL_CONFIGS[model_key])
    logger.info(f"Modelo: {model_key} | Estrategia: {strategy}")

    # Directorio de salida
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results" / "pipeline"
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    # ── Analisis individual (A1 + A2 + A3 + V2) ─────────────
    logger.info("Analizando requisitos individualmente...")
    rows = []
    for req in tqdm(requirements, desc="Requisitos", unit="req"):
        row = analyze_requirement(model, req, strategy)
        row['quality_score'] = compute_quality_score(row)
        rows.append(row)

    results_df = pd.DataFrame(rows)

    # ── Inconsistencias (V1) ─────────────────────────────────
    inconsistencies = []
    if not skip_inconsistency and len(requirements) > 1:
        logger.info(f"Analizando consistencia ({min(max_pairs, len(requirements)*(len(requirements)-1)//2)} pares)...")
        inconsistencies = analyze_inconsistencies(model, requirements, strategy, max_pairs)

    # ── Guardar resultados en subdirectorio ──────────────────
    doc_name = Path(filepath).name
    run_dir = save_pipeline_results(results_df, inconsistencies,
                                     model_key, strategy, doc_name, output)

    # ── Resumen en consola ───────────────────────────────────
    total = len(results_df)
    print(f"\n{'='*60}")
    print(f"INFORME DE ANALISIS - {total} requisitos")
    print(f"{'='*60}")
    print(f"  Funcionales:     {len(results_df[results_df['classification'] == 'F']):>3}")
    print(f"  No Funcionales:  {len(results_df[results_df['classification'] == 'NF']):>3}")
    print(f"  Ambiguos:        {len(results_df[results_df['is_ambiguous'] == True]):>3}")
    print(f"  Incompletos:     {len(results_df[results_df['is_complete'] == False]):>3}")
    print(f"  No testables:    {len(results_df[results_df['is_testable'] == False]):>3}")
    print(f"  Inconsistencias: {len(inconsistencies):>3}")
    print(f"  Calidad media:   {results_df['quality_score'].mean():>5.0f}%")
    print(f"{'='*60}")
    print(f"  Resultados: {run_dir}")
    print(f"{'='*60}\n")

    return results_df, inconsistencies


def run_pipeline_dag(filepath: str, model_key: str = "llama8b",
                     strategy: str = "few_shot", output_dir: str = None,
                     use_llm_extraction: bool = True,
                     skip_inconsistency: bool = False,
                     enable_rewriting: bool = False,
                     max_pairs: int = 50):
    """Ejecuta el pipeline DAG (prompt combinado + concurrencia).

    Wrapper que delega a dag.run_dag_pipeline() y muestra resumen en consola.
    """
    from dag import run_dag_pipeline, NODE_LABELS

    if output_dir is None:
        output_dir = str(Path(__file__).parent.parent / "results" / "pipeline")

    def progress_cb(node_name, current, total, skipped=False):
        label = NODE_LABELS.get(node_name, node_name)
        status = "saltado" if skipped else "ejecutando"
        logger.info(f"[{current}/{total}] {label} ({status})")

    ctx = run_dag_pipeline(
        filepath=filepath,
        model_key=model_key,
        strategy=strategy,
        output_dir=output_dir,
        use_llm_extraction=use_llm_extraction,
        skip_inconsistency=skip_inconsistency,
        enable_rewriting=enable_rewriting,
        max_pairs=max_pairs,
        progress_callback=progress_cb,
    )

    results_df = ctx.get('results_df', pd.DataFrame())
    inconsistencies = ctx.get('inconsistencies', [])

    if results_df.empty:
        logger.error("No se obtuvieron resultados.")
        return results_df, inconsistencies

    # Resumen en consola
    total = len(results_df)
    print(f"\n{'='*60}")
    print(f"INFORME DE ANALISIS (DAG) - {total} requisitos")
    print(f"{'='*60}")
    print(f"  Funcionales:     {len(results_df[results_df['classification'] == 'F']):>3}")
    print(f"  No Funcionales:  {len(results_df[results_df['classification'] == 'NF']):>3}")
    print(f"  Ambiguos:        {len(results_df[results_df['is_ambiguous'] == True]):>3}")
    print(f"  Incompletos:     {len(results_df[results_df['is_complete'] == False]):>3}")
    print(f"  No testables:    {len(results_df[results_df['is_testable'] == False]):>3}")
    print(f"  Inconsistencias: {len(inconsistencies):>3}")
    print(f"  Calidad media:   {results_df['quality_score'].mean():>5.0f}%")
    print(f"{'='*60}")
    if ctx.get('run_dir'):
        print(f"  Resultados: {ctx['run_dir']}")
    print(f"{'='*60}\n")

    return results_df, inconsistencies


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Pipeline completo de analisis de requisitos',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Ejemplos:
  python pipeline.py requisitos.txt
  python pipeline.py requisitos.txt --dag
  python pipeline.py requisitos.txt --dag --llm-extract
  python pipeline.py requisitos.csv --model nim_llama70b
  python pipeline.py requisitos.md --strategy chain_of_thought
  python pipeline.py requisitos.txt --skip-inconsistency
        """
    )
    parser.add_argument('file', help='Documento de requisitos (.txt, .md, .csv, .pdf)')
    parser.add_argument('--model', default='llama8b',
                        choices=list(MODEL_CONFIGS.keys()),
                        help='Modelo a usar (default: llama8b)')
    parser.add_argument('--strategy', default='few_shot',
                        choices=['question_refinement', 'cognitive_verifier',
                                 'persona_context', 'few_shot', 'chain_of_thought'],
                        help='Estrategia de prompt (default: few_shot)')
    parser.add_argument('--output', default=None, help='Directorio de salida')
    parser.add_argument('--skip-inconsistency', action='store_true',
                        help='Saltar analisis de inconsistencias (mas rapido)')
    parser.add_argument('--max-pairs', type=int, default=50,
                        help='Maximo de pares a analizar para inconsistencias')
    parser.add_argument('--dag', action='store_true',
                        help='Usar pipeline DAG (prompt combinado + concurrencia)')
    parser.add_argument('--llm-extract', action='store_true',
                        help='Usar LLM para extraer requisitos (solo con --dag)')
    parser.add_argument('--rewrite', action='store_true',
                        help='Reescribir requisitos con problemas (solo con --dag)')

    args = parser.parse_args()

    if args.dag:
        run_pipeline_dag(
            filepath=args.file,
            model_key=args.model,
            strategy=args.strategy,
            output_dir=args.output,
            use_llm_extraction=args.llm_extract,
            skip_inconsistency=args.skip_inconsistency,
            enable_rewriting=args.rewrite,
            max_pairs=args.max_pairs,
        )
    else:
        run_pipeline(
            filepath=args.file,
            model_key=args.model,
            strategy=args.strategy,
            output_dir=args.output,
            skip_inconsistency=args.skip_inconsistency,
            max_pairs=args.max_pairs,
        )
