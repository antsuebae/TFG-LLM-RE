"""
Pipeline DAG para analisis de requisitos.

Modela el pipeline como un grafo dirigido aciclico (DAG) con 7 nodos:
  1. load_document     - Carga texto desde archivo
  2. extract_requirements - Extraccion LLM de requisitos desde texto bruto
  3. combined_analysis - 4 tareas en 1 llamada LLM por requisito
  4. quality_score     - Calculo de puntuacion de calidad (sin LLM)
  5. inconsistency     - Deteccion de inconsistencias entre pares (LLM)
  6. report            - Generacion de informe (sin LLM)
  7. rewriting         - Reescritura de requisitos con problemas (LLM, opcional)

Soporta concurrencia para modelos NIM API (ThreadPoolExecutor) y
ejecucion secuencial para modelos Ollama (GPU unica).
"""

import logging
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import pandas as pd

from models import get_model
from prompts import (
    build_combined_prompt,
    build_extraction_prompt,
    build_prompt,
    parse_combined_response,
    parse_extraction_response,
    parse_response,
    REWRITE_PROMPT,
)

logger = logging.getLogger(__name__)

# ── Configuracion de modelos (compartida con pipeline.py) ────
MODEL_CONFIGS = {
    "qwen7b": {"name": "qwen2.5:7b-instruct-q5_K_M", "type": "ollama", "short_name": "qwen7b"},
    "llama8b": {"name": "llama3.1:8b-instruct-q4_K_M", "type": "ollama", "short_name": "llama8b"},
    "llama3b": {"name": "llama3.2:3b-instruct-q4_K_M", "type": "ollama", "short_name": "llama3b"},
    "nim_llama70b": {"name": "meta/llama-3.1-70b-instruct", "type": "nvidia_nim", "short_name": "nim_llama70b"},
    "nim_llama8b": {"name": "meta/llama-3.1-8b-instruct", "type": "nvidia_nim", "short_name": "nim_llama8b"},
    "nim_mistral": {"name": "mistralai/mistral-7b-instruct-v0.3", "type": "nvidia_nim", "short_name": "nim_mistral"},
}


# ============================================================
# DAG infrastructure
# ============================================================

@dataclass
class DAGNode:
    """Nodo del grafo dirigido aciclico."""
    name: str
    func: Callable[[dict], dict]  # recibe y devuelve contexto
    depends_on: list[str] = field(default_factory=list)


class DAGExecutor:
    """Ejecuta nodos en orden topologico."""

    def __init__(self, nodes: list[DAGNode], progress_callback: Optional[Callable] = None):
        self.nodes = {n.name: n for n in nodes}
        self.progress_callback = progress_callback
        self._validate()

    def _validate(self):
        """Verifica que las dependencias existen y no hay ciclos."""
        for node in self.nodes.values():
            for dep in node.depends_on:
                if dep not in self.nodes:
                    raise ValueError(f"Nodo '{node.name}' depende de '{dep}' que no existe")
        # Check for cycles via topological sort
        self._topological_order()

    def _topological_order(self) -> list[str]:
        """Calcula orden topologico (Kahn's algorithm)."""
        in_degree = {name: 0 for name in self.nodes}
        for node in self.nodes.values():
            for dep in node.depends_on:
                # dep must execute before node, so node has in_degree from dep
                pass
            in_degree[node.name] = len(node.depends_on)

        queue = deque(name for name, deg in in_degree.items() if deg == 0)
        order = []

        while queue:
            current = queue.popleft()
            order.append(current)
            for node in self.nodes.values():
                if current in node.depends_on:
                    in_degree[node.name] -= 1
                    if in_degree[node.name] == 0:
                        queue.append(node.name)

        if len(order) != len(self.nodes):
            raise ValueError("Ciclo detectado en el DAG")
        return order

    def execute(self, context: dict) -> dict:
        """Ejecuta todos los nodos en orden topologico."""
        order = self._topological_order()
        total = len(order)

        for i, name in enumerate(order):
            node = self.nodes[name]
            # Skip optional nodes not requested
            if name == 'rewriting' and not context.get('enable_rewriting', False):
                logger.info(f"[DAG] Saltando nodo opcional: {name}")
                if self.progress_callback:
                    self.progress_callback(name, i + 1, total, skipped=True)
                continue

            if name == 'inconsistency' and context.get('skip_inconsistency', False):
                context['inconsistencies'] = []
                logger.info(f"[DAG] Saltando nodo: {name}")
                if self.progress_callback:
                    self.progress_callback(name, i + 1, total, skipped=True)
                continue

            logger.info(f"[DAG] Ejecutando nodo {i+1}/{total}: {name}")
            if self.progress_callback:
                self.progress_callback(name, i + 1, total)

            start = time.time()
            context = node.func(context)
            elapsed = time.time() - start
            logger.info(f"[DAG] Nodo {name} completado en {elapsed:.1f}s")

        return context


# ============================================================
# Concurrency helper
# ============================================================

NIM_MAX_WORKERS = 2       # Hilos concurrentes para NIM (evitar 429)
NIM_STAGGER_DELAY = 0.5   # Segundos entre envios de peticiones concurrentes


def _run_concurrent_or_sequential(func, items: list, model_type: str,
                                   max_workers: int = NIM_MAX_WORKERS) -> list:
    """Ejecuta func sobre items: concurrente para NIM, secuencial para Ollama.

    Para NIM usa un ThreadPoolExecutor con max_workers limitado y un retardo
    escalonado entre envios para evitar 429 Too Many Requests.

    Args:
        func: Callable que recibe un item y devuelve resultado.
        items: Lista de items a procesar.
        model_type: 'ollama' o 'nvidia_nim'.
        max_workers: Hilos concurrentes para NIM.

    Returns:
        Lista de resultados en el mismo orden que items.
    """
    if not items:
        return []

    if model_type == 'nvidia_nim' and len(items) > 1:
        results = [None] * len(items)
        workers = min(max_workers, len(items))
        with ThreadPoolExecutor(max_workers=workers) as executor:
            future_to_idx = {}
            for idx, item in enumerate(items):
                # Stagger submissions to avoid bursts of requests
                if idx > 0:
                    time.sleep(NIM_STAGGER_DELAY)
                future_to_idx[executor.submit(func, item)] = idx
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    results[idx] = future.result()
                except Exception as e:
                    logger.error(f"Error procesando item {idx}: {e}")
                    results[idx] = None
        return results
    else:
        return [func(item) for item in items]


# ============================================================
# Node functions
# ============================================================

def _node_load_document(ctx: dict) -> dict:
    """Node 1: Carga texto del documento."""
    filepath = ctx.get('filepath')
    if not filepath:
        # Requirements already provided directly
        if 'requirements' not in ctx or not ctx['requirements']:
            raise ValueError("No se proporciono filepath ni requirements")
        return ctx

    from pipeline import load_requirements
    path = Path(filepath)
    ctx['doc_name'] = path.name
    ctx['doc_suffix'] = path.suffix.lower()

    if ctx.get('use_llm_extraction', False):
        # Read raw text for LLM extraction (Node 2 will process it)
        if path.suffix.lower() == '.csv':
            # CSV: use standard loader, no LLM extraction needed
            ctx['requirements'] = load_requirements(str(path))
            ctx['skip_extraction'] = True
        elif path.suffix.lower() == '.pdf':
            try:
                from pypdf import PdfReader
            except ImportError:
                raise ImportError("Instala pypdf: pip install pypdf")
            reader = PdfReader(str(path))
            ctx['raw_text'] = "\n".join(page.extract_text() for page in reader.pages)
            ctx['skip_extraction'] = False
        else:
            ctx['raw_text'] = path.read_text(encoding='utf-8')
            ctx['skip_extraction'] = False
    else:
        # Standard loading (regex-based)
        ctx['requirements'] = load_requirements(str(path))
        ctx['skip_extraction'] = True

    return ctx


def _node_extract_requirements(ctx: dict) -> dict:
    """Node 2: Extraccion LLM de requisitos desde texto bruto."""
    if ctx.get('skip_extraction', True):
        n = len(ctx.get('requirements', []))
        logger.info(f"[extract] Extraccion LLM saltada, {n} requisitos cargados")
        return ctx

    raw_text = ctx.get('raw_text', '')
    if not raw_text.strip():
        ctx['requirements'] = []
        return ctx

    model = ctx['model']
    model_type = ctx['model_type']

    # Split into chunks of ~3000 chars to avoid exceeding context window
    chunk_size = 3000
    chunks = []
    lines = raw_text.split('\n')
    current_chunk = []
    current_len = 0
    for line in lines:
        if current_len + len(line) > chunk_size and current_chunk:
            chunks.append('\n'.join(current_chunk))
            current_chunk = []
            current_len = 0
        current_chunk.append(line)
        current_len += len(line) + 1
    if current_chunk:
        chunks.append('\n'.join(current_chunk))

    def extract_chunk(chunk):
        prompt = build_extraction_prompt(chunk)
        resp = model.generate(prompt, max_tokens=2048)
        if resp['success']:
            return parse_extraction_response(resp['content'])
        logger.warning(f"[extract] Error en chunk: {resp['error']}")
        return []

    all_reqs = []
    chunk_results = _run_concurrent_or_sequential(extract_chunk, chunks, model_type)
    for reqs in chunk_results:
        if reqs:
            all_reqs.extend(reqs)

    # Deduplicate preserving order
    seen = set()
    unique_reqs = []
    for req in all_reqs:
        normalized = req.strip().lower()
        if normalized not in seen:
            seen.add(normalized)
            unique_reqs.append(req)

    if unique_reqs:
        ctx['requirements'] = unique_reqs
        logger.info(f"[extract] {len(unique_reqs)} requisitos extraidos por LLM desde {len(chunks)} chunks")
    else:
        # Fallback: el LLM no extrajo nada, usar parser regex
        logger.warning(f"[extract] LLM no extrajo requisitos de {len(chunks)} chunks, usando fallback regex")
        from pipeline import _parse_text_lines, _clean_pdf_artifacts
        clean_text = _clean_pdf_artifacts(raw_text)
        ctx['requirements'] = _parse_text_lines(clean_text)
        logger.info(f"[extract] Fallback regex: {len(ctx['requirements'])} requisitos")
    return ctx


def _node_combined_analysis(ctx: dict) -> dict:
    """Node 3: Analisis combinado (4 tareas en 1 llamada LLM)."""
    requirements = ctx.get('requirements', [])
    if not requirements:
        ctx['analysis_rows'] = []
        return ctx

    model = ctx['model']
    model_type = ctx['model_type']
    strategy = ctx['strategy']

    def analyze_one(req):
        try:
            prompt = build_combined_prompt(strategy, req)
            resp = model.generate(prompt, max_tokens=1024)

            if resp['success']:
                parsed = parse_combined_response(resp['content'])
                # Check if classification failed — fallback to individual calls
                if parsed['classification'] == 'INVALID':
                    logger.warning(f"[combined] Fallback a llamadas individuales para: {req[:60]}...")
                    return _fallback_individual_analysis(model, req, strategy, resp['time_seconds'])

                return {
                    'text': req,
                    'classification': parsed['classification'],
                    'is_ambiguous': parsed['is_ambiguous'],
                    'ambiguity_type': parsed['ambiguity_type'],
                    'ambiguous_words': ', '.join(parsed['ambiguous_words']),
                    'is_complete': parsed['is_complete'],
                    'missing_elements': ', '.join(parsed['missing_elements']),
                    'is_testable': parsed['is_testable'],
                    'testability_reason': parsed['testability_reason'],
                    'combined_time': resp['time_seconds'],
                }
            else:
                logger.warning(f"[combined] Error en llamada combinada: {resp['error']}")
                return _fallback_individual_analysis(model, req, strategy, 0)
        except Exception as e:
            logger.error(f"[combined] Error total analizando requisito: {e}")
            return {
                'text': req,
                'classification': 'ERROR',
                'is_ambiguous': 'ERROR',
                'ambiguity_type': '',
                'ambiguous_words': '',
                'is_complete': 'ERROR',
                'missing_elements': '',
                'is_testable': 'ERROR',
                'testability_reason': '',
                'combined_time': 0,
            }

    rows = _run_concurrent_or_sequential(analyze_one, requirements, model_type)
    # Filter out None results from errors
    ctx['analysis_rows'] = [r for r in rows if r is not None]
    return ctx


def _fallback_individual_analysis(model, req: str, strategy: str,
                                   combined_time: float) -> dict:
    """Fallback: 4 llamadas individuales cuando el prompt combinado falla."""
    from pipeline import analyze_requirement
    row = analyze_requirement(model, req, strategy)
    row['combined_time'] = combined_time
    return row


def _node_quality_score(ctx: dict) -> dict:
    """Node 4: Calculo de puntuacion de calidad (sin LLM)."""
    from pipeline import compute_quality_score
    for row in ctx.get('analysis_rows', []):
        if row.get('classification') == 'ERROR':
            row['quality_score'] = 0.0
        else:
            row['quality_score'] = compute_quality_score(row)
    return ctx


def _node_inconsistency(ctx: dict) -> dict:
    """Node 5: Deteccion de inconsistencias entre pares."""
    requirements = ctx.get('requirements', [])
    max_pairs = ctx.get('max_pairs', 50)

    if len(requirements) < 2:
        ctx['inconsistencies'] = []
        return ctx

    model = ctx['model']
    model_type = ctx['model_type']
    strategy = ctx['strategy']

    pairs = list(combinations(range(len(requirements)), 2))
    if len(pairs) > max_pairs:
        rng = np.random.default_rng(42)
        indices = rng.choice(len(pairs), size=max_pairs, replace=False)
        pairs = [pairs[i] for i in indices]

    def check_pair(pair):
        i, j = pair
        prompt = build_prompt('inconsistency', strategy,
                              requirement_a=requirements[i],
                              requirement_b=requirements[j])
        resp = model.generate(prompt, max_tokens=512)
        if resp['success']:
            parsed = parse_response('inconsistency', resp['content'])
            if parsed['is_inconsistent']:
                return {
                    'req_a_idx': i + 1,
                    'req_b_idx': j + 1,
                    'req_a': requirements[i],
                    'req_b': requirements[j],
                    'description': parsed['description'],
                }
        return None

    results = _run_concurrent_or_sequential(check_pair, pairs, model_type)
    ctx['inconsistencies'] = [r for r in results if r is not None]
    return ctx


def _node_report(ctx: dict) -> dict:
    """Node 6: Generacion de informe (sin LLM)."""
    rows = ctx.get('analysis_rows', [])
    if not rows:
        ctx['results_df'] = pd.DataFrame()
        return ctx

    ctx['results_df'] = pd.DataFrame(rows)

    # Save to disk if output_dir provided
    output_dir = ctx.get('output_dir')
    if output_dir:
        from pipeline import save_pipeline_results
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        doc_name = ctx.get('doc_name', 'documento')
        model_key = ctx.get('model_key', 'unknown')
        strategy = ctx.get('strategy', 'unknown')
        run_dir = save_pipeline_results(
            ctx['results_df'], ctx.get('inconsistencies', []),
            model_key, strategy, doc_name, output_path
        )
        ctx['run_dir'] = str(run_dir)

    return ctx


def _node_rewriting(ctx: dict) -> dict:
    """Node 7: Reescritura de requisitos con problemas (LLM, opcional)."""
    rows = ctx.get('analysis_rows', [])
    if not rows:
        ctx['corrections'] = []
        return ctx

    model = ctx['model']
    model_type = ctx['model_type']

    def rewrite_one(row):
        has_problems = (
            row.get('is_ambiguous') is True or
            row.get('is_complete') is False or
            row.get('is_testable') is False
        )
        if not has_problems:
            return {'original': row['text'], 'corrected': row['text'], 'changes_made': []}

        from pipeline import rewrite_requirement
        return rewrite_requirement(model, row['text'], row)

    corrections = _run_concurrent_or_sequential(rewrite_one, rows, model_type)
    ctx['corrections'] = [c for c in corrections if c is not None]
    return ctx


# ============================================================
# DAG builder
# ============================================================

def _build_dag_nodes() -> list[DAGNode]:
    """Construye los 7 nodos del DAG con sus dependencias."""
    return [
        DAGNode(name='load_document', func=_node_load_document, depends_on=[]),
        DAGNode(name='extract_requirements', func=_node_extract_requirements, depends_on=['load_document']),
        DAGNode(name='combined_analysis', func=_node_combined_analysis, depends_on=['extract_requirements']),
        DAGNode(name='inconsistency', func=_node_inconsistency, depends_on=['extract_requirements']),
        DAGNode(name='quality_score', func=_node_quality_score, depends_on=['combined_analysis']),
        DAGNode(name='rewriting', func=_node_rewriting, depends_on=['quality_score']),
        DAGNode(name='report', func=_node_report, depends_on=['quality_score', 'inconsistency']),
    ]


# ============================================================
# Entry points
# ============================================================

# Node name labels for progress reporting
NODE_LABELS = {
    'load_document': 'Cargando documento',
    'extract_requirements': 'Extrayendo requisitos (LLM)',
    'combined_analysis': 'Analisis combinado (4 tareas)',
    'quality_score': 'Calculando calidad',
    'inconsistency': 'Detectando inconsistencias',
    'report': 'Generando informe',
    'rewriting': 'Reescribiendo requisitos',
}


def run_dag_pipeline(filepath: str, model_key: str, strategy: str,
                     output_dir: Optional[str] = None,
                     use_llm_extraction: bool = True,
                     skip_inconsistency: bool = False,
                     enable_rewriting: bool = False,
                     max_pairs: int = 50,
                     progress_callback: Optional[Callable] = None) -> dict:
    """Ejecuta el pipeline DAG completo desde un archivo.

    Args:
        filepath: Ruta al documento de requisitos.
        model_key: Clave del modelo (e.g. 'llama8b', 'nim_llama70b').
        strategy: Estrategia de prompt.
        output_dir: Directorio de salida para informes.
        use_llm_extraction: Usar LLM para extraer requisitos (vs regex).
        skip_inconsistency: Saltar analisis de inconsistencias.
        enable_rewriting: Habilitar reescritura de requisitos con problemas.
        max_pairs: Maximo de pares para inconsistencias.
        progress_callback: Callback(node_name, current, total, skipped=False).

    Returns:
        Dict con results_df, inconsistencies, corrections, etc.
    """
    config = MODEL_CONFIGS[model_key]
    model = get_model(config)

    context = {
        'filepath': filepath,
        'model': model,
        'model_key': model_key,
        'model_type': config['type'],
        'strategy': strategy,
        'output_dir': output_dir,
        'use_llm_extraction': use_llm_extraction,
        'skip_inconsistency': skip_inconsistency,
        'enable_rewriting': enable_rewriting,
        'max_pairs': max_pairs,
    }

    nodes = _build_dag_nodes()
    executor = DAGExecutor(nodes, progress_callback=progress_callback)
    return executor.execute(context)


def run_dag_pipeline_from_requirements(requirements: list[str], model_key: str,
                                        strategy: str,
                                        output_dir: Optional[str] = None,
                                        skip_inconsistency: bool = False,
                                        enable_rewriting: bool = False,
                                        max_pairs: int = 50,
                                        doc_name: str = 'documento',
                                        progress_callback: Optional[Callable] = None) -> dict:
    """Ejecuta el pipeline DAG desde requisitos ya cargados (sin Nodes 1-2).

    Args:
        requirements: Lista de textos de requisitos.
        model_key: Clave del modelo.
        strategy: Estrategia de prompt.
        output_dir: Directorio de salida.
        skip_inconsistency: Saltar inconsistencias.
        enable_rewriting: Habilitar reescritura.
        max_pairs: Maximo de pares para inconsistencias.
        doc_name: Nombre del documento para metadatos.
        progress_callback: Callback(node_name, current, total, skipped=False).

    Returns:
        Dict con results_df, inconsistencies, corrections, etc.
    """
    config = MODEL_CONFIGS[model_key]
    model = get_model(config)

    context = {
        'requirements': requirements,
        'model': model,
        'model_key': model_key,
        'model_type': config['type'],
        'strategy': strategy,
        'output_dir': output_dir,
        'skip_inconsistency': skip_inconsistency,
        'enable_rewriting': enable_rewriting,
        'max_pairs': max_pairs,
        'doc_name': doc_name,
        'skip_extraction': True,  # requirements already provided
    }

    nodes = _build_dag_nodes()
    executor = DAGExecutor(nodes, progress_callback=progress_callback)
    return executor.execute(context)
