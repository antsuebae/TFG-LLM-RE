"""
Pipeline principal de experimentacion para tareas de ingenieria de requisitos.

Soporta 5 tareas (classification, ambiguity, completeness, inconsistency, testability),
5 modelos (2 locales + 3 NIM), 5 estrategias de prompt, y checkpoint/resume.
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json
import logging
import sys

from tqdm import tqdm
from dotenv import load_dotenv
load_dotenv(Path(__file__).parent.parent / ".env")

from prompts import build_prompt, parse_response, TASK_NAMES, STRATEGY_NAMES
from models import get_model
from metrics import calculate_metrics, calculate_binary_metrics, calculate_aggregate_metrics

# ── Logging ───────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)


class ExperimentRunner:
    """Ejecutor de experimentos de RE con soporte multi-tarea."""

    def __init__(self, config_path: str):
        """Inicializa el runner con la configuracion."""
        self.config_path = Path(config_path)
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        config_base = self.config_path.parent  # proyecto/config/
        self.results_dir = (config_base / self.config['output']['results_dir']).resolve()
        self.results_dir.mkdir(parents=True, exist_ok=True)

        self.checkpoints_dir = (config_base / self.config['output'].get('checkpoints_dir', '../results/checkpoints')).resolve()
        self.checkpoints_dir.mkdir(parents=True, exist_ok=True)

        logs_dir = (config_base / self.config['output'].get('logs_dir', '../results/logs')).resolve()
        logs_dir.mkdir(parents=True, exist_ok=True)

        # Add file handler for logging
        fh = logging.FileHandler(logs_dir / f"experiment_{datetime.now():%Y%m%d_%H%M%S}.log")
        fh.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s"))
        logging.getLogger().addHandler(fh)

        logger.info(f"Configuracion cargada desde: {config_path}")

    def _load_dataset(self, task: str) -> pd.DataFrame:
        """Carga el dataset correspondiente a la tarea."""
        tasks_config = self.config.get('tasks', {})

        if task in tasks_config:
            dataset_path = self.config_path.parent / tasks_config[task]['dataset']
        else:
            # Fallback to default dataset (classification)
            dataset_path = self.config_path.parent / self.config['dataset']['path']

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset no encontrado: {dataset_path}")

        df = pd.read_csv(dataset_path)
        logger.info(f"Dataset cargado para '{task}': {len(df)} registros desde {dataset_path}")
        return df

    def _get_task_config(self, task: str) -> dict:
        """Obtiene la configuracion de una tarea."""
        tasks_config = self.config.get('tasks', {})
        if task in tasks_config:
            return tasks_config[task]
        # Default config for classification (backward compat)
        return {
            "dataset": self.config['dataset']['path'],
            "sample_size": self.config['experiment']['sample_size'],
            "text_column": self.config['dataset']['text_column'],
            "label_column": self.config['dataset']['label_column'],
        }

    def _get_model_configs(self, models: Optional[List[str]] = None) -> List[dict]:
        """Obtiene configuraciones de modelos (locales + NIM)."""
        model_configs = []
        for m in self.config['models'].get('local', []):
            if models is None or m['short_name'] in models:
                model_configs.append(m)
        for m in self.config['models'].get('nvidia_nim', []):
            if models is None or m['short_name'] in models:
                model_configs.append(m)
        return model_configs

    def _get_strategies(self, strategies: Optional[List[str]] = None) -> List[str]:
        """Obtiene lista de estrategias a usar."""
        all_strategies = self.config.get('prompting_strategies',
                                         [p['name'] for p in self.config['patterns']])
        if strategies:
            return [s for s in all_strategies if s in strategies]
        return all_strategies

    def sample_data(self, df: pd.DataFrame, seed: int, n: Optional[int] = None) -> pd.DataFrame:
        """Genera muestra aleatoria reproducible. If n is None, uses full dataset."""
        if n is None or n >= len(df):
            return df
        return df.sample(n=n, random_state=seed)

    def _load_checkpoint(self, checkpoint_path: Path) -> List[Dict]:
        """Carga resultados desde checkpoint."""
        if checkpoint_path.exists():
            with open(checkpoint_path, 'r') as f:
                results = json.load(f)
            logger.info(f"Checkpoint cargado: {len(results)} resultados desde {checkpoint_path}")
            return results
        return []

    def _get_completed_configs(self, results: List[Dict]) -> set:
        """Extrae configuraciones ya completadas del checkpoint."""
        completed = set()
        for r in results:
            key = (r.get('model', ''), r.get('strategy', r.get('pattern', '')),
                   r.get('iteration', 0), r.get('seed', 0))
            completed.add(key)
        return completed

    def run_single_experiment(
        self,
        task: str,
        model_config: dict,
        strategy: str,
        sample: pd.DataFrame,
        task_config: dict,
        temperature: float = 0.4
    ) -> List[Dict]:
        """Ejecuta un experimento individual para una tarea."""
        model = get_model(model_config, temperature)
        results = []
        text_col = task_config['text_column']

        for _, row in sample.iterrows():
            # Build prompt based on task type
            if task == "inconsistency":
                prompt = build_prompt(
                    task, strategy,
                    requirement_a=row[text_col],
                    requirement_b=row.get('text_b', '')
                )
            else:
                prompt = build_prompt(task, strategy, requirement=row[text_col])

            response = model.generate(prompt, max_tokens=self.config['experiment']['max_tokens'])

            # Parse based on task
            if response['success']:
                parsed = parse_response(task, response['content'])
            else:
                parsed = "ERROR"

            # Build result row
            result = {
                'requirement_id': row.get('id', row.name),
                'requirement_text': row[text_col],
                'raw_response': response['content'],
                'time_seconds': response['time_seconds'],
                'tokens_per_second': response.get('tokens_per_second', 0),
                'success': response['success'],
                'error': response.get('error'),
            }

            # Task-specific fields
            if task == "classification":
                result['ground_truth'] = row[task_config['label_column']]
                result['prediction'] = parsed
            elif task == "ambiguity":
                result['ground_truth'] = row.get(task_config['label_column'], '')
                if isinstance(parsed, dict):
                    result['prediction'] = str(parsed.get('is_ambiguous', False))
                    result['ambiguity_type'] = parsed.get('ambiguity_type', 'none')
                    result['ambiguous_words'] = ','.join(parsed.get('ambiguous_words', []))
                else:
                    result['prediction'] = str(parsed)
            elif task == "completeness":
                result['ground_truth'] = row.get(task_config['label_column'], '')
                if isinstance(parsed, dict):
                    result['prediction'] = str(parsed.get('is_complete', False))
                    result['missing_elements'] = ','.join(parsed.get('missing_elements', []))
                else:
                    result['prediction'] = str(parsed)
            elif task == "inconsistency":
                result['requirement_b'] = row.get('text_b', '')
                result['ground_truth'] = row.get(task_config['label_column'], '')
                if isinstance(parsed, dict):
                    result['prediction'] = str(parsed.get('is_inconsistent', False))
                    result['conflict_description'] = parsed.get('description', 'none')
                else:
                    result['prediction'] = str(parsed)
            elif task == "testability":
                result['ground_truth'] = row.get(task_config['label_column'], '')
                if isinstance(parsed, dict):
                    result['prediction'] = str(parsed.get('is_testable', False))
                    result['reason'] = parsed.get('reason', 'vague')
                else:
                    result['prediction'] = str(parsed)

            results.append(result)

        return results

    def _compute_iteration_metrics(self, task: str, results: List[Dict]) -> Dict[str, float]:
        """Computa metricas para una iteracion segun la tarea."""
        if task == "classification":
            y_true = [r['ground_truth'] for r in results]
            y_pred = [r['prediction'] for r in results]
            return calculate_metrics(y_true, y_pred)
        else:
            # Binary tasks: compare True/False strings
            y_true = [str(r['ground_truth']).lower() == 'true' for r in results]
            y_pred = [str(r['prediction']).lower() == 'true' for r in results]
            return calculate_binary_metrics(y_true, y_pred)

    def run_full_experiment(
        self,
        task: str = "classification",
        models: Optional[List[str]] = None,
        strategies: Optional[List[str]] = None,
        dry_run: bool = False,
        resume_from: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Ejecuta el experimento completo (factorial).

        Args:
            task: Tarea a ejecutar (classification, ambiguity, etc.)
            models: Lista de modelos (None = todos)
            strategies: Lista de estrategias (None = todas)
            dry_run: Si True, usa 5 requisitos y 1 iteracion
            resume_from: Ruta a checkpoint para reanudar

        Returns:
            DataFrame con todos los resultados
        """
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_path = self.checkpoints_dir / f"checkpoint_{task}_{timestamp}.json"

        # Load checkpoint if resuming
        all_results = []
        completed_configs = set()
        if resume_from:
            resume_path = Path(resume_from)
            all_results = self._load_checkpoint(resume_path)
            completed_configs = self._get_completed_configs(all_results)
            logger.info(f"Reanudando: {len(completed_configs)} configuraciones completadas")

        # Load dataset for task
        df = self._load_dataset(task)
        task_config = self._get_task_config(task)

        # Prepare configs
        model_configs = self._get_model_configs(models)
        strategy_list = self._get_strategies(strategies)

        sample_size = 5 if dry_run else task_config.get('sample_size')
        iterations = 1 if dry_run else self.config['experiment']['iterations']
        seeds = self.config['experiment']['seeds'][:iterations]

        total_configs = len(model_configs) * len(strategy_list) * iterations
        skipped = 0

        logger.info("=" * 60)
        logger.info(f"Tarea: {task}")
        logger.info(f"Modelos: {[m['short_name'] for m in model_configs]}")
        logger.info(f"Estrategias: {strategy_list}")
        logger.info(f"Iteraciones: {iterations}, Muestra: {sample_size or 'completo'}")
        logger.info(f"Total configuraciones: {total_configs}")
        logger.info("=" * 60)

        pbar = tqdm(total=total_configs, desc=f"[{task}]", unit="config")

        for model_config in model_configs:
            for strategy in strategy_list:
                for iter_idx, seed in enumerate(seeds):
                    config_key = (model_config['short_name'], strategy, iter_idx + 1, seed)

                    # Skip if already completed (resume)
                    if config_key in completed_configs:
                        skipped += 1
                        pbar.update(1)
                        pbar.set_postfix(skipped=skipped)
                        continue

                    desc = f"{model_config['short_name']}+{strategy} i{iter_idx+1}"
                    pbar.set_description(f"[{task}] {desc}")

                    sample = self.sample_data(df, seed, sample_size)
                    results = self.run_single_experiment(
                        task, model_config, strategy, sample,
                        task_config, self.config['experiment']['temperature']
                    )

                    # Add metadata
                    for r in results:
                        r['model'] = model_config['short_name']
                        r['strategy'] = strategy
                        r['iteration'] = iter_idx + 1
                        r['seed'] = seed
                        r['task'] = task

                    all_results.extend(results)

                    # Compute and log metrics
                    metrics = self._compute_iteration_metrics(task, results)
                    avg_tps = np.mean([r.get('tokens_per_second', 0) for r in results])
                    logger.info(
                        f"{desc}: F1={metrics['f1']:.3f} Acc={metrics['accuracy']:.3f} "
                        f"Invalid={metrics.get('invalid_rate', 0):.1%} TPS={avg_tps:.1f}"
                    )

                    # Save checkpoint
                    self._save_checkpoint(all_results, checkpoint_path)
                    pbar.update(1)

        pbar.close()

        if skipped:
            logger.info(f"Configuraciones saltadas (ya completadas): {skipped}")

        # Save final results
        results_df = pd.DataFrame(all_results)
        output_path = self.results_dir / f"results_{task}_{timestamp}.csv"
        results_df.to_csv(output_path, index=False)
        logger.info(f"Resultados guardados en: {output_path}")

        # Remove checkpoint now that final results are saved
        if checkpoint_path.exists():
            checkpoint_path.unlink()
            logger.info(f"Checkpoint eliminado: {checkpoint_path}")

        return results_df

    def _save_checkpoint(self, results: List[Dict], checkpoint_path: Path):
        """Guarda checkpoint de resultados."""
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f, default=str)

    def generate_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Genera resumen de metricas agregadas."""
        summary_rows = []
        task = results_df['task'].iloc[0] if 'task' in results_df.columns else 'classification'
        strategy_col = 'strategy' if 'strategy' in results_df.columns else 'pattern'

        for model in results_df['model'].unique():
            for strategy in results_df[strategy_col].unique():
                subset = results_df[
                    (results_df['model'] == model) & (results_df[strategy_col] == strategy)
                ]

                metrics_per_iter = []
                for iteration in subset['iteration'].unique():
                    iter_data = subset[subset['iteration'] == iteration]
                    metrics = self._compute_iteration_metrics(
                        task,
                        iter_data.to_dict('records')
                    )
                    metrics_per_iter.append(metrics)

                agg = calculate_aggregate_metrics(metrics_per_iter)
                agg['model'] = model
                agg['strategy'] = strategy
                agg['task'] = task
                summary_rows.append(agg)

        summary_df = pd.DataFrame(summary_rows)
        cols = ['task', 'model', 'strategy', 'f1_mean', 'f1_std',
                'accuracy_mean', 'accuracy_std', 'precision_mean', 'recall_mean']
        available_cols = [c for c in cols if c in summary_df.columns]
        return summary_df[available_cols]


def main():
    """Punto de entrada principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline de experimentacion LLM-RE')
    parser.add_argument('--config', default=str(Path(__file__).parent.parent / 'config/experiment_config.yaml'),
                        help='Ruta al archivo de configuracion')
    parser.add_argument('--task', default='classification',
                        choices=TASK_NAMES,
                        help='Tarea a ejecutar')
    parser.add_argument('--dry-run', action='store_true',
                        help='Ejecutar prueba rapida (5 requisitos, 1 iteracion)')
    parser.add_argument('--models', nargs='+',
                        help='Modelos a evaluar (ej: qwen7b llama8b)')
    parser.add_argument('--strategies', nargs='+',
                        help='Estrategias a usar (ej: few_shot chain_of_thought)')
    parser.add_argument('--resume', type=str, default=None,
                        help='Ruta a checkpoint JSON para reanudar')

    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    results = runner.run_full_experiment(
        task=args.task,
        models=args.models,
        strategies=args.strategies,
        dry_run=args.dry_run,
        resume_from=args.resume
    )

    summary = runner.generate_summary(results)
    print("\n" + "=" * 60)
    print("RESUMEN DE RESULTADOS")
    print("=" * 60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
