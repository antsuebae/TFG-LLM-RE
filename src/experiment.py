"""
Pipeline principal de experimentacion
"""

import os
import yaml
import pandas as pd
import numpy as np
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional
import json

from prompts import build_prompt, parse_response, PROMPTS
from models import get_model
from metrics import calculate_metrics, calculate_aggregate_metrics


class ExperimentRunner:
    """Ejecutor de experimentos de clasificacion de requisitos."""

    def __init__(self, config_path: str):
        """Inicializa el runner con la configuracion."""
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)

        self.results_dir = Path(self.config['output']['results_dir'])
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Cargar dataset
        dataset_path = Path(config_path).parent / self.config['dataset']['path']
        self.df = pd.read_csv(dataset_path)
        print(f"Dataset cargado: {len(self.df)} requisitos")

    def sample_data(self, seed: int, n: int) -> pd.DataFrame:
        """Genera muestra aleatoria reproducible."""
        return self.df.sample(n=n, random_state=seed)

    def run_single_experiment(
        self,
        model_config: dict,
        pattern: str,
        sample: pd.DataFrame,
        temperature: float = 0.4
    ) -> List[Dict]:
        """Ejecuta un experimento individual."""
        model = get_model(model_config, temperature)
        results = []

        for idx, row in sample.iterrows():
            prompt = build_prompt(pattern, row[self.config['dataset']['text_column']])
            response = model.generate(prompt, max_tokens=self.config['experiment']['max_tokens'])

            prediction = parse_response(response['content']) if response['success'] else "ERROR"

            results.append({
                'requirement_id': row['id'] if 'id' in row else idx,
                'requirement_text': row[self.config['dataset']['text_column']],
                'ground_truth': row[self.config['dataset']['label_column']],
                'prediction': prediction,
                'raw_response': response['content'],
                'time_seconds': response['time_seconds'],
                'success': response['success'],
                'error': response.get('error')
            })

        return results

    def run_full_experiment(
        self,
        models: Optional[List[str]] = None,
        patterns: Optional[List[str]] = None,
        dry_run: bool = False
    ) -> pd.DataFrame:
        """
        Ejecuta el experimento completo (factorial).

        Args:
            models: Lista de modelos a evaluar (None = todos)
            patterns: Lista de patrones a usar (None = todos)
            dry_run: Si True, usa solo 5 requisitos por prueba

        Returns:
            DataFrame con todos los resultados
        """
        all_results = []
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Preparar modelos
        model_configs = []
        for m in self.config['models']['local']:
            if models is None or m['short_name'] in models:
                model_configs.append(m)
        for m in self.config['models'].get('api', []):
            if models is None or m['short_name'] in models:
                model_configs.append(m)

        # Preparar patrones
        pattern_list = [p['name'] for p in self.config['patterns']]
        if patterns:
            pattern_list = [p for p in pattern_list if p in patterns]

        sample_size = 5 if dry_run else self.config['experiment']['sample_size']
        iterations = 1 if dry_run else self.config['experiment']['iterations']
        seeds = self.config['experiment']['seeds'][:iterations]

        total_configs = len(model_configs) * len(pattern_list) * iterations
        current = 0

        print(f"\n{'='*60}")
        print(f"Iniciando experimento: {len(model_configs)} modelos x {len(pattern_list)} patrones x {iterations} iteraciones")
        print(f"Total configuraciones: {total_configs}")
        print(f"Requisitos por config: {sample_size}")
        print(f"{'='*60}\n")

        for model_config in model_configs:
            for pattern in pattern_list:
                for iter_idx, seed in enumerate(seeds):
                    current += 1
                    print(f"[{current}/{total_configs}] {model_config['short_name']} + {pattern} (iter {iter_idx+1}, seed {seed})")

                    sample = self.sample_data(seed, sample_size)
                    results = self.run_single_experiment(
                        model_config,
                        pattern,
                        sample,
                        self.config['experiment']['temperature']
                    )

                    # Agregar metadatos
                    for r in results:
                        r['model'] = model_config['short_name']
                        r['pattern'] = pattern
                        r['iteration'] = iter_idx + 1
                        r['seed'] = seed

                    all_results.extend(results)

                    # Calcular metricas parciales
                    y_true = [r['ground_truth'] for r in results]
                    y_pred = [r['prediction'] for r in results]
                    metrics = calculate_metrics(y_true, y_pred)
                    print(f"   -> F1: {metrics['f1']:.3f}, Acc: {metrics['accuracy']:.3f}, Invalid: {metrics['invalid_rate']:.1%}")

                    # Guardar checkpoint
                    self._save_checkpoint(all_results, timestamp)

        # Guardar resultados finales
        results_df = pd.DataFrame(all_results)
        output_path = self.results_dir / f"results_{timestamp}.csv"
        results_df.to_csv(output_path, index=False)
        print(f"\nResultados guardados en: {output_path}")

        return results_df

    def _save_checkpoint(self, results: List[Dict], timestamp: str):
        """Guarda checkpoint de resultados."""
        checkpoint_path = self.results_dir / f"checkpoint_{timestamp}.json"
        with open(checkpoint_path, 'w') as f:
            json.dump(results, f)

    def generate_summary(self, results_df: pd.DataFrame) -> pd.DataFrame:
        """Genera resumen de metricas agregadas."""
        summary_rows = []

        for model in results_df['model'].unique():
            for pattern in results_df['pattern'].unique():
                subset = results_df[(results_df['model'] == model) & (results_df['pattern'] == pattern)]

                metrics_per_iter = []
                for iteration in subset['iteration'].unique():
                    iter_data = subset[subset['iteration'] == iteration]
                    y_true = iter_data['ground_truth'].tolist()
                    y_pred = iter_data['prediction'].tolist()
                    metrics_per_iter.append(calculate_metrics(y_true, y_pred))

                agg = calculate_aggregate_metrics(metrics_per_iter)
                agg['model'] = model
                agg['pattern'] = pattern
                summary_rows.append(agg)

        summary_df = pd.DataFrame(summary_rows)
        return summary_df[['model', 'pattern', 'f1_mean', 'f1_std', 'accuracy_mean', 'accuracy_std',
                          'precision_mean', 'recall_mean']]


def main():
    """Punto de entrada principal."""
    import argparse

    parser = argparse.ArgumentParser(description='Pipeline de experimentacion LLM-RE')
    parser.add_argument('--config', default='../config/experiment_config.yaml', help='Ruta al archivo de configuracion')
    parser.add_argument('--dry-run', action='store_true', help='Ejecutar prueba rapida (5 requisitos)')
    parser.add_argument('--models', nargs='+', help='Modelos a evaluar (ej: qwen7b llama8b)')
    parser.add_argument('--patterns', nargs='+', help='Patrones a usar (ej: question_refinement)')

    args = parser.parse_args()

    runner = ExperimentRunner(args.config)
    results = runner.run_full_experiment(
        models=args.models,
        patterns=args.patterns,
        dry_run=args.dry_run
    )

    summary = runner.generate_summary(results)
    print("\n" + "="*60)
    print("RESUMEN DE RESULTADOS")
    print("="*60)
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
