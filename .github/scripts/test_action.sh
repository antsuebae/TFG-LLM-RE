#!/bin/bash
# Test local del GitHub Action de análisis de requisitos.
# Uso: .github/scripts/test_action.sh
#
# Requiere:
#   - NVIDIA_API_KEY en .env o como variable de entorno
#   - pip install openai

set -euo pipefail
cd "$(dirname "$0")/../.."

# Cargar .env si existe
if [ -f .env ]; then
    export $(grep -v '^#' .env | xargs)
fi

if [ -z "${NVIDIA_API_KEY:-}" ]; then
    echo "ERROR: NVIDIA_API_KEY no encontrada. Configúrala en .env o como variable de entorno."
    exit 1
fi

# PR body de prueba con requisitos buenos y malos
export PR_BODY="## Feature: Sistema de notificaciones

El sistema debe enviar notificaciones por email cuando un pedido cambie de estado.
The system should handle errors properly and be fast enough.
El sistema debe permitir al usuario configurar qué notificaciones recibir, con un máximo de 10 reglas activas. Si se excede el límite, mostrar error 'Límite alcanzado'.
El usuario puede desactivar las notificaciones de forma adecuada.
The system shall retry failed email deliveries up to 3 times with exponential backoff starting at 1 second.

## Notas
- Esto es un test local, no se postea nada en GitHub."

# Sin token ni PR number = solo imprime el comentario, no intenta postear
unset GITHUB_TOKEN 2>/dev/null || true
unset PR_NUMBER 2>/dev/null || true
unset REPO 2>/dev/null || true

echo "=== Test local del GitHub Action ==="
echo "Analizando 5 requisitos de prueba con NVIDIA NIM..."
echo ""

python3 .github/scripts/analyze_pr.py
