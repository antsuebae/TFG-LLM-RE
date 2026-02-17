"""
Patrones de prompt para clasificacion de requisitos
"""

PROMPTS = {
    "question_refinement": """Clasifica el siguiente requisito de software como F (Funcional) o NF (No Funcional).

Si la clasificacion no es clara, reformula mentalmente la pregunta para entender mejor el requisito antes de clasificar.

Requisito: "{requirement}"

Responde SOLO con una de estas opciones exactas:
- F (si es un requisito funcional - describe QUE debe hacer el sistema)
- NF (si es un requisito no funcional - describe COMO debe comportarse el sistema)

Clasificacion:""",

    "cognitive_verifier": """Analiza el siguiente requisito de software paso a paso:

Requisito: "{requirement}"

Pasos de analisis:
1. Identifica el sujeto principal del requisito
2. Determina si describe una FUNCION (que hace) o una CUALIDAD (como lo hace)
3. Considera: Las funciones son acciones concretas, las cualidades son caracteristicas del sistema

Basandote en tu analisis, clasifica como:
- F (Funcional): describe una funcion o capacidad especifica
- NF (No Funcional): describe rendimiento, seguridad, usabilidad, u otra cualidad

Clasificacion (responde solo F o NF):""",

    "persona_context": """Eres un ingeniero de requisitos senior con 10 anos de experiencia en IEEE 830 e ISO 29148.

Tu tarea es clasificar requisitos de software en dos categorias:
- F (Funcional): Requisitos que describen QUE debe hacer el sistema (funciones, caracteristicas, capacidades)
- NF (No Funcional): Requisitos que describen COMO debe ser el sistema (rendimiento, seguridad, usabilidad, mantenibilidad, etc.)

Requisito a clasificar: "{requirement}"

Como experto, proporciona tu clasificacion (responde unicamente F o NF):"""
}


def build_prompt(pattern: str, requirement: str) -> str:
    """Construye el prompt segun el patron seleccionado."""
    if pattern not in PROMPTS:
        raise ValueError(f"Patron desconocido: {pattern}. Opciones: {list(PROMPTS.keys())}")
    return PROMPTS[pattern].format(requirement=requirement)


def parse_response(response: str) -> str:
    """Extrae la clasificacion (F o NF) de la respuesta del modelo."""
    response = response.strip().upper()

    # Buscar F o NF en la respuesta
    if response in ["F", "NF"]:
        return response

    # Buscar patrones comunes
    if "NF" in response or "NO FUNCIONAL" in response or "NON-FUNCTIONAL" in response:
        return "NF"
    if response.startswith("F") or "FUNCIONAL" in response or "FUNCTIONAL" in response:
        if "NO" not in response and "NON" not in response:
            return "F"

    # Si no se puede determinar
    return "INVALID"
