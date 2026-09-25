"""
agent_chercheur.py

Agent Chercheur : traduit une demande en langage naturel (français) en un
ordre CAO structuré et validé, puis le sérialise en JSON pour l'Agent
Modélisateur.

- `interpret()` (Palier 1) : reconnaît la création d'un bloc rectangulaire
  extrudé simple et produit une `schemas.CADCommand`. Conservée telle
  quelle pour compatibilité.
- `interpret_plan()` (Palier 2) : reconnaît une demande combinant plusieurs
  opérations (bloc + trou + congé...) et produit une `schemas.CADPlan`
  (séquence ordonnée d'étapes). C'est la méthode utilisée par la CLI.

L'architecture (registre de mots-clés + extracteurs de cotes dédiés) est
conçue pour qu'ajouter une nouvelle opération au Palier 3 se limite à
enregistrer un nouvel extracteur et à l'ajouter dans `interpret_plan`,
sans toucher au reste du pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass

from schemas import (
    ActionType,
    CADCommand,
    CADPlan,
    CADStep,
    Dimensions,
    PlaneType,
    SketchType,
    StepParams,
)


class InterpretationError(ValueError):
    """Levée quand la demande en langage naturel ne peut pas être comprise."""


def _normalize(text: str) -> str:
    """Met le texte en minuscules et retire les accents.

    Simplifie l'écriture des regex ci-dessous (plus besoin de gérer
    "côté"/"cote", "épaisseur"/"epaisseur", etc.).
    """
    decomposed = unicodedata.normalize("NFKD", text)
    without_accents = "".join(c for c in decomposed if not unicodedata.combining(c))
    return without_accents.lower()


def _to_float(raw: str) -> float:
    """Convertit une valeur numérique en tolérant la virgule décimale française."""
    return float(raw.replace(",", "."))


# ---------------------------------------------------------------------------
# Extraction des cotes (largeur / hauteur / profondeur d'extrusion)
# ---------------------------------------------------------------------------

_NUM = r"(\d+(?:[.,]\d+)?)"

# "8x8 mm" / "8 x 8 mm" -> un côté carré donnant largeur ET hauteur.
_RE_SQUARE_SIDES = re.compile(rf"{_NUM}\s*x\s*{_NUM}\s*mm")

# "cote de 8 mm" / "cotes de 8 mm" (isolé, sans "8x8") -> largeur = hauteur.
_RE_SINGLE_SIDE = re.compile(rf"cote(?:s)?\s+(?:de\s+)?{_NUM}\s*mm")

# "largeur de 8 mm"
_RE_WIDTH = re.compile(rf"larg(?:eur)?\s+(?:de\s+)?{_NUM}\s*mm")

# "longueur de 8 mm"
_RE_LENGTH = re.compile(rf"long(?:ueur)?\s+(?:de\s+)?{_NUM}\s*mm")

# "2 mm de hauteur" / "2 mm de profondeur" / "2 mm d'epaisseur"
_RE_EXTRUDE_DEPTH = re.compile(rf"{_NUM}\s*mm\s+d[e']\s*(?:hauteur|profondeur|epaisseur)")

# "epaisseur 10 mm" / "epaisseur de 10 mm" (ordre inverse du pattern ci-dessus)
_RE_THICKNESS_WORD_FIRST = re.compile(rf"(?:epaisseur|hauteur|profondeur)\s+(?:de\s+)?{_NUM}\s*mm")

# "trou central de 37 mm" / "trou de 37mm" / "percage de 37 mm de diametre"
_RE_HOLE_DIAMETER = re.compile(rf"trou\w*[^.]*?{_NUM}\s*mm|percage\w*[^.]*?{_NUM}\s*mm")

# "conge de 5 mm" / "arrondi de 5 mm sur les coins"
_RE_FILLET_RADIUS = re.compile(rf"(?:conge|arrondi)\w*[^.]*?{_NUM}\s*mm")


def extract_dimensions(normalized_text: str) -> Dimensions:
    """Extrait largeur / hauteur / profondeur d'extrusion d'un texte normalisé.

    Priorité : un couple "AxB mm" explicite prime sur un côté isolé, qui
    prime lui-même sur largeur/longueur séparées.
    """
    width_mm: float | None = None
    height_mm: float | None = None

    if match := _RE_SQUARE_SIDES.search(normalized_text):
        width_mm = _to_float(match.group(1))
        height_mm = _to_float(match.group(2))
    elif match := _RE_SINGLE_SIDE.search(normalized_text):
        width_mm = height_mm = _to_float(match.group(1))
    else:
        if match := _RE_WIDTH.search(normalized_text):
            width_mm = _to_float(match.group(1))
        if match := _RE_LENGTH.search(normalized_text):
            height_mm = _to_float(match.group(1))
        # Pas de longueur distincte mentionnée : on suppose une base carrée.
        if width_mm is not None and height_mm is None:
            height_mm = width_mm

    depth_match = _RE_EXTRUDE_DEPTH.search(normalized_text) or _RE_THICKNESS_WORD_FIRST.search(normalized_text)

    if width_mm is None or height_mm is None:
        raise InterpretationError(
            "Impossible d'extraire la largeur/hauteur de l'esquisse depuis la demande."
        )
    if depth_match is None:
        raise InterpretationError(
            "Impossible d'extraire la profondeur d'extrusion depuis la demande."
        )

    return Dimensions(
        width_mm=width_mm,
        height_mm=height_mm,
        extrude_depth_mm=_to_float(depth_match.group(1)),
    )


def extract_hole_diameter(normalized_text: str) -> float | None:
    """Extrait le diamètre d'un trou/perçage mentionné dans la demande (Palier 2)."""
    match = _RE_HOLE_DIAMETER.search(normalized_text)
    if not match:
        return None
    return _to_float(match.group(1) or match.group(2))


def extract_fillet_radius(normalized_text: str) -> float | None:
    """Extrait le rayon de congé mentionné dans la demande (Palier 2)."""
    match = _RE_FILLET_RADIUS.search(normalized_text)
    return _to_float(match.group(1)) if match else None


# ---------------------------------------------------------------------------
# Reconnaissance de la forme et du plan
# ---------------------------------------------------------------------------

# Registre mot-clé -> type d'esquisse. Palier 2 : ajouter par ex.
# {"cylindre": SketchType.CIRCLE, "trou": SketchType.CIRCLE, ...}
_SHAPE_KEYWORDS: dict[str, SketchType] = {
    "bloc": SketchType.RECTANGLE,
    "cube": SketchType.RECTANGLE,
    "pave": SketchType.RECTANGLE,
    "rectangle": SketchType.RECTANGLE,
    "carre": SketchType.RECTANGLE,
    "plaque": SketchType.RECTANGLE,
}

_PLANE_KEYWORDS: dict[str, PlaneType] = {
    "top": PlaneType.TOP,
    "dessus": PlaneType.TOP,
    "front": PlaneType.FRONT,
    "face": PlaneType.FRONT,
    "right": PlaneType.RIGHT,
    "droit": PlaneType.RIGHT,
}


def detect_sketch_type(normalized_text: str) -> SketchType:
    for keyword, sketch_type in _SHAPE_KEYWORDS.items():
        if keyword in normalized_text:
            return sketch_type
    raise InterpretationError(
        "Forme non reconnue. Palier 1 ne supporte que les blocs/rectangles."
    )


def detect_plane(normalized_text: str) -> PlaneType:
    for keyword, plane in _PLANE_KEYWORDS.items():
        if f"plan {keyword}" in normalized_text:
            return plane
    return PlaneType.TOP  # Plan par défaut si non précisé.


# ---------------------------------------------------------------------------
# Agent Chercheur
# ---------------------------------------------------------------------------


@dataclass
class AgentChercheur:
    """Interprète une demande en langage naturel et produit un ordre CAO."""

    def interpret(self, request: str) -> CADCommand:
        """Palier 1 : une pièce simple (bloc rectangulaire extrudé) en une seule `CADCommand`."""
        if not request or not request.strip():
            raise InterpretationError("La demande est vide.")

        normalized_text = _normalize(request)

        sketch_type = detect_sketch_type(normalized_text)
        plane = detect_plane(normalized_text)
        dimensions = extract_dimensions(normalized_text)

        return CADCommand(
            plane=plane,
            sketch_type=sketch_type,
            dimensions=dimensions,
        )

    def interpret_plan(self, request: str) -> CADPlan:
        """Palier 2 : une demande combinant plusieurs opérations en une `CADPlan`.

        Construit toujours une base (esquisse + rectangle + extrusion), puis
        ajoute un trou (esquisse + cercle + extrusion enlèvement) et/ou un
        congé si la demande les mentionne. Pour ajouter une opération,
        ajouter son extracteur dédié et les `CADStep` correspondantes ici.
        """
        if not request or not request.strip():
            raise InterpretationError("La demande est vide.")

        normalized_text = _normalize(request)

        sketch_type = detect_sketch_type(normalized_text)
        plane = detect_plane(normalized_text)
        base_dims = extract_dimensions(normalized_text)

        steps: list[CADStep] = [
            CADStep(action=ActionType.CREATE_SKETCH, plane=plane),
            CADStep(
                action=ActionType.DRAW_RECTANGLE,
                sketch_type=sketch_type,
                params=StepParams(width_mm=base_dims.width_mm, height_mm=base_dims.height_mm),
            ),
            CADStep(action=ActionType.EXTRUDE_ADD, params=StepParams(depth_mm=base_dims.extrude_depth_mm)),
        ]

        if (hole_diameter := extract_hole_diameter(normalized_text)) is not None:
            steps.append(CADStep(action=ActionType.CREATE_SKETCH, plane=plane))
            steps.append(
                CADStep(
                    action=ActionType.DRAW_CIRCLE,
                    sketch_type=SketchType.CIRCLE,
                    params=StepParams(diameter_mm=hole_diameter),
                )
            )
            steps.append(CADStep(action=ActionType.EXTRUDE_REMOVE, params=StepParams(through_all=True)))

        if (fillet_radius := extract_fillet_radius(normalized_text)) is not None:
            steps.append(CADStep(action=ActionType.APPLY_FILLET, params=StepParams(radius_mm=fillet_radius)))

        return CADPlan(steps=steps)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Chercheur : traduit une demande en langage naturel en ordre CAO JSON.",
    )
    parser.add_argument("request", help="Demande en langage naturel, ex: 'Crée un bloc de 8x8 mm de côté et 2 mm de hauteur'")
    parser.add_argument(
        "-o", "--output",
        default="command.json",
        help="Chemin du fichier JSON de sortie (défaut: command.json)",
    )
    args = parser.parse_args()

    agent = AgentChercheur()
    try:
        plan = agent.interpret_plan(args.request)
    except InterpretationError as exc:
        print(f"[agent_chercheur] Erreur d'interprétation : {exc}", file=sys.stderr)
        sys.exit(1)

    payload = plan.model_dump(mode="json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[agent_chercheur] Plan CAO ({len(plan.steps)} étape(s)) écrit dans {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
