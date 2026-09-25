"""
agent_chercheur.py

Agent Chercheur : traduit une demande en langage naturel (français) en un
ordre CAO structuré et validé (`schemas.CADCommand`), puis le sérialise en
JSON pour l'Agent Modélisateur.

Palier 1 : reconnaît uniquement la création d'un bloc rectangulaire extrudé
sur le plan Top. L'architecture (registre de mots-clés + extracteurs de
cotes dédiés) est conçue pour qu'ajouter une forme au Palier 2 (cercle,
perçage...) se limite à enregistrer un nouveau mot-clé et un nouvel
extracteur, sans toucher au reste du pipeline.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import unicodedata
from dataclasses import dataclass

from schemas import CADCommand, Dimensions, PlaneType, SketchType


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

    depth_match = _RE_EXTRUDE_DEPTH.search(normalized_text)

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
    """Interprète une demande en langage naturel et produit un `CADCommand`."""

    def interpret(self, request: str) -> CADCommand:
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
        command = agent.interpret(args.request)
    except InterpretationError as exc:
        print(f"[agent_chercheur] Erreur d'interprétation : {exc}", file=sys.stderr)
        sys.exit(1)

    payload = command.model_dump(mode="json")

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)

    print(json.dumps(payload, indent=2, ensure_ascii=False))
    print(f"\n[agent_chercheur] Ordre CAO écrit dans {args.output}", file=sys.stderr)


if __name__ == "__main__":
    main()
