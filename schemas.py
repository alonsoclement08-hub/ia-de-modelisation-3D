"""
schemas.py

Contrat de données partagé entre l'Agent Chercheur et l'Agent Modélisateur.

Centraliser ce contrat dans un module unique garantit que les deux agents
"parlent" toujours le même langage : le Chercheur produit un objet
`CADCommand`, le Modélisateur consomme exactement le même objet. Pour le
Palier 2 (perçages, congés, formes complexes), il suffira d'étendre les
Enum ci-dessous et le modèle `Dimensions` sans casser la compatibilité
avec le Palier 1.
"""

from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Actions CAO que l'Agent Modélisateur sait exécuter.

    Palier 1 : uniquement la création de primitives extrudées.
    Palier 2 (à venir) : CREATE_HOLE, CREATE_FILLET, CREATE_PATTERN, ...
    """

    CREATE_PRIMITIVE = "CREATE_PRIMITIVE"


class PlaneType(str, Enum):
    """Plans de construction Onshape disponibles pour démarrer une esquisse."""

    TOP = "Top"
    FRONT = "Front"
    RIGHT = "Right"


class SketchType(str, Enum):
    """Types d'esquisses 2D supportés par l'esquisseur.

    Palier 1 : rectangle uniquement.
    Palier 2 : ajouter ici CIRCLE, POLYGON, SLOT, etc.
    """

    RECTANGLE = "RECTANGLE"


class Dimensions(BaseModel):
    """Cotes géométriques de la pièce, exprimées en millimètres.

    Les champs `width_mm` / `height_mm` / `extrude_depth_mm` couvrent le
    Palier 1. Les champs optionnels ci-dessous sont des emplacements
    réservés pour le Palier 2, afin de ne pas devoir changer la forme du
    JSON (rétrocompatibilité) quand ces fonctionnalités arriveront.
    """

    width_mm: float = Field(..., gt=0, description="Largeur du rectangle (mm)")
    height_mm: float = Field(..., gt=0, description="Hauteur du rectangle (mm)")
    extrude_depth_mm: float = Field(..., gt=0, description="Profondeur d'extrusion (mm)")

    # --- Réservé Palier 2 ---
    hole_diameter_mm: Optional[float] = Field(default=None, gt=0)
    fillet_radius_mm: Optional[float] = Field(default=None, gt=0)


class CADCommand(BaseModel):
    """Contrat JSON strict échangé entre les deux agents.

    Exemple (Palier 1) :
        {
          "action": "CREATE_PRIMITIVE",
          "plane": "Top",
          "sketch_type": "RECTANGLE",
          "dimensions": {
            "width_mm": 8.0,
            "height_mm": 8.0,
            "extrude_depth_mm": 2.0
          }
        }
    """

    action: ActionType = ActionType.CREATE_PRIMITIVE
    plane: PlaneType = PlaneType.TOP
    sketch_type: SketchType
    dimensions: Dimensions
