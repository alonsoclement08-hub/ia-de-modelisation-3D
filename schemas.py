"""
schemas.py

Contrat de données partagé entre l'Agent Chercheur et l'Agent Modélisateur.

Centraliser ce contrat dans un module unique garantit que les deux agents
"parlent" toujours le même langage. Deux formats cohabitent :

- `CADCommand` (Palier 1) : une pièce simple, un seul bloc extrudé. Ce
  modèle n'est plus modifié, pour rester lisible et 100% stable.
- `CADPlan` (Palier 2) : une séquence ordonnée de `CADStep`, pour les
  pièces combinant plusieurs opérations (esquisse, perçage, congé...).
  `agent_modelisateur.py` accepte les deux formats en entrée (voir
  `load_plan`) : un `command.json` Palier 1 existant continue de
  fonctionner sans modification.
"""

from __future__ import annotations

from enum import Enum

from pydantic import BaseModel, Field


class ActionType(str, Enum):
    """Actions CAO que l'Agent Modélisateur sait exécuter.

    CREATE_PRIMITIVE est le format Palier 1 (une pièce = une action).
    Les autres valeurs sont les étapes unitaires d'un `CADPlan` (Palier 2/3).
    Voir `agent_modelisateur.TOOL_BINDINGS` pour le raccourci ou le
    fallback de recherche (Alt+C) associé à chacune, et
    `agent_modelisateur.CONTEXT_VALID_ACTIONS` pour le contexte Onshape
    (esquisse / volume 3D / ...) dans lequel elle est valide.
    """

    CREATE_PRIMITIVE = "CREATE_PRIMITIVE"

    CREATE_SKETCH = "CREATE_SKETCH"
    DRAW_RECTANGLE = "DRAW_RECTANGLE"
    DRAW_CIRCLE = "DRAW_CIRCLE"

    EXTRUDE_ADD = "EXTRUDE_ADD"
    EXTRUDE_REMOVE = "EXTRUDE_REMOVE"
    REVOLVE = "REVOLVE"
    SWEEP = "SWEEP"
    LOFT = "LOFT"

    APPLY_FILLET = "APPLY_FILLET"
    APPLY_CHAMFER = "APPLY_CHAMFER"
    APPLY_SHELL = "APPLY_SHELL"
    APPLY_DRAFT = "APPLY_DRAFT"
    APPLY_HOLE_FEATURE = "APPLY_HOLE_FEATURE"
    APPLY_MIRROR = "APPLY_MIRROR"


class InterfaceContext(str, Enum):
    """États (modes) de l'interface Onshape.

    Certaines actions ne sont valides que dans un contexte précis : on ne
    peut pas dessiner un rectangle hors d'une esquisse active, ni extruder
    tant que l'esquisse n'a pas été validée. GLOBAL est le mode par défaut
    du Part Studio (rien d'actif) ; SKETCH est actif entre la création
    d'une esquisse et sa validation ; FEATURE_3D une fois qu'on a un corps
    3D, où toutes les opérations de volume ET de finition (congé,
    chanfrein, coque...) sont disponibles — Onshape n'a pas de mode
    "finition" séparé dans son UI, contrairement à SKETCH ; FINISHING
    reste ici pour le vocabulaire mais `agent_modelisateur` la traite
    comme un sous-ensemble de FEATURE_3D (voir CONTEXT_VALID_ACTIONS).
    """

    GLOBAL = "GLOBAL"
    SKETCH = "SKETCH"
    FEATURE_3D = "FEATURE_3D"
    FINISHING = "FINISHING"


class PlaneType(str, Enum):
    """Plans de construction Onshape disponibles pour démarrer une esquisse."""

    TOP = "Top"
    FRONT = "Front"
    RIGHT = "Right"


class SketchType(str, Enum):
    """Types d'esquisses 2D supportés par l'esquisseur.

    Palier 3 potentiel : POLYGON, SLOT, etc.
    """

    RECTANGLE = "RECTANGLE"
    CIRCLE = "CIRCLE"


class Dimensions(BaseModel):
    """Cotes géométriques d'une pièce Palier 1 (bloc rectangulaire extrudé), en mm."""

    width_mm: float = Field(..., gt=0, description="Largeur du rectangle (mm)")
    height_mm: float = Field(..., gt=0, description="Hauteur du rectangle (mm)")
    extrude_depth_mm: float = Field(..., gt=0, description="Profondeur d'extrusion (mm)")


class CADCommand(BaseModel):
    """Contrat JSON Palier 1 : une pièce simple, une seule action.

    Exemple :
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


class StepParams(BaseModel):
    """Paramètres numériques d'une étape de `CADPlan`.

    Tous les champs sont optionnels : chaque `ActionType` n'en utilise
    qu'un sous-ensemble (ex. APPLY_FILLET n'a besoin que de `radius_mm`).
    Les handlers de l'Agent Modélisateur valident eux-mêmes que les
    champs requis pour leur action sont bien présents.
    """

    width_mm: float | None = Field(default=None, gt=0, description="Largeur (DRAW_RECTANGLE)")
    height_mm: float | None = Field(default=None, gt=0, description="Hauteur (DRAW_RECTANGLE)")
    diameter_mm: float | None = Field(default=None, gt=0, description="Diamètre (DRAW_CIRCLE/APPLY_HOLE_FEATURE)")
    depth_mm: float | None = Field(default=None, gt=0, description="Profondeur (EXTRUDE_ADD/REMOVE)")
    through_all: bool = Field(default=False, description="Extrusion traversante (EXTRUDE_REMOVE)")
    radius_mm: float | None = Field(
        default=None, gt=0, description="Rayon de congé (APPLY_FILLET) ou distance de chanfrein (APPLY_CHAMFER)"
    )
    angle_deg: float | None = Field(default=None, description="Angle en degrés (REVOLVE/APPLY_DRAFT)")
    thickness_mm: float | None = Field(default=None, gt=0, description="Épaisseur de paroi (APPLY_SHELL)")


class CADStep(BaseModel):
    """Une étape unitaire d'un plan de construction (Palier 2)."""

    action: ActionType
    plane: PlaneType | None = None
    sketch_type: SketchType | None = None
    params: StepParams = Field(default_factory=StepParams)


class CADPlan(BaseModel):
    """Séquence ordonnée d'étapes produite par l'Agent Chercheur pour une pièce combinant
    plusieurs opérations (ex : plaque + trou + congé).

    Exemple pour "plaque 80x80mm, épaisseur 10mm, trou central 37mm" :
        {
          "steps": [
            {"action": "CREATE_SKETCH", "plane": "Top"},
            {"action": "DRAW_RECTANGLE", "sketch_type": "RECTANGLE",
             "params": {"width_mm": 80.0, "height_mm": 80.0}},
            {"action": "EXTRUDE_ADD", "params": {"depth_mm": 10.0}},
            {"action": "CREATE_SKETCH", "plane": "Top"},
            {"action": "DRAW_CIRCLE", "sketch_type": "CIRCLE",
             "params": {"diameter_mm": 37.0}},
            {"action": "EXTRUDE_REMOVE", "params": {"through_all": true}}
          ]
        }
    """

    steps: list[CADStep]
