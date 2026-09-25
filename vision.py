"""
vision.py

Localisation d'éléments d'interface par vision (Gemini), pour les clics
sur le canvas WebGL d'Onshape — qui n'a pratiquement aucune
représentation DOM exploitable (arêtes, lignes de cotation, champ de
saisie flottant...), contrairement aux panneaux HTML autour (boutons,
listes, cases à cocher), déjà ciblés par sélecteur ailleurs dans
`agent_modelisateur.py`.

Principe : avant de cliquer sur le canvas, on capture l'écran actuel, on
demande à Gemini où se trouve précisément l'élément décrit (coordonnées
normalisées 0-1000), on convertit en pixels réels, puis on clique. C'est
le mécanisme "regarder avant de cliquer" — remplace les coordonnées
calculées à l'avance (centre du canvas ± décalage fixe), qui se sont
montrées peu fiables : vue caméra jamais parfaitement normale au plan,
esquisse pas exactement centrée, etc.

Usage attendu (voir agent_modelisateur.OnshapeAgent) : uniquement pour
les étapes qui se sont montrées fragiles en test réel (cotation,
sélection de face/arête) — pas pour chaque clic du pipeline, pour garder
la latence et le coût des appels API raisonnables.
"""

from __future__ import annotations

import os

from google import genai
from google.genai import types
from pydantic import BaseModel, Field

# gemini-2.5-flash : rapide et peu coûteux, suffisant pour de la
# localisation d'éléments UI (pas besoin d'un modèle "pro" pour ça).
GEMINI_MODEL = "gemini-2.5-flash"


class ElementLocation(BaseModel):
    """Réponse structurée attendue de Gemini pour une requête de localisation.

    Les coordonnées sont normalisées 0-1000 (fraction de la largeur/hauteur
    de l'image x1000), pas des pixels absolus : l'appelant les convertit
    avec les dimensions réelles de la capture d'écran fournie.
    """

    found: bool = Field(description="true si l'élément décrit est visible sur l'image")
    x: int = Field(ge=0, le=1000, description="Position horizontale du centre de l'élément, 0 (gauche) à 1000 (droite)")
    y: int = Field(ge=0, le=1000, description="Position verticale du centre de l'élément, 0 (haut) à 1000 (bas)")
    reasoning: str = Field(description="Brève justification : ce qui a été identifié et pourquoi")


class VisionError(RuntimeError):
    """Levée quand la localisation par vision échoue (clé API absente, erreur réseau...)."""


class GeminiLocator:
    """Localise un élément décrit en langage naturel sur une capture d'écran, via Gemini."""

    def __init__(self, api_key: str | None = None) -> None:
        key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
        if not key:
            raise VisionError(
                "GEMINI_API_KEY (ou GOOGLE_API_KEY) n'est pas défini dans l'environnement : "
                "nécessaire pour localiser les éléments du canvas par vision."
            )
        self._client = genai.Client(api_key=key)

    async def locate(self, image_bytes: bytes, description: str) -> ElementLocation:
        """Demande à Gemini les coordonnées normalisées (0-1000) de `description`.

        Lève `VisionError` sur tout échec réseau/API — c'est à l'appelant
        de décider s'il se replie sur des coordonnées calculées ou échoue.
        """
        prompt = (
            "Tu regardes une capture d'écran de l'application de CAO Onshape "
            "(un logiciel de modélisation 3D dans un navigateur).\n\n"
            f"Localise précisément CET élément : {description}\n\n"
            "Réponds avec les coordonnées du CENTRE de cet élément, normalisées "
            "de 0 (bord gauche/haut de l'image) à 1000 (bord droit/bas). "
            "Si l'élément décrit n'est visible nulle part sur l'image, réponds "
            "found=false plutôt que de deviner une position approximative."
        )
        try:
            response = await self._client.aio.models.generate_content(
                model=GEMINI_MODEL,
                contents=[
                    types.Part.from_bytes(data=image_bytes, mime_type="image/png"),
                    prompt,
                ],
                config=types.GenerateContentConfig(
                    response_mime_type="application/json",
                    response_schema=ElementLocation,
                    temperature=0,
                ),
            )
        except Exception as exc:  # Erreur réseau/API Gemini, quelle qu'elle soit.
            raise VisionError(f"Appel Gemini échoué : {exc}") from exc

        if response.parsed is None:
            raise VisionError(f"Réponse Gemini non exploitable (JSON invalide) : {response.text!r}")
        return response.parsed
