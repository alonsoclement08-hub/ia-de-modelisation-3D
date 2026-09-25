"""
agent_modelisateur.py

Agent Modélisateur : lit l'ordre CAO JSON produit par l'Agent Chercheur et
le rejoue dans Onshape via Playwright, en n'utilisant que des raccourcis
clavier natifs (aucun clic sur un menu).

Palier 1 : création d'un bloc rectangulaire extrudé.
    Shift+S -> nouvelle esquisse sur le plan Top
    R       -> outil Rectangle par le centre
    D       -> cotation des deux côtés
    Shift+E -> extrusion + validation

    NOTE : la demande initiale évoquait N/R/D/E. Vérification faite contre
    la documentation officielle Onshape (cad.onshape.com/help), les
    raccourcis réels sont Shift+S (nouvelle esquisse) et Shift+E
    (extrusion) — 'N' seul correspond à "Normal to" (orientation caméra)
    et 'E' seul à la contrainte "Equal". R et D, eux, correspondent bien.

Stratégie de cotation (fidèle au workflow CAO classique) :
    1. On dessine le rectangle approximativement (taille en pixels, peu
       importe l'échelle réelle).
    2. On le cote ensuite précisément avec l'outil Dimension ('D'), qui
       est la véritable source de vérité géométrique.
    Cela évite d'avoir à connaître le facteur mm -> pixels de la vue 3D.

Le module est structuré en petites méthodes asynchrones nommées d'après
l'action métier qu'elles réalisent (`_new_sketch`, `_draw_rectangle`, ...)
et un dispatcher `ACTION_HANDLERS` fait le lien entre `CADCommand.action`
et la méthode qui l'exécute. Pour le Palier 2, il suffira :
    1. d'ajouter la nouvelle valeur dans `schemas.ActionType` ;
    2. d'écrire la méthode `_handle_xxx` correspondante sur `OnshapeAgent` ;
    3. de l'enregistrer dans `ACTION_HANDLERS`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Awaitable, Callable, ClassVar

from playwright.async_api import BrowserContext, Page, async_playwright

from schemas import ActionType, CADCommand, PlaneType

# ---------------------------------------------------------------------------
# Constantes de configuration
# ---------------------------------------------------------------------------

# Répertoire de profil Chromium persistant : y rester connecté à Onshape
# entre deux exécutions évite de refaire le login à chaque lancement.
DEFAULT_USER_DATA_DIR = Path.home() / ".onshape_agent_profile"

# Demi-taille (en pixels) du rectangle "approximatif" tracé avant cotation.
# Purement arbitraire : la cotation à l'étape suivante fixe la vraie taille.
SKETCH_DRAFT_HALF_SIZE_PX = 100

# Délai laissé à l'UI d'Onshape pour réagir après une frappe/un clic.
# Onshape anime ses panneaux ; sans ce délai les frappes suivantes peuvent
# arriver avant que le champ de saisie ne soit prêt à les recevoir.
UI_SETTLE_DELAY_S = 0.4


class ModelisationError(RuntimeError):
    """Levée quand une étape de modélisation échoue dans Onshape."""


@dataclass
class OnshapeAgent:
    """Pilote une page Onshape via des raccourcis clavier pour construire une pièce."""

    page: Page

    # -- Cycle de vie -------------------------------------------------

    async def open_document(self, document_url: str) -> None:
        """Navigue vers le document Onshape cible et attend son chargement.

        Sur un profil Chromium tout juste créé, Onshape redirige vers son
        écran de connexion. On détecte ce cas et on met le script en
        pause le temps que l'utilisateur se connecte lui-même dans la
        fenêtre ouverte (le script ne saisit jamais d'identifiants) ;
        la session est ensuite conservée dans le profil persistant pour
        les prochains lancements.
        """
        await self.page.goto(document_url)

        if await self._is_login_page():
            print(
                "\n[agent_modelisateur] Connexion Onshape requise : "
                "connectez-vous manuellement dans la fenêtre Chromium ouverte, "
                "puis revenez ici et appuyez sur Entrée pour continuer...",
                file=sys.stderr,
            )
            await asyncio.to_thread(input)
            await self.page.goto(document_url)  # Recharge le document une fois connecté.

        # Le canvas 3D d'Onshape se monte après le reste de l'UI ; on
        # attend un élément stable de l'espace de travail plutôt qu'un
        # simple délai fixe.
        await self.page.wait_for_selector("#graphicsCanvas, canvas", timeout=60_000)
        await self.page.locator("#graphicsCanvas, canvas").first.click()

    async def _is_login_page(self) -> bool:
        """Détecte si Onshape a redirigé vers l'écran de connexion."""
        url = self.page.url.lower()
        if "login" in url or "signin" in url:
            return True
        # Certains flux affichent un formulaire de connexion sans changer d'URL.
        return await self.page.locator("input[type='password']").count() > 0

    # -- Étape N : nouvelle esquisse -----------------------------------

    async def _new_sketch(self, plane: PlaneType) -> None:
        if plane is not PlaneType.TOP:
            # Palier 1 ne gère que le plan Top ; les autres plans sont
            # prévus dans le schéma mais pas encore pilotables ici.
            raise ModelisationError(
                f"Plan '{plane.value}' non encore supporté par l'Agent Modélisateur (Palier 1)."
            )

        await self.page.keyboard.press("Shift+S")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        # Après Shift+S, Onshape attend la sélection d'un plan de construction.
        # Le plan Top est cliquable dans l'arbre de fonctions ou directement
        # dans le viewport ; on cible ici l'entrée de l'arbre de features,
        # plus stable que des coordonnées de viewport codées en dur.
        top_plane_entry = self.page.get_by_text("Top", exact=True).first
        await top_plane_entry.click()
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    # -- Étape R : rectangle par le centre -----------------------------

    async def _draw_rectangle(self) -> None:
        await self.page.keyboard.press("r")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        canvas = self.page.locator("#graphicsCanvas, canvas").first
        box = await canvas.bounding_box()
        if box is None:
            raise ModelisationError("Impossible de localiser le canvas 3D d'Onshape.")

        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Rectangle "par le centre" : un clic au centre pour l'ancrer, puis
        # un second clic sur un coin pour fixer une taille de départ.
        await self.page.mouse.click(center_x, center_y)
        await self.page.mouse.click(
            center_x + SKETCH_DRAFT_HALF_SIZE_PX,
            center_y - SKETCH_DRAFT_HALF_SIZE_PX,
        )
        await self.page.keyboard.press("Escape")  # Sort de l'outil rectangle.
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    # -- Étape D : cotation des deux côtés ------------------------------

    async def _dimension_side(self, edge_x: float, edge_y: float, value_mm: float) -> None:
        """Cote une arête cliquée en (edge_x, edge_y) à `value_mm`."""
        await self.page.keyboard.press("d")
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self.page.mouse.click(edge_x, edge_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        # La cotation ouvre un champ de saisie flottant sur le canvas ;
        # il capte directement le clavier une fois l'arête sélectionnée.
        await self.page.keyboard.type(str(value_mm))
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    async def _dimension_rectangle(self, width_mm: float, height_mm: float) -> None:
        canvas = self.page.locator("#graphicsCanvas, canvas").first
        box = await canvas.bounding_box()
        if box is None:
            raise ModelisationError("Impossible de localiser le canvas 3D d'Onshape.")

        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Milieu de l'arête horizontale supérieure -> cote la largeur.
        top_edge_x = center_x + SKETCH_DRAFT_HALF_SIZE_PX / 2
        top_edge_y = center_y - SKETCH_DRAFT_HALF_SIZE_PX
        await self._dimension_side(top_edge_x, top_edge_y, width_mm)

        # Milieu de l'arête verticale gauche -> cote la hauteur.
        left_edge_x = center_x - SKETCH_DRAFT_HALF_SIZE_PX
        left_edge_y = center_y - SKETCH_DRAFT_HALF_SIZE_PX / 2
        await self._dimension_side(left_edge_x, left_edge_y, height_mm)

        await self.page.keyboard.press("Escape")  # Sort de l'outil cotation.
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    # -- Étape E : extrusion ---------------------------------------------

    async def _extrude(self, depth_mm: float) -> None:
        # Termine l'esquisse avant de pouvoir en extruder la face.
        await self.page.keyboard.press("Escape")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        canvas = self.page.locator("#graphicsCanvas, canvas").first
        box = await canvas.bounding_box()
        if box is None:
            raise ModelisationError("Impossible de localiser le canvas 3D d'Onshape.")
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Sélectionne la face de l'esquisse fermée avant d'extruder.
        await self.page.mouse.click(center_x, center_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        await self.page.keyboard.press("Shift+E")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        # La boîte de dialogue d'extrusion s'ouvre avec le champ de
        # profondeur déjà focalisé et son contenu sélectionné : on peut
        # taper directement la nouvelle valeur pour l'écraser.
        await self.page.keyboard.type(str(depth_mm))
        await self.page.keyboard.press("Enter")  # Valide la boîte de dialogue.
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    # -- Dispatch ----------------------------------------------------------

    async def _handle_create_primitive(self, command: CADCommand) -> None:
        """Gère `ActionType.CREATE_PRIMITIVE` (Palier 1 : rectangle extrudé)."""
        dims = command.dimensions

        await self._new_sketch(command.plane)
        await self._draw_rectangle()
        await self._dimension_rectangle(dims.width_mm, dims.height_mm)
        await self._extrude(dims.extrude_depth_mm)

    # Registre action -> handler. Palier 2 : ajouter par ex.
    #   ACTION_HANDLERS[ActionType.CREATE_HOLE] = OnshapeAgent._handle_create_hole
    ACTION_HANDLERS: ClassVar[dict[ActionType, Callable[["OnshapeAgent", CADCommand], Awaitable[None]]]] = {
        ActionType.CREATE_PRIMITIVE: _handle_create_primitive,
    }

    async def execute(self, command: CADCommand) -> None:
        handler = self.ACTION_HANDLERS.get(command.action)
        if handler is None:
            raise ModelisationError(f"Action non supportée : {command.action}")
        await handler(self, command)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def load_command(json_path: Path) -> CADCommand:
    """Charge et valide l'ordre CAO produit par l'Agent Chercheur."""
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    return CADCommand.model_validate(raw)


async def run(command: CADCommand, document_url: str, headless: bool, user_data_dir: Path) -> None:
    async with async_playwright() as playwright:
        # Contexte persistant : conserve la session Onshape (cookies/login)
        # d'une exécution à l'autre, comme un vrai profil utilisateur.
        context: BrowserContext = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            agent = OnshapeAgent(page=page)

            try:
                await agent.open_document(document_url)
                await agent.execute(command)
            except Exception:
                # Capture un instantané pour diagnostiquer l'état de l'UI
                # au moment de l'échec (chargement du document compris),
                # avant de relayer l'erreur.
                await page.screenshot(path="agent_modelisateur_error.png")
                raise
        finally:
            await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Agent Modélisateur : rejoue un ordre CAO JSON dans Onshape via Playwright.",
    )
    parser.add_argument("command_file", type=Path, help="Fichier JSON produit par agent_chercheur.py")
    parser.add_argument("document_url", help="URL du document Onshape cible")
    parser.add_argument(
        "--headless",
        action="store_true",
        help="Lance Chromium en mode headless (défaut: fenêtre visible)",
    )
    parser.add_argument(
        "--user-data-dir",
        type=Path,
        default=DEFAULT_USER_DATA_DIR,
        help="Répertoire de profil Chromium persistant (login Onshape conservé)",
    )
    args = parser.parse_args()

    try:
        command = load_command(args.command_file)
    except Exception as exc:
        print(f"[agent_modelisateur] Ordre CAO invalide : {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[agent_modelisateur] Exécution de l'ordre : {command.model_dump_json()}", file=sys.stderr)
    asyncio.run(run(command, args.document_url, args.headless, args.user_data_dir))
    print("[agent_modelisateur] Pièce créée avec succès.", file=sys.stderr)


if __name__ == "__main__":
    main()
