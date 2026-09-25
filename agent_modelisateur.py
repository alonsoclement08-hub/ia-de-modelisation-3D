"""
agent_modelisateur.py

Agent Modélisateur : lit l'ordre CAO JSON produit par l'Agent Chercheur et
le rejoue dans Onshape via Playwright, en n'utilisant que des raccourcis
clavier natifs (aucun clic sur un menu).

Raccourcis utilisés (vérifiés contre cad.onshape.com/help) :
    Shift+S -> nouvelle esquisse
    R       -> rectangle par le centre
    C       -> cercle par le centre
    D       -> cotation -> clic sur l'élément -> saisie -> Enter
    Shift+E -> extrusion (Add ou Remove selon le dialogue)
    Shift+F -> congé (fillet)

    NOTE : la demande initiale du Palier 1 évoquait N/R/D/E pour la
    séquence de base. 'N' seul correspond en réalité à "Normal to"
    (orientation caméra, qu'on utilise bien, cf. `_new_sketch`) et 'E'
    seul à la contrainte "Equal" ; les vrais raccourcis pour nouvelle
    esquisse et extrusion sont Shift+S et Shift+E.

Stratégie de cotation (fidèle au workflow CAO classique) :
    1. On dessine la géométrie approximativement (taille en pixels, peu
       importe l'échelle réelle), après avoir orienté la caméra "Normal
       to" le plan pour que les axes écran correspondent aux axes du plan.
    2. On la cote ensuite précisément avec l'outil Dimension ('D'), qui
       est la véritable source de vérité géométrique. Le champ de saisie
       est explicitement vidé (Ctrl/Cmd+A) avant de taper la valeur, pour
       éviter qu'elle ne se mélange à une valeur mesurée/snappée par
       Onshape (ex: 79.70mm au lieu de 80mm) — voir `_type_exact_value`.

Deux niveaux d'API cohabitent :
    - `execute(CADCommand)` (Palier 1) : une pièce simple en une seule
      action, via `ACTION_HANDLERS`. Conservé tel quel.
    - `execute_plan(CADPlan)` (Palier 2) : une séquence d'étapes
      (esquisse, rectangle/cercle, extrusion add/remove, congé...), via
      `STEP_HANDLERS`. C'est le chemin utilisé par la CLI (`load_plan`
      accepte aussi bien un `command.json` Palier 1 qu'un plan Palier 2).

Pour ajouter une opération au Palier 3 : ajouter la valeur dans
`schemas.ActionType`, écrire la méthode `_step_xxx` correspondante sur
`OnshapeAgent`, puis l'enregistrer dans `STEP_HANDLERS`.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Awaitable, Callable, ClassVar

from playwright.async_api import BrowserContext, Page, async_playwright

from schemas import ActionType, CADCommand, CADPlan, CADStep, PlaneType, StepParams

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

# Dossier des captures de contrôle prises après chaque étape (Shift+S, R,
# D, Shift+E). Playwright ne détecte pas les échecs "silencieux" (un clic
# qui atterrit au bon endroit mais ne déclenche pas l'action attendue) :
# ces captures sont le seul moyen de vérifier visuellement que chaque
# étape a réellement eu l'effet voulu dans l'UI Onshape.
DEBUG_SCREENSHOT_DIR = Path("debug_screenshots")


class ModelisationError(RuntimeError):
    """Levée quand une étape de modélisation échoue dans Onshape."""


@dataclass
class OnshapeAgent:
    """Pilote une page Onshape via des raccourcis clavier pour construire une pièce."""

    page: Page
    _step_counter: int = field(default=0, repr=False)

    async def _snapshot(self, step_name: str) -> None:
        """Capture un instantané après une étape, pour vérifier visuellement son effet réel."""
        DEBUG_SCREENSHOT_DIR.mkdir(exist_ok=True)
        self._step_counter += 1
        path = DEBUG_SCREENSHOT_DIR / f"{self._step_counter}_{step_name}.png"
        await self.page.screenshot(path=str(path))
        print(f"[agent_modelisateur] Capture de contrôle : {path}", file=sys.stderr)

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

        await self._dismiss_overlays()

        # Onshape a plusieurs <canvas> sur la page (mini-cube d'orientation,
        # icônes...) : document.querySelector('canvas') attrape le premier
        # trouvé dans le DOM, pas forcément le grand canvas 3D. On attend
        # donc qu'AU MOINS UN canvas dépasse une taille plausible pour une
        # zone de dessin, ce qui filtre les petites icônes.
        await self.page.wait_for_function(
            """
            () => Array.from(document.querySelectorAll('canvas')).some(c => {
                const r = c.getBoundingClientRect();
                return r.width > 200 && r.height > 200;
            })
            """,
            timeout=90_000,
        )

        box = await self._get_graphics_canvas_box()
        await self.page.mouse.click(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)

    async def _get_graphics_canvas_box(self) -> dict:
        """Retourne le bounding box du plus grand `<canvas>` de la page.

        Sert de zone de dessin/clic pour toutes les étapes suivantes. On ne
        peut pas se fier à "le premier <canvas> du DOM" (cf. `open_document`) :
        on prend systématiquement le plus grand par surface.
        """
        box = await self.page.evaluate(
            """
            () => {
                let best = null;
                for (const c of document.querySelectorAll('canvas')) {
                    const r = c.getBoundingClientRect();
                    const area = r.width * r.height;
                    if (!best || area > best.width * best.height) {
                        best = { x: r.x, y: r.y, width: r.width, height: r.height };
                    }
                }
                return best;
            }
            """
        )
        if box is None or box["width"] == 0 or box["height"] == 0:
            raise ModelisationError("Impossible de localiser le canvas 3D d'Onshape.")
        return box

    async def _dismiss_overlays(self) -> None:
        """Ferme les popups (accueil, cookies, nouveautés) qui peuvent masquer le canvas."""
        for label in ("Got it", "Skip", "Close", "Accept", "OK", "J'ai compris", "Fermer", "Accepter"):
            button = self.page.get_by_role("button", name=label, exact=False)
            try:
                if await button.count() > 0:
                    await button.first.click(timeout=2_000)
            except Exception:
                pass  # Overlay absent ou non cliquable : on continue sans bloquer.

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
        await self._snapshot("01_apres_shift_s")

        # Après Shift+S, Onshape attend la sélection d'un plan de construction.
        # Le plan Top est cliquable dans l'arbre de fonctions ou directement
        # dans le viewport ; on cible ici l'entrée de l'arbre de features,
        # plus stable que des coordonnées de viewport codées en dur.
        top_plane_entry = self.page.get_by_text("Top", exact=True).first
        await top_plane_entry.click()
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("02_plan_top_clique")

        # Sans cela, la caméra reste en vue isométrique par défaut : un
        # rectangle "par pixels d'écran" n'y est ni carré, ni même
        # axé horizontalement/verticalement, ce qui fait ensuite échouer
        # la cotation et la sélection de face (clics au mauvais endroit).
        # On réoriente donc la vue perpendiculairement au plan Top, comme
        # le ferait un utilisateur humain avant de dessiner.
        await self.page.keyboard.press("n")
        await asyncio.sleep(1.0)  # L'animation de rotation de caméra est plus longue qu'un simple settle.
        await self._snapshot("02b_vue_normale_au_plan")

    # -- Étape R : rectangle par le centre -----------------------------

    async def _draw_rectangle(self) -> None:
        await self.page.keyboard.press("r")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        box = await self._get_graphics_canvas_box()
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
        await self._snapshot("03_rectangle_dessine")

    # -- Verrouillage strict d'une valeur numérique -----------------------

    async def _type_exact_value(self, value: float) -> None:
        """Vide le champ de saisie actif puis y tape `value`, sans valider.

        Onshape pré-remplit souvent le champ de cotation avec la valeur
        mesurée au moment du clic (ex: 79.70mm). Taper par-dessus sans
        vider le champ produit un mélange (ex: "79.708" au lieu de "80") :
        c'est le bug de snapping décrit dans la demande. Ctrl/Cmd+A
        sélectionne tout le contenu existant pour que la frappe suivante
        le remplace intégralement. La validation (Enter) reste à la
        charge de l'appelant, pour que le verrouillage soit explicite à
        chaque site d'appel.
        """
        await self.page.keyboard.press("ControlOrMeta+A")
        await self.page.keyboard.type(f"{value:g}")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    # -- Étape D : cotation ------------------------------------------------

    async def _dimension_side(self, edge_x: float, edge_y: float, value_mm: float) -> None:
        """Cote un élément cliqué en (edge_x, edge_y) à `value_mm`, verrouillée."""
        await self.page.keyboard.press("d")
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self.page.mouse.click(edge_x, edge_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        # La cotation ouvre un champ de saisie flottant sur le canvas ;
        # il capte directement le clavier une fois l'élément sélectionné.
        await self._type_exact_value(value_mm)
        await self.page.keyboard.press("Enter")  # Verrouille la cote à la valeur exacte tapée.
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    async def _dimension_rectangle(self, width_mm: float, height_mm: float) -> None:
        box = await self._get_graphics_canvas_box()
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
        await self._snapshot("04_cotation_terminee")

    # -- Étape C : cercle par le centre (Palier 2, perçages) ----------------

    async def _draw_circle(self) -> None:
        await self.page.keyboard.press("c")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Cercle par le centre : clic au centre pour l'ancrer, puis clic
        # sur la circonférence pour fixer un rayon de départ approximatif.
        await self.page.mouse.click(center_x, center_y)
        await self.page.mouse.click(center_x + SKETCH_DRAFT_HALF_SIZE_PX, center_y)
        await self.page.keyboard.press("Escape")  # Sort de l'outil cercle.
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("cercle_dessine")

    async def _dimension_circle(self, diameter_mm: float) -> None:
        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # NON VÉRIFIÉ CONTRE L'APP RÉELLE : cliquer sur la circonférence
        # avec l'outil Dimension donne en général un rayon dans les
        # logiciels de CAO, parfois un diamètre selon le point cliqué.
        # `diameter_mm` est ici tapé tel quel ; si Onshape l'interprète
        # comme un rayon, corriger en passant `diameter_mm / 2` ou en
        # cliquant un point diamétralement opposé (deux points opposés
        # de la circonférence) pour forcer une cotation de diamètre.
        edge_x = center_x + SKETCH_DRAFT_HALF_SIZE_PX
        edge_y = center_y
        await self._dimension_side(edge_x, edge_y, diameter_mm)

        await self.page.keyboard.press("Escape")  # Sort de l'outil cotation.
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("cercle_cote")

    # -- Étape Shift+E : extrusion (ajout ou enlèvement de matière) --------

    async def _extrude(self, *, depth_mm: float | None, remove: bool, through_all: bool) -> None:
        # Termine l'esquisse avant de pouvoir en extruder la face.
        await self.page.keyboard.press("Escape")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Sélectionne la face de l'esquisse fermée avant d'extruder.
        await self.page.mouse.click(center_x, center_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("face_selectionnee")

        await self.page.keyboard.press("Shift+E")
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("dialogue_extrusion")

        if remove:
            # NON VÉRIFIÉ CONTRE L'APP RÉELLE : bascule le dialogue en mode
            # "Remove" (enlèvement de matière) en ciblant le texte du
            # bouton/option, faute de sélecteur DOM stable connu.
            remove_toggle = self.page.get_by_text("Remove", exact=False)
            if await remove_toggle.count() > 0:
                await remove_toggle.first.click()
                await asyncio.sleep(UI_SETTLE_DELAY_S)

            if through_all:
                through_all_option = self.page.get_by_text("Through all", exact=False)
                if await through_all_option.count() > 0:
                    await through_all_option.first.click()
                    await asyncio.sleep(UI_SETTLE_DELAY_S)
            elif depth_mm is not None:
                await self._type_exact_value(depth_mm)
        else:
            if depth_mm is None:
                raise ModelisationError("EXTRUDE_ADD nécessite une profondeur (depth_mm).")
            # La boîte de dialogue d'extrusion s'ouvre avec le champ de
            # profondeur déjà focalisé : on le vide puis on tape la valeur
            # exacte pour éviter tout mélange avec une valeur par défaut.
            await self._type_exact_value(depth_mm)

        await self.page.keyboard.press("Enter")  # Valide la boîte de dialogue.
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("extrusion_validee")

    # -- Étape Shift+F : congé (fillet) -------------------------------------

    async def _apply_fillet(self, radius_mm: float) -> None:
        """Applique un congé sur les arêtes du dessus de la pièce.

        NON VÉRIFIÉ CONTRE L'APP RÉELLE : sélectionne les 4 coins de la
        face supérieure en cliquant à proximité de chacun, une fois
        l'outil Fillet actif. La géométrie exacte des clics (quels coins,
        quel décalage, faut-il maintenir Shift pour une multi-sélection)
        n'a pas pu être validée en conditions réelles ; à ajuster lors du
        premier test, comme le reste du pipeline de cotation/sélection.
        """
        await self.page.keyboard.press("Shift+F")
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("outil_fillet_actif")

        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        corner_offsets = (
            (-SKETCH_DRAFT_HALF_SIZE_PX, -SKETCH_DRAFT_HALF_SIZE_PX),
            (SKETCH_DRAFT_HALF_SIZE_PX, -SKETCH_DRAFT_HALF_SIZE_PX),
            (SKETCH_DRAFT_HALF_SIZE_PX, SKETCH_DRAFT_HALF_SIZE_PX),
            (-SKETCH_DRAFT_HALF_SIZE_PX, SKETCH_DRAFT_HALF_SIZE_PX),
        )
        for dx, dy in corner_offsets:
            await self.page.mouse.click(center_x + dx, center_y + dy)
            await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("aretes_selectionnees")

        await self._type_exact_value(radius_mm)
        await self.page.keyboard.press("Enter")  # Verrouille le rayon à la valeur exacte tapée.
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("fillet_valide")

    # -- Dispatch Palier 1 (CADCommand) -------------------------------------

    async def _handle_create_primitive(self, command: CADCommand) -> None:
        """Gère `ActionType.CREATE_PRIMITIVE` (Palier 1 : rectangle extrudé)."""
        dims = command.dimensions

        await self._new_sketch(command.plane)
        await self._draw_rectangle()
        await self._dimension_rectangle(dims.width_mm, dims.height_mm)
        await self._extrude(depth_mm=dims.extrude_depth_mm, remove=False, through_all=False)

    ACTION_HANDLERS: ClassVar[dict[ActionType, Callable[["OnshapeAgent", CADCommand], Awaitable[None]]]] = {
        ActionType.CREATE_PRIMITIVE: _handle_create_primitive,
    }

    async def execute(self, command: CADCommand) -> None:
        """Exécute une `CADCommand` Palier 1 (une pièce simple, une action)."""
        handler = self.ACTION_HANDLERS.get(command.action)
        if handler is None:
            raise ModelisationError(f"Action non supportée : {command.action}")
        await handler(self, command)

    # -- Dispatch Palier 2 (CADPlan) -----------------------------------------

    async def _step_create_sketch(self, step: CADStep) -> None:
        await self._new_sketch(step.plane or PlaneType.TOP)

    async def _step_draw_rectangle(self, step: CADStep) -> None:
        params = step.params
        if params.width_mm is None or params.height_mm is None:
            raise ModelisationError("DRAW_RECTANGLE nécessite width_mm et height_mm.")
        await self._draw_rectangle()
        await self._dimension_rectangle(params.width_mm, params.height_mm)

    async def _step_draw_circle(self, step: CADStep) -> None:
        params = step.params
        if params.diameter_mm is None:
            raise ModelisationError("DRAW_CIRCLE nécessite diameter_mm.")
        await self._draw_circle()
        await self._dimension_circle(params.diameter_mm)

    async def _step_extrude_add(self, step: CADStep) -> None:
        params = step.params
        if params.depth_mm is None:
            raise ModelisationError("EXTRUDE_ADD nécessite depth_mm.")
        await self._extrude(depth_mm=params.depth_mm, remove=False, through_all=False)

    async def _step_extrude_remove(self, step: CADStep) -> None:
        params = step.params
        await self._extrude(depth_mm=params.depth_mm, remove=True, through_all=params.through_all)

    async def _step_apply_fillet(self, step: CADStep) -> None:
        params = step.params
        if params.radius_mm is None:
            raise ModelisationError("APPLY_FILLET nécessite radius_mm.")
        await self._apply_fillet(params.radius_mm)

    # Registre action -> handler d'étape. Palier 3 : ajouter la nouvelle
    # valeur dans schemas.ActionType, écrire _step_xxx, puis l'enregistrer ici.
    STEP_HANDLERS: ClassVar[dict[ActionType, Callable[["OnshapeAgent", CADStep], Awaitable[None]]]] = {
        ActionType.CREATE_SKETCH: _step_create_sketch,
        ActionType.DRAW_RECTANGLE: _step_draw_rectangle,
        ActionType.DRAW_CIRCLE: _step_draw_circle,
        ActionType.EXTRUDE_ADD: _step_extrude_add,
        ActionType.EXTRUDE_REMOVE: _step_extrude_remove,
        ActionType.APPLY_FILLET: _step_apply_fillet,
    }

    async def execute_plan(self, plan: CADPlan) -> None:
        """Exécute un `CADPlan` Palier 2 (séquence ordonnée d'étapes)."""
        for i, step in enumerate(plan.steps, start=1):
            handler = self.STEP_HANDLERS.get(step.action)
            if handler is None:
                raise ModelisationError(f"Action non supportée dans un plan : {step.action}")
            print(
                f"[agent_modelisateur] Étape {i}/{len(plan.steps)} : {step.action.value}",
                file=sys.stderr,
            )
            await handler(self, step)


# ---------------------------------------------------------------------------
# Point d'entrée
# ---------------------------------------------------------------------------


def load_plan(json_path: Path) -> CADPlan:
    """Charge un ordre CAO JSON, Palier 1 ou Palier 2.

    Accepte soit une `CADPlan` ({"steps": [...]}, Palier 2), soit une
    `CADCommand` isolée (format Palier 1 d'origine) — auquel cas elle est
    convertie en un plan équivalent à 3 étapes. Un `command.json` généré
    par une version antérieure de l'Agent Chercheur continue donc de
    fonctionner sans modification.
    """
    raw = json.loads(json_path.read_text(encoding="utf-8"))
    if "steps" in raw:
        return CADPlan.model_validate(raw)

    legacy = CADCommand.model_validate(raw)
    return CADPlan(
        steps=[
            CADStep(action=ActionType.CREATE_SKETCH, plane=legacy.plane),
            CADStep(
                action=ActionType.DRAW_RECTANGLE,
                sketch_type=legacy.sketch_type,
                params=StepParams(width_mm=legacy.dimensions.width_mm, height_mm=legacy.dimensions.height_mm),
            ),
            CADStep(action=ActionType.EXTRUDE_ADD, params=StepParams(depth_mm=legacy.dimensions.extrude_depth_mm)),
        ]
    )


async def run(plan: CADPlan, document_url: str, headless: bool, user_data_dir: Path) -> None:
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
                await agent.execute_plan(plan)
            except Exception:
                # Capture un instantané pour diagnostiquer l'état de l'UI
                # au moment de l'échec (chargement du document compris),
                # avant de relayer l'erreur.
                await page.screenshot(path="agent_modelisateur_error.png")
                if not headless:
                    # En mode visible, on laisse la fenêtre ouverte le temps
                    # d'inspecter la page réelle (DevTools compris) plutôt
                    # que de la fermer immédiatement dans le `finally`.
                    print(
                        "\n[agent_modelisateur] Erreur : capture enregistrée dans "
                        "agent_modelisateur_error.png. La fenêtre reste ouverte pour "
                        "inspection — appuyez sur Entrée ici pour la fermer.",
                        file=sys.stderr,
                    )
                    await asyncio.to_thread(input)
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
        plan = load_plan(args.command_file)
    except Exception as exc:
        print(f"[agent_modelisateur] Ordre CAO invalide : {exc}", file=sys.stderr)
        sys.exit(1)

    print(f"[agent_modelisateur] Exécution du plan ({len(plan.steps)} étape(s)) : {plan.model_dump_json()}", file=sys.stderr)
    asyncio.run(run(plan, args.document_url, args.headless, args.user_data_dir))
    print("[agent_modelisateur] Pièce créée avec succès.", file=sys.stderr)


if __name__ == "__main__":
    main()
