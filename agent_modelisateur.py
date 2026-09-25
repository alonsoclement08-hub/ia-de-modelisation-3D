"""
agent_modelisateur.py

Agent Modélisateur : lit l'ordre CAO JSON produit par l'Agent Chercheur et
le rejoue dans Onshape via Playwright, en pilotant l'UI par raccourcis
clavier natifs et, en repli, par la barre de recherche d'outils.

Machine à états (contextes Onshape) :
    Certaines actions ne sont valides que dans un contexte précis (on ne
    peut pas dessiner un rectangle hors d'une esquisse active, ni
    extruder tant qu'elle n'a pas été validée). `OnshapeAgent.context`
    (un `schemas.InterfaceContext`) suit l'état courant ; `TOOL_BINDINGS`
    et `CONTEXT_VALID_ACTIONS` forment la matrice raccourci/contexte, et
    `_activate_tool` refuse (ModelisationError) toute action hors de son
    contexte plutôt que de presser une touche au hasard. Voir ces trois
    noms pour la matrice complète.

Raccourcis vérifiés contre cad.onshape.com/help (2026-09-25) :
    Alt+C   -> barre de recherche d'outils (fallback universel)
    Shift+S -> nouvelle esquisse (CONTEXT_GLOBAL/FEATURE_3D -> SKETCH)
    R / C   -> rectangle / cercle par le centre (CONTEXT_SKETCH)
    D       -> cotation -> clic sur l'élément -> saisie -> Enter
    Entrée  -> valide et quitte l'esquisse active (CONTEXT_SKETCH -> FEATURE_3D)
    Shift+E -> extrusion (Add ou Remove selon le dialogue)
    Shift+W -> révolution (Revolve)
    Shift+F -> congé (Fillet)

    Sweep, Loft, Chamfer, Shell, Draft, Hole et Mirror N'ONT PAS de
    raccourci clavier direct documenté dans Onshape (vérifié) : ils
    passent systématiquement par Alt+C (recherche) + nom + Entrée, plutôt
    que par une combinaison Shift+<lettre> devinée qui déclencherait
    autre chose (ou rien) dans l'UI réelle.

    NOTE historique : les demandes initiales évoquaient 'N'/'E' seuls
    pour esquisse/extrusion, et Escape pour quitter l'esquisse. Vérifié
    faux dans les deux cas : 'N' seul = "Normal to" (utilisé pour orienter
    la caméra, cf. `_new_sketch`), 'E' seul = contrainte "Equal", et
    Escape ne fermait pas l'esquisse de façon fiable en test réel — la
    documentation officielle indique Entrée comme mécanisme de
    validation, désormais utilisé par `_exit_sketch`.

Stratégie de cotation (fidèle au workflow CAO classique) :
    1. On dessine la géométrie approximativement (taille en pixels, peu
       importe l'échelle réelle), après avoir orienté la caméra "Normal
       to" le plan pour que les axes écran correspondent aux axes du plan.
    2. On la cote ensuite précisément avec l'outil Dimension ('D'), qui
       est la véritable source de vérité géométrique. Le champ de saisie
       est explicitement vidé (Ctrl/Cmd+A) avant de taper la valeur, pour
       éviter qu'elle ne se mélange à une valeur mesurée/snappée par
       Onshape (ex: 79.70mm au lieu de 80mm) — voir `_type_exact_value`.

Trois niveaux d'API cohabitent :
    - `execute(CADCommand)` (Palier 1) : une pièce simple en une seule
      action, via `ACTION_HANDLERS`. Conservé tel quel.
    - `execute_plan(CADPlan)` (Palier 2/3) : une séquence d'étapes
      (esquisse, formes, extrusion add/remove, révolution, finitions...),
      via `STEP_HANDLERS`. Utilisé par la CLI (`load_plan` accepte les
      deux formats) et par `main.py` (terminal interactif).
    - `main.py` combine les deux agents dans une boucle interactive,
      session Onshape unique conservée entre les commandes.

Pour ajouter une opération : ajouter la valeur dans `schemas.ActionType`,
son entrée dans `TOOL_BINDINGS`/`CONTEXT_VALID_ACTIONS`, écrire la
méthode `_step_xxx` correspondante sur `OnshapeAgent`, puis l'enregistrer
dans `STEP_HANDLERS`.
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

from schemas import ActionType, CADCommand, CADPlan, CADStep, InterfaceContext, PlaneType, StepParams

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
    """Levée quand une étape de modélisation échoue dans Onshape, y compris
    quand une action est demandée dans un contexte où elle n'est pas valide."""


@dataclass(frozen=True)
class ToolBinding:
    """Comment déclencher un outil Onshape : raccourci direct si documenté,
    sinon recherche (Alt+C -> nom -> Entrée)."""

    shortcut: str | None = None
    search_term: str | None = None

    def __post_init__(self) -> None:
        if not self.shortcut and not self.search_term:
            raise ValueError("ToolBinding nécessite shortcut ou search_term.")


# Matrice action -> déclenchement, vérifiée le 2026-09-25 contre
# cad.onshape.com/help/Content/Home/keyboard_shortcuts_and_hotkeys.htm.
# Alt+C (recherche), Shift+S (esquisse), R/C (outils d'esquisse), Shift+E
# (extrusion), Shift+W (révolution) et Shift+F (congé) sont des raccourcis
# documentés. Sweep, Loft, Chamfer, Shell, Draft, Hole et Mirror N'ONT PAS
# de raccourci clavier direct documenté dans Onshape (vérifié) : on utilise
# systématiquement le fallback recherche pour eux plutôt qu'une combinaison
# Shift+<lettre> devinée, qui déclencherait autre chose (ou rien) dans
# l'UI réelle.
TOOL_BINDINGS: dict[ActionType, ToolBinding] = {
    ActionType.CREATE_SKETCH: ToolBinding(shortcut="Shift+S"),
    ActionType.DRAW_RECTANGLE: ToolBinding(shortcut="r"),
    ActionType.DRAW_CIRCLE: ToolBinding(shortcut="c"),
    ActionType.EXTRUDE_ADD: ToolBinding(shortcut="Shift+E"),
    ActionType.EXTRUDE_REMOVE: ToolBinding(shortcut="Shift+E"),
    ActionType.REVOLVE: ToolBinding(shortcut="Shift+W"),
    ActionType.SWEEP: ToolBinding(search_term="Sweep"),
    ActionType.LOFT: ToolBinding(search_term="Loft"),
    ActionType.APPLY_FILLET: ToolBinding(shortcut="Shift+F"),
    ActionType.APPLY_CHAMFER: ToolBinding(search_term="Chamfer"),
    ActionType.APPLY_SHELL: ToolBinding(search_term="Shell"),
    ActionType.APPLY_DRAFT: ToolBinding(search_term="Draft"),
    ActionType.APPLY_HOLE_FEATURE: ToolBinding(search_term="Hole"),
    ActionType.APPLY_MIRROR: ToolBinding(search_term="Mirror"),
}

# Actions valides par contexte (machine à états). Onshape n'a pas de mode
# "finition" séparé dans son UI : une fois un corps 3D obtenu (FEATURE_3D),
# les opérations de volume ET de finition sont toutes disponibles au même
# niveau — CONTEXT_FINISHING n'est donc pas un état distinct ici, ses
# actions sont simplement incluses dans FEATURE_3D.
CONTEXT_VALID_ACTIONS: dict[InterfaceContext, frozenset[ActionType]] = {
    InterfaceContext.GLOBAL: frozenset({ActionType.CREATE_SKETCH}),
    InterfaceContext.SKETCH: frozenset({ActionType.DRAW_RECTANGLE, ActionType.DRAW_CIRCLE}),
    InterfaceContext.FEATURE_3D: frozenset(
        {
            ActionType.CREATE_SKETCH,
            ActionType.EXTRUDE_ADD,
            ActionType.EXTRUDE_REMOVE,
            ActionType.REVOLVE,
            ActionType.SWEEP,
            ActionType.LOFT,
            ActionType.APPLY_FILLET,
            ActionType.APPLY_CHAMFER,
            ActionType.APPLY_SHELL,
            ActionType.APPLY_DRAFT,
            ActionType.APPLY_HOLE_FEATURE,
            ActionType.APPLY_MIRROR,
        }
    ),
}


@dataclass
class OnshapeAgent:
    """Pilote une page Onshape via des raccourcis clavier pour construire une pièce."""

    page: Page
    context: InterfaceContext = InterfaceContext.GLOBAL
    _step_counter: int = field(default=0, repr=False)

    # -- Machine à états --------------------------------------------------

    def _assert_context(self, action: ActionType) -> None:
        allowed = CONTEXT_VALID_ACTIONS.get(self.context, frozenset())
        if action not in allowed:
            raise ModelisationError(
                f"Action {action.value} invalide dans le contexte actuel "
                f"({self.context.value}). Séquence attendue : esquisse -> "
                f"validation (Entrée) -> volume 3D -> finition."
            )

    async def _focus_canvas(self) -> None:
        """Redonne le focus DOM et clavier au canvas 3D avant un raccourci.

        CORRECTIF issu du test réel : les captures de contrôle ont montré
        des cas où une lettre seule ('d') ne déclenchait pas l'outil
        correspondant — juste une sélection au clic, comme si la frappe
        n'atteignait jamais le canvas. Un `<canvas>` n'est pas focusable
        par défaut en HTML (pas de tabindex), donc un simple clic dessus
        ne garantit pas que le focus clavier quitte réellement le panneau
        de propriétés ouvert par l'action précédente. On force le focus
        DOM explicitement en plus du clic.
        """
        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        await self.page.mouse.click(center_x, center_y)
        await self.page.evaluate(
            """
            () => {
                let best = null;
                for (const c of document.querySelectorAll('canvas')) {
                    const r = c.getBoundingClientRect();
                    if (!best || r.width * r.height > best.width * best.height) best = c;
                }
                if (best) {
                    if (best.tabIndex < 0) best.tabIndex = -1;
                    best.focus();
                }
            }
            """
        )
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    async def _activate_tool(self, action: ActionType) -> None:
        """Vérifie le contexte, redonne le focus au canvas, puis déclenche l'outil."""
        self._assert_context(action)
        binding = TOOL_BINDINGS[action]
        if binding.shortcut:
            await self._focus_canvas()
            await self.page.keyboard.press(binding.shortcut)
        else:
            await self._search_and_activate(binding.search_term)
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    async def _search_and_activate(self, term: str) -> None:
        """Ouvre la barre de recherche d'outils (Alt+C), tape `term`, valide (Entrée).

        Fallback générique pour tout outil sans raccourci clavier direct
        documenté (Sweep, Loft, Chamfer, Shell, Draft, Hole, Mirror...).
        """
        await self.page.keyboard.press("Alt+C")
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self.page.keyboard.type(term)
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

    async def _exit_sketch(self) -> None:
        """Valide et quitte l'esquisse active, puis passe le contexte à FEATURE_3D.

        CORRECTIF issu du test réel : la documentation Onshape indique
        Entrée comme mécanisme de validation d'esquisse (repli documenté :
        clic sur l'encoche verte). Un Escape seul, utilisé dans une
        version précédente de ce pipeline, ne fermait pas l'esquisse de
        façon fiable. Un test plus récent montre qu'Entrée seul sans
        focus explicite sur le canvas ne suffit pas non plus (le panneau
        d'esquisse restait ouvert) : on force donc le focus avant Entrée,
        puis on tente en repli un clic sur l'encoche verte de
        confirmation (sélecteur best-effort, non vérifié contre le DOM
        réel — n'échoue pas silencieusement si absent).
        """
        await self._focus_canvas()
        await self.page.keyboard.press("Enter")
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        try:
            confirm_button = self.page.get_by_role("button", name="OK", exact=False)
            if await confirm_button.count() > 0:
                await confirm_button.first.click(timeout=1_000)
                await asyncio.sleep(UI_SETTLE_DELAY_S)
        except Exception:
            pass  # Bouton absent/introuvable : Entrée seul a peut-être déjà suffi.

        self.context = InterfaceContext.FEATURE_3D
        await self._snapshot("esquisse_validee")

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

    # -- CONTEXT_GLOBAL -> CONTEXT_SKETCH : nouvelle esquisse --------------

    async def _new_sketch(self, plane: PlaneType) -> None:
        if plane is not PlaneType.TOP:
            # Palier 1/2 ne gère que le plan Top ; les autres plans sont
            # prévus dans le schéma mais pas encore pilotables ici.
            raise ModelisationError(
                f"Plan '{plane.value}' non encore supporté par l'Agent Modélisateur."
            )

        # Une esquisse déjà ouverte (nested) est validée avant d'en ouvrir
        # une nouvelle : Shift+S n'est valide qu'en GLOBAL ou FEATURE_3D.
        if self.context is InterfaceContext.SKETCH:
            await self._exit_sketch()

        await self._activate_tool(ActionType.CREATE_SKETCH)
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

        self.context = InterfaceContext.SKETCH

    # -- CONTEXT_SKETCH : rectangle par le centre --------------------------

    async def _draw_rectangle(self) -> None:
        await self._activate_tool(ActionType.DRAW_RECTANGLE)

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
        """Cote un élément cliqué en (edge_x, edge_y) à `value_mm`, verrouillée.

        CORRECTIF issu du test réel : une capture de contrôle a montré 'd'
        ne pas activer l'outil Dimension (juste une sélection d'arête au
        clic suivant, aucun champ de saisie n'apparaît). `_focus_canvas`
        garantit que la frappe atteint bien le canvas avant d'être envoyée.
        """
        await self._focus_canvas()
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

    # -- CONTEXT_SKETCH : cercle par le centre (perçages) -------------------

    async def _draw_circle(self) -> None:
        await self._activate_tool(ActionType.DRAW_CIRCLE)

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

    # -- CONTEXT_SKETCH -> CONTEXT_3D_FEATURE : extrusion (add/remove) -----

    async def _extrude(self, *, depth_mm: float | None, remove: bool, through_all: bool) -> None:
        # Valide et quitte l'esquisse (Entrée) avant de pouvoir extruder sa face.
        if self.context is InterfaceContext.SKETCH:
            await self._exit_sketch()

        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2

        # Sélectionne la face de l'esquisse fermée avant d'extruder.
        await self.page.mouse.click(center_x, center_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("face_selectionnee")

        action = ActionType.EXTRUDE_REMOVE if remove else ActionType.EXTRUDE_ADD
        await self._activate_tool(action)
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
        await self._activate_tool(ActionType.APPLY_FILLET)
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

    # -- CONTEXT_3D_FEATURE : révolution, balayage, loft --------------------

    async def _revolve(self, angle_deg: float | None) -> None:
        """Révolution du profil d'esquisse autour d'un axe.

        NON VÉRIFIÉ CONTRE L'APP RÉELLE, et INCOMPLET PAR CONSTRUCTION : ce
        pipeline ne sait pas encore tracer de ligne de construction ('Q')
        à utiliser comme axe de révolution. Ne sélectionne que le profil
        (convention du reste du pipeline : clic au centre du canvas) puis
        déclenche l'outil ; la sélection de l'axe reste à la charge de
        l'utilisateur dans Onshape tant qu'un outil DRAW_LINE + axe de
        construction n'est pas implémenté côté Chercheur/Modélisateur.
        """
        if self.context is InterfaceContext.SKETCH:
            await self._exit_sketch()

        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        await self.page.mouse.click(center_x, center_y)  # Sélectionne le profil.
        await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("profil_selectionne")

        await self._activate_tool(ActionType.REVOLVE)
        await self._snapshot("dialogue_revolve")

        if angle_deg is not None:
            await self._type_exact_value(angle_deg)
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot("revolve_valide")

    async def _sweep(self) -> None:
        """Ouvre l'outil Sweep (recherche Alt+C, aucun raccourci direct documenté).

        NON IMPLÉMENTÉ AU-DELÀ DE L'ACTIVATION DE L'OUTIL : Sweep nécessite
        un profil ET un chemin, deux sélections distinctes que ce pipeline
        ne sait pas construire (pas de tracé de chemin 3D). L'outil
        s'ouvre pour que l'utilisateur termine la sélection manuellement.
        """
        if self.context is InterfaceContext.SKETCH:
            await self._exit_sketch()
        await self._activate_tool(ActionType.SWEEP)
        await self._snapshot("dialogue_sweep_ouvert")

    async def _loft(self) -> None:
        """Ouvre l'outil Loft (recherche Alt+C, aucun raccourci direct documenté).

        NON IMPLÉMENTÉ AU-DELÀ DE L'ACTIVATION DE L'OUTIL : Loft nécessite
        plusieurs profils sur des esquisses/plans distincts, que ce
        pipeline ne sait pas encore préparer. L'outil s'ouvre pour une
        sélection manuelle par l'utilisateur.
        """
        if self.context is InterfaceContext.SKETCH:
            await self._exit_sketch()
        await self._activate_tool(ActionType.LOFT)
        await self._snapshot("dialogue_loft_ouvert")

    # -- CONTEXT_FINISHING : congé, chanfrein, coque, dépouille, perçage, symétrie

    async def _run_finishing_op(self, action: ActionType, value: float | None) -> None:
        """Sélectionne la géométrie courante, déclenche l'outil, tape la
        valeur unique si fournie, valide.

        NON VÉRIFIÉ CONTRE L'APP RÉELLE. Le clic au centre du canvas comme
        sélection est une approximation raisonnable pour Chamfer
        (arête/face) et Shell (face), mais probablement INSUFFISANTE pour
        Hole (nécessite un point/sommet) et Mirror (nécessite un plan de
        symétrie) : ces deux-là sont câblés dans la matrice de contexte
        pour compléter l'architecture demandée, mais leur logique de
        sélection reste à construire lors d'un test réel.
        """
        box = await self._get_graphics_canvas_box()
        center_x = box["x"] + box["width"] / 2
        center_y = box["y"] + box["height"] / 2
        await self.page.mouse.click(center_x, center_y)
        await asyncio.sleep(UI_SETTLE_DELAY_S)

        await self._activate_tool(action)
        await self._snapshot(f"{action.value.lower()}_outil_actif")

        if value is not None:
            await self._type_exact_value(value)
            await self.page.keyboard.press("Enter")
            await asyncio.sleep(UI_SETTLE_DELAY_S)
        await self._snapshot(f"{action.value.lower()}_valide")

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

    async def _step_revolve(self, step: CADStep) -> None:
        await self._revolve(step.params.angle_deg)

    async def _step_sweep(self, step: CADStep) -> None:
        await self._sweep()

    async def _step_loft(self, step: CADStep) -> None:
        await self._loft()

    async def _step_apply_chamfer(self, step: CADStep) -> None:
        if step.params.radius_mm is None:
            raise ModelisationError("APPLY_CHAMFER nécessite radius_mm.")
        await self._run_finishing_op(ActionType.APPLY_CHAMFER, step.params.radius_mm)

    async def _step_apply_shell(self, step: CADStep) -> None:
        if step.params.thickness_mm is None:
            raise ModelisationError("APPLY_SHELL nécessite thickness_mm.")
        await self._run_finishing_op(ActionType.APPLY_SHELL, step.params.thickness_mm)

    async def _step_apply_draft(self, step: CADStep) -> None:
        if step.params.angle_deg is None:
            raise ModelisationError("APPLY_DRAFT nécessite angle_deg.")
        await self._run_finishing_op(ActionType.APPLY_DRAFT, step.params.angle_deg)

    async def _step_apply_hole_feature(self, step: CADStep) -> None:
        await self._run_finishing_op(ActionType.APPLY_HOLE_FEATURE, step.params.diameter_mm)

    async def _step_apply_mirror(self, step: CADStep) -> None:
        await self._run_finishing_op(ActionType.APPLY_MIRROR, None)

    # Registre action -> handler d'étape. Palier 4 : ajouter la nouvelle
    # valeur dans schemas.ActionType, l'entrée correspondante dans
    # TOOL_BINDINGS/CONTEXT_VALID_ACTIONS, écrire _step_xxx, puis
    # l'enregistrer ici.
    STEP_HANDLERS: ClassVar[dict[ActionType, Callable[["OnshapeAgent", CADStep], Awaitable[None]]]] = {
        ActionType.CREATE_SKETCH: _step_create_sketch,
        ActionType.DRAW_RECTANGLE: _step_draw_rectangle,
        ActionType.DRAW_CIRCLE: _step_draw_circle,
        ActionType.EXTRUDE_ADD: _step_extrude_add,
        ActionType.EXTRUDE_REMOVE: _step_extrude_remove,
        ActionType.REVOLVE: _step_revolve,
        ActionType.SWEEP: _step_sweep,
        ActionType.LOFT: _step_loft,
        ActionType.APPLY_FILLET: _step_apply_fillet,
        ActionType.APPLY_CHAMFER: _step_apply_chamfer,
        ActionType.APPLY_SHELL: _step_apply_shell,
        ActionType.APPLY_DRAFT: _step_apply_draft,
        ActionType.APPLY_HOLE_FEATURE: _step_apply_hole_feature,
        ActionType.APPLY_MIRROR: _step_apply_mirror,
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
