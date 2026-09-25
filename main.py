"""
main.py

Interface terminal interactive : ouvre UNE session Onshape persistante et
exécute, une par une, des demandes en langage naturel tapées au clavier.

Combine les deux agents dans une boucle read-eval-print :
    demande FR -> AgentChercheur.interpret_plan() -> CADPlan
                -> OnshapeAgent.execute_plan() -> retour pas à pas

Différence avec `agent_modelisateur.py` en CLI (qui ouvre un navigateur,
exécute UN plan, puis ferme) : ici le même navigateur/la même page/le même
`OnshapeAgent` (donc le même contexte de machine à états, cf.
`agent_modelisateur.InterfaceContext`) restent ouverts entre les
commandes. On peut enchaîner "crée un bloc de 8x8...", puis "ajoute un
congé de 2 mm" sur la même pièce sans tout refermer/rouvrir.

Une commande qui échoue n'interrompt pas la session : l'erreur est
affichée, une capture de diagnostic est prise, et l'utilisateur peut soit
corriger manuellement dans la fenêtre Onshape, soit reformuler sa demande.
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

from playwright.async_api import async_playwright

from agent_chercheur import AgentChercheur, InterpretationError
from agent_modelisateur import DEFAULT_USER_DATA_DIR, OnshapeAgent

_EXIT_COMMANDS = {"quit", "exit", "q", ":q"}
_REPL_ERROR_SCREENSHOT = Path("main_repl_error.png")


async def _handle_request(agent: OnshapeAgent, chercheur: AgentChercheur, request: str) -> None:
    """Interprète et exécute une demande, sans jamais laisser une exception remonter."""
    try:
        plan = chercheur.interpret_plan(request)
    except InterpretationError as exc:
        print(f"[main] Demande non comprise : {exc}")
        return

    print(f"[main] Plan ({len(plan.steps)} étape(s)) :")
    for i, step in enumerate(plan.steps, start=1):
        params = step.params.model_dump(exclude_none=True, exclude_defaults=True)
        print(f"    {i}. {step.action.value}" + (f"  {params}" if params else ""))

    try:
        await agent.execute_plan(plan)
    except Exception as exc:
        # Capture large et volontaire : une ModelisationError métier (ex.
        # action hors contexte) et une erreur Playwright bas niveau (ex.
        # sélecteur introuvable) doivent toutes deux laisser la session
        # utilisable pour la commande suivante, pas la faire crasher.
        try:
            await agent.page.screenshot(path=str(_REPL_ERROR_SCREENSHOT))
            print(f"[main] Capture de diagnostic : {_REPL_ERROR_SCREENSHOT}")
        except Exception:
            pass
        print(f"[main] Échec à l'exécution : {exc}")
        print(
            "[main] La session Onshape reste ouverte : corrigez manuellement "
            "si besoin, puis tapez une nouvelle consigne."
        )
    else:
        print("[main] Terminé.\n")


async def _repl(agent: OnshapeAgent, chercheur: AgentChercheur) -> None:
    print(
        "\nSession Onshape ouverte. Tapez une consigne en français, ex :\n"
        "  Crée un bloc de 8x8 mm de côté et 2 mm de hauteur\n"
        "  Crée une plaque de 80x80 mm, épaisseur 10 mm, avec un trou central "
        "de 37 mm et un congé de 5 mm sur les coins.\n"
        "('quit' ou Ctrl+C pour quitter)\n"
    )
    while True:
        try:
            request = (await asyncio.to_thread(input, "cao> ")).strip()
        except EOFError:
            print("\n[main] Fin de session.")
            return

        if not request:
            continue
        if request.lower() in _EXIT_COMMANDS:
            print("[main] Fin de session.")
            return

        await _handle_request(agent, chercheur, request)


async def _main_async(document_url: str, headless: bool, user_data_dir: Path) -> None:
    async with async_playwright() as playwright:
        # Contexte persistant : conserve la session Onshape (cookies/login)
        # d'une exécution à l'autre, comme un vrai profil utilisateur.
        context = await playwright.chromium.launch_persistent_context(
            user_data_dir=str(user_data_dir),
            headless=headless,
        )
        try:
            page = context.pages[0] if context.pages else await context.new_page()
            agent = OnshapeAgent(page=page)

            print(f"[main] Ouverture du document : {document_url}")
            await agent.open_document(document_url)

            await _repl(agent, AgentChercheur())
        finally:
            await context.close()


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Terminal interactif : pilote Onshape en langage naturel, "
            "une session ouverte pour plusieurs consignes successives."
        ),
    )
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
        asyncio.run(_main_async(args.document_url, args.headless, args.user_data_dir))
    except KeyboardInterrupt:
        print("\n[main] Interrompu.")
        sys.exit(0)


if __name__ == "__main__":
    main()
