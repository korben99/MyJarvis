"""Verrou GPU : une acquisition annulée ne doit pas emporter le verrou avec elle.

Le 09/09/2026, Jarvis a servi `/docs` pendant trois heures sans produire un seul token, sans
rien journaliser. Un client s'était déconnecté pendant qu'une requête attendait le GPU.

`asyncio.to_thread(_infer_lock.acquire)` n'est pas annulable : un `threading.Lock.acquire()`
bloquant ne s'interrompt pas. À l'annulation, la coroutine part — mais le thread continue,
finit par prendre le verrou, et plus personne ne le relâche, puisque le `finally` qui s'en
charge appartient au bloc que la coroutine n'atteindra jamais. Toute inférence ultérieure se
bloque alors, chat comme tâche de fond.

Ces tests exercent la vraie fonction, sans modèle : le verrou est un `threading.Lock` de la
bibliothèque standard, et mlx est simulé par conftest.
"""

import asyncio
import threading
import time

import pytest

import llm.local as local


@pytest.fixture(autouse=True)
def _etat_verrou_propre():
    """Rend le module à son état neutre : ces globales sont partagées par la suite."""
    yield
    if local._infer_lock.locked():
        try:
            local._infer_lock.release()
        except RuntimeError:
            pass
    with local._chat_waiters_lock:
        local._chat_waiters = 0
    local._bg_wakeup = None
    local._bg_loop = None


def _attendre_liberation(delai: float = 3.0) -> bool:
    """Le thread orphelin s'exécute hors boucle asyncio : on lui laisse le temps d'aboutir."""
    fin = time.monotonic() + delai
    while time.monotonic() < fin:
        if local._infer_lock.acquire(blocking=False):
            local._infer_lock.release()
            return True
        time.sleep(0.02)
    return False


def test_acquisition_annulee_ne_confisque_pas_le_verrou(caplog):
    """Le scénario exact de l'incident : occupé → en attente → déconnexion → relâche."""

    async def scenario():
        assert local._infer_lock.acquire(blocking=False), "verrou déjà pris"

        tache = asyncio.create_task(local._acquire_infer_lock_chat(time.time()))
        await asyncio.sleep(0.2)  # la requête est bloquée sur l'acquisition
        assert not tache.done(), "l'attente ne s'est pas produite, le scénario est vide"

        tache.cancel()  # ← le client se déconnecte
        with pytest.raises(asyncio.CancelledError):
            await tache

        local._infer_lock.release()  # la génération en cours se termine
        await asyncio.sleep(0.3)  # laisse le rappel de fin s'exécuter

    with caplog.at_level("WARNING", logger="jarvis-llm-local"):
        asyncio.run(scenario())

    assert _attendre_liberation(), (
        "le verrou est resté pris après annulation — toute inférence suivante se bloquerait"
    )
    assert any("orpheline" in m for m in caplog.messages), (
        "l'acquisition orpheline n'a pas eu lieu : le test ne prouve rien"
    )


def test_le_compteur_de_chat_revient_a_zero():
    """Un attendeur qui disparaît sans se décompter starverait les appels de fond."""

    async def scenario():
        assert local._infer_lock.acquire(blocking=False)
        tache = asyncio.create_task(local._acquire_infer_lock_chat(time.time()))
        await asyncio.sleep(0.2)
        assert local._chat_waiters == 1, "l'attendeur ne s'est pas déclaré"
        tache.cancel()
        with pytest.raises(asyncio.CancelledError):
            await tache
        assert local._chat_waiters == 0, "attendeur fantôme"
        local._infer_lock.release()
        await asyncio.sleep(0.3)

    asyncio.run(scenario())
    assert _attendre_liberation()


def test_chemin_nominal_rend_le_verrou_au_caller():
    """Sans annulation, la fonction rend le verrou pris — le caller doit pouvoir le relâcher."""

    async def scenario():
        attendu = await local._acquire_infer_lock_chat(time.time())
        assert local._infer_lock.locked(), "le verrou n'est pas pris au retour"
        assert attendu >= 0.0
        local._infer_lock.release()

    asyncio.run(scenario())
    assert local._chat_waiters == 0


def test_attente_reelle_puis_acquisition():
    """Le verrou occupé par un tiers est bien attendu, puis obtenu — pas de faux positif."""

    async def scenario():
        assert local._infer_lock.acquire(blocking=False)
        threading.Timer(0.3, local._infer_lock.release).start()

        t0 = time.monotonic()
        await local._acquire_infer_lock_chat(time.time())
        assert time.monotonic() - t0 >= 0.25, "obtenu sans attendre : le verrou n'était pas tenu"
        assert local._infer_lock.locked()
        local._infer_lock.release()

    asyncio.run(scenario())
