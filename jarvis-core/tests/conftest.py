"""Bootstrap partagé de la suite de tests.

**Le principe qui gouverne ce fichier : on importe le VRAI `config.py`.**

La suite précédente reconstruisait un faux module `config` en listant ses constantes à la
main. Chaque constante ajoutée au code sans être recopiée ici cassait la collecte pytest
avec un `ImportError` sans rapport avec le test concerné. La dérive était arrivée à
69 constantes manquantes sur 125, et la suite ne démarrait plus du tout.

Ici, `config` est importé tel quel : il ne peut plus diverger du code. Ce qu'on maîtrise,
c'est son ENVIRONNEMENT — les variables lues par `os.getenv` sont posées avant l'import,
et pointent sur des chemins jetables et un `users_list.json` de test. Aucune donnée réelle
n'entre dans les tests, et aucun seuil de production n'est deviné.

Seules les feuilles lourdes ou indisponibles hors du Mac de production sont simulées :
mlx (le moteur d'inférence), torch, transformers, sentence-transformers. Tout le reste du
code de Jarvis est exercé pour de bon.
"""

import json
import os
import pathlib
import sys
import types
from unittest.mock import MagicMock

import pytest

# ── Chemins ───────────────────────────────────────────────────────────────────
# Déduits de la position de ce fichier, jamais codés en dur : la suite précédente
# faisait `sys.path.insert(0, "/opt/jarvis/jarvis-core/src")` et ne tournait donc que
# sur une seule installation.
TESTS_DIR = pathlib.Path(__file__).resolve().parent
SRC_DIR = TESTS_DIR.parent / "src"
FIXTURES_DIR = TESTS_DIR / "fixtures"

sys.path.insert(0, str(SRC_DIR))


# ── Environnement de test, posé AVANT l'import de config ─────────────────────
def _prepare_env(tmp_root: pathlib.Path) -> None:
    """Pointe toutes les sorties disque de config vers un dossier jetable."""
    # Le .env de l'exploitant ne doit jamais entrer dans les tests : il porte ses clés
    # réelles et ses interrupteurs, et ferait passer ou échouer la suite selon la machine.
    os.environ["JARVIS_ENV_FILE"] = str(FIXTURES_DIR / "test.env")
    os.environ["USERS_LIST"] = str(FIXTURES_DIR / "users_list.json")
    os.environ["JARVIS_DATA"] = str(tmp_root)
    os.environ["SELF_MEMORY_PATH"] = str(tmp_root / "jarvis-self.json")
    os.environ["PROMPT_DATA_DIR"] = str(tmp_root / "prompts")
    os.environ["AGENT_WORKSPACE"] = str(tmp_root / "agent_workspace")
    os.environ["ROUTER_DATA_DIR"] = str(tmp_root / "router")
    os.environ["TRADE_DATA_DIR"] = str(tmp_root / "trade")
    # Pas de clé : aucun test ne doit pouvoir sortir sur le réseau par inadvertance.
    os.environ.setdefault("OPENAI_API_KEY", "")
    os.environ.setdefault("TAVILY_API_KEY", "")
    os.environ.setdefault("HF_TOKEN", "")


_TMP_ROOT = pathlib.Path(
    os.environ.get("PYTEST_TMP_ROOT") or "/tmp/jarvis-tests"
)
_TMP_ROOT.mkdir(parents=True, exist_ok=True)
_prepare_env(_TMP_ROOT)


# ── Feuilles lourdes simulées ────────────────────────────────────────────────
# mlx est une dépendance dure de `llm/local.py`, lui-même importé sans condition par
# `helpers`. Un `MagicMock` posé sur le seul nom `mlx` ne suffit pas : les imports de la
# forme `from mlx_lm.models.cache import …` exigent que CHAQUE sous-module existe dans
# sys.modules. On les enregistre donc un par un.
_HEAVY_MODULES = (
    "mlx", "mlx.core", "mlx.nn", "mlx.utils",
    "mlx_lm", "mlx_lm.models", "mlx_lm.models.cache",
    "mlx_lm.sample_utils", "mlx_lm.utils", "mlx_lm.generate",
    "mlx_vlm", "mlx_vlm.utils",
    "sentence_transformers", "torch", "transformers",
)
for _name in _HEAVY_MODULES:
    sys.modules.setdefault(_name, MagicMock())


# `qdrant_client.models.PointIdsList` est construit par le code de rétraction ; un
# MagicMock rendrait l'assertion sur son contenu inexploitable. On fournit donc une vraie
# classe, minimale, qui garde ses points lisibles.
class PointIdsList:
    def __init__(self, points):
        self.points = points


if "qdrant_client" not in sys.modules:
    _qdrant_models = types.ModuleType("qdrant_client.models")
    _qdrant_models.PointIdsList = PointIdsList
    _qdrant_models.Filter = MagicMock()
    _qdrant_models.FieldCondition = MagicMock()
    _qdrant_models.MatchValue = MagicMock()
    _qdrant = types.ModuleType("qdrant_client")
    _qdrant.models = _qdrant_models
    _qdrant.QdrantClient = MagicMock()
    sys.modules["qdrant_client"] = _qdrant
    sys.modules["qdrant_client.models"] = _qdrant_models


# ── Le vrai config ────────────────────────────────────────────────────────────
# Aliasé : le hook `pytest_configure` DOIT nommer son paramètre `config`, et un import
# du même nom au niveau module le masquerait.
import config as jarvis_config  # noqa: E402


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: nécessite un serveur sur :8000")
    config.addinivalue_line(
        "markers", "benchmark: banc de mesure — charge le modèle réel, Jarvis arrêté"
    )


# ── Cloisonnement : aucun test unitaire ne touche la mémoire de production ───

def _est_integration(item) -> bool:
    return item.get_closest_marker("integration") is not None


@pytest.fixture(autouse=True)
def pas_de_reseau(request, monkeypatch):
    """Coupe toute connexion sortante pour les tests NON marqués `integration`.

    C'est la garantie que la suite rapide ne peut pas écrire dans la mémoire de
    production. Redis (6379) et Qdrant (6333) tournent sur la machine de développement :
    un `get_redis()` oublié dans un patch se connecterait au vrai magasin et y écrirait
    sous un code utilisateur de test — pollution silencieuse d'une mémoire qui, elle, est
    conservée 90 jours.

    Plutôt que de compter sur la discipline de chaque test, on ferme la porte : toute
    connexion lève ici, en nommant l'adresse visée.
    """
    if _est_integration(request.node):
        return

    import socket

    def _refus(self, adresse):
        raise AssertionError(
            f"Connexion réseau interdite dans un test unitaire → {adresse}. "
            "Simulez le magasin (patch de get_redis / get_qdrant), ou marquez le test "
            "@pytest.mark.integration s'il doit vraiment parler au serveur."
        )

    monkeypatch.setattr(socket.socket, "connect", _refus)


# ── Fixtures ──────────────────────────────────────────────────────────────────

@pytest.fixture(scope="session")
def cfg():
    """Le module config réel, chargé sur l'environnement de test."""
    return jarvis_config


@pytest.fixture(scope="session")
def users():
    """Les utilisateurs de test, tels que config les a chargés."""
    return json.loads((FIXTURES_DIR / "users_list.json").read_text())


@pytest.fixture
def alice():
    """Code de l'utilisateur administrateur de test."""
    return "ALICE1"


@pytest.fixture
def bob():
    """Code de l'utilisateur non-administrateur de test."""
    return "BOB2"


@pytest.fixture
def session_id():
    """Identifiant de session horodaté.

    Les sessions Redis vivent 90 jours. Un identifiant fixe fait qu'un test rejoue sur le
    contexte laissé par son exécution précédente, et l'analyzer rejette alors des faits
    qu'il aurait acceptés sur une session vierge. L'horodatage rend chaque exécution
    indépendante.
    """
    import time
    return f"qa_{int(time.time() * 1000)}"


@pytest.fixture
def fake_qdrant():
    """Client Qdrant simulé, avec un `query_points` qui rend des points scorés."""
    client = MagicMock()

    def points(*scored):
        result = MagicMock()
        result.points = [
            _point(pid, score) for pid, score in scored
        ]
        client.query_points.return_value = result
        return result

    client.set_points = points
    points()
    return client


def _point(point_id, score, payload=None):
    p = MagicMock()
    p.id = point_id
    p.score = score
    p.payload = payload or {}
    return p


@pytest.fixture
def make_point():
    """Fabrique un point Qdrant scoré, pour les tests de rappel et de rétraction."""
    return _point
