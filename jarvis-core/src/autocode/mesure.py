"""Phase 4 — ce que le travail vaut. Pur Python, aucun appel LLM.

Le contrat est le même pour toute tâche : **rendre la preuve la plus forte disponible**.
Les preuves forment une échelle, et le verdict dit jusqu'où l'agent est monté.

    corrigé       un test bascule rouge → vert        le défaut ET sa réparation
    reproduit     un test échoue                      le défaut existe
    gardé         un test passe, sans rien modifier   la propriété tient et reste gardée
    signalé       un constat sourcé, sans test        ce qui ne se teste pas mais se situe
    rien trouvé   ni test ni constat
    rejeté        mécaniquement irrecevable

Les quatre premiers rangs reposent sur la falsifiabilité : le code est le seul livrable
dont la réussite se vérifie par machine, et un modèle confiant ne peut pas fabriquer un
test qui échoue avant et passe après.

Le cinquième existe parce que tout ne se teste pas. Une capacité dangereuse, une garde
absente, un invariant qui ne tient que par habitude n'ont rien à faire basculer : ils
existent ou non. Sans ce rang, une trouvaille intestable serait rendue « rien trouvé » et
perdue. Ce qui reste alors vérifiable, c'est le SOURÇAGE — le transcript dit quels fichiers
ont réellement été ouverts, et un constat qui cite un chemin jamais lu est écarté. Ça
n'établit pas qu'une conclusion est juste ; ça élimine le rapport assuré rédigé sur du code
que personne n'a regardé.

Tout s'exécute dans le bac à sable de `agent/shell.py`, et c'est délibéré : ces commandes
font tourner du code que l'agent vient d'écrire. Les lancer en direct donnerait
l'exécution arbitraire sous le compte de l'utilisateur par le simple fait d'écrire un
fichier de test.
"""

import ast
import os
import re
import shutil
import sys

from config import AUTOCODE_MAX_DIFF_LINES
from helpers import get_logger

from .constats import est_protege

logger = get_logger("jarvis-autocode")

CORRIGE = "corrigé"
REPRODUIT = "reproduit"
# Aucun défaut, mais un test qui garde désormais la propriété. C'est une issue à part
# entière et non une nuance de `rien trouvé` : elle laisse un patch à relire et à
# appliquer, là où l'autre ne laisse rien. Les confondre faisait annoncer « rien trouvé »
# en tête d'un rapport qui proposait un diff — un relecteur referme à la première ligne.
GARDE = "gardé"
# Un constat sourcé, sans test. Le rang le plus bas qui laisse quelque chose à lire.
SIGNALE = "signalé"
RIEN_TROUVE = "rien trouvé"
REJETE = "rejeté"

# Livrable des constats intestables, à la racine du workspace — hors de `repo/`, donc
# absent du patch : ce n'est pas une modification du dépôt, c'est une observation.
FICHIER_REVUE = "revue.md"

# Un constat : un titre en `##`, et quelque part sous lui un `chemin:ligne` qui le situe.
#
# La citation se reconnaît à sa SEULE forme — un chemin de source suivi d'un numéro —, où
# qu'elle se trouve et quelle que soit la mise en forme autour. Exiger une ligne d'un
# gabarit précis revenait à courir après la rédaction du modèle : la même consigne a
# produit `fichier: chemin:12`, `**fichier:** chemin:12-18`, puis
# ``- **Fichier** : `chemin:12` ``, et chaque variante jetait des constats valides.
#
# L'extension de source est ce qui distingue une citation d'un `budget:40` : sans elle, un
# mot suivi d'un nombre compterait.
_TITRE_CONSTAT = re.compile(r"^##\s+(?P<titre>.+?)\s*$", re.M)
_CITATION = re.compile(
    r"(?P<chemin>[\w./-]+\.(?:py|jinja|json|sh|md|toml|yml)):(?P<ligne>\d+)"
)

_TIMEOUT = 300.0

# Ce que le test neuf a fait quand on l'a rejoué.
ASSERTION = "assertion"  # tombe sur une assertion — le défaut est démontré
ERREUR = "erreur"        # tombe à la collecte ou dans une fixture — ne démontre rien
PASSE = "passe"          # ne tombe pas

# Un fichier de test ajouté par le patch. La convention pytest du dépôt est stricte.
_EST_TEST = re.compile(r"(^|/)tests?/test_[A-Za-z0-9_]+\.py$")

# Ligne de bilan de pytest : « 2 failed, 2 errors in 0.01s ». `failed` est invariable,
# `error` se met au pluriel — les deux formes sont acceptées.
_BILAN = re.compile(r"(\d+)\s+(failed|errors?|passed)\b")

# `fichier:ligne:colonne: message`. Compter toute ligne contenant « : » surestimerait :
# sur une erreur de syntaxe, pyflakes ajoute la ligne source et son curseur.
_SIGNALEMENT = re.compile(r"^\S+:\d+:\d+: ")


def _py(module: str, *args: str) -> str:
    """Commande d'un module Python de l'interpréteur de Jarvis.

    `sys.executable` et non un chemin codé en dur : c'est celui du venv, sur n'importe
    quelle installation.
    """
    return " ".join([sys.executable, "-m", module, *args])


def comptes_pytest(sortie: str) -> dict[str, int]:
    """Échecs, erreurs et succès lus dans le bilan de pytest.

    La DISTINCTION est tout l'intérêt : un test qui tombe sur une assertion démontre le
    défaut qu'il vise ; un test qui tombe à l'import ne démontre que sa propre inaptitude
    à tourner. Les deux rendent un code de sortie non nul, et les confondre reviendrait à
    accepter une faute de frappe comme reproduction.
    """
    comptes = {"failed": 0, "error": 0, "passed": 0}
    for nombre, genre in _BILAN.findall(sortie):
        comptes["error" if genre.startswith("error") else genre] += int(nombre)
    return comptes


# Ligne de `--tb=line` : « /chemin/test_x.py:41: AssertionError: message ». Trois
# segments et non deux, ce qui la distingue d'un signalement pyflakes.
_EXCEPTION_DU_TB = re.compile(r"^\S+:\d+: ([A-Za-z_][\w.]*)", re.M)


def _issue(sortie: str) -> str:
    """Ce que le test neuf a fait : tombé sur une assertion, planté, ou passé.

    Le décompte de pytest ne suffit pas. `failed` couvre TOUTE exception levée dans le
    corps du test — un `FileExistsError` dû à un fixture bâclé y figure au même titre
    qu'une assertion qui tombe. Seules les erreurs de COLLECTE comptent comme `error`.

    Mesuré sur le cinquième cycle réel : un test dont le helper recréait un lien
    symbolique déjà présent plantait sur `FileExistsError`, et le décompte seul l'aurait
    présenté comme un défaut reproduit. On lit donc le TYPE de l'exception, que `--tb=line`
    expose, et on n'accepte que `AssertionError`.
    """
    comptes = comptes_pytest(sortie)
    if comptes["error"]:
        return ERREUR
    if not comptes["failed"]:
        return PASSE
    types = _EXCEPTION_DU_TB.findall(sortie)
    # Sans trace exploitable, on s'en tient au décompte : refuser faute de preuve
    # écarterait des reproductions valables sur un simple défaut de format.
    if types and any(t != "AssertionError" for t in types):
        return ERREUR
    return ASSERTION


def _code_sortie(sortie: str) -> int:
    """Le code rendu par `shell.executer` en tête de sa sortie. -1 s'il est absent."""
    trouve = re.match(r"\[code de sortie (-?\d+)\]", sortie.strip())
    return int(trouve.group(1)) if trouve else -1


# ── Lecture du diff ───────────────────────────────────────────────────────


def lire_diff(diff: str) -> dict[str, dict]:
    """Un enregistrement par fichier : {lignes, cree}.

    Deux pièges, tous deux traités ici plutôt qu'en trois regex séparées :

    · Une SUPPRESSION porte `+++ /dev/null`. N'attribuer les lignes qu'au côté droit la
      rendrait invisible — le fichier n'apparaîtrait ni dans les fichiers touchés ni dans
      les sources, et un patch qui efface un module se lirait « rien trouvé ».
    · `---` et `+++` ne sont des EN-TÊTES qu'avant le premier `@@` d'un fichier. Après,
      ce sont des lignes de contenu : une ligne source `++ x` apparaît `+++ x` dans le
      diff, et serait lue comme un en-tête de fichier.
    """
    fichiers: dict[str, dict] = {}
    ancien = courant = None
    dans_hunk = False

    for ligne in diff.splitlines():
        if ligne.startswith("diff --git "):
            ancien = courant = None
            dans_hunk = False
        elif ligne.startswith("@@"):
            dans_hunk = True
        elif not dans_hunk and (m := re.match(r"^--- (?:a/)?(.+)$", ligne)):
            ancien = None if m.group(1) == "/dev/null" else m.group(1)
            courant = None
        elif not dans_hunk and (m := re.match(r"^\+\+\+ (?:b/)?(.+)$", ligne)):
            courant = ancien if m.group(1) == "/dev/null" else m.group(1)
            if courant:
                fichiers.setdefault(courant, {"lignes": 0, "cree": ancien is None})
        elif dans_hunk and courant and ligne.startswith(("+", "-")):
            fichiers[courant]["lignes"] += 1

    return fichiers


def tests_ajoutes(fichiers: dict[str, dict]) -> list[str]:
    """Fichiers de test que le patch CRÉE.

    Créés et non modifiés : un test préexistant qu'on retouche ne prouve rien, il peut
    avoir été assoupli pour passer.
    """
    return sorted(
        nom for nom, info in fichiers.items() if info["cree"] and _EST_TEST.search(nom)
    )


# ── Sourçage des constats écrits ──────────────────────────────────────────


def lire_revue(workspace: str) -> str:
    """Les constats écrits par l'agent, ou une chaîne vide s'il n'en a pas écrit."""
    try:
        with open(os.path.join(workspace, FICHIER_REVUE), encoding="utf-8") as f:
            return f.read()
    except OSError:
        return ""


def _mesurer_les_constats(workspace: str, m: dict) -> None:
    """Relève les constats écrits et les `fichier:ligne` qui les situent.

    Les citations sont COMPTÉES, pas vérifiées. Un recoupement automatique contre les
    fichiers réellement ouverts a été essayé : sur six déclenchements il a rejeté cinq
    revues valides — un chemin cité par son seul nom de fichier, un résultat de recherche
    tronqué dans la trace, une forme de rédaction non prévue — pour une seule référence
    réellement fausse. Un garde qui se trompe cinq fois sur six ne protège de rien : il
    détruit le livrable et fait porter le doute sur l'agent au lieu de l'instrument.

    Ce qu'une citation garantit est donc plus modeste, et honnête : le constat est SITUÉ,
    donc vérifiable en dix secondes par qui le lit. C'est au relecteur d'ouvrir le fichier.
    """
    texte = lire_revue(workspace)
    citees = [
        (c.group("chemin"), int(c.group("ligne"))) for c in _CITATION.finditer(texte)
    ]
    m["constats_ecrits"] = [t.group("titre") for t in _TITRE_CONSTAT.finditer(texte)]
    m["citations"] = [f"{c}:{n}" for c, n in citees]


# ── Vacuité ───────────────────────────────────────────────────────────────


def _vaine(noeud: ast.Assert) -> bool:
    """Une assertion dont la condition ne référence rien ne peut rien prouver."""
    return not any(
        isinstance(f, (ast.Name, ast.Call, ast.Attribute)) for f in ast.walk(noeud.test)
    )


def test_vain(source: str) -> bool:
    """True si ce fichier de test ne peut rien démontrer.

    `assert False` échoue sur HEAD comme sur l'arbre travaillé : sans ce contrôle, il
    passerait pour une reproduction. Le garde attrape les formes vides — aucune assertion,
    ou des assertions qui ne touchent à rien. Il n'attrape PAS un test qui référence
    quelque chose mais affirme une chose hors sujet : ce jugement-là reste humain, et il
    coûte dix secondes de lecture.
    """
    try:
        arbre = ast.parse(source)
    except SyntaxError:
        return True
    asserts = [n for n in ast.walk(arbre) if isinstance(n, ast.Assert)]
    return not asserts or all(_vaine(a) for a in asserts)


# ── Exécution confinée ────────────────────────────────────────────────────


async def _lancer(task: dict, cmd: str) -> str:
    from agent import shell as sh

    return await sh.executer(task, cmd, _TIMEOUT, compter=False)


async def _pytest(task: dict, arbre: str, *cibles: str) -> str:
    """`--tb=line` : une ligne par échec, avec le TYPE de l'exception.

    C'est ce qui permet de séparer une assertion qui tombe d'un test qui plante — pytest
    les compte tous deux en `failed`. Format court à dessein : la trace complète d'une
    suite rouge remplirait la sortie bien avant d'être utile.
    """
    args = cibles or ("jarvis-core/tests/", "-m", "'not integration'")
    return await _lancer(task, f"cd {arbre} || exit 1\n" + _py("pytest", *args, "-q", "--tb=line"))


def _nb_pyflakes(sortie: str) -> int:
    return sum(1 for ligne in sortie.splitlines() if _SIGNALEMENT.match(ligne))


async def _pyflakes(task: dict, arbre: str) -> str:
    return await _lancer(
        task,
        f"cd {arbre} || exit 1\n"
        + _py("pyflakes", "jarvis-core/src/*.py", "jarvis-core/src/*/*.py"),
    )


# ── Mesure ────────────────────────────────────────────────────────────────


async def mesurer(task: dict, diff: str) -> dict:
    """Toutes les mesures d'un patch.

    Le workspace contient `repo/` (l'arbre travaillé) et `repo_ref/` (une copie vierge de
    HEAD), préparés par `chantier.py`.
    """
    workspace = task["workspace"]
    par_fichier = lire_diff(diff)
    fichiers = sorted(par_fichier)
    neufs = tests_ajoutes(par_fichier)

    m: dict = {
        "fichiers": fichiers,
        "tests_ajoutes": neufs,
        "sources_touchees": [f for f in fichiers if not _EST_TEST.search(f)],
        "lignes_diff": sum(i["lignes"] for i in par_fichier.values()),
        # Le plafond borne un CORRECTIF, pas une preuve : un test de reproduction un peu
        # long reste une preuve valable, et le rejeter perdrait la trouvaille.
        "lignes_source": sum(
            i["lignes"] for f, i in par_fichier.items() if not _EST_TEST.search(f)
        ),
        "fichiers_proteges": [f for f in fichiers if est_protege(f)],
    }

    compile_sortie = await _lancer(
        task, "cd repo || exit 1\n" + _py("compileall", "-q", "jarvis-core/src")
    )
    m["compile_ok"] = _code_sortie(compile_sortie) == 0
    m["compile_sortie"] = compile_sortie

    apres = await _pyflakes(task, "repo")
    avant = await _pyflakes(task, "repo_ref")
    m["pyflakes_avant"] = _nb_pyflakes(avant)
    m["pyflakes_apres"] = _nb_pyflakes(apres)
    m["pyflakes_sortie"] = apres

    await _eprouver_le_test(task, workspace, neufs, m)
    _mesurer_les_constats(workspace, m)

    # La suite complète ne dit quelque chose que si le test neuf passe. S'il échoue encore
    # — reproduction sans correctif —, elle est rouge PAR CONSTRUCTION : ce qu'il faut
    # savoir alors, c'est s'il a cassé autre chose, donc on la rejoue sans lui.
    suite = await _pytest(task, "repo")
    m["suite_ok"] = _code_sortie(suite) == 0
    m["suite_sortie"] = suite

    if neufs and m.get("apres") != PASSE:
        hors = await _lancer(
            task,
            "cd repo || exit 1\n"
            + _py("pytest", "jarvis-core/tests/", "-m", "'not integration'", "-q",
                  *[f"--ignore={c}" for c in neufs]),
        )
        m["suite_hors_test_ok"] = _code_sortie(hors) == 0
        m["suite_hors_test_sortie"] = hors
    return m


async def _eprouver_le_test(task: dict, workspace: str, neufs: list[str], m: dict) -> None:
    """Rejoue le test neuf sur l'arbre vierge puis sur l'arbre travaillé.

    Les deux mesures ensemble disent la distance parcourue, et aucune ne suffit seule :
    « rouge avant » sans « après » ne distingue pas un défaut prouvé d'un test cassé,
    « vert après » sans « avant » ne distingue pas un correctif d'un test complaisant.
    """
    m["avant"] = m["apres"] = None
    m["vain"] = None
    if not neufs:
        return

    m["vain"] = any(
        test_vain(contenu)
        for contenu in (_lire(os.path.join(workspace, "repo", c)) for c in neufs)
        if contenu is not None
    )

    # Le test neuf est copié DANS l'arbre vierge : c'est la seule façon de le faire courir
    # contre le code d'avant.
    for chemin in neufs:
        source = os.path.join(workspace, "repo", chemin)
        cible = os.path.join(workspace, "repo_ref", chemin)
        try:
            os.makedirs(os.path.dirname(cible), exist_ok=True)
            shutil.copy2(source, cible)
        except OSError as exc:
            logger.warning("autocode: test %s non copié vers l'arbre vierge (%s)", chemin, exc)
            return

    sortie_avant = await _pytest(task, "repo_ref", *neufs)
    m["avant"], m["avant_sortie"] = _issue(sortie_avant), sortie_avant

    sortie_apres = await _pytest(task, "repo", *neufs)
    m["apres"], m["apres_sortie"] = _issue(sortie_apres), sortie_apres


def _lire(chemin: str) -> str | None:
    try:
        with open(chemin, encoding="utf-8") as f:
            return f.read()
    except OSError:
        return None


# ── Verdict ───────────────────────────────────────────────────────────────


def _bascule(m: dict) -> bool:
    """Le test neuf est-il passé du rouge au vert ? C'est la preuve, et la seule."""
    return m["apres"] == PASSE and m["avant"] in (ASSERTION, ERREUR)


def garde_proposee(m: dict) -> bool:
    """Un test qui PASSE des deux côtés, sans rien modifier : un garde, pas une preuve.

    Il ne démontre aucun défaut — et c'est justement ce qu'on voulait savoir quand le
    constat demandait de vérifier qu'une propriété tient. Le test n'en vaut pas moins :
    la suite ne le contenait pas, et il empêchera désormais la propriété de se rompre en
    silence. Le distinguer d'une bredouille complète évite de jeter le seul livrable
    qu'une vérification réussie puisse produire.
    """
    return bool(
        m["tests_ajoutes"]
        and m["avant"] == PASSE
        and not m["sources_touchees"]
        and not m["vain"]
    )


def verdict(m: dict) -> tuple[str, list[str]]:
    """(verdict, motifs). CALCULÉ, jamais rendu par un modèle.

    Les gardes mécaniques d'abord : on ne demande au LLM que ce qu'aucun compteur ne
    tranche. Il rédige le rapport, il ne décide pas de son issue — c'est ce qui empêche un
    patch cassé d'être présenté comme un succès par une prose convaincante.
    """
    motifs: list[str] = []
    a_corrige = m["apres"] == PASSE

    # Un fichier protégé peut être EXAMINÉ — le parcours s'arrête alors à la preuve. Ce qui
    # est refusé, c'est de le modifier.
    if m["fichiers_proteges"]:
        motifs.append(f"fichiers protégés modifiés : {', '.join(m['fichiers_proteges'])}")
    if not m["compile_ok"]:
        motifs.append("la compilation échoue")
    if m["vain"]:
        motifs.append("le test ajouté ne référence rien — il ne peut rien démontrer")
    if m["avant"] == ERREUR and not a_corrige:
        # Une erreur sur HEAD ne distingue pas « la chose manque » d'« mon test est cassé ».
        # Quand le test PASSE ensuite, la seconde moitié du contrat lève l'ambiguïté : le
        # symbole était bien absent, il existe maintenant — c'est la forme normale d'un
        # défaut « la fonction manque ». Sans « après », rien ne la lève, et une faute de
        # frappe passerait pour une preuve.
        motifs.append(
            "le test ajouté tombe sur une erreur (import, collecte ou fixture) et non sur "
            "une assertion, et rien ne le corrige — il démontre qu'il ne tourne pas, pas "
            "qu'un défaut existe"
        )
    if a_corrige and m["lignes_source"] > AUTOCODE_MAX_DIFF_LINES:
        motifs.append(
            f"correctif de {m['lignes_source']} lignes (plafond {AUTOCODE_MAX_DIFF_LINES})"
        )
    if a_corrige and not m["suite_ok"]:
        motifs.append("la suite unitaire est rouge")
    if not a_corrige and m["tests_ajoutes"] and m.get("suite_hors_test_ok") is False:
        motifs.append("le test ajouté en fait tomber d'autres — la preuve n'est pas isolée")
    if m["sources_touchees"] and not _bascule(m):
        # Le contrat n'a qu'une forme, et elle commence par la preuve. Une source modifiée
        # que rien ne fait basculer du rouge au vert n'est pas « rien trouvé » — c'est une
        # modification que rien n'étaye, qu'il n'y ait PAS de test ou qu'il y en ait un qui
        # passait déjà. Le diff reste dans le dossier : refuser n'est pas jeter.
        motifs.append(
            f"sources modifiées sans test qui bascule : {', '.join(m['sources_touchees'])}"
        )
    if m["tests_ajoutes"] and m["avant"] is None:
        motifs.append("le test ajouté n'a pas pu être éprouvé contre HEAD")
    if motifs:
        return REJETE, motifs

    if not m["tests_ajoutes"]:
        # Rien de testable, mais peut-être quelque chose de situé. Le rang le plus bas qui
        # laisse de quoi lire — sans lui, une trouvaille intestable serait perdue.
        if m["constats_ecrits"] and m["citations"]:
            return SIGNALE, [
                f"{len(m['constats_ecrits'])} constat(s) sans test, "
                f"{len(m['citations'])} citation(s) vérifiée(s) contre ce qui a été ouvert"
            ]
        # Bredouille, et c'est une issue légitime : un constat dit « ceci a été observé »,
        # pas « voici le défaut ». Un modèle poussé à rapporter une prise en fabriquerait une.
        return RIEN_TROUVE, ["ni test ni constat sourcé — rien n'a pu être étayé"]
    if m["avant"] == PASSE:
        if garde_proposee(m):
            return GARDE, [
                f"aucun défaut — la propriété tient. {', '.join(m['tests_ajoutes'])} la "
                "garde désormais : elle ne pourra plus se rompre en silence"
            ]
        return RIEN_TROUVE, ["le test ajouté passe déjà sur HEAD — il décrit ce qui marchait"]

    if m["pyflakes_apres"] > m["pyflakes_avant"]:
        motifs.append(
            f"pyflakes passe de {m['pyflakes_avant']} à {m['pyflakes_apres']} signalements"
        )

    if a_corrige:
        return CORRIGE, motifs + [
            "test rouge sur HEAD et vert avec le correctif, suite verte"
        ]
    return REPRODUIT, motifs + [
        f"défaut prouvé par {', '.join(m['tests_ajoutes'])}, non corrigé"
    ]
