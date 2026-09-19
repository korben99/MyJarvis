"""Auto-correction nocturne : constats, gardes mécaniques, calcul du verdict.

Ce qui est testé ici est exactement ce qui NE doit pas dépendre d'un modèle. Le cycle
comporte deux appels LLM, et aucun des deux ne décide quoi que ce soit : le premier choisit
dans une liste fermée — dont la fermeture est vérifiée ici —, le second rédige à partir
d'un verdict déjà calculé — dont le calcul est vérifié ici.

Un garde qui s'ouvre en silence ne se voit nulle part. C'est pourquoi les cas de REFUS sont
plus nombreux que les cas de succès.
"""

import logging
import pathlib

import pytest

from autocode import constats, mesure
from autocode.mesure import CORRIGE, REJETE, REPRODUIT, RIEN_TROUVE


# ── Constats ajoutés à la main ───────────────────────────────────────────────

_AJOUTS = """\
# Constats

## Format

```markdown
### un exemple de la documentation
cible: chemin/exemple.py
```

## Constats

### Le back-fill du convlog écrase satisfaction sans garde
cible: jarvis-core/src/analyzer.py
notes: voir la fenêtre session["msgs"]

### Deux cibles d'un coup
cible: jarvis-core/src/agent/store.py, jarvis-core/src/agent/sandbox.py

### Un constat sans cible
notes: rien à viser
"""


class TestAjoutsALaMain:
    def test_le_titre_est_le_constat(self):
        lus = constats.parser(_AJOUTS)
        assert lus[0]["constat"] == "Le back-fill du convlog écrase satisfaction sans garde"
        assert lus[0]["cible"] == ["jarvis-core/src/analyzer.py"]
        assert lus[0]["notes"] == 'voir la fenêtre session["msgs"]'

    def test_une_cible_multiple_est_decoupee(self):
        assert len(constats.parser(_AJOUTS)[1]["cible"]) == 2

    def test_un_constat_sans_cible_est_ecarte(self):
        """Un constat qui ne nomme aucun fichier n'envoie l'agent nulle part."""
        assert all("sans cible" not in c["constat"] for c in constats.parser(_AJOUTS))

    def test_un_titre_de_prose_ne_declenche_pas_d_avertissement(self, caplog):
        """Le fichier documente son propre format, et ses sous-titres sont des `###` comme
        les constats. Les signaler remplirait le journal d'un avertissement par section et
        par nuit ; un titre SANS aucun champ est de la prose, pas un constat manqué."""
        with caplog.at_level(logging.WARNING, logger="jarvis-autocode"):
            resultat = constats.parser("### Un sous-titre de documentation\n\ndu texte.\n")
        assert resultat == []
        assert not caplog.records

    def test_un_constat_avec_des_notes_mais_sans_cible_est_signale(self, caplog):
        """Celui-là, on a voulu l'écrire, et il n'envoie nulle part."""
        with caplog.at_level(logging.WARNING, logger="jarvis-autocode"):
            resultat = constats.parser("### Un vrai oubli\nnotes: mais aucune cible\n")
        assert resultat == []
        assert any("sans cible" in r.message for r in caplog.records)

    def test_un_exemple_en_bloc_de_code_n_est_pas_un_constat(self):
        """Le fichier documente son propre format. Sans cette garde, l'exemple qu'il en
        donne devient un objectif de nuit."""
        assert all("documentation" not in c["constat"] for c in constats.parser(_AJOUTS))

    def test_l_identifiant_est_derive_du_texte(self):
        """Dérivé et non choisi : un identifiant réutilisé pour un autre sujet ferait
        hériter le nouveau du sommeil de l'ancien."""
        lus = constats.parser(_AJOUTS)
        assert lus[0]["id"].startswith("MAIN-")
        assert lus[0]["id"] == constats.parser(_AJOUTS)[0]["id"]
        assert lus[0]["id"] != lus[1]["id"]

    def test_un_fichier_vide_ne_leve_pas(self):
        assert constats.parser("") == []


class TestZoneDAjout:
    """Le fichier porte son mode d'emploi ET les constats. Sans frontière explicite, un
    `###` de documentation devenait un candidat, et seule l'absence de champs l'écartait —
    une garde par défaut plutôt qu'une frontière."""

    _FICHIER = (
        "# Mode d'emploi\n\n"
        "### Un titre de documentation\n"
        "cible: ceci/ne/doit/pas/etre/lu.py\n\n"
        f"{constats.SEPARATEUR}\n\n"
        "### Un vrai constat\n"
        "cible: jarvis-core/src/rag.py\n"
    )

    def test_seule_la_zone_apres_le_separateur_est_lue(self):
        lus = constats.parser(self._FICHIER)
        assert [c["constat"] for c in lus] == ["Un vrai constat"]

    def test_sans_separateur_le_fichier_entier_est_lu(self):
        """Compatibilité : un fichier qui n'en porte pas garde l'ancien comportement."""
        sans = self._FICHIER.replace(constats.SEPARATEUR, "")
        assert len(constats.parser(sans)) == 2

    def test_le_separateur_est_un_commentaire_markdown(self):
        """Invisible au rendu : le fichier reste lisible tel quel."""
        assert constats.SEPARATEUR.startswith("<!--") and constats.SEPARATEUR.endswith("-->")

    def test_le_fichier_reel_delimite_sa_zone(self):
        from config import AUTOCODE_POOL_FILE

        try:
            texte = pathlib.Path(AUTOCODE_POOL_FILE).read_text(encoding="utf-8")
        except OSError:
            pytest.skip("aucun fichier de constats sur cette installation")
        assert constats.SEPARATEUR in texte, "la zone d'ajout doit être délimitée"


class TestListeFermee:
    """Un id inventé est un refus, jamais un rapprochement vers le plus ressemblant."""

    def test_un_id_connu_est_retrouve(self):
        lus = constats.parser(_AJOUTS)
        assert constats.par_id(lus, lus[1]["id"])["cible"][0].endswith("store.py")

    def test_un_id_inconnu_rend_none(self):
        assert constats.par_id(constats.parser(_AJOUTS), "MAIN-deadbeef") is None

    def test_un_id_approchant_ne_matche_pas(self):
        lus = constats.parser(_AJOUTS)
        assert constats.par_id(lus, lus[0]["id"][:-1]) is None


# ── Constats tirés du journal ────────────────────────────────────────────────

_TRACEBACK = """\
2026-09-14 10:00:00  jarvis-self               ERROR  Cycle en échec
Traceback (most recent call last):
  File "/opt/jarvis/jarvis-core/src/self/engine.py", line 12, in _reflechir
    resultat = etape["reason"]
  File "/opt/jarvis/jarvis-core/src/agent/sandbox.py", line 60, in resolve
    raise SandboxError(chemin)
KeyError: 'reason'
2026-09-14 10:00:01  jarvis-api                INFO  suite du journal
"""


@pytest.fixture
def journal_lisible(monkeypatch):
    monkeypatch.setattr(constats, "_fenetres_incidents", lambda *a, **k: [])
    monkeypatch.setattr(constats.os.path, "isfile", lambda _: True)


class TestNormalisation:
    """Un message d'exception peut recopier une génération entière — donc, selon l'appel,
    un extrait de conversation. Il part dans le prompt, le titre et le courriel."""

    def test_la_charge_json_est_coupee(self):
        depuis = constats.normaliser(
            "Invalid JSON in LLM response: {'focus': 'un extrait de conversation'}"
        )
        assert depuis == "Invalid JSON in LLM response"
        assert "conversation" not in depuis

    def test_un_errno_n_est_pas_de_la_charge(self):
        """Couper sur « [ » viderait le message : un crochet ouvre bien plus souvent un
        [Errno N] qu'un tableau JSON."""
        assert "No such file" in constats.normaliser(
            "[Errno 2] No such file or directory: '/opt/x/y.json'"
        )

    def test_les_chiffres_et_chemins_sont_gommes(self):
        """Deux occurrences du même défaut doivent partager une empreinte, sinon le
        cooldown ne tient sur aucune des deux."""
        assert constats.normaliser("échec ligne 41 de /opt/a.py") == \
               constats.normaliser("échec ligne 77 de /opt/b.py")

    def test_un_message_entierement_charge_utile_ne_devient_pas_vide(self):
        assert constats.normaliser("{'tout': 'est charge utile'}").strip()

    def test_le_message_est_borne(self):
        assert len(constats.normaliser("x" * 500)) <= constats._MESSAGE_MAX_CARS


class TestDepuisLeJournal:
    def test_la_frame_la_plus_interne_est_retenue(self, journal_lisible):
        """Les frames suivantes appartiennent aux bibliothèques, la première n'est que le
        point d'entrée : c'est la dernière frame du projet qui désigne le code à lire."""
        trouves = constats._depuis_tracebacks(_TRACEBACK)
        assert len(trouves) == 1
        assert trouves[0]["cible"] == ["jarvis-core/src/agent/sandbox.py"]
        assert "KeyError: 'reason'" in trouves[0]["constat"]

    def test_un_fichier_disparu_est_ignore(self, monkeypatch):
        """`self.py` et `helpers.py` ont été scindés en paquets ; leurs tracebacks sont
        encore au journal. Viser un chemin mort enverrait lire ce qui n'existe plus."""
        monkeypatch.setattr(constats, "_fenetres_incidents", lambda *a, **k: [])
        monkeypatch.setattr(constats.os.path, "isfile", lambda _: False)
        assert constats._depuis_tracebacks(_TRACEBACK) == []

    def test_un_traceback_trop_vieux_est_ignore(self, journal_lisible):
        assert constats._depuis_tracebacks(_TRACEBACK.replace("2026-09-14", "2020-01-01")) == []

    def test_deux_occurrences_se_regroupent(self, journal_lisible):
        """Sinon la liste se remplirait de doublons et le cooldown ne tiendrait sur aucun."""
        trouves = constats._depuis_tracebacks(_TRACEBACK + _TRACEBACK)
        assert len(trouves) == 1 and trouves[0]["poids"] == 2

    def test_le_constat_a_la_meme_forme_qu_un_ajout_manuel(self, journal_lisible):
        """Une source de plus ne doit pas être une machinerie de plus : le choix, la mesure
        et le journal ne connaissent qu'un seul genre de constat."""
        auto = constats._depuis_tracebacks(_TRACEBACK)[0]
        manuel = constats.parser(_AJOUTS)[0]
        assert set(auto) == set(manuel)
        assert auto["id"].startswith("SIG-")
        assert "SIG-" in constats.rendre([auto])


# ── Gardes mécaniques ────────────────────────────────────────────────────────


class TestGardes:
    """Appelés AVANT le modèle : on ne lui présente jamais une cible impraticable."""

    @pytest.fixture(autouse=True)
    def _sans_redis(self, monkeypatch):
        monkeypatch.setattr(constats.store, "en_cooldown", lambda _: False)

    def test_une_cible_trop_longue_est_ecartee(self, monkeypatch):
        """Au-delà, la lecture ne tient plus en un appel — et la pagination est le mode
        d'échec le mieux établi de la boucle."""
        monkeypatch.setattr(constats, "_lignes_du_fichier", lambda _: 1800)
        entree = {"id": "X", "cible": ["jarvis-core/src/llm/local.py"]}
        assert "trop longue" in constats.motif_exclusion(entree, set())

    def test_un_constat_deja_traite_sort_de_la_liste(self):
        assert constats.motif_exclusion({"id": "X", "cible": []}, {"X"}) == "déjà traité"

    def test_un_constat_en_sommeil_est_ecarte(self, monkeypatch):
        monkeypatch.setattr(constats.store, "en_cooldown", lambda _: True)
        assert "sommeil" in constats.motif_exclusion({"id": "X", "cible": []}, set())

    def test_un_fichier_protege_reste_examinable(self):
        """Protéger, c'est interdire la MODIFICATION, pas la lecture. Le parcours peut
        s'arrêter à la preuve, qui ne modifie rien — écarter ces fichiers ferait des plus
        sensibles les seuls qu'on ne puisse jamais examiner."""
        entree = {"id": "X", "cible": ["jarvis-core/src/prompts.py"]}
        assert constats.motif_exclusion(entree, set()) is None
        assert constats.est_protege("jarvis-core/src/prompts.py")

    def test_une_cible_inexistante_reste_eligible(self, monkeypatch):
        """Un fichier à créer n'est pas trop long pour une lecture qu'on ne fera pas."""
        monkeypatch.setattr(constats, "_lignes_du_fichier", lambda _: 0)
        assert constats.motif_exclusion({"id": "X", "cible": ["src/neuf.py"]}, set()) is None


# ── Lecture du bilan pytest ──────────────────────────────────────────────────


class TestComptesPytest:
    """Bilans relevés sur pytest 8 — `failed` est invariable, `error` se met au pluriel."""

    @pytest.mark.parametrize(
        "sortie, attendu",
        [
            ("1 failed in 0.01s", {"failed": 1, "error": 0, "passed": 0}),
            ("2 failed in 0.01s", {"failed": 2, "error": 0, "passed": 0}),
            ("1 error in 0.01s", {"failed": 0, "error": 1, "passed": 0}),
            ("2 errors in 0.01s", {"failed": 0, "error": 2, "passed": 0}),
            ("2 failed, 2 errors in 0.01s", {"failed": 2, "error": 2, "passed": 0}),
            ("108 passed in 1.81s", {"failed": 0, "error": 0, "passed": 108}),
            ("107 passed, 33 deselected in 1.78s", {"failed": 0, "error": 0, "passed": 107}),
        ],
    )
    def test_les_bilans_reels_sont_lus(self, sortie, attendu):
        assert mesure.comptes_pytest(f"[code de sortie 1]\n{sortie}") == attendu

    def test_un_bilan_absent_ne_leve_pas(self):
        assert mesure.comptes_pytest("") == {"failed": 0, "error": 0, "passed": 0}


class TestTypeDeLEchec:
    """`failed` couvre TOUTE exception levée dans le corps d'un test — un helper bâclé qui
    plante y figure au même titre qu'une assertion qui tombe. Mesuré sur le cinquième cycle
    réel : un `FileExistsError` serait passé pour un défaut reproduit."""

    def test_une_assertion_qui_tombe_demontre(self):
        assert mesure._issue(
            "[code de sortie 1]\n/x/test_a.py:2: AssertionError: attendu 3\n1 failed in 0.01s"
        ) == mesure.ASSERTION

    def test_un_test_qui_plante_ne_demontre_rien(self):
        assert mesure._issue(
            "[code de sortie 1]\n/x/test_b.py:41: FileExistsError: existe\n1 failed in 0.01s"
        ) == mesure.ERREUR

    def test_un_seul_echec_parasite_suffit_a_disqualifier(self):
        """Un lot mêlant une vraie assertion et un test cassé n'est pas une preuve
        propre : le second peut masquer ce que le premier prétend montrer."""
        assert mesure._issue(
            "[code de sortie 1]\n/x/a.py:2: AssertionError: x\n"
            "/x/b.py:4: KeyError: y\n2 failed in 0.01s"
        ) == mesure.ERREUR

    def test_une_erreur_de_collecte_reste_une_erreur(self):
        assert mesure._issue(
            "[code de sortie 1]\nERROR test_c.py\n1 error in 0.01s"
        ) == mesure.ERREUR

    def test_une_suite_verte_ne_demontre_rien(self):
        assert mesure._issue("[code de sortie 0]\n108 passed in 1.8s") == mesure.PASSE

    def test_sans_trace_exploitable_on_s_en_tient_au_decompte(self):
        """Refuser faute de trace écarterait des reproductions valables sur un simple
        défaut de format de sortie."""
        assert mesure._issue(
            "[code de sortie 1]\n1 failed in 0.01s"
        ) == mesure.ASSERTION


# ── Lecture du diff ──────────────────────────────────────────────────────────

_DIFF = """\
diff --git a/jarvis-core/src/rag.py b/jarvis-core/src/rag.py
--- a/jarvis-core/src/rag.py
+++ b/jarvis-core/src/rag.py
@@ -1,3 +1,4 @@
 import os
+import sys
-import json
diff --git a/jarvis-core/tests/test_rag_neuf.py b/jarvis-core/tests/test_rag_neuf.py
--- /dev/null
+++ b/jarvis-core/tests/test_rag_neuf.py
@@ -0,0 +1,2 @@
+def test_neuf():
+    assert False
"""


class TestLectureDuDiff:
    def test_les_lignes_sont_comptees_par_fichier(self):
        lu = mesure.lire_diff(_DIFF)
        assert {f: i["lignes"] for f, i in lu.items()} == {
            "jarvis-core/src/rag.py": 2,
            "jarvis-core/tests/test_rag_neuf.py": 2,
        }

    def test_seuls_les_tests_CREES_comptent(self):
        """Un test préexistant qu'on retouche ne prouve rien : il a pu être assoupli."""
        assert mesure.tests_ajoutes(mesure.lire_diff(_DIFF)) == [
            "jarvis-core/tests/test_rag_neuf.py"
        ]

    def test_un_fichier_source_neuf_n_est_pas_un_test(self):
        diff = "--- /dev/null\n+++ b/jarvis-core/src/neuf.py\n@@ -0,0 +1 @@\n+x = 1\n"
        assert mesure.tests_ajoutes(mesure.lire_diff(diff)) == []

    def test_une_suppression_est_comptee(self):
        """Une suppression porte `+++ /dev/null`. N'attribuer les lignes qu'au côté droit
        la rendrait invisible : un patch qui efface un module se lirait « rien trouvé »."""
        diff = (
            "diff --git a/jarvis-core/src/rag.py b/jarvis-core/src/rag.py\n"
            "deleted file mode 100644\n"
            "--- a/jarvis-core/src/rag.py\n+++ /dev/null\n"
            "@@ -1,2 +0,0 @@\n-import os\n-x = 1\n"
        )
        lu = mesure.lire_diff(diff)
        assert lu == {"jarvis-core/src/rag.py": {"lignes": 2, "cree": False}}
        assert not mesure._EST_TEST.search("jarvis-core/src/rag.py")

    def test_une_ligne_de_contenu_n_est_pas_lue_comme_un_entete(self):
        """Une ligne source « ++ x » apparaît « +++ x » dans le diff. Hors en-tête, c'est
        du contenu — la confondre ferait compter les lignes sur un fichier inventé."""
        diff = (
            "diff --git a/jarvis-core/src/a.py b/jarvis-core/src/a.py\n"
            "--- a/jarvis-core/src/a.py\n+++ b/jarvis-core/src/a.py\n"
            "@@ -1,2 +1,3 @@\n"
            "+++ b/faux.py\n"
            "--- a/faux.py\n"
            " contexte\n"
        )
        assert sorted(mesure.lire_diff(diff)) == ["jarvis-core/src/a.py"]
        assert mesure.lire_diff(diff)["jarvis-core/src/a.py"]["lignes"] == 2


# ── Vacuité ──────────────────────────────────────────────────────────────────


class TestVacuite:
    """`assert False` échoue sur HEAD comme après : sans ce garde, il passerait pour une
    preuve."""

    @pytest.mark.parametrize("source", [
        "def test_x():\n    pass\n",
        "def test_x():\n    assert True\n",
        "def test_x():\n    assert 1 == 1\n",
        "import os\n\ndef test_x():\n    assert 2 + 2 == 4\n",
        "def test_x(:\n",  # ne compile pas
    ])
    def test_les_formes_vides_sont_refusees(self, source):
        assert mesure.test_vain(source)

    @pytest.mark.parametrize("source", [
        "from agent.sandbox import resolve\n\ndef test_x():\n    assert resolve('t', 'x') is None\n",
        "from x import f\n\ndef test_a():\n    assert True\n\ndef test_b():\n    assert f()\n",
    ])
    def test_un_test_qui_touche_quelque_chose_passe(self, source):
        assert not mesure.test_vain(source)


# ── Calcul du verdict ────────────────────────────────────────────────────────


def _m(**surcharges) -> dict:
    """Un parcours nominal : test rouge sur HEAD, vert après, suite verte."""
    base = {
        "fichiers": ["jarvis-core/src/rag.py", "jarvis-core/tests/test_n.py"],
        "sources_touchees": ["jarvis-core/src/rag.py"],
        "tests_ajoutes": ["jarvis-core/tests/test_n.py"],
        "lignes_diff": 14, "lignes_source": 8,
        "fichiers_proteges": [],
        "compile_ok": True, "suite_ok": True,
        "pyflakes_avant": 3, "pyflakes_apres": 3,
        "avant": mesure.ASSERTION, "apres": mesure.PASSE, "vain": False,
        "constats_ecrits": [], "citations": [],
        "citations_hors_lecture": [], "citations_hors_bornes": [],
    }
    base.update(surcharges)
    return base


class TestVerdict:
    """CALCULÉ, jamais rendu par un modèle : c'est ce qui empêche une prose convaincante
    de présenter un patch cassé comme un succès."""

    def test_rouge_puis_vert_est_corrige(self):
        assert mesure.verdict(_m())[0] == CORRIGE

    def test_rouge_puis_rouge_est_reproduit(self):
        """Une preuve sans correctif est un résultat : elle transforme un soupçon en fait."""
        verdict, motifs = mesure.verdict(_m(
            apres=mesure.ASSERTION, sources_touchees=[], lignes_source=0,
            fichiers=["jarvis-core/tests/test_n.py"], suite_ok=False,
            suite_hors_test_ok=True,
        ))
        assert verdict == REPRODUIT
        assert any("non corrigé" in x for x in motifs)

    def test_la_suite_rouge_ne_rejette_pas_une_preuve(self):
        """Elle EST rouge quand le test neuf tombe encore. La juger telle quelle
        rejetterait toute reproduction valable."""
        assert mesure.verdict(_m(
            apres=mesure.ASSERTION, sources_touchees=[], lignes_source=0,
            fichiers=["jarvis-core/tests/test_n.py"], suite_ok=False,
            suite_hors_test_ok=True,
        ))[0] == REPRODUIT

    def test_casser_d_autres_tests_fait_rejeter(self):
        assert mesure.verdict(_m(
            apres=mesure.ASSERTION, sources_touchees=[], lignes_source=0,
            fichiers=["jarvis-core/tests/test_n.py"], suite_ok=False,
            suite_hors_test_ok=False,
        ))[0] == REJETE

    def test_aucun_test_est_rien_trouve(self):
        """Un constat dit « ceci a été observé », pas « voici le défaut »."""
        assert mesure.verdict(_m(
            tests_ajoutes=[], sources_touchees=[], fichiers=[], lignes_source=0,
            avant=None, apres=None, vain=None,
        ))[0] == RIEN_TROUVE

    def test_un_test_qui_passe_sans_rien_toucher_devient_un_garde(self):
        """`rien trouvé` et `gardé` sont deux issues, pas une nuance : la seconde laisse un
        patch à relire. Les confondre annonçait « rien trouvé » en tête d'un rapport qui
        proposait un diff — un relecteur referme à la première ligne."""
        assert mesure.verdict(_m(
            avant=mesure.PASSE, apres=mesure.PASSE,
            sources_touchees=[], fichiers=["jarvis-core/tests/test_n.py"], lignes_source=0,
        ))[0] == mesure.GARDE

    def test_un_test_vain_qui_passe_ne_garde_rien(self):
        assert mesure.verdict(_m(
            avant=mesure.PASSE, apres=mesure.PASSE, vain=True,
            sources_touchees=[], fichiers=["jarvis-core/tests/test_n.py"], lignes_source=0,
        ))[0] == REJETE


    def test_une_erreur_sans_correctif_fait_rejeter(self):
        """Un import cassé rend un code de sortie non nul tout autant qu'une assertion.
        Sans « après » pour lever l'ambiguïté, une faute de frappe passerait pour une
        preuve."""
        verdict, motifs = mesure.verdict(_m(
            avant=mesure.ERREUR, apres=mesure.ERREUR, sources_touchees=[],
            lignes_source=0, fichiers=["jarvis-core/tests/test_n.py"],
            suite_ok=False, suite_hors_test_ok=True,
        ))
        assert verdict == REJETE
        assert any("assertion" in x for x in motifs)

    def test_une_erreur_rattrapee_par_le_correctif_est_acceptee(self):
        """« La fonction manque » produit forcément un ImportError sur HEAD. Le test qui
        passe ENSUITE lève l'ambiguïté : le symbole était bien absent, il existe
        maintenant. Rejeter ce cas exclurait toute une famille de défauts légitimes."""
        assert mesure.verdict(_m(avant=mesure.ERREUR, apres=mesure.PASSE))[0] == CORRIGE

    def test_un_test_vain_fait_rejeter(self):
        assert mesure.verdict(_m(vain=True))[0] == REJETE

    def test_un_correctif_sans_test_fait_rejeter(self):
        """Le contrat commence par la preuve. Un correctif sans test n'est pas « rien
        trouvé » : c'est une modification que rien n'étaye."""
        verdict, motifs = mesure.verdict(_m(
            tests_ajoutes=[], avant=None, apres=None, vain=None,
        ))
        assert verdict == REJETE
        assert any("bascule" in x for x in motifs)

    def test_un_correctif_avec_un_test_qui_ne_bascule_pas_fait_rejeter(self):
        """Même trou, autre forme : le test existe mais passait DÉJÀ. Rien ne dit que la
        modification serve à quelque chose."""
        verdict, motifs = mesure.verdict(_m(avant=mesure.PASSE, apres=mesure.PASSE))
        assert verdict == REJETE
        assert any("bascule" in x for x in motifs)

    def test_un_test_non_eprouve_fait_rejeter(self):
        assert mesure.verdict(_m(avant=None))[0] == REJETE

    def test_un_fichier_protege_modifie_fait_rejeter(self):
        assert mesure.verdict(
            _m(fichiers_proteges=["jarvis-core/src/config.py"])
        )[0] == REJETE

    def test_une_compilation_en_echec_fait_rejeter(self):
        assert mesure.verdict(_m(compile_ok=False))[0] == REJETE

    def test_une_suite_rouge_avec_correctif_fait_rejeter(self):
        assert mesure.verdict(_m(suite_ok=False))[0] == REJETE

    def test_le_plafond_borne_le_correctif_pas_la_preuve(self):
        """Un test de reproduction un peu long reste une preuve valable ; le rejeter
        perdrait la trouvaille."""
        assert mesure.verdict(_m(lignes_source=9000))[0] == REJETE
        assert mesure.verdict(_m(
            apres=mesure.ASSERTION, lignes_source=0, sources_touchees=[],
            fichiers=["jarvis-core/tests/test_n.py"], lignes_diff=9000,
            suite_ok=False, suite_hors_test_ok=True,
        ))[0] == REPRODUIT

    def test_une_regression_pyflakes_est_signalee_sans_rejeter(self):
        verdict, motifs = mesure.verdict(_m(pyflakes_avant=3, pyflakes_apres=7))
        assert verdict == CORRIGE
        assert any("pyflakes" in x for x in motifs)

    def test_moins_de_signalements_pyflakes_ne_penalise_pas(self):
        """On juge une RÉGRESSION, pas un état : le dépôt en a déjà."""
        assert mesure.verdict(_m(pyflakes_avant=7, pyflakes_apres=3))[0] == CORRIGE

    def test_un_rejet_prime_sur_une_issue_positive(self):
        assert mesure.verdict(_m(compile_ok=False, vain=True))[0] == REJETE


class TestGardeProposee:
    """Un constat peut demander de VÉRIFIER qu'une propriété tient. Si elle tient, le test
    ne démontre aucun défaut — et c'est la réponse attendue. Le jeter perdrait le seul
    livrable qu'une vérification réussie puisse produire."""

    def _verification(self, **s):
        base = _m(avant=mesure.PASSE, apres=mesure.PASSE, sources_touchees=[],
                  fichiers=["jarvis-core/tests/test_n.py"], lignes_source=0, vain=False)
        base.update(s)
        return base

    def test_un_test_qui_passe_sans_toucher_aux_sources_est_garde(self):
        m = self._verification()
        assert mesure.garde_proposee(m)
        verdict, motifs = mesure.verdict(m)
        assert verdict == mesure.GARDE
        assert any("garde" in x for x in motifs)

    def test_garde_et_rien_trouve_sont_deux_issues_distinctes(self):
        """L'une laisse un patch à relire, l'autre ne laisse rien — et le cycle les traite
        différemment : créneau de décision contre cooldown."""
        assert mesure.GARDE != mesure.RIEN_TROUVE

    def test_un_test_vain_n_est_pas_un_garde(self):
        """`assert True` passe aussi, et ne garde rien."""
        assert not mesure.garde_proposee(self._verification(vain=True))

    def test_une_source_touchee_annule_le_garde(self):
        """Le test ne garde plus HEAD : il décrit un arbre déjà modifié."""
        assert not mesure.garde_proposee(
            self._verification(sources_touchees=["jarvis-core/src/rag.py"])
        )

    def test_sans_test_il_n_y_a_rien_a_garder(self):
        assert not mesure.garde_proposee(
            self._verification(tests_ajoutes=[], avant=None, apres=None, vain=None)
        )

    def test_un_test_qui_bascule_n_est_pas_un_garde(self):
        """Celui-là prouve un défaut : c'est une autre issue, pas une garde."""
        assert not mesure.garde_proposee(_m())


class TestConstatSource:
    """Tout ne se teste pas. Une capacité dangereuse ou une garde absente n'ont rien à
    faire basculer : sans ce rang, la trouvaille serait rendue « rien trouvé » et perdue.
    Ce qui reste vérifiable, c'est que le constat cite ce qui a vraiment été ouvert."""

    def _revue(self, **s):
        base = _m(tests_ajoutes=[], sources_touchees=[], fichiers=[], lignes_source=0,
                  lignes_diff=0, avant=None, apres=None, vain=None,
                  constats_ecrits=["Le module X expose un arrêt brutal"],
                  citations=["jarvis-core/src/x.py:12"])
        base.update(s)
        return base

    def test_un_constat_source_sans_test_est_signale(self):
        verdict, motifs = mesure.verdict(self._revue())
        assert verdict == mesure.SIGNALE
        assert any("constat" in x for x in motifs)

    def test_la_citation_survit_a_l_emphase_et_aux_plages_de_lignes(self):
        """Un modèle qui rédige en markdown écrit `**fichier:**` et `:292-295`. Refuser
        ces formes jetait six constats sourcés au motif qu'ils portaient des astérisques."""
        texte = (
            "## Un titre\n"
            "**fichier:** jarvis-core/src/main.py:292-295\n"
            "## Un autre\n"
            "fichier: jarvis-core/src/vitals.py:64\n"
        )
        trouves = [
            (c.group("chemin"), c.group("ligne"))
            for c in mesure._CITATION.finditer(texte)
        ]
        assert trouves == [
            ("jarvis-core/src/main.py", "292"),
            ("jarvis-core/src/vitals.py", "64"),
        ]

    def test_un_chemin_sans_numero_de_ligne_n_est_pas_une_citation(self):
        """Sans ligne, le constat n'est pas situable — donc pas vérifiable en dix secondes,
        ce qui est tout ce qu'une citation promet."""
        assert not mesure._CITATION.findall("**fichier:** jarvis-core/src/deps.py\n")

    def test_une_reference_en_pleine_prose_compte(self):
        """La garantie ne vient pas de l'endroit où la citation est écrite, mais du
        recoupement avec ce qui a été ouvert. Exiger un gabarit de ligne revenait à courir
        après la rédaction du modèle, qui a produit quatre formes en huit exécutions."""
        assert mesure._CITATION.findall(
            "vu dans le fichier jarvis-core/src/x.py:12 au passage\n"
        ) == [("jarvis-core/src/x.py", "12")]

    def test_les_quatre_formes_deja_produites_sont_reconnues(self):
        """Chacune a été rendue par une exécution réelle, sous la même consigne."""
        formes = (
            "fichier: jarvis-core/src/a.py:12\n",
            "**fichier:** jarvis-core/src/a.py:12-18\n",
            "- **Fichier** : `jarvis-core/src/a.py:12`\n",
            "**fichier :** `jarvis-core/src/a.py:12-18`\n",
        )
        for forme in formes:
            assert mesure._CITATION.findall(forme) == [("jarvis-core/src/a.py", "12")], forme

    def test_un_mot_suivi_d_un_nombre_n_est_pas_une_citation(self):
        """Sans extension de source, ce n'est pas un chemin — sinon `budget:40` compterait."""
        assert not mesure._CITATION.findall("budget:40 — pas 12:30\n")

    def test_une_citation_n_est_plus_recoupee_automatiquement(self):
        """Le recoupement contre les fichiers ouverts a rejeté cinq revues valides sur six
        déclenchements : un garde qui se trompe cinq fois sur six détruit le livrable et
        fait porter le doute sur l'agent au lieu de l'instrument. C'est au relecteur
        d'ouvrir le fichier — la citation ne promet que de lui dire où."""
        verdict, _ = mesure.verdict(self._revue())
        assert verdict == mesure.SIGNALE

    def test_un_constat_sans_citation_ne_signale_rien(self):
        assert mesure.verdict(self._revue(citations=[]))[0] == RIEN_TROUVE

    def test_sans_constat_ni_test_c_est_rien_trouve(self):
        assert mesure.verdict(self._revue(constats_ecrits=[], citations=[]))[0] == RIEN_TROUVE

    def test_un_test_prime_sur_un_constat(self):
        """L'échelle descend : une preuve exécutable vaut mieux qu'une phrase située."""
        assert mesure.verdict(_m(constats_ecrits=["x"], citations=["a.py:1"]))[0] == CORRIGE


# ── Jeu d'outils et budgets par origine ──────────────────────────────────────


class TestOutilsParOrigine:
    """Le jeu d'outils est fonction de la TÂCHE : activer une capacité pour l'une ne doit
    pas l'activer pour l'autre."""

    def test_l_autocoding_a_verify(self):
        from agent.tools import VERIFY, names_for

        assert VERIFY in names_for({"origin": "autocode"})

    def test_une_tache_humaine_n_a_pas_verify(self):
        from agent.tools import VERIFY, names_for

        assert VERIFY not in names_for({"origin": "human"})

    def test_l_autocoding_n_a_ni_web_ni_rag_ni_shell(self):
        from agent.tools import names_for

        outils = names_for({"origin": "autocode"})
        assert not outils & {"web_search", "fetch_url", "search_docs", "threat_intel", "shell"}

    def test_l_autocoding_garde_de_quoi_travailler(self):
        from agent.tools import FINISH, PLAN, names_for

        assert {PLAN, "read_file", "write_file", "list_dir", FINISH} <= names_for(
            {"origin": "autocode"}
        )

    def test_une_origine_absente_vaut_humaine(self):
        from agent.tools import TOOL_SCHEMAS, schemas_for

        assert schemas_for({}) is TOOL_SCHEMAS


class TestBudgets:
    def test_le_budget_de_la_tache_prime(self):
        from agent.loop import _max_steps, _timeout_minutes

        assert _max_steps({"max_steps": 40}) == 40
        assert _timeout_minutes({"timeout_minutes": 90}) == 90

    def test_un_budget_absent_retombe_sur_la_constante(self):
        from agent.loop import _max_steps
        from config import AGENT_MAX_STEPS

        assert _max_steps({}) == AGENT_MAX_STEPS
        assert _max_steps({"max_steps": 0}) == AGENT_MAX_STEPS


# ── Le cycle ne lève jamais ──────────────────────────────────────────────────


class TestRevue:
    """Un cycle sans cible désignée. Même échelle de preuves, même canal — seule la
    manière d'arriver devant le code change."""

    def test_l_objectif_d_une_revue_ne_designe_aucune_cible(self):
        from autocode import chantier

        objectif = chantier.construire_objectif(
            {"id": "REVUE-2026-01-01", "constat": "", "cible": [],
             "origine": chantier.ORIGINE_REVUE}
        )
        assert "revue.md" in objectif.lower()
        assert "CONSTAT :" not in objectif

    def test_un_constat_designe_garde_le_contrat_ordinaire(self):
        from autocode import chantier

        objectif = chantier.construire_objectif(
            {"id": "SIG-1", "constat": "le cache ne purge pas",
             "cible": ["jarvis-core/src/exemple.py"], "origine": "traceback"}
        )
        assert "le cache ne purge pas" in objectif

    def test_un_perimetre_borne_l_objectif_de_revue(self):
        """Le dépôt est trop grand pour quarante pas : restreindre le champ concentre la
        revue au lieu de la laisser suivre sa première impulsion."""
        from autocode import chantier

        objectif = chantier.construire_objectif(
            {"id": "REVUE-x", "origine": chantier.ORIGINE_REVUE,
             "perimetre": "jarvis-core/src/memory"}
        )
        assert "repo/jarvis-core/src/memory" in objectif
        assert objectif.startswith("PÉRIMÈTRE")

    def test_une_revue_sans_perimetre_ne_le_mentionne_pas(self):
        from autocode import chantier

        objectif = chantier.construire_objectif(
            {"id": "REVUE-x", "origine": chantier.ORIGINE_REVUE, "perimetre": ""}
        )
        assert "PÉRIMÈTRE" not in objectif
        assert "jarvis-core/src/exemple.py" in objectif

    def test_une_revue_saute_le_recueil_et_le_choix(self, monkeypatch):
        """Il n'y a rien à recueillir ni à départager : la cible d'une revue est
        l'absence de cible."""
        import asyncio

        import autocode

        def _interdit(*a, **k):
            raise AssertionError("phase sautée appelée")

        monkeypatch.setattr(autocode, "AUTOCODE_ENABLED", True)
        monkeypatch.setattr(autocode, "AGENT_ENABLED", True)
        monkeypatch.setattr(autocode, "USER_ADMINS", {"ADMIN"})
        monkeypatch.setattr(autocode.store, "patch_en_attente", lambda: None)
        monkeypatch.setattr(autocode.constats, "recueillir", _interdit)
        monkeypatch.setattr(autocode.choix, "choisir", _interdit)

        rendu = asyncio.run(autocode.run_nightly_autocode(dry_run=True, revue=True))
        assert rendu["cible"].startswith("REVUE-")
        assert rendu["origine"] == autocode.chantier.ORIGINE_REVUE

    def test_le_verrou_d_une_revue_est_distinct_de_celui_de_la_nuit(self, monkeypatch):
        """Sinon une revue lancée à la main serait refusée au motif que le cycle nocturne
        est déjà passé, et lui volerait son créneau du lendemain."""
        import asyncio

        import autocode

        pris = []

        monkeypatch.setattr(autocode, "AUTOCODE_ENABLED", True)
        monkeypatch.setattr(autocode, "AGENT_ENABLED", True)
        monkeypatch.setattr(autocode, "USER_ADMINS", {"ADMIN"})
        monkeypatch.setattr(autocode.store, "patch_en_attente", lambda: None)
        monkeypatch.setattr(autocode.store, "prendre_verrou",
                            lambda cle: pris.append(cle) or False)

        asyncio.run(autocode.run_nightly_autocode(revue=True))
        assert pris and pris[0].endswith("-revue")

    def test_le_rapport_d_une_revue_renvoie_aux_constats_pas_a_git_apply(self):
        from autocode import bilan, mesure

        rapport = bilan.rendre_rapport(
            {"id": "REVUE-2026-01-01", "constat": "revue de maintenance",
             "origine": "revue"},
            _m(constats_ecrits=["Le cache n'est jamais purgé"],
               citations=["jarvis-core/src/exemple.py:128"]),
            mesure.SIGNALE, ["1 constat sans test"], {}, {"steps": 12}, "autocode/2026-01-01",
        )
        assert "git apply" not in rapport
        assert mesure.FICHIER_REVUE in rapport
        assert "1, 1 situé(s) par fichier:ligne" in rapport

    def test_un_verdict_signale_ne_promet_pas_un_git_apply_dans_le_chat(self, monkeypatch):
        """Accepter une revue, c'est l'avoir lue — il n'y a pas de patch à appliquer."""
        import autocode

        monkeypatch.setattr(autocode, "USER_ADMINS", {"ADMIN"})
        monkeypatch.setattr(autocode.store, "patch_en_attente", lambda: {
            "constat_id": "REVUE-2026-01-01", "verdict": autocode.mesure.SIGNALE,
            "dossier": "autocode/2026-01-01",
        })
        monkeypatch.setattr(autocode.store, "trancher", lambda *a: True)

        reponse = autocode.handle_autocode_command(
            "accepte le patch REVUE-2026-01-01", "ADMIN"
        )
        assert "git apply" not in reponse
        assert "autocode/2026-01-01" in reponse


class TestCycleRobuste:
    """Le job planifié n'a personne pour rattraper une exception : Redis indisponible
    ferait disparaître le cycle sans un mot."""

    def test_une_panne_de_magasin_rend_un_compte_rendu(self, monkeypatch):
        import asyncio

        import autocode

        def _tombe():
            raise ConnectionError("magasin injoignable")

        monkeypatch.setattr(autocode, "AUTOCODE_ENABLED", True)
        monkeypatch.setattr(autocode, "AGENT_ENABLED", True)
        monkeypatch.setattr(autocode.store, "patch_en_attente", _tombe)

        rendu = asyncio.run(autocode.run_nightly_autocode(dry_run=True))
        assert rendu["lance"] is False
        assert "ConnectionError" in rendu["motif"]

    def test_un_interrupteur_ferme_court_circuite_tout(self, monkeypatch):
        import asyncio

        import autocode

        monkeypatch.setattr(autocode, "AUTOCODE_ENABLED", False)
        assert asyncio.run(autocode.run_nightly_autocode())["lance"] is False

    def test_les_appels_de_cadrage_ont_leur_propre_journal(self, monkeypatch):
        """Sans ça, choisir un constat et rédiger le rapport tombaient dans prompts.log,
        au milieu du trafic de chat — alors que la boucle qu'ils encadrent écrit dans
        agent-prompts.log. Les deux moitiés d'un cycle dans deux fichiers sans rapport."""
        import asyncio

        import autocode
        from llm.local import _AUTOCODE_PROMPTS_LOG_PATH, _cycle_journal

        vus = []

        async def _espion(dry_run, revue=False, perimetre=""):
            vus.append(_cycle_journal.get())
            return {"lance": False, "motif": "espion"}

        monkeypatch.setattr(autocode, "_cycle", _espion)
        asyncio.run(autocode.run_nightly_autocode(dry_run=True))

        assert vus == [_AUTOCODE_PROMPTS_LOG_PATH]
        assert _cycle_journal.get() is None, "le journal doit être rendu après le cycle"

    def test_le_journal_est_rendu_meme_sur_erreur(self, monkeypatch):
        """Un journal laissé posé détournerait ensuite le trafic de chat."""
        import asyncio

        import autocode
        from llm.local import _cycle_journal

        async def _tombe(dry_run):
            raise RuntimeError("panne")

        monkeypatch.setattr(autocode, "_cycle", _tombe)
        asyncio.run(autocode.run_nightly_autocode(dry_run=True))
        assert _cycle_journal.get() is None

    @pytest.mark.parametrize("statut", ["cancelled", "failed", "interrupted"])
    def test_une_tache_non_terminee_ne_produit_aucun_verdict(self, monkeypatch, statut):
        """Mesurer une tâche annulée rendrait « rien trouvé » sur un diff vide, et ce
        verdict poserait un cooldown de 30 jours sur un constat que personne n'a examiné."""
        import asyncio

        import autocode

        pose = []
        task = {"id": "t1", "workspace": "/tmp/x", "user_code": "A", "status": statut}
        monkeypatch.setattr(autocode.chantier, "lancer", _async(task))
        monkeypatch.setattr(autocode.chantier, "attendre", _async(task))
        monkeypatch.setattr(autocode.chantier, "nettoyer", _async(None))
        monkeypatch.setattr(autocode.store, "poser_cooldown", lambda c: pose.append(c))
        monkeypatch.setattr(autocode.store, "journaliser", lambda e: pose.append("journal"))

        rendu = asyncio.run(autocode._executer("A", {"id": "C1", "constat": "x"}, [], []))
        assert rendu["motif"] == statut
        assert pose == [], "ni cooldown ni journal sur une tâche non terminée"


def _async(valeur):
    async def _f(*a, **k):
        return valeur
    return _f


# ── Décision humaine — le chemin de retour ───────────────────────────────────


class TestCommandeDecision:
    """Sans ce chemin, on n'a pas ajouté une boucle mais un émetteur : le cycle
    reproposerait indéfiniment ce qui vient d'être écarté."""

    @pytest.fixture
    def commande(self, monkeypatch):
        import autocode

        tranches = []
        en_vol = {"constat_id": "SIG-a1b2c3d4", "constat": "un constat",
                  "verdict": CORRIGE, "dossier": "/tmp/autocode/2026-09-15-SIG-a1b2c3d4"}
        monkeypatch.setattr(autocode.store, "patch_en_attente", lambda: en_vol)
        monkeypatch.setattr(
            autocode.store, "trancher",
            lambda cid, dec: (tranches.append((cid, dec)), True)[1],
        )
        monkeypatch.setattr(autocode, "USER_ADMINS", {"ADMIN1"})
        return autocode.handle_autocode_command, tranches

    def test_un_message_ordinaire_n_est_pas_une_commande(self, commande):
        handle, _ = commande
        assert handle("il fait beau aujourd'hui", "ADMIN1") is None
        assert handle("explique-moi ce patch de sécurité linux", "ADMIN1") is None

    def test_accepter_tranche_en_acceptation(self, commande):
        handle, tranches = commande
        assert "SIG-a1b2c3d4" in handle("accepte le patch SIG-a1b2c3d4", "ADMIN1")
        assert tranches == [("SIG-a1b2c3d4", "accepte")]

    def test_rejeter_tranche_en_rejet(self, commande):
        handle, tranches = commande
        handle("rejette le patch SIG-a1b2c3d4", "ADMIN1")
        assert tranches == [("SIG-a1b2c3d4", "rejete")]

    def test_l_id_est_insensible_a_la_casse(self, commande):
        """Le message est passé en minuscules, pas les identifiants."""
        handle, tranches = commande
        handle("Accepte le patch sig-A1B2C3D4", "ADMIN1")
        assert tranches == [("SIG-a1b2c3d4", "accepte")]

    def test_un_id_qui_ne_correspond_pas_ne_tranche_rien(self, commande):
        handle, tranches = commande
        assert "SIG-a1b2c3d4" in handle("accepte le patch SIG-99999999", "ADMIN1")
        assert tranches == []

    def test_un_id_manquant_ne_tranche_rien(self, commande):
        handle, tranches = commande
        assert "identifiant" in handle("accepte le patch", "ADMIN1")
        assert tranches == []

    def test_un_non_administrateur_ne_tranche_rien(self, commande):
        handle, tranches = commande
        assert "administrateur" in handle("accepte le patch SIG-a1b2c3d4", "AUTRE")
        assert tranches == []

    def test_la_consultation_ne_tranche_rien(self, commande):
        handle, tranches = commande
        assert "SIG-a1b2c3d4" in handle("montre les patchs en attente", "ADMIN1")
        assert tranches == []
