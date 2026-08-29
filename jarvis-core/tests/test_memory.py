"""Mémoire : rétraction autobiographique, convlog, reconstruction des projets.

Deux pièges documentés sont gardés ici :

  • `update_user_projects` reconstruit chaque entrée CHAMP PAR CHAMP. Tout champ absent de
    la liste blanche est effacé à la fusion nocturne suivante. Le champ `due_at` a déjà
    disparu ainsi une fois.
  • La rétraction supprime dans Qdrant. Un seuil trop bas efface des souvenirs valides, et
    l'opération est irréversible.
"""

import json
from datetime import datetime, timedelta, timezone
from unittest.mock import MagicMock, patch

import pytest

import memory
from memory import episodic, vectors


# ── Rétraction autobiographique ───────────────────────────────────────────────

class TestRetractAutobiographicalEvent:

    def _run(self, points, user="ALICE1", **kwargs):
        with patch("memory.vectors._invalidate_timeline_cache") as inv, \
             patch("memory.vectors.get_qdrant") as qdrant, \
             patch("memory.vectors.get_embed_model") as embed:
            embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
            qdrant.return_value.query_points.return_value.points = points
            n = vectors.retract_autobiographical_event(user, "requête", **kwargs)
            return n, qdrant.return_value, inv

    def test_sans_resultat_ne_supprime_rien(self):
        n, qdrant, _ = self._run([])
        assert n == 0
        qdrant.delete.assert_not_called()

    def test_sous_le_seuil_ne_supprime_pas(self, make_point):
        """Le seuil par défaut est 0.88 — plus strict que l'archivage (0.78), parce que la
        rétraction supprime pour de bon."""
        n, qdrant, _ = self._run([make_point("abc-123", 0.80)])
        assert n == 0
        qdrant.delete.assert_not_called()

    def test_au_dessus_du_seuil_supprime_et_invalide_le_cache(self, make_point):
        n, qdrant, inv = self._run([make_point("abc-123", 0.92)])
        assert n == 1
        qdrant.delete.assert_called_once()
        inv.assert_called_once_with("ALICE1")

    def test_seuls_les_points_au_dessus_du_seuil_partent(self, make_point):
        n, qdrant, _ = self._run([
            make_point("garde", 0.50),
            make_point("garde-aussi", 0.85),
            make_point("supprime", 0.95),
        ])
        assert n == 1
        qdrant.delete.assert_called_once()

    def test_le_seuil_est_reglable(self, make_point):
        pts = [make_point("abc", 0.90)]
        assert self._run(pts, threshold=0.95)[0] == 0
        assert self._run(pts, threshold=0.80)[0] == 1

    def test_le_defaut_est_plus_strict_que_larchivage(self):
        """L'archivage est réversible, la rétraction non : leurs seuils ne doivent pas
        se croiser."""
        import inspect
        sig_r = inspect.signature(vectors.retract_autobiographical_event)
        sig_a = inspect.signature(vectors.archive_autobiographical_event)
        assert sig_r.parameters["threshold"].default > sig_a.parameters["threshold"].default

    def test_la_suppression_est_cloisonnee_par_utilisateur(self, make_point):
        """Un filtre utilisateur absent effacerait la mémoire de quelqu'un d'autre."""
        with patch("memory.vectors._invalidate_timeline_cache"), \
             patch("memory.vectors.get_qdrant") as qdrant, \
             patch("memory.vectors.get_embed_model") as embed:
            embed.return_value.encode.return_value.tolist.return_value = [0.1] * 384
            qdrant.return_value.query_points.return_value.points = [make_point("x", 0.99)]
            vectors.retract_autobiographical_event("BOB2", "un fait")
            appel = json.dumps(str(qdrant.return_value.query_points.call_args))
            assert "BOB2" in appel


# ── Journal de conversation ───────────────────────────────────────────────────

class TestLogConversation:

    def _entree(self, user_msg="bonjour", **kwargs):
        capture = {}
        r = MagicMock()
        r.zadd.side_effect = lambda key, mapping: capture.update(mapping)
        with patch("memory.episodic.get_redis", return_value=r):
            episodic.log_conversation(
                user_code="ALICE1",
                session_id="sess-test",
                user_msg=user_msg,
                assistant_msg="réponse.",
                importance=kwargs.pop("importance", 0.0),
                **kwargs,
            )
        assert capture, "aucune entrée convlog écrite"
        return json.loads(next(iter(capture)))

    def test_le_champ_satisfaction_est_toujours_present(self):
        """`satisfaction` s'écrase sans garde au back-fill : il doit exister dès l'écriture."""
        assert "satisfaction" in self._entree()

    def test_la_session_est_conservee(self):
        assert self._entree()["session_id"] == "sess-test"

    def test_lentree_porte_un_horodatage(self):
        e = self._entree()
        assert isinstance(e["timestamp"], (int, float))
        assert e["timestamp"] > 0


# ── Reconstruction des projets ────────────────────────────────────────────────

class TestUpdateUserProjects:
    """La liste blanche de `update_user_projects` est un piège à part entière."""

    def _ecrit(self, projects):
        with patch("memory.projects.redis_set_json") as w:
            memory.update_user_projects("ALICE1", projects)
            assert w.called, "rien n'a été écrit"
            return w.call_args[0][1]

    def _projet(self, **kw):
        base = {
            "name": "Projet test",
            "status": "in_progress",
            "first_mentioned": "2026-01-01",
            "last_update": "2026-08-01",
        }
        base.update(kw)
        return base

    @pytest.mark.parametrize(
        "champ,valeur",
        [
            ("description", "une description"),
            ("due_at", "2026-09-15"),
            ("updates", [{"date": "2026-08-01", "summary": "avancée"}]),
        ],
    )
    def test_les_champs_de_la_liste_blanche_survivent(self, champ, valeur):
        """Le test qui aurait attrapé la disparition de `due_at`."""
        out = self._ecrit([self._projet(**{champ: valeur})])
        assert champ in out[0], (
            f"`{champ}` a été effacé par la reconstruction. Tout champ absent de la liste "
            "blanche de update_user_projects disparaît à la fusion nocturne suivante."
        )
        assert out[0][champ] == valeur

    def test_les_champs_de_base_sont_toujours_ecrits(self):
        out = self._ecrit([self._projet()])
        for champ in ("name", "status", "first_mentioned", "last_update"):
            assert champ in out[0]

    def test_un_champ_inconnu_est_bien_ecarte(self):
        """Comportement voulu — le test existe pour que le suivant soit lisible."""
        out = self._ecrit([self._projet(champ_invente="x")])
        assert "champ_invente" not in out[0]

    def test_les_updates_sont_plafonnees_a_vingt(self):
        updates = [{"date": f"2026-01-{i:02d}", "summary": str(i)} for i in range(1, 26)]
        out = self._ecrit([self._projet(updates=updates)])
        assert len(out[0]["updates"]) == 20
        assert out[0]["updates"][-1]["summary"] == "25", "le plafond doit garder les plus récentes"

    def test_un_projet_termine_et_ancien_est_purge(self, cfg):
        vieux = (
            datetime.now(timezone.utc) - timedelta(days=cfg.DONE_PROJECT_TTL_DAYS + 5)
        ).isoformat()
        out = self._ecrit([self._projet(status="done", last_update=vieux)])
        assert out == []

    def test_un_projet_termine_recent_est_garde(self, cfg):
        recent = (datetime.now(timezone.utc) - timedelta(days=1)).isoformat()
        out = self._ecrit([self._projet(status="done", last_update=recent)])
        assert len(out) == 1

    def test_un_projet_en_cours_nest_jamais_purge(self):
        out = self._ecrit([self._projet(status="in_progress", last_update="2020-01-01")])
        assert len(out) == 1

    def test_une_date_illisible_ne_fait_pas_tomber_lecriture(self):
        out = self._ecrit([self._projet(status="done", last_update="pas-une-date")])
        assert len(out) == 1, "une date corrompue ne doit pas supprimer le projet"
