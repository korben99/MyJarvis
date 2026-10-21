"""Tests pour le module cti : gardes et silences implémentatifs."""

import pytest
from unittest.mock import AsyncMock, patch


class TestRecentActivity:
    """Le silence implémentatif dans recent_activity quand les données
    sont reçues mais toutes filtrées (dict de valeurs non-dict)."""

    @pytest.mark.asyncio
    async def test_recent_activity_data_but_no_valid_entries(self):
        """Si recent_activity reçoit un dict dont les valeurs ne sont pas
        des dicts valides, elle ne doit pas retourner un header vide.

        Le code actuel fait :
          posts = list(dict.values())  # donne une liste non vide
          lines = [ ... for p in posts if isinstance(p, dict)]  # empty
          return "Publications récentes :\n" + "\n".join([])  # header vide
        Ce qu'il devrait faire : signaler qu'aucune publication valide
        n'a pu être extraite, au lieu de rester muet."""
        from agent import cti

        # Simuler un retour RansomLook : dict avec AUCUNE valeur dict
        fake_posts = {
            "some_id": "scraped_text",
            "another_id": 42,
            "last_id": ["a", "list", "not", "a", "dict"],
        }

        with patch.object(cti, "_fetch_json", new_callable=AsyncMock) as mock_fetch:
            mock_fetch.return_value = fake_posts

            result = await cti.recent_activity()

            assert result != "Publications récentes (tous groupes) :\n", (
                "recent_activity ne doit pas retourner un header vide "
                "quand aucune donnée valide n'est disponible."
            )
