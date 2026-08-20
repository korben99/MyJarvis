import json
import logging
import os

from dotenv import load_dotenv

load_dotenv("/opt/jarvis/.env")

logger = logging.getLogger("jarvis-config")

# ── Shared OpenAI credentials (fallback for any tier not explicitly configured) ──
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
OPENAI_API_URL = os.getenv(
    "OPENAI_API_URL", os.getenv("OPENAI_URL", "https://api.openai.com/v1")
)

# ── Tier 1 — Router model (fast, cheap intent classifier) ─────────────────
# Now:    GPT-4.1-nano          → set nothing, uses OPENAI credentials
# Future: Qwen3-7B via mlx-lm  → set ROUTER_API_URL + ROUTER_API_KEY + ROUTER_MODEL
# Disable LLM router entirely:   set ROUTER_MODEL=""  (falls back to embedding router)
ROUTER_API_URL = os.getenv("ROUTER_API_URL") or OPENAI_API_URL
ROUTER_API_KEY = os.getenv("ROUTER_API_KEY") or OPENAI_API_KEY
ROUTER_MODEL = os.getenv(
    "ROUTER_MODEL", "gpt-4.1-nano"
)  # empty string intentionally disables LLM router
ROUTER_TIMEOUT = float(os.getenv("ROUTER_TIMEOUT") or "6")

# ── Tier 2 — Primary model (standard chat, trading, briefing, self-reflection) ──
# Now:    GPT-4o-mini              → set nothing, uses OPENAI credentials
PRIMARY_MODEL = os.getenv("PRIMARY_MODEL", "gpt-4o-mini")
PRIMARY_API_URL = os.getenv("PRIMARY_API_URL") or OPENAI_API_URL
PRIMARY_API_KEY = os.getenv("PRIMARY_API_KEY") or OPENAI_API_KEY
PRIMARY_TIMEOUT = float(os.getenv("PRIMARY_TIMEOUT") or "120")

# ── Tier 3 — Reasoning model (complex queries only, cloud-gated) ─────────
# Only reached when the router sets use_reasoning=True.
REASONING_MODEL = os.getenv("REASONING_MODEL") or PRIMARY_MODEL
REASONING_API_URL = os.getenv("REASONING_API_URL") or OPENAI_API_URL
REASONING_API_KEY = os.getenv("REASONING_API_KEY") or OPENAI_API_KEY
REASONING_TIMEOUT = float(os.getenv("REASONING_TIMEOUT") or "180")

# ── Vision model (image description — first stage of two-stage pipeline) ──
# Set to a vision-capable model (Qwen2.5-VL, gpt-4o, gpt-5.1, …).
# Leave empty to disable image support (images will be ignored with a warning).
VISION_MODEL = os.getenv("VISION_MODEL", "mlx-community/Qwen2.5-VL-7B-Instruct-4bit")
VISION_API_URL = os.getenv("VISION_API_URL") or OPENAI_API_URL
VISION_API_KEY = os.getenv("VISION_API_KEY") or OPENAI_API_KEY
VISION_TIMEOUT = float(os.getenv("VISION_TIMEOUT") or "60")

# ── Ninja-patch chat template (Qwen3.6 only) ─────────────────────────────
# Chemin local du fichier chat_template.optional.jinja téléchargé depuis HF.
# Téléchargement : voir scripts/download_models.py (section TEMPLATES).
# Remplace le chat_template par défaut du tokenizer Qwen3.6 :
#   enable_thinking=False → aucun tag <think> (vs standard : <think>\n\n</think>\n\n)
# Configurable via env QWEN36_NINJA_TEMPLATE pour pointer vers un autre chemin.
# Ne s'applique qu'à Qwen3.6 (is_qwen36) : le fichier est lié à ce tokenizer précis.
QWEN36_NINJA_TEMPLATE = os.getenv(
    "QWEN36_NINJA_TEMPLATE",
    "/opt/jarvis/models/templates/qwen36_ninja.jinja",
)

# ── Familles Qwen3 à architecture hybride ────────────────────────────────
# Marqueurs de version reconnus par is_qwen3_hybrid(). Surchargeable par env
# (liste séparée par des virgules) pour accueillir une génération suivante sans
# toucher au code — la seule condition est qu'elle partage l'architecture.
QWEN3_HYBRID_VERSIONS = tuple(
    v.strip().lower()
    for v in os.getenv("QWEN3_HYBRID_VERSIONS", "qwen3.5,qwen3.6,qwen3.8").split(",")
    if v.strip()
)

# Effort de réflexion pour Qwen3.8 (template qwen3_8 : low / medium / xhigh).
# Vide = on ne passe pas le kwarg, le modèle applique son propre défaut (xhigh).
# À budget de réflexion constant, xhigh se fait couper par ThinkingBudgetProcessor
# plus souvent : passer à "medium" si les réponses arrivent tronquées après bascule.
QWEN38_REASONING_EFFORT = os.getenv("QWEN38_REASONING_EFFORT", "").strip().lower()
if QWEN38_REASONING_EFFORT and QWEN38_REASONING_EFFORT not in ("low", "medium", "xhigh"):
    logger.warning(
        "QWEN38_REASONING_EFFORT=%r invalide (low|medium|xhigh) — ignoré",
        QWEN38_REASONING_EFFORT,
    )
    QWEN38_REASONING_EFFORT = ""

# ── Local LLM mode — Apple Silicon / mlx-lm (M4 Pro) ─────────────────────
# Activé par LLM_LOCAL=yes dans .env.
# Écrase Router et Primary pour pointer vers les serveurs mlx-lm locaux.
LLM_LOCAL = os.getenv("LLM_LOCAL", "").lower() in ("yes", "true", "1")

if LLM_LOCAL:
    # Mode import direct MLX — pas de serveurs HTTP mlx-lm.
    # helpers.py route vers call_llm_local / call_llm_local_async directement.
    # Les API_URL / API_KEY ne sont pas utilisées pour l'inférence en mode local.
    # Stock default — downloadable by scripts/download_models.py out of the box.
    # A custom LoRA-fine-tuned router (see scripts/router_lora_adapterv1.py) is a
    # later optional upgrade, not required for a working install.
    ROUTER_MODEL = os.getenv(
        "ROUTER_MODEL_LOCAL", "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
    )
    PRIMARY_MODEL = os.getenv(
        "PRIMARY_MODEL_LOCAL", "spicyneuron/Qwen3.6-35B-A3B-MLX-5.4bit"
    )
    REASONING_MODEL = os.getenv("REASONING_MODEL_LOCAL") or PRIMARY_MODEL
    VISION_MODEL = os.getenv(
        "VISION_MODEL_LOCAL", "lmstudio-community/Qwen3-VL-8B-Instruct-MLX-5bit"
    )
    logger.info(
        "Mode LLM local activé (import direct MLX) — router: %s  primary: %s  vision: %s (local)",
        ROUTER_MODEL,
        PRIMARY_MODEL,
        VISION_MODEL,
    )


# True when mlx_vlm local inference is active for image description.
VISION_LOCAL = LLM_LOCAL and bool(VISION_MODEL)


# ── Model compatibility helpers ───────────────────────────────────────────
def is_qwen(model: str) -> bool:
    """True for Qwen models served locally via mlx-lm or Ollama."""
    return "qwen" in (model or "").lower()


def is_qwen25(model: str) -> bool:
    """True for Qwen2.5.x models — no thinking mode, deterministic JSON output."""
    m = (model or "").lower()
    return "qwen2.5" in m or "qwen25" in m


def is_qwen3(model: str) -> bool:
    """True for Qwen3.x models — supports enable_thinking + thinking_budget in apply_chat_template.
    llm_local._build_prompt catches TypeError if a kwarg isn't available (e.g. old tokenizer version
    or Qwen3.5 open-source where thinking_budget is Alibaba Cloud API only)."""
    return "qwen3" in (model or "").lower()


def is_qwen3_hybrid(model: str) -> bool:
    """True for the Qwen3 "hybrid" generations (3.5 / 3.6 / 3.8 — same architecture).

    Subset of is_qwen3() — always check is_qwen3_hybrid() BEFORE is_qwen3() to avoid
    shadowing. Ces générations partagent (source : ms-swift, template qwen3_5/3_6/3_8)
    une architecture identique, d'où un comportement commun côté inférence :
      - Profil d'échantillonnage dédié (temp_think=1.0 vs 0.6 pour Qwen3 de base).
      - Attention hybride full/linear → ArraysCache non trimmable, cf. llm/local.py:62.
      - Aucun support de <budget_remaining> : le budget de réflexion n'est pas piloté
        par le template mais par ThinkingBudgetProcessor, au niveau des logits.

    La liste est surchargeable par env (QWEN3_HYBRID_VERSIONS) pour accueillir une
    génération suivante sans changement de code.
    """
    m = (model or "").lower()
    return any(v in m for v in QWEN3_HYBRID_VERSIONS)


def is_qwen36(model: str) -> bool:
    """True specifically for Qwen3.6.x models (e.g. spicyneuron/Qwen3.6-35B-A3B-MLX-*).

    Portée volontairement étroite : ne sert plus qu'à décider de l'application du ninja
    patch, qui est un fichier Jinja lié au tokenizer 3.6 précis (QWEN36_NINJA_TEMPLATE).
    Pour tout trait de comportement partagé par la génération, utiliser is_qwen3_hybrid().
    """
    return "qwen3.6" in (model or "").lower()


def is_qwen38(model: str) -> bool:
    """True for Qwen3.8.x models (Qwen3.8-27B, Qwen3.8-35B-A3B, Qwen3.8-2.4T-A95B).

    Sous-ensemble de is_qwen3_hybrid(). Deux écarts vs 3.5/3.6, documentés par le
    template qwen3_8 de ms-swift :
      - `reasoning_effort` accepté : xhigh (défaut modèle) / medium / low.
        Voir QWEN38_REASONING_EFFORT — xhigh produit des blocs de réflexion nettement
        plus longs, qui se feront couper par ThinkingBudgetProcessor aux budgets actuels.
      - Le contenu de réflexion des tours passés est conservé par défaut au rendu.
        Sans effet ici : l'historique Jarvis ne stocke que des réponses déjà nettoyées
        de leur bloc <think>.
    """
    return "qwen3.8" in (model or "").lower()


def is_hermes(model: str) -> bool:
    """True for NousResearch Hermes models (Llama base, purpose-built structured output).
    No thinking mode; optimal at temperature=0 with no repetition/frequency penalty."""
    return "hermes" in (model or "").lower()


# ══════════════════════════════════════════════════════════════════════════
#  LLM BUDGETS — token budgets et timeouts
#
#  ThinkingBudgetProcessor (llm_local.py) force </think> via logits après exactement
#  THINKING_BUDGET_* tokens — hard cut avec phase soft (boost progressif sur les 10%
#  finaux). La valeur est précise : un budget trop court tronque le raisonnement.
#
#  Timeout formula : max_tokens / TOKEN_SPEED_TPS * TIMEOUT_MARGIN
#  llm_timeout(n) applique cette formule avec un plancher à 10 s.
# ══════════════════════════════════════════════════════════════════════════

# ── Vitesse de génération (mesurée, MLX + queue lock, conservateur) ──────
TOKEN_SPEED_TPS = float(os.getenv("TOKEN_SPEED_TPS", "50"))  # tok/s
TIMEOUT_MARGIN = float(os.getenv("TIMEOUT_MARGIN", "1.3"))  # marge sécurité


def llm_timeout(max_tokens: int) -> float:
    """Timeout en secondes pour un appel LLM local, plancher à 10 s."""
    return max(10.0, max_tokens / TOKEN_SPEED_TPS * TIMEOUT_MARGIN)


# ── no_think — output seul, pas de bloc think ────────────────────────────
MAX_TOKENS_TINY = int(os.getenv("MAX_TOKENS_TINY", "80"))  # ticker, query rewriter
MAX_TOKENS_SHORT = int(os.getenv("MAX_TOKENS_SHORT", "300"))  # router, normalize, judge
MAX_TOKENS_COMPACT = int(
    os.getenv("MAX_TOKENS_COMPACT", "600")
)  # push, cleaning, consolidate
MAX_TOKENS_MEDIUM = int(
    os.getenv("MAX_TOKENS_MEDIUM", "1000")
)  # analyzer, reflection, alerts
MAX_TOKENS_NO_THINK = int(
    os.getenv("MAX_TOKENS_NO_THINK", "1500")
)  # chat simple, nightly self
MAX_TOKENS_BRIEFING = int(os.getenv("MAX_TOKENS_BRIEFING", "3000"))  # briefing assembly

# ── think — THINKING_BUDGET_* + headroom réponse ────────────────────────
# Le ThinkingBudgetProcessor coupe le think exactement à THINKING_BUDGET_* tokens.
# max_tokens doit toujours > thinking_budget + taille_réponse_attendue.
THINKING_BUDGET_COMPACT = int(
    os.getenv("THINKING_BUDGET_COMPACT", "1024")
)  # classification, décision binaire
THINKING_BUDGET_MEDIUM = int(
    os.getenv("THINKING_BUDGET_MEDIUM", "2048")
)  # raisonnement modéré
THINKING_BUDGET_DEEP = int(
    os.getenv("THINKING_BUDGET_DEEP", "4000")
)  # créativité, analyse longue

MAX_TOKENS_THINK_COMPACT = THINKING_BUDGET_COMPACT + 1024  # prune, action_review
MAX_TOKENS_THINK_MEDIUM = THINKING_BUDGET_MEDIUM + 3000  # trading thresholds
MAX_TOKENS_SYNTHESIS = int(os.getenv("MAX_TOKENS_SYNTHESIS", "8000"))  # chat web/RAG
MAX_TOKENS_REASONING = int(
    os.getenv("MAX_TOKENS_REASONING", "10000")
)  # refine_prompt, chat reasoning

# ── Hard cap (kill switch runaway — indépendant des budgets ci-dessus) ───
MAX_TOKENS_HARD_CAP = int(os.getenv("MAX_TOKENS_HARD_CAP", "16000"))

# Sentinel température : None = utilise le profil du modèle (défini dans llm_local._model_profile).
# Passer 0.0 signifie greedy réel. Passer une valeur >0 signifie température explicite.
DEFAULT_TEMP: float | None = None

# ── Historique conversationnel (chat.py) ─────────────────────────────────
HIST_CONV_TOKEN_BUDGET = int(os.getenv("HIST_CONV_TOKEN_BUDGET", "1000"))
SESSION_SUMMARY_TOKENS = int(os.getenv("SESSION_SUMMARY_TOKENS", "600"))
HIST_CONV_SUMMARIZE_THRESHOLD = int(os.getenv("HIST_CONV_SUMMARIZE_THRESHOLD", "1500"))
PROFILE_NARRATIVE_TOKENS = int(os.getenv("PROFILE_NARRATIVE_TOKENS", "600"))

# ThinkingBudgetProcessor : activé via USE_THINKING_BUDGET_PROCESSOR=yes
USE_THINKING_BUDGET_PROCESSOR = (
    os.getenv("USE_THINKING_BUDGET_PROCESSOR", "no").lower() == "yes"
)


def tokens_param(model: str) -> str:
    """
    Return the correct token-limit parameter name for the model.
    OpenAI o1/o3/gpt-5.x require 'max_completion_tokens'.
    All others (gpt-4o, Qwen via mlx-lm, Gemini-compat, etc.) use 'max_tokens'.

    Usage: json={..., **{tokens_param(REASONING_MODEL): 512}, ...}
    """
    _needs_completion = ("o1-", "o3-", "gpt-5", "gpt-4.5")
    return (
        "max_completion_tokens"
        if any(x in (model or "") for x in _needs_completion)
        else "max_tokens"
    )


# ── External search APIs ──────────────────────────────────────────────────
# Tavily: primary web search backend (designed for LLM agents).
# Leave empty to disable — DDG deep pipeline is used as fallback.
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "")

# ── Infrastructure ────────────────────────────────────────────────────────
REDIS_URL = os.getenv("REDIS_URL", "redis://redis:6379")
QDRANT_URL = os.getenv("QDRANT_URL", "http://qdrant:6333")
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "open-webui_knowledge")
QDRANT_MEMORY_COLLECTION = os.getenv("QDRANT_MEMORY_COLLECTION", "jarvis_memory")
RAG_TOP_K = int(os.getenv("RAG_TOP_K", os.getenv("QDRANT_TOP_K", "8")))
# Seuil de l'ÉTAPE 1 du RAG : score sémantique minimal pour qu'un document soit adopté
# comme cible (rag.py, repli sémantique global). C'est le seul vrai verrou de la chaîne —
# une fois un document adopté, l'étape 2 rend toujours des extraits, y compris par son
# repli à score_threshold=0.0.
#
# Relevé de 0.40 à 0.55 le 19/08/2026, sur mesure et non au jugé. À 0.40, la requête
# « Zero Bytes » (renseignement) adoptait une facture de piscine à 0.41 et injectait ses
# extraits dans le contexte. Sur cette base : hors sujet plafonne à 0.41-0.52, une vraie
# correspondance monte à 0.72-0.73. 0.55 sépare les deux avec de la marge des deux côtés.
#
# Effet de bord à surveiller : une question légitime mais formulée de loin peut désormais
# ne plus trouver de document du tout. Repasser à 0.4 par env si le rappel se dégrade.
RAG_SCORE_THRESHOLD = float(os.getenv("RAG_SCORE_THRESHOLD", "0.55"))
OWUI_MAX_DOC_CHARS = int(os.getenv("OWUI_MAX_DOC_CHARS", "80000"))

# ── Features ──────────────────────────────────────────────────────────────
SELF_MEMORY_PATH = os.getenv("SELF_MEMORY_PATH", "/app/data/jarvis-self.json")
EMBED_MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"

# ── Google Services (Gmail read+send, Calendar read) ──────────────────────
GOOGLE_CLIENT_ID = os.getenv("GOOGLE_CLIENT_ID", "")
GOOGLE_CLIENT_SECRET = os.getenv("GOOGLE_CLIENT_SECRET", "")
GOOGLE_CALENDAR_ID = os.getenv("GOOGLE_CALENDAR_ID", "primary")

# ── Morning Briefing ──────────────────────────────────────────────────────
BRIEFING_ENABLED = os.getenv("BRIEFING_ENABLED", "true").lower() == "true"
BRIEFING_TIME = os.getenv("BRIEFING_TIME", "07:30")  # HH:MM
BRIEFING_TIMEZONE = os.getenv("BRIEFING_TIMEZONE", "Europe/Paris")

# ── Proto-self reflection loop ─────────────────────────────────────────────
REFLECTION_INTERVAL_HOURS = int(os.getenv("REFLECTION_INTERVAL_HOURS", "6"))
CONV_ANALYSIS_INTERVAL_MINUTES = int(os.getenv("CONV_ANALYSIS_INTERVAL_MINUTES", "60"))
MAX_CHAIN_ITERATIONS = int(
    os.getenv("MAX_CHAIN_ITERATIONS", "3")
)  # max actions per reflection cycle

# ── Boucle agentique (agent/) ─────────────────────────────────────────────
# Régime DISTINCT du proto-self : le proto-self observe et propose sans jamais agir sur le
# monde, l'agent agit — mais uniquement sur commande humaine explicite, jamais déclenché
# par le cycle de réflexion. Désactivé par défaut : AGENT_ENABLED=false ⇒ aucun worker
# démarré, aucune route montée, Jarvis est strictement celui d'avant.
AGENT_ENABLED = os.getenv("AGENT_ENABLED", "false").lower() in ("yes", "true", "1")

# Racine des espaces de travail. Une tâche = un sous-dossier {task_id}. C'est la SEULE
# zone où l'agent peut écrire (voir agent/sandbox.py).
AGENT_WORKSPACE = os.getenv("AGENT_WORKSPACE", "/opt/jarvis/agent_workspace")

# Budgets d'une tâche. max_steps borne la dérive, le timeout borne le temps réel.
AGENT_MAX_STEPS = int(os.getenv("AGENT_MAX_STEPS", "20"))
AGENT_TASK_TIMEOUT_MINUTES = int(os.getenv("AGENT_TASK_TIMEOUT_MINUTES", "45"))

# Plafond de tokens PAR PAS. Volontairement bas : le lock GPU background ne cède qu'ENTRE
# deux générations, donc ce plafond est le pire cas d'attente imposé à un tour de chat.
# Mesuré ~50 tok/s sur Qwen3.6-35B-A3B-5bit ⇒ 1200 tok ≈ 24 s. Ne pas monter sans mesurer.
# Les deux se partagent le MÊME budget : réflexion + sortie visible + appel d'outil.
# Relevés le 19/08/2026 (1200/600) une fois le raisonnement du tour précédent réinjecté
# dans le contexte — à ce moment-là la réflexion sert vraiment à quelque chose, et 600
# tokens devenaient courts pour arbitrer entre huit outils sur un contexte de 17 k tokens.
# Le coût est une latence de chat : (AGENT_STEP_MAX_TOKENS ÷ 50 tok/s) secondes d'attente
# au pire pour un tour de conversation, soit ~44 s ici. La fenêtre de calme
# (AGENT_QUIET_SECONDS) rend le cas rare, elle ne le supprime pas.
# Le journal « think=N car » de chaque pas dit si la réflexion est coupée : si N frôle
# systématiquement AGENT_THINKING_BUDGET × 4, c'est qu'elle l'est.
AGENT_STEP_MAX_TOKENS = int(os.getenv("AGENT_STEP_MAX_TOKENS", "2200"))
AGENT_THINKING_BUDGET = int(os.getenv("AGENT_THINKING_BUDGET", "1000"))

# Plafond du pas d'ÉCRITURE. Un livrable transite par le paramètre `content` de write_file :
# il est donc généré à l'intérieur du bloc <tool_call>, et AGENT_STEP_MAX_TOKENS le coupe en
# plein milieu — le bloc n'est jamais fermé, aucun appel n'est détecté, le pas est perdu
# (mesuré le 19/08/2026 : 6 pas consécutifs perdus sur une note de synthèse). Ce budget-là
# n'est dépensé qu'en cas de troncature avérée, et sans raisonnement : ~2 min de GPU au pire.
AGENT_WRITE_MAX_TOKENS = int(os.getenv("AGENT_WRITE_MAX_TOKENS", "6000"))

# Taille annoncée au modèle pour UN appel write_file. Une consigne « écris en plusieurs
# fois » sans chiffre est inapplicable : il ne peut pas connaître son propre plafond.
#
# Dérivé de AGENT_STEP_MAX_TOKENS et NON de AGENT_WRITE_MAX_TOKENS, à dessein : l'annoncer
# sur le budget de reprise ferait tronquer CHAQUE écriture, et la reprise à 6000 tokens
# (~2 min de GPU tenu) deviendrait le régime normal au lieu de l'exception — précisément ce
# que le plafond par pas protège. Marge 2:1 : 2 caractères par token là où le français en
# markdown en fait plutôt 3 à 4, le reste absorbe le bloc <tool_call> et le raisonnement.
#
# Écritures trop fragmentées à l'usage ? Monter AGENT_STEP_MAX_TOKENS — le coût se paie en
# latence de chat : (tokens ÷ 50) secondes d'attente au pire pour un tour de conversation.
AGENT_WRITE_MAX_CHARS = int(
    os.getenv(
        "AGENT_WRITE_MAX_CHARS",
        str(max(AGENT_STEP_MAX_TOKENS - AGENT_THINKING_BUDGET, 400) * 2),
    )
)

# Fenêtre de calme : on ne démarre un pas que si aucun chat n'a demandé le GPU depuis N s.
# Complète le lock, qui ne sait rien du tour de chat à venir. 0 = désactivée.
AGENT_QUIET_SECONDS = float(os.getenv("AGENT_QUIET_SECONDS", "45"))

# Troncature des sorties d'outil injectées dans le contexte (caractères).
# Relevé de 6000 à 15000 le 19/08/2026, sur mesure. À 6000, une recherche web de 9984
# caractères perdait 40 % de sa matière — pendant qu'un contexte de tâche COMPLET pesait
# 7784 tokens, soit 24 % du plafond pratique du modèle (~32 k). On affamait l'agent article
# par article en laissant les trois quarts de la place vides.
# Le vrai garde-fou reste global : _CONTEXT_SOFT_CAP dans loop.py élide les plus vieux
# résultats quand la somme dépasse le budget, ce qui est le bon endroit pour arbitrer.
AGENT_MAX_TOOL_OUTPUT = int(os.getenv("AGENT_MAX_TOOL_OUTPUT", "15000"))

# Plafond d'UNE lecture de fichier (read_file). Plus haut que le plafond général : un
# fichier source doit tenir en UNE lecture. La pagination est ce que le modèle rate le
# plus mal — mesuré le 19/08/2026 sur vitals.py (25 168 caractères, donc 2 lectures à
# 15 000) : l'indication « reprends avec offset=318 » a été ignorée quatre fois de suite,
# le modèle rejouant la même lecture jusqu'à épuiser la tâche. À 32 000, le fichier passe
# d'un coup et le problème ne se pose plus. Coût : ~8 000 tokens de contexte pour un gros
# fichier, sur les ~32 000 praticables — le contexte total d'une tâche mesurait 7 800.
AGENT_READ_MAX_CHARS = int(os.getenv("AGENT_READ_MAX_CHARS", "32000"))

# Plafond d'extraction d'UNE page lue par l'agent (fetch_url). Distinct de
# _PAGE_MAX_CHARS (6000), calibré pour le budget de contexte du chat : l'agent lit une
# source pour en tirer dates et chiffres exacts, et un article de presse dépasse
# largement 6000 caractères — c'est précisément la troncature qui l'a fait écrire des
# dates de 2024/2025 sur des faits de 2026.
AGENT_PAGE_MAX_CHARS = int(os.getenv("AGENT_PAGE_MAX_CHARS", "14000"))

# Score minimal pour qu'un extrait de la base documentaire soit rendu à l'agent.
# Bien plus strict que RAG_SCORE_THRESHOLD (0.4), qui sert le chat : là, un extrait
# faiblement pertinent est au pire une phrase de trop dans un contexte que l'utilisateur
# relit. Pour l'agent, c'est du bruit qu'il ne peut PAS identifier comme tel et qui le fait
# diverger — mesuré le 19/08/2026 : 5 appels, 19 435 caractères de règlement intérieur et
# de factures sur une tâche de renseignement.
# Abaissé de 0.60 à 0.35 le 19/08/2026, après contre-exemple. 0.60 venait d'une
# calibration sur 5 requêtes DESCRIPTIVES ; il rejetait les requêtes par TITRE ou
# acronyme, que l'embedding multilingue encode mal : « VIT Conscienceness AI » → 0.53 et
# « Consciensness AI TACL » → 0.45 sur un corpus de 10 documents pourtant consacrés au
# sujet. Or l'étape 1 du RAG avait identifié le bon document PAR SON TITRE, sans seuil —
# et ce plancher défaisait ce travail.
#
# Il est de toute façon devenu redondant : depuis que RAG_SCORE_THRESHOLD est passé à
# 0.55, le bruit est bloqué à la source (« Zero Bytes » ne fait plus adopter aucun
# document). Ce plancher ne sert donc plus que de dernier filet, d'où sa valeur basse.
# Les scores restent affichés à l'agent : c'est à lui de juger de la pertinence.
AGENT_DOCS_MIN_SCORE = float(os.getenv("AGENT_DOCS_MIN_SCORE", "0.35"))

# Rétention des enregistrements de tâche dans Redis.
AGENT_TASK_TTL = int(os.getenv("AGENT_TASK_TTL_DAYS", "30")) * 86400

# Envoi du livrable par courriel au demandeur, à la fin de la tâche. Le push iOS annonce,
# le courriel transporte : plafonné à 500 caractères et lu sur un écran verrouillé, une
# notification ne peut pas porter un rapport. Destinataire = le demandeur lui-même, depuis
# son propre compte Google ; jamais un tiers.
AGENT_EMAIL_REPORT = os.getenv("AGENT_EMAIL_REPORT", "true").lower() in ("yes", "true", "1")

# Plafond du corps du courriel. Au-delà, le document est tronqué avec un renvoi vers le
# workspace — un rapport de plusieurs centaines de kilooctets ne se lit pas dans un client
# de messagerie, et certains les rejettent.
AGENT_EMAIL_MAX_CHARS = int(os.getenv("AGENT_EMAIL_MAX_CHARS", "120000"))

# ── Shell agentique (Phase 2) ─────────────────────────────────────────────
# Désactivé par défaut. Un shell sous le compte de l'utilisateur est la capacité la plus
# dangereuse de la boucle : elle s'active sciemment, jamais par héritage de configuration.
# Confinement : seatbelt (écriture workspace+/tmp, pas de réseau, secrets illisibles),
# liste noire, quota et délai. Voir agent/shell.py.
AGENT_SHELL_ENABLED = os.getenv("AGENT_SHELL_ENABLED", "false").lower() in ("yes", "true", "1")

# Délai par commande. Court volontairement : une commande qui dépasse est presque toujours
# une commande interactive ou en boucle, pas un traitement long légitime.
AGENT_SHELL_TIMEOUT = float(os.getenv("AGENT_SHELL_TIMEOUT", "60"))

# Quota de commandes par tâche — borne les dégâts d'une boucle que les autres détecteurs
# laisseraient passer (arguments variables à chaque appel).
AGENT_SHELL_MAX_CALLS = int(os.getenv("AGENT_SHELL_MAX_CALLS", "25"))

# Réseau dans le shell. Coupé par défaut : l'agent a déjà web_search et fetch_url, qui
# passent par du code journalisé et borné. Un `curl` libre est le chemin d'exfiltration le
# plus court qui soit. Ne l'ouvrir que pour une tâche qui l'exige vraiment (pip, git clone).
AGENT_SHELL_NETWORK = os.getenv("AGENT_SHELL_NETWORK", "false").lower() in ("yes", "true", "1")

# Racines lisibles en plus du workspace (lecture seule, jamais d'écriture). Sert la
# lecture du code source prévue en Phase 1 d'Autocoding (ROADMAP).
AGENT_READONLY_ROOTS = tuple(
    p.strip()
    for p in os.getenv(
        "AGENT_READONLY_ROOTS",
        "/opt/jarvis/jarvis-core/src,/opt/jarvis/scripts,/opt/jarvis/DOCS",
    ).split(",")
    if p.strip()
)
# ── Autocoding — prompt self-modification ─────────────────────────────────
# Number of times a knowledge gap must be flagged before a prompt-refine is triggered
REFINE_PROMPT_THRESHOLD = int(os.getenv("REFINE_PROMPT_THRESHOLD", "3"))
# Stores prompt_proposals.json + prompt_overrides.json (inside the /app/data volume)
PROMPT_DATA_DIR = os.path.join(os.path.dirname(SELF_MEMORY_PATH), "prompts")

# ── Router training data collector ────────────────────────────────────────
# JSONL file of (message, routing_json) pairs for future LoRA fine-tuning.
# Mount on host: ./RouterData:/app/router_data  (see docker-compose.yml)
ROUTER_DATA_DIR = os.getenv("ROUTER_DATA_DIR", "/app/router_data")

# ── Conversation storage limits ───────────────────────────────────────────
CHAT_MAX_MESSAGES = int(
    os.getenv("CHAT_MAX_MESSAGES", "100")
)  # server-side Redis LTRIM cap
IOS_MAX_MESSAGES = int(
    os.getenv("IOS_MAX_MESSAGES", "50")
)  # messages returned to iOS app
CHAT_LOG_TTL = (
    int(os.getenv("CHAT_LOG_TTL_DAYS", "90")) * 86400
)  # raw log expiry (seconds)

# ── Memory thresholds ─────────────────────────────────────────────────────
IMPORTANCE_THRESHOLD = 0.35
RECALL_MEMORY_SIMILARITY_THRESHOLD = 0.7
AUTOBIO_IMPORTANCE_THRESHOLD = 0.45
NOVELTY_THRESHOLD = 0.25
PROJECT_THRESHOLD = 0.6

# Fenêtre de récence pour le scoring des souvenirs autobiographiques dans search_memory().
# Les souvenirs épisodiques utilisent une fenêtre de 30 jours (hardcodée).
# Les autobiographiques (milestones durables) méritent une fenêtre plus longue pour rester
# pertinents dans le score de rappel même après plusieurs mois.
AUTOBIO_RECENCY_WINDOW_DAYS = int(os.getenv("AUTOBIO_RECENCY_WINDOW_DAYS", "365"))

# Fenêtre de rétention épisodique : les souvenirs épisodiques plus récents que cette valeur
# sont EXEMPTÉS de la consolidation mensuelle — ils restent accessibles en recall pendant
# cette période avant d'être compressés en autobio.
# 45 jours = fenêtre confortable (6 semaines de contexte moyen terme).
# Min raisonnable : 30j. Max : 60j.
EPISODIC_RETENTION_DAYS = int(os.getenv("EPISODIC_RETENTION_DAYS", "45"))

# Seuil de similarité cosine au-dessus duquel un nouvel événement autobiographique est
# considéré comme un doublon et ignoré (pas de stockage Qdrant).
# 0.85 = très similaire en sens mais formulation différente tolérée.
# Augmenter vers 0.95 pour être plus permissif (moins de dédup).
AUTOBIO_DEDUP_THRESHOLD = float(os.getenv("AUTOBIO_DEDUP_THRESHOLD", "0.82"))

# Décroissance mémorielle mensuelle des souvenirs autobiographiques.
# DECAY_FACTOR        : multiplicateur appliqué à importance à chaque passe (0.85 = -15 %/mois).
# DECAY_THRESHOLD     : en dessous, le souvenir est supprimé de Qdrant.
# DECAY_DURABLE_MIN   : importance initiale >= cette valeur → exempt de décroissance.
# CONSOLIDATION_IMPORTANCE : score assigné aux milestones issus de la consolidation mensuelle.
#                            Doit être == DECAY_DURABLE_MIN pour que ces souvenirs soient permanents.
MEMORY_DECAY_FACTOR = float(os.getenv("MEMORY_DECAY_FACTOR", "0.85"))
MEMORY_DECAY_THRESHOLD = float(os.getenv("MEMORY_DECAY_THRESHOLD", "0.15"))
MEMORY_DECAY_DURABLE_MIN = float(os.getenv("MEMORY_DECAY_DURABLE_MIN", "1.0"))
MEMORY_CONSOLIDATION_IMPORTANCE = float(
    os.getenv("MEMORY_CONSOLIDATION_IMPORTANCE", "1.0")
)

# Durée de rétention des projets "done" dans la liste Redis active (en jours).
# Passé ce délai, un projet terminé est retiré automatiquement au prochain update.
# Les projets importants sont de toute façon consolidés vers Qdrant (mémoire autobiographique)
# lors du nightly review, donc l'information n'est pas perdue.
DONE_PROJECT_TTL_DAYS = int(os.getenv("DONE_PROJECT_TTL_DAYS", "180"))

# Nombre maximum d'entrées conservées dans le journal de croissance (growth_log) de jarvis-self.json.
# Chaque entrée représente un résumé quotidien par utilisateur (1 entrée/utilisateur/jour).
# Avec 2 utilisateurs actifs, 180 = environ 3 mois de journal avant rotation.
GROWTH_LOG_MAX_ENTRIES = int(os.getenv("GROWTH_LOG_MAX_ENTRIES", "180"))


# ══════════════════════════════════════════════════
#  USER MANAGEMENT — loaded from users_list.json
# ══════════════════════════════════════════════════

USERS_LIST_PATH = os.getenv("USERS_LIST", "/app/data/users_list.json")

# Full user objects keyed by access code
USERS: dict[str, dict] = {}

try:
    with open(USERS_LIST_PATH, encoding="utf-8") as _f:
        _raw: list[dict] = json.load(_f)
    for _u in _raw:
        code = _u.get("code", "").strip()
        if code:
            USERS[code] = _u
    logger.info("Loaded %d users from %s", len(USERS), USERS_LIST_PATH)
except FileNotFoundError:
    logger.error("users_list.json not found at %s — no users loaded", USERS_LIST_PATH)
except json.JSONDecodeError as _e:
    logger.error("users_list.json is invalid JSON: %s", _e)

# ── Derived dicts (backward-compatible with existing code) ────────────────
# code → firstname
USER_CODES: dict[str, str] = {code: u["firstname"] for code, u in USERS.items()}

# codes with admin: true — allowed to approve/reject prompt proposals
USER_ADMINS: set[str] = {code for code, u in USERS.items() if u.get("admin") is True}

# code → email (empty string = no email delivery)
USER_EMAILS: dict[str, str] = {code: u.get("mail", "") for code, u in USERS.items()}

# email → code (reverse index — used by the OpenWebUI proxy to identify users by email)
EMAIL_TO_CODE: dict[str, str] = {
    u.get("mail", "").lower(): code for code, u in USERS.items() if u.get("mail")
}

# code → city for weather
USER_CITIES: dict[str, str] = {
    code: u.get("city", "Paris") for code, u in USERS.items()
}

# code → timezone name (IANA)
USER_TIMEZONES: dict[str, str] = {
    code: u.get("timezone", "Europe/Paris") for code, u in USERS.items()
}

# codes with trading enabled
USER_TRADING: list[str] = [code for code, u in USERS.items() if u.get("trading", False)]

# codes with a connected Google account (Gmail + Calendar access)
# Only these users receive calendar/Gmail data in briefings and chat.
USER_GOOGLE: set[str] = {code for code, u in USERS.items() if u.get("google", False)}

# Per-user Google OAuth refresh tokens.
# Loaded from GOOGLE_REFRESH_TOKEN_<CODE> env vars for each user with "google": true.
# No fallback — each user must have their own token (prevents cross-account calendar leaks).
# To enable a user: set "google": true in users_list.json AND add their token to .env.
GOOGLE_USER_TOKENS: dict[str, str] = {}
for _code in USER_GOOGLE:
    _tok = os.getenv(f"GOOGLE_REFRESH_TOKEN_{_code}", "")
    if _tok:
        GOOGLE_USER_TOKENS[_code] = _tok

# ── APNs (Apple Push Notification service) ───────────────────────────────────
# Key and identifiers from developer.apple.com → Certificates, IDs & Profiles → Keys
APNS_KEY_ID   = os.getenv("APNS_KEY_ID",   "")   # 10-char alphanumeric key ID
APNS_TEAM_ID  = os.getenv("APNS_TEAM_ID",  "")   # 10-char team ID
APNS_BUNDLE_ID = os.getenv("APNS_BUNDLE_ID", "com.sebastienviou.JarvisApp")
APNS_KEY_PATH = os.getenv("APNS_KEY_PATH", "")   # absolute path to AuthKey_XXXXXX.p8
APNS_ENV      = os.getenv("APNS_ENV",      "production")  # "production" or "sandbox"

