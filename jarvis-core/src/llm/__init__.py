"""Sous-système LLM de Jarvis — regroupé par commodité.

    client        client HTTP OpenAI-compatible + streaming (stream_openai, describe_images…)
    local         inférence MLX locale (call_llm_local*, stream_local, preload_models…)
    router        routage de requête vers le bon modèle (llm_route, RouterResult)
    embed_router  routage par embedding (embed_route, preload_embed_router)

Simple regroupement de modules : chacun garde son identité, importé via son chemin
(`from llm.local import call_llm_local`). Pas de ré-export ici.
"""
