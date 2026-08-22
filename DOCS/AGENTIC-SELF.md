# Enquêtes autonomes — brancher l'agentic sur les actions de `self/`

> Jarvis voit ce qui ne fonctionne pas. Aujourd'hui il ne peut qu'en parler.
> Ce document décrit comment lui donner le droit d'aller voir.

Statut : **plan, non implémenté**. Rédigé le 21/08/2026 à partir de l'audit logique de
`self/` et `memory/`. Complète `DOCS/AGENTIC.md` (le chantier `agent/`, phases 0 à 4) et
`DOCS/ROADMAP.md` (le proto-self et l'autocoding).

---

## Le point de départ

La brique agentique existe et fonctionne. `agent/` est validé sur trois registres — web
sourcé, analyse de journaux au shell, lecture de code — avec bac à sable noyau, budgets
indépendants, reprise après redémarrage, double restitution push + courriel. En
production : `AGENT_ENABLED=true`, `AGENT_SHELL_ENABLED=true`.

Ce qui manque n'est pas la boucle, c'est le **déclencheur**. Et le déclencheur est la
partie dangereuse.

### Le cas d'école qui motive tout le reste

L'audit du 21/08 a mesuré que **la moitié des conversations sortent sans
`memory_summary`**, et que la cause est un verdict incohérent de l'analyseur : il extrait
des `project_updates` et déclare simultanément `memory_summary: null`, ce qui force
`importance` à 0 et supprime mécaniquement l'écriture du vecteur épisodique.

`REFLECTION_PROMPT` dit explicitement au modèle : *« un taux élevé AVEC une activité
récente peut indiquer un bug d'analyse ou de prompt »*. Le signal est dans son contexte
depuis des semaines. La seule chose qu'il puisse en faire est `alert_admin`. Il ne l'a
jamais fait — et même s'il l'avait fait, il n'aurait eu qu'un symptôme à signaler, pas un
diagnostic.

Instruire ce défaut demandait de lire `analyzer.py`, de croiser `analyzer-prompts.log`
avec le convlog, de décomposer l'ESS et de recouper avec les projets en base. C'est
exactement une tâche d'agent, et c'est hors de portée du catalogue d'actions actuel.

---

## L'invariant qui change

`AGENTIC.md` et `ROADMAP.md` posent aujourd'hui :

> *« le cycle de réflexion ne crée jamais de tâche agentique »*

Ce plan lève la clause. **Il faut le dire explicitement dans les deux documents** — un
invariant qui disparaît sans être nommé revient sous forme de bug six mois plus tard.

Ce qui le remplace n'est pas « rien », c'est un invariant plus étroit et plus utile :

> Le cycle de réflexion peut ouvrir une **enquête** : une tâche agentique en **lecture
> seule**, choisie dans un catalogue fermé de modèles d'enquête, dont l'unique produit est
> un rapport. Aucune tâche déclenchée par Jarvis lui-même ne modifie quoi que ce soit hors
> de son workspace. Toute modification reste sur les rails d'approbation humaine
> existants.

---

## Décision 1 — Un catalogue d'enquêtes, pas un objectif libre

La tentation naturelle est de donner au LLM de réflexion un champ `objective` libre.

C'est exactement le motif du constat **S9** de l'audit : un champ texte piloté par un
modèle, sans ancrage mécanique, dérive. Le seuil « ×3 » de `refine_prompt` n'a jamais été
lu en code ; le seul garde réel portait sur le nom de prompt ; le modèle en est sorti en
changeant de cible. Bilan mesuré : **11 propositions rejetées sur 13**, dont **4 sur un
même sujet** visant quatre prompts différents.

Une tâche agentique coûte cent fois plus cher qu'une proposition de prompt. On ne rejoue
pas ce motif à cette échelle.

Chaque modèle d'enquête est donc lié à un signal **déjà présent dans
`gather_global_context()`** :

| Signal déclencheur | Modèle d'enquête | Outils |
|---|---|---|
| `null_rate_7d` élevé + activité récente | Diagnostiquer pourquoi N % des conversations sortent sans `memory_summary` : lire `analyzer.py`, croiser `analyzer-prompts.log` et le convlog, identifier ce que le modèle renvoie à la place | `read_file, list_dir, plan, write_file, finish` |
| `norm_anomalies > 0` | Recenser les vecteurs non normalisés de `jarvis_memory`, dater leur écriture, remonter au chemin d'écriture responsable | `read_file, list_dir, plan, write_file, finish` |
| incident `degradation_interne` | Analyser `jarvis-api.log` sur la fenêtre de l'incident, classer les erreurs par famille, proposer une cause | `plan, write_file, finish` **+ shell — obligatoire (2,7 Mo)** |
| `cve_conseil` non vide | Vérifier la disponibilité réelle des versions correctives, rédiger le plan de mise à jour image par image | `web_search, fetch_url, plan, write_file, finish` |
| `health[service] ≠ ok` | Diagnostiquer l'indisponibilité du service nommé : configuration, journaux, état du conteneur | `read_file, plan, write_file, finish` |

Chaque modèle porte sa propre clé de cooldown, son propre quota, et la partie **fixe** de
l'objectif. Le LLM ne remplit qu'une fente — code utilisateur, nom de service, fenêtre
d'incident — et rédige son `reason`, qui part dans le courriel de restitution pour que
l'administrateur juge le **déclenchement** autant que la conclusion.

---

## Décision 2 — Déclenchement sur front, jamais sur niveau

C'est le constat **S8** appliqué avant de faire le dégât.

Le cycle tourne toutes les `REFLECTION_INTERVAL_HOURS` (5 h en production) et les signaux
ci-dessus sont des **états persistants**, pas des événements. Un déclenchement sur niveau
produirait près de cinq enquêtes par jour sur le même symptôme, indéfiniment — la
pathologie que `jarvis:self:gap_counts` a déjà démontrée, jamais décrémenté et donc
bloquant le garde en position ouverte pour toujours.

- Chaque tâche mémorise l'**empreinte** du signal : `null_rate_7d` arrondi au dixième,
  `at` de l'incident, hash de la liste CVE, nom du service.
- Refus si une tâche d'origine `self` porte la même empreinte, quel que soit son statut.
- Refus si une tâche d'origine `self` est encore ouverte, toutes empreintes confondues.
- Quota global `SELF_AGENT_MAX_PER_DAY`, à **1** au départ.
- Cooldown de 7 jours par modèle d'enquête.

---

## Décision 3 — L'origine détermine les outils

Aujourd'hui `TOOL_SCHEMAS` est filtré **une seule fois à l'import** (`agent/tools.py`,
selon `AGENT_SHELL_ENABLED`). Le jeu d'outils doit devenir une **fonction de la tâche**.
C'est le seul changement structurel à faire dans `agent/` ; tout le reste est additif.

- `store.create_task(user_code, objective, origin="human"|"self", template=…, fingerprint=…)`
- `tools.schemas_for(task)` remplace la constante ; `loop._generate` et `loop._conclure`
  la consomment
- Origine `self`, phases 1–2 : **pas de shell**, même si `AGENT_SHELL_ENABLED=true`. Le
  shell autonome est un régime distinct, il arrive mesuré (phase 3).
- Origine `self` : pas de `web_search` / `fetch_url` sauf pour le modèle CVE, où c'est le
  cœur de la tâche.

---

## Décision 4 — Refermer la boucle

Aujourd'hui une tâche se termine en push + courriel + enregistrement Redis à TTL de
30 jours. La réflexion ne les voit pas.

**Sans le chemin de retour, on n'a pas ajouté une boucle, on a ajouté un émetteur.**

- `gather_global_context()` lit `list_tasks(origin="self")` et rend un bloc `<enquetes>` :
  en cours, terminées depuis le dernier cycle, avec leur `result` (le résumé, pas le
  livrable).
- Le modèle peut alors choisir `alert_admin` **avec un diagnostic en main** — c'est
  l'action qu'il possède déjà et à laquelle il n'a aujourd'hui rien de substantiel à
  mettre.
- Une enquête en échec → `vitals.mark_incident("enquete_echouee", …)`. L'échec devient un
  fait du système, pas une ligne de journal.
- À terme le rapport devient l'entrée de `refine_prompt` : un diagnostic avec ses preuves,
  au lieu des compteurs de lacunes actuels. S9 montre que l'entrée actuelle est trop
  maigre pour produire des propositions acceptables.

Ces trois points reprennent la Phase 4 de `AGENTIC.md`, qui les listait déjà sans les
détailler.

---

## Point d'insertion dans le cycle

- Nouvelle action `open_investigation` dans `_SELF_ACTIONS` **uniquement** — jamais en
  phase utilisateur. C'est « agir sur soi » au sens strict.
- Placée dans `_SELF_REVIEW_REQUIRED`, avec une branche `_build_review_context` montrant :
  valeur courante du signal, empreinte de la dernière tâche du même modèle, quota du jour,
  tâches ouvertes.
- Conformément au constat **S14** : **les gardes mécaniques d'abord** — `AGENT_ENABLED`,
  `SELF_AGENT_ENABLED`, quota, cooldown, empreinte, fenêtre nocturne. Le contesteur n'est
  appelé que s'ils passent tous, et on ne lui pose que la question qu'aucun compteur ne
  tranche : *ce signal vaut-il sept minutes de GPU*.
- Fenêtre d'exclusion 22 h 30 – 01 h 00 tant que **S18** n'est pas corrigé : la revue
  nocturne y fait cinq appels LLM par utilisateur en priorité chat, qui affameraient le
  worker.

---

## Ordre d'exécution

### Phase 0 — prérequis, aucun agentic encore

Les quatre correctifs ci-dessous ne sont pas du confort : ce sont les conditions pour que
le déclencheur ne reproduise pas, à cent fois le coût, les pannes que `refine_prompt` a
déjà démontrées.

- [ ] **S1** — l'effacement de `jarvis-self.json` sur erreur de lecture transitoire. Rien
      d'autre ne doit être mis en ligne tant qu'un écrivain autonome supplémentaire peut
      déclencher ce chemin.
- [ ] **S8 + S11 + S3** — garde sur front, purge des lacunes d'avril, versionnage du
      journal de réflexion. Sans ça la nouvelle action hérite de la pathologie de boucle.
- [ ] **S18** — `call_llm_bg` sur les cinq chemins planifiés.
- [ ] **S6** — résultat d'action structuré. On s'apprête à ajouter une action dont la
      sortie sera « enquête `<id>` ouverte », que l'heuristique de chaîne classerait en
      échec.

### Phase 1 — le déclencheur, en lecture seule

- [ ] `origin`, `template`, `fingerprint` sur l'enregistrement de tâche
- [ ] `tools.schemas_for(task)` à la place de la constante de module
- [ ] `_action_open_investigation` : table de modèles, empreinte, quota, cooldown, gardes
      mécaniques puis auto-contestation
- [ ] Drapeau `SELF_AGENT_ENABLED=false` par défaut, **distinct** de `AGENT_ENABLED`. Deux
      capacités, deux interrupteurs.
- [ ] Premier modèle mis en service : celui de `null_rate_7d`. Signal le mieux établi, le
      plus reproductible, et le seul dont on connaît déjà l'importance.

### Phase 2 — le chemin de retour

- [ ] Bloc `<enquetes>` dans `gather_global_context` et `REFLECTION_PROMPT`
- [ ] Échec d'enquête → `vitals.mark_incident`
- [ ] Courriel de restitution reformulé en « enquête autonome », citant le signal
      déclencheur et le `reason` du modèle

### Phase 3 — le shell, une fois mesuré

- [ ] Activer `shell` pour le seul modèle d'analyse de journaux — celui qui en a réellement
      besoin (`jarvis-api.log` pèse 2,7 Mo, hors de portée de `read_file`) et celui qui est
      déjà validé en conditions réelles par la phase 2 d'`AGENTIC.md`. Après relecture
      manuelle d'au moins dix exécutions des phases 1–2.

### Phase 4 — ce pour quoi tout cela existe

- [ ] Le rapport d'enquête devient l'entrée de `refine_prompt`, puis de `code_proposals` :
      un diagnostic sourcé au lieu d'un compteur de lacunes. C'est le seul chemin plausible
      pour que l'autocoding produise des propositions acceptées — le taux actuel de 2 sur
      13 mesure la pauvreté de l'entrée, pas celle du modèle.

---

## Risques

| Risque | Parade |
|---|---|
| Emballement du déclenchement sur un état persistant | Empreinte + quota journalier + cooldown par modèle + drapeau séparé. S8 traité *avant* plutôt qu'après. |
| L'agent enquête sur lui-même : il lit ses propres journaux, y voit ses propres erreurs, et en tire des conclusions autoréférentielles | Exclure les lignes du logger `jarvis-agent` du périmètre du modèle d'analyse de journaux. Le modèle l'a fait spontanément une fois — *« pour éviter les biais autocentrés »* — mais on ne construit pas un garde-fou sur une bonne surprise. |
| Un diagnostic faux, formulé avec assurance, remonte en `alert_admin` | Le rapport porte toujours ses preuves brutes (commandes exécutées, fichiers lus) et le courriel cite le signal déclencheur. L'humain garde la décision : *signaler plutôt qu'interdire*, la règle qu'`AGENTIC.md` a tirée de ses douze pannes. |
| Contention GPU avec la revue nocturne | S18, plus la fenêtre d'exclusion 22 h 30 – 01 h 00 en attendant. |
| Une enquête survit à l'entrée de journal qui l'a déclenchée (TTL de tâche 30 j vs 30 entrées de journal) | Stocker le déclencheur complet **dans** l'enregistrement de tâche, jamais par référence. |

---

## Critère d'arrêt — à fixer maintenant, pas après

Sur les **dix premières enquêtes** déclenchées par Jarvis : combien ont produit un rapport
sur lequel l'administrateur a agi ?

> **Si la réponse est inférieure à trois, ce sont les modèles d'enquête qui sont mauvais,
> pas le mécanisme** — et il faut les réécrire avant d'en ajouter.

C'est très exactement le test que `refine_prompt` a échoué : 2 acceptations sur 13, sur
dix-huit semaines, sans que le chiffre soit jamais regardé. Le poser avant de démarrer est
ce qui distingue une capacité d'une accumulation.
