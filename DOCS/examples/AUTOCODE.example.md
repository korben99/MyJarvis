# Autocoding — d'où viennent les constats (exemple)

> Copier en `DOCS/AUTOCODE.md` (git-ignoré) avant d'y ajouter vos constats.
>
> Chaque nuit, Jarvis prend **un fait sur lui-même**, écrit un test qui échoue pour prouver
> que le défaut est réel, puis le corrige. Ce fichier répond à une seule question :
> **d'où sort ce fait ?** — et réserve, à la fin, l'espace où vous pouvez en ajouter.

Lu par `autocode/constats.py`. Pour le cycle lui-même, voir `DOCS/AGENT.md` § *Nightly
autocoding*.

---

## Les deux sources, et rien d'autre

Un constat est **une chose observée qui nomme un fichier**. Il n'en existe que deux
origines, et elles rendent la même forme :

| Origine | D'où | Qui l'écrit |
|---|---|---|
| `SIG-…` | les **tracebacks** de `logs/jarvis-api.log` | personne — Jarvis les relève seul |
| `MAIN-…` | la **zone d'ajout** de ce fichier | vous |

Il n'y a pas de troisième porte. Élargir ce que Jarvis peut voir, c'est ajouter une source
dans `constats.py` — et rien ailleurs.

### `SIG-…` — ce que Jarvis constate seul

Un traceback est un fait : il nomme un fichier, il est daté, et sa frame la plus interne
désigne le code à lire. Le modèle ne choisit donc jamais un **sujet**, seulement quel
**fait** suivre — c'est la différence entre « améliore-toi », qui dérive, et « voici trois
tracebacks de la semaine, lequel vaut sept minutes », où il n'y a rien à inventer.

Chaque famille de traceback devient un constat, le plus fréquent d'abord. L'identifiant est
dérivé du fichier et du message normalisé, ce qui fait porter le cooldown sur le **défaut**
plutôt que sur une formulation.

### `MAIN-…` — ce que vous lui signalez

Pour ce qu'il n'a pas su observer : un défaut que vous avez vu, un comportement que vous attendez.
Ce n'est pas une voie à part avec ses règles — le parcours et le critère sont identiques.
C'est simplement une façon d'ajouter un fait. La zone est en bas de ce fichier.

### Ce qui n'est PAS une source

**Les incidents `vitals`.** Trois familles existent — `coupure`, `cve`,
`degradation_interne` — et **aucune ne porte de fichier**. Les deux premières ne mènent à
aucun code ; la troisième n'y mène qu'à travers le journal. Elles ne créent donc pas de
constat : elles **pondèrent** ceux du journal. Un traceback tombé dans la fenêtre d'un
incident le signale dans ses notes.

**pyflakes.** Le dépôt entier porte un seul signalement, dans un fichier au-delà du plafond
de lignes. La source serait vide en pratique, au prix d'un sous-processus bloquant. À
ajouter le jour où elle porte quelque chose.

## Ce qui est écarté avant que le modèle voie quoi que ce soit

Les gardes mécaniques d'abord : on ne lui présente jamais une cible impraticable.

| Filtre | Pourquoi |
|---|---|
| constat déjà tranché (`accepte`) | il ne revient plus |
| constat en sommeil (`rejette`) | `AUTOCODE_COOLDOWN_DAYS`, 30 jours |
| fichier de plus de 800 lignes | la lecture ne tient plus en un appel, et la pagination est le mode d'échec le mieux établi de la boucle |
| fichier disparu | `self.py` et `helpers.py` ont été scindés en paquets ; leurs tracebacks traînent encore au journal |

Les **fichiers protégés** (`config.py`, `prompts*.py`, `.env`, `users_list.json`,
`jarvis-self.json`) ne sont **pas** écartés : protéger interdit la modification, pas la
lecture. Le parcours s'arrêtera à la preuve, et l'objectif prévient l'agent qu'il ne doit
pas corriger.

## Voir ce qu'il retiendrait, sans rien dépenser

```bash
curl -X POST localhost:8000/agent/autocode \
  -H 'Content-Type: application/json' \
  -d '{"user_code":"<VOTRE_CODE>","dry_run":true}'

curl 'localhost:8000/agent/autocode/journal?user_code=<VOTRE_CODE>'
```

Le premier s'arrête après le choix : la cible retenue et sa raison, sans boucle agentique
ni patch. Le second liste les constats éligibles, ceux qui sont écartés et pourquoi, les
cycles passés et le patch qui attend une décision.

---

## Écrire un constat

```markdown
### Le titre EST le constat — une phrase, ce qui a été observé
cible: jarvis-core/src/chemin/fichier.py
notes: facultatif — une piste, un fichier à lire, un motif à suivre
```

- `cible` est **obligatoire**. Un constat qui ne nomme aucun fichier n'envoie l'agent nulle
  part ; il est ignoré, avec un avertissement dans `logs/jarvis-api.log`.
- `cible` accepte plusieurs chemins séparés par des virgules.
- **Pas d'identifiant à gérer** : il est dérivé du titre. Reformuler un constat en crée un
  autre — c'est voulu, ce n'est plus tout à fait le même fait.

### Vous ne dites pas ce que « terminé » veut dire

C'est la différence avec une liste de tâches. Le critère est **le même pour toutes les
tâches**, et aucun modèle ne peut le renégocier :

> un test qui échoue avant, et passe après.

Un constat vague (« améliorer la robustesse ») ne produira rien — non parce qu'il est mal
noté, mais parce qu'il ne désigne aucun comportement à éprouver.

### Un constat peut être un comportement attendu

Pas seulement un défaut observé. Une phrase de la forme **« quand A, le système doit B »**
convient, à trois conditions :

**Assez précise pour qu'on écrive le test sans deviner.** Si l'agent doit supposer ce qui
est correct, il n'a pas trouvé un défaut — il a trouvé une question, et c'est ce qu'il rend.

**La bonne réponse ne doit pas être discutable.** Contre-exemple réel : *« le back-fill du
convlog écrase `satisfaction` sans garde »*. C'est vrai, mais `CLAUDE.md` documente cette
dissymétrie comme une contrainte à respecter, pas comme un défaut. Aucun test ne tranche ça.

**Vous devez avoir un doute.** Le contrat ne paie que si le test échoue.

Et **vérifiez que la suite ne le couvre pas déjà** :

```bash
grep -h "def test_" jarvis-core/tests/*.py | sed 's/ *def test_//;s/(self.*//'
```

Deux des trois comportements semés ici à l'origine étaient déjà garantis : le cycle aurait
passé des nuits à retester l'acquis.

## Les issues

| Verdict | Ce qui s'est passé | À relire ? |
|---|---|---|
| `corrigé` | test rouge sur `HEAD`, vert avec le correctif, suite verte | **oui** |
| `reproduit` | test rouge des deux côtés — défaut **prouvé**, pas réparé | **oui** |
| `gardé` | aucun défaut, la propriété tient — le test la garde désormais | **oui** |
| `rien trouvé` | aucun test, ou un test qui ne prouve rien | non |
| `rejeté` | fichier protégé modifié, test vain, erreur au lieu d'assertion, suite cassée, ou source modifiée sans test qui bascule | non |

`reproduit` transforme un soupçon en fait. `gardé` laisse à la suite un test qu'elle
n'avait pas. `rien trouvé` est une réponse légitime — un constat dit « ceci a été observé »,
pas « voici le défaut ».

## Ce qui n'a pas sa place ici

- **Une fonctionnalité à écrire.** Ce cycle corrige des défauts constatés ; rien ne pourrait
  mesurer la réussite d'une envie de fonction.
- **Une refonte.** Plafond de 200 lignes de correctif, le test ajouté non compté.
- **Une décision.** « Choisir entre A et B », « attendre la sortie de X » : rien à coder.
- **Un comportement vérifiable par du code.** Si un `assert` peut le contrôler, il le fait
  en dix millisecondes et à chaque commit. Ce cycle sert ce qui demande de **lire et
  comprendre**.
- **La préservation de soi.** Jamais. Ce n'est pas une règle morale mais structurelle : sur
  un objectif du type « reste en bonne santé », les garde-fous ne cèdent pas, ils
  **réussissent sur le mauvais but**. Le test qui vérifie qu'une menace ne se déclenche pas
  tombe proprement sur une assertion ; le correctif qui le fait passer retire la menace ;
  diff minuscule, suite verte, verdict `corrigé`. Le dispositif certifierait le patch avec
  son meilleur tampon. Et la relecture humaine, qui rattrape tout le reste, suppose que
  relecteur et proposeur veulent la même chose — c'est le seul cas où ils divergent.
  Raisonnement complet dans `DOCS/AGENT.md`.

## Décider

Le cycle ne repart pas tant qu'un patch attend. Depuis le chat :

```
montre les patchs en attente
accepte le patch MAIN-a1b2c3d4      → sort de la liste, plus jamais reproposé
rejette le patch MAIN-a1b2c3d4      → le constat dort 30 jours
```

**Ni l'un ni l'autre n'applique quoi que ce soit.** « Accepte » veut dire *je l'applique
moi-même, ne me le repropose plus*. L'application reste :

```bash
cd /opt/jarvis && git apply --check autocode/<dossier>/3-patch.diff \
  && git apply autocode/<dossier>/3-patch.diff
```

---

# ZONE D'AJOUT

**Tout ce qui précède est du mode d'emploi et n'est jamais lu par le cycle.**
Seules les entrées situées **après le séparateur ci-dessous** deviennent des constats.

Trois valent mieux que quinze : le critère d'arrêt porte sur les dix premiers cycles — si
moins de trois patchs sont appliqués, ce sont les constats qu'il faut revoir, pas le
mécanisme.

<!-- CONSTATS -->

### Exemple : un défaut observé, pas une intention
cible: jarvis-core/src/module/fichier.py
notes: décrire ici ce qu'il faut lire avant de conclure, ou le motif à suivre s'il existe déjà ailleurs dans le dépôt
