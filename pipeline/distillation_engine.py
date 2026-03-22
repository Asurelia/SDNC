"""
Moteur de distillation teacher → student via ProjectionBridge.

Le ProjectionBridge est le SEUL module avec backprop dans tout SDNC.
Tous les autres apprentissages restent locaux (STDP, PC, Hebbian).

Flux par épisode :
  1. TeacherLoader.get_representations_and_targets(teacher_model, tokenizer, prompt) → teacher_reps[4] + soft_targets
  2. ProjectionBridge(teacher_reps[i]) → projected_reps[i]
  3. Student brain forward avec projected_reps
  4. MSE(projected, student_reps.detach()) + KL(student_logits, soft_targets)
  5. Backprop UNIQUEMENT sur ProjectionBridge
  6. Student brain apprend via STDP / PC / hippocampe (règles locales)
"""

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


# ══════════════════════════════════════════════════════════════════
#  ProjectionBridge — seul module entraînable par backprop
# ══════════════════════════════════════════════════════════════════

class ProjectionBridge(nn.Module):
    """
    Projette les représentations du teacher vers l'espace du student.

    Architecture : Linear(teacher_dim, bottleneck) → LayerNorm(bottleneck) → GELU → Linear(bottleneck, student_dim) → LayerNorm(student_dim)

    Seul module avec backprop dans tout le pipeline SDNC.
    Le cast float32 en entrée assure la stabilité des gradients.

    Tenseurs :
        Entrée  : (batch, seq_len, teacher_dim)  — ex: (1, S, 5120)
        Sortie  : (batch, seq_len, student_dim)   — ex: (1, S, 2560)
    """

    def __init__(
        self,
        teacher_dim: int = 5120,
        bottleneck: int = 1024,
        student_dim: int = 2560,
    ):
        super().__init__()
        self.down = nn.Linear(teacher_dim, bottleneck)
        self.mid_norm = nn.LayerNorm(bottleneck)
        self.act = nn.GELU()
        self.up = nn.Linear(bottleneck, student_dim)
        self.norm = nn.LayerNorm(student_dim)

    def forward(self, teacher_repr: torch.Tensor) -> torch.Tensor:
        """(batch, seq_len, teacher_dim) → (batch, seq_len, student_dim)"""
        orig_dtype = teacher_repr.dtype
        h = self.down(teacher_repr.float())
        h = self.mid_norm(h)
        h = self.act(h)
        h = self.up(h)
        return self.norm(h).to(orig_dtype)


# ══════════════════════════════════════════════════════════════════
#  Corpus de distillation — 300 prompts, 11 catégories
# ══════════════════════════════════════════════════════════════════

DISTILLATION_CORPUS = [
    # ── Raisonnement logique (20) ────────────────────────────────
    "Si tous les A sont B et tous les B sont C, alors tous les A sont C. Pourquoi ?",
    "Un nombre est pair s'il est divisible par 2. 17 est-il pair ? Explique.",
    "Si je dis 'Il pleut donc le sol est mouillé', quelle est la contraposée ?",
    "Résous ce syllogisme : Tous les mammifères respirent. La baleine est un mammifère.",
    "Explique la différence entre corrélation et causalité avec un exemple.",
    "Si A implique B et non-B est vrai, que peut-on déduire sur A ?",
    "Un paradoxe est-il une contradiction ? Donne un exemple de paradoxe résolvable.",
    "Explique le raisonnement par l'absurde avec un exemple mathématique.",
    "Qu'est-ce qu'un biais de confirmation ? Comment l'éviter dans un raisonnement ?",
    "Si P alors Q. P est faux. Peut-on conclure sur Q ?",
    "Explique la différence entre déduction et induction en logique.",
    "Le paradoxe du menteur : 'Cette phrase est fausse'. Analyse-le.",
    "Qu'est-ce qu'un argument valide mais non sound en logique formelle ?",
    "Explique le modus ponens et le modus tollens avec des exemples.",
    "Si on a 3 chapeaux (2 rouges, 1 bleu) et 3 personnes, décris le raisonnement.",
    "Qu'est-ce que le rasoir d'Occam et comment l'appliquer ?",
    "Explique pourquoi 'absence de preuve n'est pas preuve d'absence'.",
    "Analyse ce raisonnement : 'Les cygnes que j'ai vus sont blancs, donc tous les cygnes sont blancs'.",
    "Qu'est-ce qu'une tautologie ? Donne trois exemples.",
    "Explique le paradoxe de Simpson avec un exemple concret.",
    "Construis un raisonnement formel pour montrer que la somme de deux nombres impairs est toujours paire.",
    "Qu'est-ce qu'une preuve par contraposée et en quoi diffère-t-elle d'une preuve par contradiction ?",
    "Analyse le sophisme de la pente glissante et donne un exemple réel où il est employé à tort.",
    "Explique la notion de complétude et de cohérence dans un système formel selon Gödel.",
    "Qu'est-ce qu'un dilemme du prisonnier et que révèle-t-il sur la coopération rationnelle ?",
    "Comment la logique floue diffère-t-elle de la logique classique binaire et où est-elle utilisée ?",
    "Démontre par récurrence que la somme des n premiers entiers vaut n(n+1)/2.",
    "Qu'est-ce que le paradoxe de Russell et comment a-t-il remis en question les fondements des mathématiques ?",
    "Explique la différence entre une preuve constructive et une preuve non constructive.",
    "Qu'est-ce que la logique modale et comment traite-t-elle les notions de nécessité et de possibilité ?",

    # ── Sciences physique/biologie/chimie (30) ───────────────────
    "Explique la dualité onde-particule de la lumière.",
    "Comment fonctionne la photosynthèse en termes d'énergie ?",
    "Qu'est-ce que l'entropie et pourquoi augmente-t-elle toujours ?",
    "Décris le cycle de Krebs et son rôle dans le métabolisme.",
    "Comment fonctionne la liaison hydrogène dans l'eau ?",
    "Explique la relativité restreinte en termes simples.",
    "Qu'est-ce que la mitose et en quoi diffère-t-elle de la méiose ?",
    "Comment fonctionne un transistor MOSFET ?",
    "Décris le mécanisme de la transcription de l'ADN en ARN.",
    "Qu'est-ce que la supraconductivité et à quelle température se produit-elle ?",
    "Explique le principe de superposition en mécanique quantique.",
    "Comment fonctionne l'électrolyse de l'eau ?",
    "Qu'est-ce que la pression osmotique et son rôle biologique ?",
    "Décris la structure de l'atome selon le modèle quantique.",
    "Explique la différence entre fusion et fission nucléaire.",
    "Comment les neurotransmetteurs traversent-ils la synapse ?",
    "Qu'est-ce que la catalyse enzymatique ?",
    "Explique l'effet Doppler pour le son et la lumière.",
    "Comment fonctionne le système immunitaire adaptatif ?",
    "Qu'est-ce que la cristallographie aux rayons X ?",
    "Décris le fonctionnement d'une pile à combustible.",
    "Explique la sélection naturelle avec un exemple concret.",
    "Comment fonctionne la résonance magnétique nucléaire (RMN) ?",
    "Qu'est-ce que la chimie verte et ses principes fondamentaux ?",
    "Explique le principe d'incertitude d'Heisenberg.",
    "Comment fonctionne la PCR (Polymerase Chain Reaction) ?",
    "Qu'est-ce que l'intrication quantique et pourquoi est-elle utile ?",
    "Décris le cycle de l'azote dans la nature.",
    "Explique la théorie des cordes en termes accessibles.",
    "Comment fonctionne le CRISPR-Cas9 pour l'édition génomique ?",
    "Explique le mécanisme de la transduction du signal via les protéines G et les seconds messagers.",
    "Qu'est-ce que la biologie des systèmes et comment modélise-t-elle les réseaux biologiques ?",
    "Décris le rôle de l'ARN non codant (ARNlnc, miRNA) dans la régulation épigénétique.",
    "Comment fonctionne la spectroscopie de masse et quelles informations donne-t-elle sur une molécule ?",
    "Explique la thermodynamique de la liaison protéine-ligand et le concept d'énergie libre de Gibbs.",
    "Qu'est-ce que la mécanique des fluides de Navier-Stokes et quand devient-elle chaotique ?",
    "Décris le phénomène de fluorescence et son exploitation en microscopie STED.",
    "Comment la sélection sexuelle diffère-t-elle de la sélection naturelle selon Darwin et ses successeurs ?",
    "Explique la chimie des réactions redox et leur rôle dans la respiration cellulaire.",
    "Qu'est-ce que la topologie de bande dans les isolants topologiques et pourquoi est-elle robuste ?",

    # ── Mathématiques (20) ───────────────────────────────────────
    "Démontre que la racine de 2 est irrationnelle.",
    "Explique la transformée de Fourier et ses applications.",
    "Qu'est-ce qu'un espace vectoriel et donne un exemple concret.",
    "Explique le théorème fondamental de l'algèbre.",
    "Comment fonctionne la descente de gradient en optimisation ?",
    "Qu'est-ce que la convergence d'une série et le test du ratio ?",
    "Explique la différence entre un groupe et un anneau en algèbre.",
    "Décris l'algorithme de Dijkstra pour le plus court chemin.",
    "Qu'est-ce qu'une distribution de probabilité normale et pourquoi est-elle importante ?",
    "Explique le théorème de Bayes avec un exemple médical.",
    "Comment calcule-t-on le déterminant d'une matrice 3×3 ?",
    "Qu'est-ce que la décomposition en valeurs singulières (SVD) ?",
    "Explique les nombres complexes et leur représentation géométrique.",
    "Décris l'intégration par parties avec un exemple.",
    "Qu'est-ce que la complexité algorithmique O(n log n) ?",
    "Explique le concept de limite en analyse mathématique.",
    "Comment fonctionne l'algorithme de multiplication de Karatsuba ?",
    "Qu'est-ce qu'un espace de Hilbert et son importance en physique quantique ?",
    "Explique la conjecture de Goldbach et son état actuel.",
    "Décris le problème du voyageur de commerce et les approches de résolution.",
    "Explique la théorie des graphes spectraux et son lien avec les valeurs propres du laplacien.",
    "Qu'est-ce que la cohomologie de de Rham et comment généralise-t-elle l'intégration ?",
    "Décris la méthode des éléments finis et son application à la résolution d'EDP.",
    "Comment fonctionne la cryptographie sur courbes elliptiques et pourquoi est-elle plus efficace que RSA ?",
    "Explique la théorie des catégories et la notion de foncteur avec un exemple concret.",
    "Qu'est-ce que le théorème de compacité de Tychonoff et ses conséquences en topologie ?",
    "Décris le problème de Riemann P≠NP et pourquoi sa résolution transformerait l'informatique.",
    "Comment les transformées de Laplace et de Z sont-elles reliées et quand utilise-t-on chacune ?",
    "Explique la théorie des probabilités bayésiennes non paramétriques avec les processus de Dirichlet.",
    "Qu'est-ce que la géométrie différentielle riemannienne et son rôle en relativité générale ?",

    # ── Programmation Python (20) ────────────────────────────────
    "Explique la différence entre une liste et un tuple en Python.",
    "Comment fonctionne un décorateur en Python ? Donne un exemple.",
    "Qu'est-ce qu'un générateur Python et quand l'utiliser ?",
    "Explique le GIL (Global Interpreter Lock) en Python.",
    "Comment implémenter un pattern singleton en Python ?",
    "Décris les différences entre asyncio et threading en Python.",
    "Qu'est-ce qu'un context manager et comment en créer un ?",
    "Explique la résolution MRO (Method Resolution Order) en Python.",
    "Comment fonctionne la méta-programmation avec les métaclasses ?",
    "Décris les bonnes pratiques pour la gestion des erreurs en Python.",
    "Qu'est-ce que le duck typing et comment Python l'utilise ?",
    "Explique les compréhensions de listes, sets et dicts en Python.",
    "Comment optimiser un programme Python qui traite de gros fichiers ?",
    "Qu'est-ce que le module dataclasses et quand l'utiliser ?",
    "Explique la sérialisation avec pickle et ses dangers.",
    "Comment fonctionne le garbage collector de Python ?",
    "Décris le pattern observer en Python avec un exemple.",
    "Qu'est-ce que le typing module et les type hints en Python ?",
    "Comment utiliser multiprocessing pour paralléliser du calcul ?",
    "Explique les closures et le scope en Python.",
    "Comment implémenter un système de cache LRU en Python pur sans utiliser functools.lru_cache ?",
    "Explique les protocoles Python 3.8+ (Protocol, runtime_checkable) et leur différence avec les ABC.",
    "Comment fonctionne l'inspection de bytecode avec le module dis et quand est-ce utile ?",
    "Décris le pattern command en Python et son utilisation dans les systèmes d'annulation/rétablissement.",
    "Qu'est-ce que le descripteur Python et comment implémenter __get__, __set__, __delete__ ?",
    "Explique comment fonctionne le module ast pour analyser et transformer du code Python.",
    "Comment utiliser ctypes pour appeler des fonctions C depuis Python et quelles précautions prendre ?",
    "Décris les stratégies de test property-based avec Hypothesis et leurs avantages sur les tests unitaires.",
    "Qu'est-ce que le pattern structural subtyping et comment le raisonner avec mypy strict ?",
    "Comment implémenter un interpréteur d'expressions arithmétiques avec un visiteur AST en Python ?",

    # ── Philosophie et éthique (15) ──────────────────────────────
    "Qu'est-ce que le problème du tramway et ses implications éthiques ?",
    "Explique la différence entre utilitarisme et déontologie.",
    "Qu'est-ce que le test de Turing et est-il encore pertinent ?",
    "Décris l'allégorie de la caverne de Platon et son interprétation moderne.",
    "Qu'est-ce que le problème de la conscience phénoménale (qualia) ?",
    "Explique le concept de libre arbitre vs déterminisme.",
    "Qu'est-ce que l'éthique de la vertu d'Aristote ?",
    "Décris le paradoxe du bateau de Thésée et ses implications pour l'identité.",
    "Qu'est-ce que le solipsisme et comment le réfuter ?",
    "Explique la philosophie de l'esprit et le problème corps-esprit.",
    "Qu'est-ce que l'impératif catégorique de Kant ?",
    "Décris le concept de Dasein chez Heidegger.",
    "Qu'est-ce que le pari de Pascal et ses critiques ?",
    "Explique la pensée de Wittgenstein sur les limites du langage.",
    "Qu'est-ce que l'existentialisme de Sartre ? 'L'existence précède l'essence'.",
    "Comment Spinoza conçoit-il le rapport entre substance, attribut et mode dans l'Éthique ?",
    "Qu'est-ce que le problème de la démarcation de Popper et ses limites selon Kuhn et Lakatos ?",
    "Explique la notion de jeux de langage chez le second Wittgenstein et son impact sur la philosophie de l'esprit.",
    "Qu'est-ce que le naturalisme moral et comment répond-il à la guillotine de Hume ?",
    "Décris la tension entre justice procédurale et justice substantielle dans la théorie rawlsienne.",

    # ── Neurosciences et IA (25) ─────────────────────────────────
    "Explique le fonctionnement d'un réseau de neurones transformer.",
    "Qu'est-ce que l'attention multi-têtes et pourquoi est-elle efficace ?",
    "Décris la potentialisation à long terme (LTP) dans les synapses.",
    "Comment fonctionne le mécanisme de backpropagation et est-il biologiquement plausible ?",
    "Qu'est-ce que le predictive coding dans le cerveau selon Karl Friston ?",
    "Explique la différence entre apprentissage supervisé et auto-supervisé.",
    "Décris le rôle de l'hippocampe dans la consolidation de la mémoire.",
    "Qu'est-ce que le Sparse Distributed Memory de Kanerva ?",
    "Explique comment fonctionne STDP (Spike-Timing-Dependent Plasticity).",
    "Qu'est-ce qu'un réseau de neurones spiking (SNN) et ses avantages ?",
    "Décris le rôle de la dopamine dans l'apprentissage par renforcement.",
    "Qu'est-ce que le free energy principle de Karl Friston ?",
    "Explique les réseaux de neurones CfC (Closed-form Continuous-time).",
    "Qu'est-ce que le cortex cingulaire antérieur et son rôle dans la détection de conflit ?",
    "Décris la théorie de la mémoire de travail de Baddeley.",
    "Comment fonctionne la distillation de connaissances (knowledge distillation) ?",
    "Qu'est-ce que le continual learning et le problème de l'oubli catastrophique ?",
    "Explique le mécanisme de gating dans les LSTM et GRU.",
    "Décris les oscillations gamma et thêta dans le cerveau et leur rôle cognitif.",
    "Qu'est-ce que l'architecture mixture of experts (MoE) ?",
    "Explique le concept de plasticité synaptique et la règle de Hebb.",
    "Comment fonctionne la normalisation de couche (LayerNorm) ?",
    "Qu'est-ce que le retrieval-augmented generation (RAG) ?",
    "Décris le fonctionnement du connectome de C. elegans et son lien avec les NCP.",
    "Qu'est-ce que la théorie de l'information intégrée (IIT) de Tononi ?",
    "Explique l'hypothèse du cerveau bayésien et comment elle unifie perception et action.",
    "Comment les oscillations thalamocorticales génèrent-elles les différents stades du sommeil ?",
    "Décris le rôle du cervelet dans l'apprentissage prédictif des séquences motrices.",
    "Qu'est-ce que la théorie du codage sparse et pourquoi le cortex visuel V1 l'implémenterait-il ?",
    "Explique la dynamique des attracteurs dans les réseaux de Hopfield et leur capacité mémorielle.",
    "Comment les architectures de type Mamba (SSM sélectif) se comparent-elles aux transformers pour les longues séquences ?",
    "Décris le mécanisme de l'attention linéaire et ses avantages computationnels sur l'attention quadratique.",
    "Qu'est-ce que le continual learning meta-learning (MAML, Reptile) et comment évite-t-il l'oubli ?",
    "Explique le concept de world model dans les agents RL modernes (Dreamer, MuZero).",
    "Comment fonctionne le score matching et son lien avec les modèles de diffusion génératifs ?",

    # ── Langue française (15) ────────────────────────────────────
    "Explique la différence entre le passé simple et le passé composé.",
    "Qu'est-ce que le subjonctif et quand l'utiliser en français ?",
    "Décris les figures de style : métaphore, métonymie, synecdoque.",
    "Explique la concordance des temps en français.",
    "Qu'est-ce que le discours indirect libre en littérature ?",
    "Décris les règles d'accord du participe passé avec avoir.",
    "Explique la différence entre 'amener' et 'apporter'.",
    "Qu'est-ce qu'un alexandrin et comment le reconnaître ?",
    "Décris les niveaux de langue : soutenu, courant, familier.",
    "Explique l'utilisation du conditionnel passé en français.",
    "Qu'est-ce que la négation restrictive 'ne... que' ?",
    "Décris le rôle des connecteurs logiques dans un argumentaire.",
    "Explique la valeur des temps narratifs en français.",
    "Qu'est-ce qu'un mot-valise et donne des exemples.",
    "Décris la règle de proximité pour l'accord des adjectifs.",
    "Explique l'emploi du gérondif et son accord en français contemporain.",
    "Qu'est-ce que la deixis spatiale et temporelle et comment s'exprime-t-elle en français ?",
    "Décris le système aspecto-temporel du français en comparant indicatif et subjonctif dans les subordonnées.",
    "Comment distinguer les propositions relatives explicatives et déterminatives et quel est leur impact sur le sens ?",
    "Qu'est-ce que la focalisation clivée et pseudo-clivée en français et quel effet stylistique produit-elle ?",

    # ── Questions sur SDNC (20) ──────────────────────────────────
    "Décris l'architecture de SDNC : Qwen gelé + CfC + SNN + STDP.",
    "Comment le cerveau CfC prédit-il les représentations de la couche suivante ?",
    "Explique le rôle de l'hippocampe (SDM) dans SDNC.",
    "Qu'est-ce que le StepScheduler et comment simule-t-il les rythmes cérébraux ?",
    "Comment l'ACC détecte-t-il les conflits dans SDNC ?",
    "Explique le mécanisme de sleep consolidation dans SDNC.",
    "Comment les InjectionGates modulent-elles la génération de Qwen ?",
    "Qu'est-ce que le predictive coding hiérarchique dans SDNC ?",
    "Comment la dopamine module-t-elle l'apprentissage STDP dans SDNC ?",
    "Décris le flux complet d'un forward pass dans BrainHybridModel.",
    "Comment SDNC gère-t-il la mémoire épisodique sans oubli ?",
    "Explique la différence entre phase gamma et phase thêta dans le scheduler.",
    "Comment le arousal_boost affecte-t-il les fréquences d'apprentissage ?",
    "Décris le mécanisme de gate alpha adaptatif dans les InjectionGates.",
    "Comment SDNC mesure-t-il la saillance pour décider quoi mémoriser ?",
    "Qu'est-ce que la projection bridge et son rôle dans la distillation ?",
    "Comment le modèle teacher enrichit-il l'apprentissage du student ?",
    "Décris les 5 métriques du benchmark SDNC.",
    "Comment SDNC évite-t-il l'oubli catastrophique ?",
    "Explique pourquoi le Qwen est gelé et comment le cerveau CfC compense.",
    "Comment le ProjectionBridge garantit-il qu'aucun gradient ne remonte vers le teacher ni le student LLM ?",
    "Décris précisément comment l'ACC dans SDNC modifie la fréquence d'apprentissage STDP en cas de conflit détecté.",
    "Explique la stratégie de consolidation sélective : quels critères déterminent qu'un souvenir hippocampique est transféré au CfC ?",
    "Comment SDNC gère-t-il les représentations multimodales si on voulait étendre le système à la vision ?",
    "Décris le protocole de benchmark utilisé pour comparer Qwen pur vs Qwen+CfC sur lm-eval.",

    # ── Conflits logiques pour ACC (15) ──────────────────────────
    "2+2=5 donc la Terre est plate. Analyse cette erreur logique.",
    "Si A>B et B>A alors A=B. Est-ce correct ?",
    "L'eau bout à 50°C à pression normale. Corrige cette affirmation.",
    "Tous les chats sont des chiens donc mon chat est un chien. Pourquoi c'est faux ?",
    "Le cervelet contrôle la mémoire épisodique. Est-ce vrai ?",
    "Python est un langage compilé comme le C. Corrige.",
    "La tour Eiffel est à Berlin. Quel est le problème ?",
    "STDP signifie Static Training Data Protocol. Corrige cette erreur.",
    "La Terre est le centre de l'univers selon la science moderne.",
    "Les neurones communiquent uniquement par contact direct.",
    "Si demain c'est hier alors aujourd'hui n'existe pas. Analyse.",
    "Le Soleil tourne autour de la Terre. Pourquoi est-ce faux ?",
    "L'ADN est une protéine. Corrige cette erreur fondamentale.",
    "Qwen3-4B a 175 milliards de paramètres. Corrige.",
    "L'entropie d'un système isolé diminue toujours. Est-ce correct ?",
    "Les CfC sont des réseaux récurrents discrets équivalents aux RNN classiques. Corrige cette affirmation.",
    "La rétropropagation est biologiquement implémentée telle quelle dans le cortex. Analyse cette erreur.",
    "STDP renforce indifféremment toutes les connexions synaptiques actives. Qu'est-ce qui est faux ?",
    "Le modèle de Hopfield ne peut stocker que des états binaires et ne peut pas être étendu. Corrige.",
    "La distillation de connaissances nécessite que teacher et student aient la même architecture. Est-ce vrai ?",
    "Le predictive coding de Friston est identique à la backpropagation gradient descendant. Analyse.",
    "Un réseau de neurones spiking consomme nécessairement plus d'énergie qu'un réseau dense. Corrige.",
    "Le cortex préfrontal n'a aucun rôle dans la régulation émotionnelle. Corrige.",
    "LayerNorm et BatchNorm sont interchangeables pour les séquences de longueur variable. Est-ce correct ?",
    "L'hippocampe chez l'adulte ne produit aucun nouveau neurone : la neurogenèse adulte est un mythe. Analyse.",

    # ── Sujets variés (20) ──────────────────────────────────────
    "Explique comment fonctionne un moteur électrique.",
    "Décris l'histoire de l'Internet en 5 étapes clés.",
    "Qu'est-ce que la blockchain et comment garantit-elle l'intégrité ?",
    "Explique le fonctionnement d'un GPS.",
    "Décris le système solaire et ses particularités.",
    "Qu'est-ce que la cryptographie asymétrique (RSA) ?",
    "Explique comment fonctionne une impression 3D.",
    "Décris les principes de l'agriculture biologique.",
    "Qu'est-ce que le machine learning et ses types principaux ?",
    "Explique le concept de smart contract dans Ethereum.",
    "Décris le fonctionnement d'un réfrigérateur.",
    "Qu'est-ce que la méthode scientifique et ses étapes ?",
    "Explique comment fonctionne la compression vidéo (H.264).",
    "Décris les principes de la musique classique : harmonie, mélodie, rythme.",
    "Qu'est-ce que l'effet de serre et son impact sur le climat ?",
    "Explique le principe de fonctionnement d'un laser.",
    "Décris la structure d'une étoile (du noyau à la couronne).",
    "Qu'est-ce que la réalité augmentée et ses applications ?",
    "Explique comment fonctionne la reconnaissance faciale.",
    "Décris les grandes extinctions de masse et leurs causes.",

    # ── Dialogue multilingue (25) ────────────────────────────────
    "Explain in English the key differences between transformer self-attention and cross-attention mechanisms.",
    "In English: describe the vanishing gradient problem and how residual connections solve it.",
    "Explica en español la diferencia entre aprendizaje supervisado y no supervisado con ejemplos prácticos.",
    "¿Cómo funciona la normalización por lotes (Batch Normalization) y por qué estabiliza el entrenamiento?",
    "Erkläre auf Deutsch das Konzept der Aufmerksamkeit (Attention) in neuronalen Netzen.",
    "Was ist der Unterschied zwischen einem Autoencoder und einem Variational Autoencoder?",
    "请用中文解释梯度消失问题以及残差网络如何解决这个问题。",
    "用中文描述Transformer架构中多头注意力机制的工作原理。",
    "In English: what is the difference between online learning and batch learning in machine learning?",
    "Describe in English the role of temperature in softmax during language model inference.",
    "Explique en français pourquoi les modèles de langage massivement paramétrés peuvent généraliser hors-distribution.",
    "In English: explain the intuition behind contrastive learning and the SimCLR framework.",
    "Explica en español qué es el ajuste fino (fine-tuning) y cuándo preferirlo al entrenamiento desde cero.",
    "In English: what is LoRA (Low-Rank Adaptation) and why does it reduce GPU memory requirements?",
    "Erkläre auf Deutsch, warum die Tokenisierung mit BPE (Byte-Pair Encoding) für mehrsprachige Modelle vorteilhaft ist.",
    "In English: describe the RLHF pipeline (Reinforcement Learning from Human Feedback) and its limitations.",
    "用中文解释强化学习中的策略梯度定理及其在PPO算法中的应用。",
    "Explique en français la différence entre perplexité et précision comme métriques d'évaluation des LLM.",
    "In English: what is speculative decoding and how does it accelerate autoregressive generation?",
    "Explica en español el concepto de embedding posicional y por qué los transformers lo nécessitent.",
    "In English: compare sparse attention patterns (Longformer, BigBird) against full attention for long documents.",
    "Erkläre auf Deutsch den Unterschied zwischen deterministischem und stochastischem Decoding bei Sprachmodellen.",
    "In English: what is the dead neuron problem in ReLU networks and how does Leaky ReLU address it?",
    "用中文说明知识蒸馏中软标签与硬标签的区别以及温度参数的作用。",
    "In English: explain the concept of emergence in large language models and why it is debated.",
]

assert len(DISTILLATION_CORPUS) == 300, (
    f"Le corpus doit contenir 300 prompts, trouvé {len(DISTILLATION_CORPUS)}"
)


# ══════════════════════════════════════════════════════════════════
#  DistillationEngine
# ══════════════════════════════════════════════════════════════════

class DistillationEngine:
    """
    Moteur de distillation teacher → student via ProjectionBridge.

    Le TeacherLoader fournit les représentations du teacher via son API dédiée.
    Les bridges les projettent vers l'espace du student.
    Le student apprend par règles locales (STDP, PC, hippocampe).

    Backprop UNIQUEMENT sur les bridges — jamais sur le teacher ni le student LLM.
    """

    def __init__(
        self,
        config,
        teacher_loader,
        teacher_model,
        teacher_tokenizer,
        student_brain,
        device: torch.device,
    ):
        """
        Initialise le moteur de distillation.

        Args:
            config: DualModelConfig avec les hyperparamètres.
            teacher_loader: TeacherLoader — gère le forward teacher et l'extraction des reps.
            teacher_model: Modèle teacher (Qwen ou autre LLM gelé).
            teacher_tokenizer: Tokenizer associé au teacher.
            student_brain: BrainHybridModel (modèle student + cerveau CfC).
            device: Device cible pour les bridges.
        """
        self.config = config
        self.teacher_loader = teacher_loader
        self.teacher_model = teacher_model
        self.teacher_tokenizer = teacher_tokenizer
        self.student = student_brain
        self.device = device

        # 4 bridges — un par couche d'intercept
        self.bridges = nn.ModuleList([
            ProjectionBridge(
                teacher_dim=config.teacher_hidden,
                bottleneck=config.distill_projection_dim,
                student_dim=config.student_hidden,
            ).to(device)
            for _ in range(len(config.teacher_layers))
        ])

        # Optimiseur — UNIQUEMENT pour les bridges
        self.optimizer = torch.optim.AdamW(
            self.bridges.parameters(),
            lr=config.sleep_distill_lr,
            weight_decay=1e-5,
        )

        self.total_episodes = 0
        self.loss_history = []

    def _align_seq_len(
        self,
        teacher_rep: torch.Tensor,
        student_rep: torch.Tensor,
    ) -> tuple:
        """
        Aligne les dimensions seq_len entre teacher et student.

        Tronque au min(S_teacher, S_student).
        """
        s_t = teacher_rep.shape[1]
        s_s = student_rep.shape[1]
        s_min = min(s_t, s_s)
        return teacher_rep[:, :s_min, :], student_rep[:, :s_min, :]

    def distill_episode(self, prompt: str) -> dict:
        """
        Un épisode de distillation sur un prompt.

        Flux :
          1. TeacherLoader.get_representations_and_targets(teacher_model, tokenizer, prompt) → reps + soft_targets
          2. Bridge[i](teacher_reps[i]) → projected[i]
          3. Student reps (détachés — pas de backprop sur student)
          4. MSE(projected, student_reps.detach()) → bridge loss
          5. Backprop sur bridges uniquement
          6. Student brain forward avec projected reps (apprentissage local)

        Args:
            prompt: Texte d'entrée pour l'épisode de distillation.

        Returns:
            dict avec distill_errors, total_loss, mean_distill_loss, student_mean_error, student_conflict, episode.
        """
        # 1. Teacher forward via TeacherLoader (un seul pass)
        teacher_reps, soft_targets = self.teacher_loader.get_representations_and_targets(
            self.teacher_model,
            self.teacher_tokenizer,
            prompt,
            layers=self.config.teacher_layers,
            temperature=self.config.distill_temperature,
        )

        # 2. Student reps (détachés)
        student_reps = self.student.llm.get_layer_representations(
            prompt,
            layers=self.config.student_layers,
        )

        # 3. Projection + MSE loss
        self.optimizer.zero_grad()

        distill_errors = []
        projected_reps = []

        for i, bridge in enumerate(self.bridges):
            projected = bridge(teacher_reps[i].float())
            projected_reps.append(projected)

            # Aligner seq_len
            proj_aligned, stud_aligned = self._align_seq_len(
                projected, student_reps[i].float()
            )

            # MSE — student détaché : backprop sur bridge seulement
            mse = F.mse_loss(proj_aligned, stud_aligned.detach())
            distill_errors.append(mse)

        # 4. Loss totale
        mean_distill = sum(distill_errors) / len(distill_errors)
        total_loss = mean_distill  # KL ajouté si logits disponibles

        # 5. Backprop sur bridges uniquement
        total_loss.backward()

        # Gradient clipping pour stabilité
        torch.nn.utils.clip_grad_norm_(self.bridges.parameters(), max_norm=1.0)
        self.optimizer.step()

        # 6. Student brain forward avec projected reps (apprentissage local)
        # Détacher les projected reps pour éviter de backprop dans le student
        detached_projected = [p.detach() for p in projected_reps]

        # Aligner seq_len de tous les projected aux student_reps
        aligned_projected = []
        for i in range(len(detached_projected)):
            proj_a, _ = self._align_seq_len(detached_projected[i], student_reps[i])
            aligned_projected.append(proj_a)

        student_output = self.student.forward_with_external_reps(
            layer_reps=aligned_projected,
            prompt=prompt,
            learn=True,
        )

        self.total_episodes += 1
        loss_val = total_loss.item()
        self.loss_history.append(loss_val)

        return {
            'distill_errors': [e.item() for e in distill_errors],
            'total_loss': loss_val,
            'mean_distill_loss': mean_distill.item(),
            'student_mean_error': student_output.get('mean_error', 0.0),
            'student_conflict': student_output.get('conflict_score', 0.0),
            'episode': self.total_episodes,
        }

    def run_sleep_distillation(self, n_steps: int = 200) -> dict:
        """
        Lance n_steps épisodes de distillation sur le corpus.

        Tous les 20 steps : sleep consolidation du student.

        Args:
            n_steps: Nombre d'épisodes de distillation.

        Returns:
            dict avec métriques agrégées.
        """
        corpus = DISTILLATION_CORPUS.copy()
        random.shuffle(corpus)

        all_losses = []
        all_distill = []

        print(f"\n{'='*60}")
        print(f" Distillation sommeil — {n_steps} épisodes")
        print(f"{'='*60}")

        for step in range(n_steps):
            prompt = corpus[step % len(corpus)]

            try:
                metrics = self.distill_episode(prompt)
                all_losses.append(metrics['total_loss'])
                all_distill.append(metrics['mean_distill_loss'])
            except RuntimeError as e:
                if "out of memory" in str(e).lower():
                    print(f"  [OOM step {step}] — skip + empty_cache")
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                    continue
                raise

            # Sleep consolidation tous les 20 steps
            if (step + 1) % 20 == 0:
                self.student._sleep_consolidation()

            # Progression ASCII
            if (step + 1) % 10 == 0:
                recent = all_losses[-10:]
                mean_loss = sum(recent) / len(recent)
                bar_len = 30
                progress = int(bar_len * (step + 1) / n_steps)
                bar = "█" * progress + "░" * (bar_len - progress)
                ram_info = ""
                try:
                    import psutil
                    vm = psutil.virtual_memory()
                    ram_info = f" | RAM={vm.used/1e9:.1f}/{vm.total/1e9:.1f}GB ({vm.percent:.0f}%)"
                except ImportError:
                    pass
                print(
                    f"  [{bar}] {step+1}/{n_steps} "
                    f"| loss={mean_loss:.4f} "
                    f"| bridge_grad={self._bridge_grad_norm():.2e}"
                    f"{ram_info}"
                )

        # Métriques agrégées
        result = {
            'n_episodes': len(all_losses),
            'mean_loss': sum(all_losses) / len(all_losses) if all_losses else 0.0,
            'final_loss': all_losses[-1] if all_losses else 0.0,
            'loss_reduction': (
                (all_losses[0] - all_losses[-1]) / all_losses[0]
                if all_losses and all_losses[0] > 1e-8 else 0.0
            ),
        }

        print(f"\n  Distillation terminée — loss finale : {result['final_loss']:.4f}")
        print(f"  Réduction loss : {result['loss_reduction']*100:.1f}%")

        return result

    def _bridge_grad_norm(self) -> float:
        """Norme des gradients des bridges (monitoring)."""
        total = 0.0
        for p in self.bridges.parameters():
            if p.grad is not None:
                total += p.grad.data.norm(2).item() ** 2
        return total ** 0.5

    def emergency_save(self, path: str):
        """
        Sauvegarde d'urgence des bridges — appelable depuis try/finally.

        Sauvegarde le state_dict des bridges et l'optimizer.
        """
        try:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            torch.save({
                'bridges': self.bridges.state_dict(),
                'optimizer': self.optimizer.state_dict(),
                'total_episodes': self.total_episodes,
                'loss_history': self.loss_history[-100:],  # garder les 100 derniers
            }, path)
            print(f"[DistillationEngine] Sauvegarde d'urgence : {path}")
        except Exception as e:
            print(f"[DistillationEngine] ERREUR sauvegarde d'urgence : {e}")

    def load_bridges(self, path: str):
        """Charge les bridges depuis un checkpoint."""
        ckpt = torch.load(path, map_location=self.device, weights_only=False)
        self.bridges.load_state_dict(ckpt['bridges'])
        if 'optimizer' in ckpt:
            self.optimizer.load_state_dict(ckpt['optimizer'])
        self.total_episodes = ckpt.get('total_episodes', 0)
        self.loss_history = ckpt.get('loss_history', [])
        print(f"[DistillationEngine] Bridges chargés — {self.total_episodes} épisodes")

    def save_bridges(self, path: str):
        """Sauvegarde complète des bridges."""
        torch.save({
            'bridges': self.bridges.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'total_episodes': self.total_episodes,
            'loss_history': self.loss_history,
        }, path)
        print(f"[DistillationEngine] Bridges sauvegardés : {path}")
