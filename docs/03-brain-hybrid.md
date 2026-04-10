# Partie 3 — Brain-Hybrid : Simulation cérébrale complète

## Table des matières

1. [BrainHybridModel — Assemblage complet](#1-brainhybridmodel)
2. [BrainModule — Module CfC + SNN hybride](#2-brainmodule)
3. [STDP — Plasticité synaptique](#3-stdp)
4. [InjectionGate — Greffe CfC → Qwen](#4-injectiongate)
5. [HierarchicalPC — Codage prédictif](#5-hierarchicalpc)
6. [ACCModule — Détection de conflit](#6-accmodule)
7. [HippocampalMemory — Mémoire épisodique](#7-hippocampalmemory)
8. [StepScheduler — Rythmes cérébraux](#8-stepscheduler)
9. [DistilledInputLayer — Fusion teacher-student](#9-distilledinputlayer)
10. [Data Flow — Flux de données complet](#10-data-flow)

---

## 1. BrainHybridModel

**Fichier** : `brain_hybrid/model.py`

### 1.1 Rôle

BrainHybridModel est un **cerveau artificiel complet** qui greffe des modules neuronaux adaptatifs sur un LLM gelé. Il orchestre la cognition, la mémoire épisodique, la détection de conflit et les rythmes cérébraux.

### 1.2 Architecture globale

```
┌─────────────────────────────────────────────────────────────────┐
│                      BrainHybridModel                            │
│                                                                 │
│  ┌──────────┐     ┌───────────────────────────────────────┐    │
│  │ Qwen3-4B  │     │  Cerveau adaptatif                    │    │
│  │ (gelé)    │     │                                       │    │
│  │           │     │  ┌─────────┐  ┌─────────┐            │    │
│  │ Couche 8  │◄───►│  │Module 0  │  │ STDP 0  │            │    │
│  │           │     │  └─────────┘  └─────────┘            │    │
│  │ Couche 16 │◄───►│  ┌─────────┐  ┌─────────┐            │    │
│  │           │     │  │Module 1  │  │ STDP 1  │            │    │
│  │ Couche 24 │◄───►│  ┌─────────┐  ┌─────────┐            │    │
│  │           │     │  │Module 2  │  │ STDP 2  │            │    │
│  │ Couche 32 │◄───►│  ┌─────────┐  ┌─────────┐            │    │
│  │           │     │  │Module 3  │  │ STDP 3  │            │    │
│  └──────────┘     │  └─────────┘  └─────────┘            │    │
│                    │                                       │    │
│                    │  ┌──────────┐  ┌──────────────┐       │    │
│                    │  │   ACC     │  │ StepScheduler │       │    │
│                    │  │ (conflit) │  │ (rythmes)     │       │    │
│                    │  └──────────┘  └──────────────┘       │    │
│                    │                                       │    │
│                    │  ┌──────────────┐  ┌────────────┐     │    │
│                    │  │ HierarchicalPC│  │ Hippocampe  │     │    │
│                    │  │ (prédictif)   │  │ (SDM)       │     │    │
│                    │  └──────────────┘  └────────────┘     │    │
│                    │                                       │    │
│                    │  ┌──────────────────────────────────┐  │    │
│                    │  │  4 × InjectionGate (CfC → Qwen)  │  │    │
│                    │  └──────────────────────────────────┘  │    │
│                    └───────────────────────────────────────┘    │
│                                                                 │
│  ┌──────────────────┐  ┌──────────────────┐                    │
│  │ HippocampalPrefix │  │ DistilledInput    │                    │
│  │ Injector          │  │ Layer (teacher)    │                    │
│  └──────────────────┘  └──────────────────┘                    │
└─────────────────────────────────────────────────────────────────┘
```

### 1.3 Composants assemblés

| Composant | Quantité | Fichier | Rôle |
|-----------|----------|---------|------|
| QwenWrapper | 1 | `llm/qwen_wrapper.py` | LLM gelé, extraction hidden states |
| BrainModule | 4 | `core/brain_module.py` | CfC + SNN par couche d'intercept |
| STDPLearning | 4 | `core/stdp.py` | Plasticité synaptique par module |
| InjectionGate | 4 | `core/injection.py` | Gate low-rank CfC → Qwen |
| BrainHookManager | 1 | `core/injection.py` | Gestion des forward hooks |
| HippocampalMemory | 1 | `memory/hippocampus.py` | Sparse Distributed Memory |
| HippocampalPrefixInjector | 1 | `core/injection.py` | Mémoire → prefix tokens |
| ACCModule | 1 | `core/acc.py` | Détection de conflit |
| HierarchicalPC | 1 | `core/predictive_coding.py` | Codage prédictif 4 couches |
| StepScheduler | 1 | `utils/step_scheduler.py` | Rythmes cérébraux |
| DistilledInputLayer | 1 | `core/distilled_input.py` | Fusion teacher-student |

---

## 2. BrainModule

**Fichier** : `brain_hybrid/core/brain_module.py`

### 2.1 Architecture d'un module

Chaque BrainModule est greffé sur une couche de Qwen et combine deux types de réseaux neuronaux biologiquement inspirés :

```
Layer representation (1, seq_len, 2560)
            │
            ▼
    ┌────────────────┐
    │  input_proj      │  Linear(2560, 64)
    │  2560 → 64       │  Nécessaire : CfC ne gère pas les grandes dimensions
    └────────┬───────┘
             │
      ┌──────┴──────┐
      │              │
      ▼              ▼
┌──────────┐  ┌──────────┐
│  CfC      │  │  SNN      │
│  (liquid)  │  │  (spikes) │
│           │  │           │
│  NCP:     │  │  LIF:     │
│  4 sensory│  │  Leaky    │
│  16 inter │  │  Integrate│
│  12 cmd   │  │  & Fire   │
│  8 motor  │  │           │
│           │  │  Surrogate│
│  État     │  │  gradient │
│  persiste │  │           │
└────┬─────┘  └────┬─────┘
     │              │
     ▼              ▼
  cfc_out        spikes
  (64-dim)      (64-dim)
     │              │
     └──────┬───────┘
            │
            ▼
    ┌────────────────┐
    │  output_proj     │  Linear(64, 2560)
    │  64 → 2560       │  Retour vers l'espace Qwen
    └────────┬───────┘
             │
             ▼
    prediction (1, seq_len, 2560)
    + spikes (1, seq_len, 64)
```

### 2.2 CfC — Continuous-time Flow Cell

Le CfC est un réseau neuronal liquide basé sur le connectome du ver C. elegans :

```python
wiring = NCP(
    inter_neurons=16,              # Neurones intermédiaires
    command_neurons=12,            # Neurones de commande
    motor_neurons=8,               # Neurones moteurs (output)
    sensory_fanout=4,              # Connexions sensory → inter
    inter_fanout=3,                # Connexions inter → command
    recurrent_command_synapses=3,  # Récurrence dans command
    motor_fanin=4,                 # Connexions command → motor
)
cfc = CfC(input_size=64, units=wiring, mode="pure")
```

**Propriétés** :
- Dynamiques **continues** dans le temps (pas de pas discrets)
- État **persistant** entre les appels (mémoire à long terme)
- Câblage **sparse** inspiré de la biologie
- Paramètres ~500× moins nombreux qu'un transformer équivalent

### 2.3 SNN — Spiking Neural Network

Le SNN utilise des neurones Leaky Integrate-and-Fire avec gradient surrogate :

```python
spike_fn = surrogate.fast_sigmoid(slope=25)
lif = snn.Leaky(beta=0.95, spike_grad=spike_fn, init_hidden=True)
```

**Propriétés** :
- Communication par **spikes** (0 ou 1), pas par activations continues
- `beta=0.95` : décroissance du potentiel membranaire (95% rétention)
- Surrogate gradient : permet le calcul de gradient à travers les spikes

### 2.4 Traces pré/post synaptiques

Chaque module maintient des traces pour le STDP :

```python
self.pre_trace = None   # Activation avant le CfC
self.post_trace = None  # Activation après le CfC

def forward(self, x):
    projected = self.input_proj(x)      # (B, S, 64)
    self.pre_trace = projected.detach()  # Sauvegarde pré-synaptique
    
    cfc_out, self.cfc_state = self.cfc(projected, hx=self.cfc_state)
    spk, self.mem = self.lif(projected)
    
    self.post_trace = cfc_out.detach()   # Sauvegarde post-synaptique
    
    prediction = self.output_proj(cfc_out)
    return prediction, spk
```

---

## 3. STDP — Spike-Timing-Dependent Plasticity

**Fichier** : `brain_hybrid/core/stdp.py` (103 lignes)

### 3.1 Principe biologique

```
    Pré s'active AVANT post → RENFORCEMENT (LTP)
    ─────────────────────────────────────────────
    temps:  ──────▶
    pré:    █
    post:        █
    synapse: ───▲─── (force augmentée)

    Post s'active AVANT pré → AFFAIBLISSEMENT (LTD)
    ─────────────────────────────────────────────
    temps:  ──────▶
    post:   █
    pré:         █
    synapse: ───▼─── (force diminuée)
```

### 3.2 Neuromodulation dopaminergique

Le signal dopamine module l'amplitude de l'apprentissage STDP :

```python
def update_dopamine(self, prediction_error, threshold=0.3):
    normalized = min(1.0, abs(prediction_error) / threshold)
    self.dopamine_signal = 0.9 * self.dopamine_signal + 0.1 * normalized
    # Lissage exponentiel : évite les fluctuations brutales
```

```
Erreur de prédiction élevée
    → Surprise
    → Dopamine haute (≈ 1.0)
    → Apprentissage STDP amplifié
    → "Cet événement est important, retiens-le"

Erreur de prédiction faible
    → Rien de nouveau
    → Dopamine basse (≈ 0.0)
    → Apprentissage STDP minimal
    → "Situation déjà connue, rien à retenir"
```

### 3.3 Application STDP

```python
def apply(self, weight, pre_trace, post_trace):
    # Moyennes dimensionnelles
    pre_mean = pre_trace.mean(0).mean(0)
    post_mean = post_trace.mean(0).mean(0)
    
    # Interpolation pour matcher les dimensions des poids CfC
    pre_r = interpolate(pre_mean, size=in_dim)
    post_r = interpolate(post_mean, size=out_dim)
    
    # LTP : post_r ⊗ pre_r  (outer product)
    ltp = post_r.unsqueeze(1) * pre_r.unsqueeze(0)
    
    # LTD : pre_as_out ⊗ post_as_in  (inversé pour l'asymétrie)
    ltd = pre_as_out.unsqueeze(1) * post_as_in.unsqueeze(0)
    
    # Delta modulé par la dopamine
    delta = lr_plus * dopamine * ltp - lr_minus * dopamine * ltd
    delta = clamp(delta, -1e-3, 1e-3)  # Stabilité numérique
    
    weight.data += delta
```

### 3.4 Paramètres

| Paramètre | Valeur par défaut | Rôle |
|-----------|-------------------|------|
| `lr_plus` | 0.01 | Taux de renforcement (LTP) |
| `lr_minus` | 0.01 | Taux d'affaiblissement (LTD) |
| `dopamine_signal` | 0.5 (initial) | Modulateur d'apprentissage [0, 1] |
| `threshold` | 0.3 | Seuil de normalisation dopamine |

---

## 4. InjectionGate — Greffe CfC → Qwen

**Fichier** : `brain_hybrid/core/injection.py` (135 lignes)

### 4.1 Concept

Les InjectionGates permettent au cerveau adaptatif d'**influencer les représentations internes du LLM** pendant la génération de texte. C'est la principale innovation architecturale de Brain-Hybrid.

### 4.2 Architecture du gate

```
CfC prediction (1, seq_len, 4096)
            │
            ▼
    ┌────────────────┐
    │  LayerNorm       │  Normalisation
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  down_proj       │  Linear(4096, 16)  — compression low-rank
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  tanh             │  Borne la sortie [-1, 1] — stabilité
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  up_proj         │  Linear(16, 4096)  — retour dimension Qwen
    └────────┬───────┘
             │
             ▼
    ┌────────────────┐
    │  × alpha          │  Scalaire appris (init=0.001, max=0.1)
    └────────┬───────┘
             │
             ▼
        delta (1, seq_len, 4096)
```

**Formule** : `delta = alpha × up_proj(tanh(down_proj(norm(prediction))))`

### 4.3 BrainHookManager — Injection via forward hooks

Le manager installe des hooks PyTorch sur les couches de Qwen :

```python
class BrainHookManager:
    def register_hooks(self, qwen_model, brain_modules, injection_gates, layer_indices):
        layers = self._get_decoder_layers(qwen_model)
        
        for i, layer_idx in enumerate(layer_indices):
            module = brain_modules[i]
            gate = injection_gates[i]
            
            def make_hook(mod, gt):
                def hook_fn(layer_module, layer_input, layer_output):
                    hidden_states = layer_output[0]
                    prediction, _ = mod(hidden_states)       # CfC prédit
                    delta = gt(prediction)                    # Gate transforme
                    modified = hidden_states + delta           # Injection résiduelle
                    return (modified,) + layer_output[1:]
                return hook_fn
            
            handle = layer.register_forward_hook(make_hook(module, gate))
```

### 4.4 HippocampalPrefixInjector

Convertit un souvenir de l'hippocampe en **tokens virtuels** préfixés aux embeddings :

```
Recalled memory (4096,)
        │
        ▼
    memory_to_prefix: Linear(4096, 4 × 4096)
        │
        ▼
    reshape → (1, 4, 4096)   # 4 tokens virtuels
        │
        ▼
    × prefix_gate (init=0.01)  # Amplitude faible au début
        │
        ▼
    Préfixé aux embeddings d'entrée de Qwen
```

### 4.5 Sécurité de l'injection

| Mécanisme | Rôle | Valeur |
|-----------|------|--------|
| `tanh` | Borne les deltas à [-1, 1] | Structurel |
| `alpha` initial | Force initiale quasi-nulle | 0.001 |
| `alpha` max | Force maximale bornée | 0.1 |
| `prefix_gate` | Amplitude des prefix tokens | 0.01 |
| `LayerNorm` | Normalisation avant injection | Structurel |
| Low-rank (16) | Compression 4096→16→4096 | Structurel |

L'injection démarre presque invisible et grandit progressivement pendant l'entraînement.

---

## 5. HierarchicalPC — Codage prédictif hiérarchique

**Fichier** : `brain_hybrid/core/predictive_coding.py` (138 lignes)

### 5.1 Principe (Karl Friston, Free Energy Principle)

Chaque couche du cerveau **prédit** l'activité de la couche inférieure. Seule l'**erreur de prédiction** remonte. Le cerveau est une machine à minimiser la surprise.

```
Couche 3 (module 3, couche Qwen 32)
    │ prédit ↓              ↑ erreur
    ▼                        │
Couche 2 (module 2, couche Qwen 24)
    │ prédit ↓              ↑ erreur
    ▼                        │
Couche 1 (module 1, couche Qwen 16)
    │ prédit ↓              ↑ erreur
    ▼                        │
Couche 0 (module 0, couche Qwen 8)
```

### 5.2 PredictiveCodingLayer

```python
class PredictiveCodingLayer:
    module: BrainModule          # Le module CfC+SNN
    pc_lr: float = 0.001        # Taux d'apprentissage PC
    prediction: Tensor | None    # Dernière prédiction top-down
    error: Tensor | None         # Erreur = actual - predicted
    precision: float = 1.0       # 1/variance(error) — confiance
```

#### Calcul de l'erreur et de la précision

```python
def compute_error(self, actual, predicted):
    self.error = actual - predicted               # Erreur brute
    self.precision = 1.0 / (error.var() + 1e-8)  # Confiance
    return self.error
```

**Précision** : inverse de la variance de l'erreur. Si l'erreur est stable (faible variance), la précision est haute — le module est confiant dans ses prédictions.

### 5.3 Mise à jour PC locale

```python
def pc_update(self, stdp_learner, dopamine_signal):
    modulation = pc_lr × precision × dopamine × error_magnitude
    
    for param in module.cfc.parameters():
        # 1. Appliquer STDP standard (pre/post traces)
        stdp_learner.apply(param, pre_trace, post_trace)
        
        # 2. Perturbation PC additionnelle (proportionnelle à la précision)
        noise = randn_like(param) * modulation * 1e-4
        param.data += noise.clamp(-1e-4, 1e-4)
```

### 5.4 HierarchicalPC — Orchestration des 4 couches

```python
class HierarchicalPC:
    layers: List[PredictiveCodingLayer]  # 4 couches PC
    
    def forward(self, layer_reps: List[Tensor]) -> List[Tensor]:
        errors = []
        for i, pc_layer in enumerate(self.layers):
            prediction, spikes = pc_layer.module(layer_reps[i])
            if i < len(layer_reps) - 1:
                error = pc_layer.compute_error(layer_reps[i+1], prediction)
                errors.append(error)
        return errors  # 3 erreurs (entre couches adjacentes)
    
    def update_all(self, errors, dopamine_signals, stdp_learners):
        for i, pc_layer in enumerate(self.layers):
            pc_layer.pc_update(stdp_learners[i], dopamine_signals[i])
```

---

## 6. ACCModule — Cortex Cingulaire Antérieur

**Fichier** : `brain_hybrid/core/acc.py` (166 lignes)

### 6.1 Rôle

L'ACC détecte les **conflits** entre signaux internes (erreurs de prédiction divergentes, dopamine élevée, entropie) et déclenche des réponses adaptatives.

### 6.2 Architecture

```
Input (9 dimensions) :
  ├── 4 erreurs de prédiction (une par module)
  ├── 4 signaux dopamine (un par STDP learner)
  └── 1 entropie d'action

MLP sans backprop :
  fc1: Linear(9, 32) → ReLU
  fc2: Linear(32, 32) → ReLU
  fc3: Linear(32, 1) → Sigmoid → conflict_score ∈ [0, 1]

Apprentissage : Hebbian local (pas de backprop)
  Δw = lr × conflict_score × (input - mean_input)
```

### 6.3 Trois états possibles

| État | Condition | Réponse |
|------|-----------|---------|
| **Conflit** | `score > 0.7` | Arousal boost + prompt enrichi |
| **Exploration** | `0.4 < score ≤ 0.7` | Légère augmentation dopamine |
| **Stable** | `score ≤ 0.4` | Mode normal |

### 6.4 Réponse au conflit

Quand un conflit est détecté :

1. **Arousal boost** : le StepScheduler double toutes les fréquences pendant 50 steps
2. **Prompt enrichi** : un fragment est préfixé au prompt Qwen

```python
if is_conflict:
    fragment = f"[CONFLIT DÉTECTÉ: erreur={mean_err:.3f}, arousal={score:.2f}] "
```

### 6.5 Tendance historique

```python
def trend(self):
    recent = history[-10:]
    older = history[-20:-10]
    if recent_mean > older_mean * 1.1: return "rising"
    if recent_mean < older_mean * 0.9: return "falling"
    return "stable"
```

---

## 7. HippocampalMemory — Sparse Distributed Memory

**Fichier** : `brain_hybrid/memory/hippocampus.py` (131 lignes)

### 7.1 Principe (Kanerva, 1988)

La SDM est fondamentalement différente d'un vectorstore classique :

| Aspect | Vectorstore | SDM |
|--------|------------|-----|
| Adresses | Continues (embeddings) | Binaires (aléatoires) |
| Stockage | 1 location par mémoire | Distribué sur N locations |
| Lecture | Cosine similarity exacte | Vote majoritaire par voisinage |
| Robustesse | Fragile (dépend de la distance) | Robuste (récupération partielle) |

### 7.2 Architecture

```
HippocampalMemory
│
├── addresses: (10000, 256)    # Adresses binaires aléatoires (fixes)
├── contents: (10000, 2560)    # Contenu accumulé (additive writing)
├── access_counts: (10000,)    # Compteurs d'accès par location
│
├── activation_radius: 115     # Rayon de Hamming pour l'activation
│
└── metadata: []               # Métadonnées des souvenirs
```

### 7.3 Opérations

#### Écriture distribuée

```python
def write(self, embedding, content, metadata=None):
    # 1. Convertir embedding → adresse binaire
    address = to_binary_address(embedding)
    # Binarisation : flat > median(flat) → 1/0
    
    # 2. Trouver les locations actives (Hamming ≤ 115)
    active = get_active_locations(address)
    # diffs = (addresses - address).abs().sum(dim=1) ≤ 115
    
    # 3. Écriture additive distribuée
    contents[active] += content
    access_counts[active] += 1
```

#### Lecture par reconstruction

```python
def read(self, embedding):
    address = to_binary_address(embedding)
    active = get_active_locations(address)
    
    if len(active) == 0:
        return zeros(content_dim)
    
    # Vote majoritaire : moyenne des contenus actifs
    return contents[active].mean(dim=0)
```

#### Consolidation (sommeil)

```python
def consolidate(self, threshold=5):
    # Élagage synaptique : supprime les locations peu accédées
    low_access = access_counts < threshold
    contents[low_access] = 0
    access_counts[low_access] = 0
```

### 7.4 Propriétés clés

- **Jamais remis à zéro** — Principe fondamental #4
- **Écriture additive** — Les souvenirs s'accumulent, ne s'écrasent pas
- **Récupération partielle** — Un fragment suffit pour reconstruire un souvenir complet
- **Consolidation pendant le sommeil** — Les traces faibles sont élaguées

---

## 8. StepScheduler — Rythmes cérébraux

**Fichier** : `brain_hybrid/utils/step_scheduler.py` (100 lignes)

### 8.1 Fréquences par module

```
         Fréquence (steps)
Module        │
SNN           │█  (1 — chaque step)
Hippocampe    │██████  (6 — rythme thêta)
CfC           │██████████  (10)
Qwen enrichi  │████████████████████████████████████████████████████████████████████████████████████████████████████  (100)
Sleep         │████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████  (5000)
```

| Module | Fréquence | Analogie biologique |
|--------|-----------|---------------------|
| SNN | 1 step | Activité gamma rapide (~40Hz) |
| Hippocampe | 6 steps | Oscillation thêta (~6Hz) |
| CfC | 10 steps | Rythme alpha (~10Hz) |
| Qwen enrichi | 100 steps | Traitement cortical rare |
| Sleep | 5000 steps | Consolidation nocturne |

### 8.2 Phases cérébrales

```python
def get_phase(self):
    if step % 6 == 0: return "theta"   # Rythme hippocampique
    if step % 2 == 0: return "gamma"   # Activité rapide
    return "idle"                       # Repos
```

### 8.3 Arousal boost

Quand l'ACC détecte un conflit :

```python
def arousal_boost(self):
    # Double toutes les fréquences pendant 50 steps
    self.current_freqs = {
        name: max(1, freq // 2)
        for name, freq in self.base_freqs.items()
    }
    # SNN: 1→1, CfC: 10→5, Hippocampe: 6→3, etc.
    # = plus d'activité cérébrale sous stress
```

---

## 9. DistilledInputLayer — Fusion teacher-student

**Fichier** : `brain_hybrid/core/distilled_input.py` (118 lignes)

### 9.1 Rôle

Fusionne les représentations brutes du student (Qwen3-4B) avec les représentations projetées du teacher (Qwen3-32B via ProjectionBridge).

### 9.2 Architecture

```
Student reps (B, S, 2560)         Teacher projected (B, S, 2560)
        │                                  │
        │                                  │
        └──────────┬───────────────────────┘
                   │
                   ▼
           ┌──────────────┐
           │  concat + gate │  Linear(5120, 1) → Sigmoid
           └──────┬───────┘
                  │
                  ▼
        enriched = α × student + (1-α) × gate × teacher
```

### 9.3 Paramètres appris

| Paramètre | Init | Rôle |
|-----------|------|------|
| `alpha[i]` (×4) | 0.8 | Poids du student (1-α = poids teacher) |
| `gate` | Linear(5120, 1) | Quand le teacher est fiable |

### 9.4 Interprétation

```
Alpha = 0.8 (début) : 80% student, 20% teacher
    → Le student domine, le teacher guide légèrement

Alpha → 0.5 (convergé) : 50/50
    → Les deux sources contribuent également

Alpha → 0.2 (si teacher dominant) : 20% student, 80% teacher
    → Le teacher porte l'essentiel de l'information
```

Le gate **ferme automatiquement** si le signal teacher est peu fiable (cosine similarity faible).

---

## 10. Data Flow — Flux de données complet

### 10.1 Cycle complet d'un step

```python
# === 1. Rythmes cérébraux ===
scheduler.step()

# === 2. Extraire les hidden states de Qwen ===
layer_reps = llm.get_layer_representations(prompt, layers=[8, 16, 24, 32])
# layer_reps[0]: (1, S, 2560) — couche 8
# layer_reps[1]: (1, S, 2560) — couche 16
# layer_reps[2]: (1, S, 2560) — couche 24
# layer_reps[3]: (1, S, 2560) — couche 32

# === 3. Enrichissement par teacher (optionnel) ===
if bridge_available:
    enriched_reps, fusion_stats = distilled_input(layer_reps, teacher_projected)

# === 4. Codage prédictif hiérarchique ===
pc_errors = hierarchical_pc.forward(layer_reps)
# pc_errors[0]: erreur couche 8→16
# pc_errors[1]: erreur couche 16→24
# pc_errors[2]: erreur couche 24→32

# === 5. BrainModules forward ===
for i, module in enumerate(brain_modules):
    if scheduler.should_update("cfc"):
        prediction, spikes = module(layer_reps[i])
        error = prediction - layer_reps[i+1]
        
        # STDP + dopamine
        stdp_learners[i].update_dopamine(error.abs().mean())
        if learn:
            stdp_learners[i].apply(
                module.cfc.parameters(),
                module.pre_trace,
                module.post_trace,
            )

# === 6. ACC — détection de conflit ===
acc_output = acc.forward(
    prediction_errors=[e.abs().mean().item() for e in pc_errors],
    dopamine_signals=[s.dopamine_signal for s in stdp_learners],
)
if acc_output.is_conflict:
    scheduler.arousal_boost()

# === 7. Hippocampe (si scheduler le permet) ===
if scheduler.should_update("hippocampus"):
    # Écriture si saillant
    mean_error = sum(e.abs().mean() for e in pc_errors) / len(pc_errors)
    if mean_error > config.salience_threshold:
        hippocampus.write(
            embedding=layer_reps[-1].mean(dim=1).squeeze(),
            content=layer_reps[-1].mean(dim=1).squeeze(),
            metadata={"step": scheduler.global_step, "error": mean_error.item()},
        )

# === 8. Génération avec injection ===
if acc_output.is_conflict:
    prompt = acc_output.conflict_prompt_fragment + prompt

hook_manager.register_hooks(llm.model, brain_modules, injection_gates, [8, 16, 24, 32])
response = llm.generate(prompt)
hook_manager.remove_hooks()

# === 9. Sleep replay (si scheduler le déclenche) ===
if scheduler.should_update("sleep"):
    for _ in range(config.sleep_replay_count):
        random_memory = hippocampus.read(random_embedding)
        if random_memory.norm() > 0:
            hierarchical_pc.forward([random_memory] * 4)  # Replay
    hippocampus.consolidate()
```

### 10.2 Diagramme de séquence

```
    Scheduler    Qwen(gelé)    BrainModules    STDP    PC    ACC    Hippocampe
        │            │             │            │       │      │        │
        │ step()     │             │            │       │      │        │
        │──────►     │             │            │       │      │        │
        │            │             │            │       │      │        │
        │      get_layer_reps()    │            │       │      │        │
        │     ──────────►          │            │       │      │        │
        │     ◄──────────          │            │       │      │        │
        │     layer_reps[4]        │            │       │      │        │
        │            │             │            │       │      │        │
        │            │   forward() │            │       │      │        │
        │            │ ──────────► │            │       │      │        │
        │            │ ◄────────── │            │       │      │        │
        │            │ pred+spikes │            │       │      │        │
        │            │             │  apply()   │       │      │        │
        │            │             │ ─────────► │       │      │        │
        │            │             │            │       │      │        │
        │            │             │            │forward()     │        │
        │            │             │            │ ────► │      │        │
        │            │             │            │ ◄──── │      │        │
        │            │             │            │errors │      │        │
        │            │             │            │       │forward()      │
        │            │             │            │       │ ───► │        │
        │            │             │            │       │ ◄─── │        │
        │            │             │            │       │score │        │
        │            │             │            │       │      │        │
        │    [if conflict: arousal_boost()]      │       │      │        │
        │────────────────────────────────────────────────────►│        │
        │            │             │            │       │      │        │
        │    [if salient: write()]  │            │       │      │ write()│
        │──────────────────────────────────────────────────────────────►│
        │            │             │            │       │      │        │
        │   generate_with_hooks()  │            │       │      │        │
        │     ──────────►          │            │       │      │        │
        │     ◄──────────          │            │       │      │        │
        │     response             │            │       │      │        │
```
