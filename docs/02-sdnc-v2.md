# Partie 2 — SDNC v2 : Circuits Prédictifs

## Table des matières

1. [SDNCModel — Le modèle principal](#1-sdncmodel)
2. [PredictiveCircuitBank — Banque de circuits prédictifs](#2-predictivecircuitbank)
3. [GlobalState — État contextuel persistant](#3-globalstate)
4. [TemporalStream — Flux temporel continu](#4-temporalstream)
5. [SparseRouter — Routage sparse](#5-sparserouter)
6. [Hebbian Learning — Apprentissage local](#6-hebbian-learning)
7. [TemporalIntegrator — Intégrateur CfC](#7-temporalintegrator)
8. [Encoders — Extracteurs de features](#8-encoders)
9. [Memory Systems — Mémoire épisodique](#9-memory-systems)
10. [Data Flow — Flux de données complet](#10-data-flow)

---

## 1. SDNCModel

**Fichier** : `sdnc/model.py` (413 lignes)

### 1.1 Rôle

SDNCModel est le point d'entrée principal du système SDNC v2. Il orchestre l'ensemble du flux de traitement prédictif : encodage → contexte temporel → routage sparse → circuits prédictifs → apprentissage local.

### 1.2 Architecture

```
         Input (images/text/audio/signals)
              │
              ▼
    ┌─────────────────┐
    │  Encoders (gelés) │  CLIP ViT-B/32, CLIP Text, Whisper, Signal
    │  → 512-dim        │
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  GlobalState.read │  Lecture de l'a priori contextuel
    │  + résidu         │  prior_biased = features + prior
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  TemporalStream   │  Enrichissement par l'historique récent
    │  .get_context()   │  Attention temporelle apprise
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  TemporalIntegrator│  Compression CfC (état persistant)
    │  (CfC + proj)     │  512-dim → 512-dim
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │  SparseRouter     │  Sélection top-k circuits
    │  (TokenChoice     │  TokenChoice : input → top-k circuits
    │   ou ExpertChoice)│  ExpertChoice : circuit → top-k inputs
    └────────┬────────┘
             │
             ▼
    ┌─────────────────┐
    │ PredictiveCircuit │  Double tête :
    │ Bank              │  • Processing CfC → circuit_output (32-dim)
    │                   │  • Prediction CfC → predicted input (512-dim)
    │                   │  → prediction_error = actual - predicted
    └────────┬────────┘
             │
             ├──► TemporalStream.push(features)     [pour le prochain appel]
             ├──► GlobalState.write(prediction_error) [nouvelle connaissance]
             ├──► GlobalState.apply_decay()           [oubli graduel]
             │
             ▼
    ┌─────────────────┐
    │  circuit_proj     │  Projection → espace de classification
    │  32-dim → 128-dim │  L2-normalisé
    └────────┬────────┘
             │
             ▼
        circuit_repr (128-dim)
        → Classification prototypique
```

### 1.3 Modes d'opération

#### Forward (inférence)

```python
model = SDNCModel(SDNCConfig())
result = model(images=batch_images)

# Résultat :
result["circuit_repr"]          # (batch, 128) — représentation dans l'espace circuit
result["prediction"]            # (batch, 512) — ce que les circuits prédisaient
result["prediction_error"]      # (batch, 512) — actual - predicted
result["prediction_error_norm"] # scalaire — norme moyenne de l'erreur
result["active_indices"]        # (batch, k) — quels circuits étaient actifs
result["weights"]               # (batch, k) — poids de routage
result["global_state_summary"]  # dict — état du GlobalState
result["stream_stats"]          # dict — statistiques du flux temporel
```

#### Learn (entraînement)

Deux signaux d'apprentissage simultanés :

```python
result = model.learn(labels=labels, images=batch_images)
```

**Signal 1 — Prototypical Loss (backprop)** :
- Calcul de prototypes par classe dans l'espace circuit (128-dim)
- Distance euclidienne query → prototypes
- NLL loss + prediction error loss + entropy regularization
- Backprop sur : router, circuit_bank, circuit_proj, integrator, global_state, temporal_stream

**Signal 2 — Hebbian Update (local, sans backprop)** :
- Règle d'Oja appliquée aux poids CfC du circuit de traitement
- Learning rate modulé par la norme de l'erreur de prédiction
- Plus l'erreur est grande → plus l'apprentissage est fort

#### Recognize (few-shot)

```python
logits = model.recognize(
    support_labels=support_labels,  # (n_way * k_shot,)
    support_kwargs={"images": support_images},
    query_kwargs={"images": query_images},
)
predictions = logits.argmax(dim=-1)
```

### 1.4 Paramètres entraînables vs gelés

| Composant | Entraînable | Méthode |
|-----------|-------------|---------|
| Vision encoder (CLIP) | Non | Gelé |
| Text encoder (CLIP) | Non | Gelé |
| Audio encoder (Whisper) | Non (sauf proj) | Gelé + proj trainable |
| Signal encoder | Oui | Backprop |
| SparseRouter | Oui | Backprop |
| PredictiveCircuitBank | Oui | Backprop + Hebbian |
| GlobalState (query/key/write) | Oui | Backprop |
| TemporalStream (fusion/attn) | Oui | Backprop |
| TemporalIntegrator (CfC) | Oui | Backprop |
| circuit_proj | Oui | Backprop |

---

## 2. PredictiveCircuitBank

**Fichier** : `sdnc/core/predictive_circuits.py`

### 2.1 Concept

Chaque circuit possède **deux têtes CfC** :

```
                    Input (512-dim)
                         │
              ┌──────────┴──────────┐
              │                      │
    ┌─────────▼─────────┐  ┌───────▼──────────┐
    │  Processing Head   │  │  Prediction Head   │
    │  (CfC principal)   │  │  (CfC prédicteur)  │
    │                    │  │                    │
    │  input_dim → 32    │  │  proc_state → 512  │
    │  NCP wiring:       │  │  NCP wiring:       │
    │  48 inter, 32 cmd  │  │  24 inter, 16 cmd  │
    └─────────┬─────────┘  └───────┬──────────┘
              │                      │
              ▼                      ▼
      circuit_output (32)    prediction (512)
              │                      │
              └──────────┬───────────┘
                         │
                         ▼
              prediction_error = actual - prediction
```

### 2.2 Innovation : circuits partagés avec états indépendants

La banque utilise **un seul circuit** (partagé) mais **N paires d'états** indépendants :

```python
# Architecture interne
self.circuit = PredictiveCircuit(config)  # UN seul circuit (poids partagés)
self.proc_states = torch.zeros(n_circuits, proc_state_size)  # N états processing
self.pred_states = torch.zeros(n_circuits, pred_state_size)  # N états prediction
```

**Pourquoi ?** Économie massive de paramètres. Au lieu de 1000 circuits × (poids processing + poids prediction), on a 1 circuit × poids + 1000 × (état processing + état prediction). Les poids sont partagés, la spécialisation émerge des états.

### 2.3 Inter-circuit wiring

Les circuits ne sont pas isolés — ils développent des connexions entre eux via apprentissage Hebbian :

```python
# Matrice de connexions inter-circuits (n_circuits × n_circuits)
self.inter_circuit_weights = torch.zeros(n_circuits, n_circuits)

# Renforcement : les circuits co-activés développent des connexions
strengthen_co_active_circuits(
    co_activation_matrix,     # Quels circuits étaient actifs ensemble ?
    connection_weights,        # Matrice actuelle
    lr=1e-3,                  # Taux d'apprentissage
    decay=0.999,              # Oubli graduel
)
```

### 2.4 Tracking des erreurs de prédiction

Chaque circuit maintient un historique de ses erreurs :

```python
self.prediction_error_history = torch.zeros(n_circuits)  # Erreur moyenne par circuit
self.activation_history = torch.zeros(n_circuits)         # Nombre d'activations
```

Cela permet de diagnostiquer quels circuits prédisent bien (erreur faible) et lesquels ont besoin de plus d'entraînement.

---

## 3. GlobalState

**Fichier** : `sdnc/core/global_state.py` (120 lignes)

### 3.1 Concept

Le GlobalState est l'**a priori contextuel** — ce que le système "sait déjà" avant de voir une nouvelle entrée. C'est l'équivalent de la mémoire de travail persistante.

### 3.2 Structure

```
GlobalState
│
├── state: (n_slots=8, dim=512)     # Mémoire à slots multiples
│
├── READ (attention légère)
│   ├── query_proj: Linear(512, 512)  # Projette l'input en query
│   ├── key_proj: Linear(512, 512)    # Projette les slots en keys
│   └── scale: 512^(-0.5)            # Facteur d'échelle
│
├── WRITE (mise à jour gated)
│   ├── write_gate: Linear(1024, 512) → Sigmoid  # Combien mettre à jour
│   └── write_value: Linear(1024, 512) → Tanh    # Quoi écrire
│
└── DECAY
    └── decay_rate: 0.995             # Oubli graduel
```

### 3.3 Opérations

#### Lecture (Read)

```python
# L'input "interroge" le GlobalState via attention
context = global_state.read(encoder_features)  # (batch, 512)

# Internement :
# 1. state_snapshot = state.detach().clone()   (pas de gradient sur le state)
# 2. q = query_proj(x)                        (batch, 512)
# 3. k = key_proj(state_snapshot)              (n_slots, 512)
# 4. attn = softmax(q @ k^T * scale)          (batch, n_slots)
# 5. context = attn @ state_snapshot           (batch, 512)
```

#### Écriture (Write)

```python
# L'erreur de prédiction est écrite dans le GlobalState
global_state.write(prediction_error)

# Internement (pour chaque slot) :
# 1. x_mean = prediction_error.mean(dim=0)    (représentatif du batch)
# 2. combined = cat([slot, x_mean])            (1024,)
# 3. gate = sigmoid(write_gate(combined))      (512,) — combien mettre à jour
# 4. value = tanh(write_value(combined))       (512,) — quoi écrire
# 5. state[i] = (1-gate)*slot + gate*value     (mise à jour gated)
```

#### Oubli (Decay)

```python
global_state.apply_decay()
# state *= 0.995  — empêche la saturation
```

### 3.4 Propriétés clés

- **Persistant** : jamais reset pendant l'entraînement (sauf `reset()` explicite)
- **Détaché du graphe** : la lecture utilise `detach().clone()` pour éviter les conflits inplace
- **Partagé** : le write prend la moyenne du batch — l'état est global, pas per-sample
- **Buffer PyTorch** : enregistré comme `register_buffer` — pas de gradient

---

## 4. TemporalStream

**Fichier** : `sdnc/core/temporal_stream.py` (137 lignes)

### 4.1 Concept

Le TemporalStream est un **buffer circulaire** qui stocke les N dernières entrées. Chaque circuit voit l'input courant **dans le contexte de l'historique récent**, pas de manière isolée.

### 4.2 Structure

```
TemporalStream
│
├── buffer: (history_len=16, dim=512)     # Ring buffer des dernières entrées
├── write_idx: int                         # Position d'écriture courante
├── filled: int                            # Nombre d'entrées écrites
│
├── position_encoding: Embedding(16, 512)  # Encodage positionnel appris
├── temporal_attn: Linear(512, 1)          # Attention : quelles entrées comptent ?
│
└── fusion: Sequential                     # Fusion current + history
    ├── Linear(1024, 512)
    ├── GELU
    └── Linear(512, 512)
```

### 4.3 Opérations

#### Push (enregistrement)

```python
temporal_stream.push(encoder_features)
# 1. x_repr = features.mean(dim=0)    (moyenne du batch)
# 2. buffer[write_idx] = x_repr
# 3. write_idx = (write_idx + 1) % 16  (circulaire)
# 4. filled = min(filled + 1, 16)
```

#### Get Context (fusion avec historique)

```python
context = temporal_stream.get_context(prior_biased)  # (batch, 512)

# Internement :
# 1. history = get_history()                    (T, 512) ordonné old→new
# 2. pos_encoded = history + position_encoding  (T, 512)
# 3. attn_scores = temporal_attn(pos_encoded)   (T, 1)
# 4. attn_weights = softmax(attn_scores)        (T,)
# 5. history_summary = sum(attn * pos_encoded)  (512,)
# 6. fused = fusion(cat[input, history_summary]) (batch, 512)
```

### 4.4 Buffer circulaire — ordre de lecture

```
Exemple : buffer de taille 4, write_idx=2, filled=4

buffer = [C, D, _, _]  ← write_idx pointe ici (position 2)
                ^
Historique ordonné : [_, _, C, D] → les plus anciens d'abord

Après wrap :
buffer = [C, D, E, F]  write_idx=0 (wrappé)
Historique : [E, F, C, D] → réordonné [buffer[2:], buffer[:2]]
```

---

## 5. SparseRouter

**Fichier** : `sdnc/core/sparse_router.py` (165 lignes)

### 5.1 Deux modes de routage

Le système propose deux stratégies de routage, chacune avec ses avantages :

#### TokenChoice (Phase 1)

Chaque **input** choisit ses top-k circuits :

```
Input batch (B=4)         Circuits (N=1000)
    │                          │
    │    gate: Linear(512, 1000)
    ▼                          ▼
Scores (B, 1000) ───► topk(k=3) ───► indices (B, 3) + weights (B, 3)

Avantage : Simple, rapide
Risque   : Déséquilibre — certains circuits monopolisent
Solution : expert_bias qui pénalise les circuits suractivés
```

```python
# Correction du déséquilibre
def update_bias(self, gamma=0.001):
    target_count = self.activation_counts.sum() / self.n_circuits
    self.expert_bias -= gamma * (self.activation_counts - target_count).sign()
```

#### ExpertChoice (Phase 2+)

Chaque **circuit** choisit ses top-k inputs — inversion du paradigme :

```
Score matrix S (B × N_circuits)
    │
    ▼ softmax over tokens (column-wise)
    │
    ▼ flatten → topk global → (circuit_idx, token_idx) pairs
    │
    ▼ group by circuit → assignments + weights
```

**Garanties** :
- Équilibre parfait structurel (pas de circuit monopolisant)
- Chaque token reçoit un nombre borné d'experts
- Total compute = `capacity_factor × batch_size`

### 5.2 Entropie de routage

L'entropie du routeur mesure la diversité des circuits activés :

```python
def entropy_loss(self, x):
    probs = softmax(self.gate(x))
    entropy = -(probs * log(probs)).sum(dim=-1).mean()
    return -entropy  # Minimiser → maximiser l'entropie → plus de diversité
```

### 5.3 Ratio de sparsité

```python
# TokenChoice : k/N = 3/1000 = 0.3%
# ExpertChoice : capacity_factor/N = 1.25/1000 = 0.125%
```

Seule une fraction infime des circuits est active à chaque instant — comme dans le cerveau.

---

## 6. Hebbian Learning

**Fichier** : `sdnc/core/hebbian_update.py` (223 lignes)

### 6.1 Règle d'Oja

La règle d'Oja est une version stabilisée de la règle de Hebb :

```
Hebb classique : Δw = lr × post × pre        (instable — les poids explosent)
Oja stabilisé  : Δw = lr × post × (pre - post^T × w)
                                    └────────────┘
                                    terme de stabilisation
```

```python
def oja_update(weight, pre, post, lr=1e-4, mask=None):
    pre_mean = pre.mean(0)
    post_mean = post.mean(0)
    residual = pre_mean - post_mean @ weight       # stabilisation
    delta = lr * post_mean.unsqueeze(1) * residual.unsqueeze(0)  # outer product
    if mask is not None:
        delta = delta * (mask != 0).float()        # respecter la sparsité NCP
    return weight + delta
```

### 6.2 Application aux CfC

Les CfC internes ont une structure NCP à 3 couches :

```
Layer 0 : sensory → inter    (input + inter_state → inter_output)
Layer 1 : inter → command    (inter_output + command_state → command_output)
Layer 2 : command → motor    (command_output + motor_state → motor_output)
```

L'update Hebbian traverse chaque couche séquentiellement, en respectant les `sparsity_mask` du câblage NCP.

### 6.3 Élagage synaptique

```python
def synaptic_pruning(module, threshold=1e-5):
    # Les connexions dont le poids absolu < threshold sont mises à zéro
    # Les connexions structurellement nulles (sparsity_mask) sont préservées
    mask = param.abs() > threshold
    if name in sparsity_masks:
        structural_mask = sparsity_masks[name] != 0
        mask = mask | (~structural_mask)
    param.data *= mask.float()
```

### 6.4 Renforcement inter-circuits

```python
def strengthen_co_active_circuits(co_activation_matrix, connection_weights, lr, decay):
    normalized = co_activation_matrix / co_activation_matrix.sum()
    return decay * connection_weights + lr * normalized
```

Les circuits fréquemment co-activés développent des connexions plus fortes — exactement comme les assemblées neuronales de Hebb.

---

## 7. TemporalIntegrator

**Fichier** : `sdnc/core/integrator.py` (73 lignes)

### 7.1 Rôle

L'intégrateur temporel compresse et synchronise les entrées multimodales dans le temps. Son état CfC **persiste entre les appels** — il encode "ce qui se passe maintenant" en tenant compte de tout le passé.

### 7.2 Architecture

```
TemporalIntegrator
│
├── CfC (AutoNCP wiring)
│   ├── input_size: 512 (input_dim)
│   ├── units: 64 (integrator_state_dim)
│   ├── output_size: 16 (state_dim // 4)
│   └── sparsity_level: 0.5
│
├── proj: Linear(16, 512)          # Projection retour vers input_dim
│
└── state: (batch, 64)             # État persistant
    └── .detach() après chaque appel  # Pas d'accumulation de gradient
```

### 7.3 Persistance

```python
def forward(self, x):
    # L'état survit entre les appels
    cfc_out, self.state = self.cfc(x.unsqueeze(1), hx=self.state)
    self.state = self.state.detach()  # Coupe le gradient mais garde l'information
    return self.proj(cfc_out)
```

---

## 8. Encoders

### 8.1 Vision Encoder (CLIP ViT-B/32)

**Fichier** : `sdnc/encoders/vision_encoder.py`

```
Images (batch, 3, 224, 224)
    │
    ▼ CLIP ViT-B/32 (gelé)
    │
    ▼ L2 normalize
    │
    ▼ (batch, 512)
```

- **Lazy loading** : le modèle CLIP n'est chargé qu'au premier appel
- **Gelé** : `requires_grad = False` sur tous les paramètres
- **Output** : embeddings normalisés L2 de dimension 512

### 8.2 Text Encoder (CLIP Text)

**Fichier** : `sdnc/encoders/text_encoder.py`

```
Text tokens (batch, seq_len)
    │
    ▼ CLIP text encoder (gelé)
    │
    ▼ L2 normalize
    │
    ▼ (batch, 512)
```

### 8.3 Audio Encoder (Whisper tiny)

**Fichier** : `sdnc/encoders/audio_encoder.py`

```
Audio (batch, mel_features)
    │
    ▼ Whisper tiny encoder (gelé) → 384-dim
    │
    ▼ proj: Linear(384, 512)  ← seule partie entraînable
    │
    ▼ (batch, 512)
```

### 8.4 Signal Encoder

**Fichier** : `sdnc/encoders/signal_encoder.py`

```
Signals (batch, 8)
    │
    ▼ Linear(8, 64) → GELU → Linear(64, 512)  ← entièrement entraînable
    │
    ▼ (batch, 512)
```

### 8.5 Dispatch multimodal

```python
def encode(self, **kwargs):
    if kwargs.get("images") is not None:
        return self.vision_encoder(kwargs["images"])
    elif kwargs.get("text_tokens") is not None:
        return self.text_encoder(kwargs["text_tokens"])
    elif kwargs.get("audio") is not None:
        return self.audio_encoder(kwargs["audio"])
    elif kwargs.get("signals") is not None:
        return self.signal_encoder(kwargs["signals"])
```

---

## 9. Memory Systems

### 9.1 Episodic Memory

**Fichier** : `sdnc/memory/episodic_memory.py` (128 lignes)

Stocke les représentations passées pour la récupération par similarité :

```python
class EpisodicMemory:
    embedding_dim: int          # Dimension des embeddings (128 = circuit_repr_dim)
    max_memories: int = 10000   # Capacité maximale
    episodes: list[Episode]     # Liste des épisodes stockés
    _embeddings_cache: Tensor   # Cache pour recherche batch rapide

class Episode:
    embedding: Tensor           # Vecteur (128-dim)
    label: int | None           # Label de classe optionnel
    activation_pattern: Tensor  # Quels circuits étaient actifs
    strength: float = 1.0       # Force du souvenir
```

#### Opérations

| Opération | Description | Complexité |
|-----------|-------------|------------|
| `store()` | Ajoute un épisode, éviction FIFO si plein | O(1) |
| `retrieve(query, top_k)` | Cosine similarity → top-k | O(N) |
| `consolidate(threshold)` | Fusionne les épisodes similaires > threshold | O(N²) |

### 9.2 Salience Detector

**Fichier** : `sdnc/memory/consolidation.py`

Filtre les épisodes par importance avant stockage :

```python
class SalienceDetector:
    threshold: float = 0.7  # Seuil minimum de saillance
    
    def is_salient(self, embedding) -> bool:
        # Un épisode est saillant si sa norme dépasse le seuil
        return embedding.norm() > self.threshold
```

---

## 10. Data Flow — Flux de données complet

### 10.1 Forward pass détaillé

```python
# Étape par étape avec dimensions

# 1. Encode (gelé)
encoder_features = encode(images=batch)           # (B, 512)

# 2. Read global state
prior_context = global_state.read(encoder_features) # (B, 512)
prior_biased = encoder_features + prior_context      # (B, 512) résidu

# 3. Temporal stream
stream_context = temporal_stream.get_context(prior_biased) # (B, 512)

# 4. Integrate (CfC persistant)
integrated = integrator(stream_context)              # (B, 512)

# 5a. Route (TokenChoice)
indices, weights = router(integrated)                # (B, 3), (B, 3)

# 5b. Route (ExpertChoice)
(assignments, weights, circuit_map, counts), top_w, token_counts = router(integrated)

# 6. Predictive circuits
circuit_result = circuit_bank.forward_token_choice(integrated, indices, weights)
# circuit_result["circuit_output"]     → (B, 32)
# circuit_result["prediction"]         → (B, 512)
# circuit_result["prediction_error"]   → (B, 512)
# circuit_result["hidden_states"]      → (B, state_size)

# 7. Push to stream
temporal_stream.push(encoder_features)

# 8. Write to global state
global_state.write(circuit_result["prediction_error"])
global_state.apply_decay()

# 9. Project to circuit space
circuit_repr = circuit_proj(circuit_result["circuit_output"]) # (B, 128)
circuit_repr = L2_normalize(circuit_repr)                      # (B, 128)
```

### 10.2 Learn — Double signal

```
                    ┌────────────────────────────────────┐
                    │        SIGNAL 1 : Backprop          │
                    │                                    │
                    │  Prototypical loss in circuit space │
                    │  + prediction error loss            │
                    │  + entropy regularization           │
                    │                                    │
                    │  ► Mise à jour : router, bank,     │
                    │    proj, integrator, global_state,  │
                    │    temporal_stream                  │
                    └────────────────────────────────────┘

                    ┌────────────────────────────────────┐
                    │     SIGNAL 2 : Hebbian (local)      │
                    │                                    │
                    │  Oja's rule sur les poids CfC       │
                    │  LR ∝ ||prediction_error||          │
                    │  Plus d'erreur → plus d'apprentissage│
                    │                                    │
                    │  ► Mise à jour : poids CfC du       │
                    │    processing_cfc (dans circuit)    │
                    └────────────────────────────────────┘

                    ┌────────────────────────────────────┐
                    │  SIGNAL 3 : Inter-circuit wiring    │
                    │                                    │
                    │  Co-activation → renforcement       │
                    │  + decay graduel                    │
                    │                                    │
                    │  ► Mise à jour : matrice de         │
                    │    connexions inter-circuits        │
                    └────────────────────────────────────┘
```

### 10.3 Recognize — Classification few-shot

```
Support set (5 classes × 1 shot)          Query set (5 × 15 queries)
         │                                         │
         ▼ forward()                               ▼ forward()
         │                                         │
    support_repr (5, 128)                    query_repr (75, 128)
         │                                         │
         ▼ moyenne par classe                      │
         │                                         │
    prototypes (5, 128)                            │
         │                                         │
         └──────────── cdist ──────────────────────┘
                         │
                    dists (75, 5)
                         │
                    -dists → logits
                         │
                    argmax → predictions
```
