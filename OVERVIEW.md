# Panoramica del Progetto Meta-Token

## Indice
1. [Introduzione](#introduzione)
2. [Architettura](#architettura)
3. [File Principali](#file-principali)
4. [Pipeline Completa](#pipeline-completa)
5. [Meta-Token: Concetti Chiave](#meta-token-concetti-chiave)

## Introduzione

Il progetto Meta-Token è un sistema innovativo per il training di modelli di linguaggio che generano non solo testo, ma anche meta-informazioni strutturali e semantiche per ogni token.

### Obiettivi
- **Interpretabilità**: Rendere esplicito il processo di ragionamento del modello
- **Controllabilità**: Permettere un controllo fine-grained sulla generazione
- **Struttura**: Insegnare al modello a seguire flussi conversazionali strutturati

## Architettura

```
┌─────────────────────────────────────────────────────────┐
│                    Input Layer                          │
│  Word Embedding + Ruolo Embedding + Semantico Embedding │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│              GPT-2 Transformer Layers                   │
│  (Pre-trained: GroNLP/gpt2-small-italian)              │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│                  Output Heads                           │
│  ┌──────────┐  ┌──────────┐  ┌──────────────┐         │
│  │  Word    │  │  Ruolo   │  │  Semantico   │         │
│  │  Head    │  │  Head    │  │  Head        │         │
│  └──────────┘  └──────────┘  └──────────────┘         │
└─────────────────────────────────────────────────────────┘
```

### Evoluzione del Progetto

#### Fase 1: From Scratch (V1-V4)
- Training da zero su dataset piccoli
- **Problema**: Overfitting estremo
- **Lezione**: Necessità di transfer learning

#### Fase 2: Transfer Learning (V5-V9)
- Uso di GPT-2 pre-allenato
- Meta-token solo in input
- Predizione solo delle parole
- **Risultato**: Generazione fluente e grammaticale

#### Fase 3: Autonomia (V10-V13)
- Reintroduzione predizione meta-token
- 3 output heads (word, ruolo, semantico)
- **Risultato**: Modello autonomo che pianifica e struttura la risposta

## File Principali

### Script di Training

#### `minimal.py`
- Versione base con training da zero
- Architettura transformer semplificata
- Uso: Comprensione dei concetti base

#### `minimal_9.py`
- **Raccomandato per iniziare**
- Transfer learning con GPT-2 italiano
- Meta-token come contesto in input
- Predizione solo parole
- Output: `model_v9_checkpoints/`

#### `minimal_10.py`
- **Avanzato**: Predizione autonoma
- Richiede checkpoint V9
- 3 teste di predizione
- Output: `model_v10_final/`

### Generazione Dataset

#### `crea_dataset.py`
Generatore di dataset ibrido avanzato.

**Caratteristiche**:
- Template-based con variabilità
- Contenuto dinamico ricco
- Classificazione semantica automatica
- Output in formato JSON

**File di Configurazione**:
- `generator_data/vocabs.json`: Vocabolari per tipo semantico
- `generator_data/rich_content.json`: Contenuti dinamici (fatti, spiegazioni)
- `generator_data/templates.json`: Template conversazionali
- `generator_data/semantic_map.json`: Mapping parola→semantica

### Inferenza

#### `inference_example.py`
Script per generazione con modello allenato.

**Uso**:
```bash
python inference_example.py \
  --model-path model_v10_final \
  --prompt "Spiegami i transformer" \
  --max-tokens 50
```

## Pipeline Completa

### 1. Generazione Dataset
```bash
python crea_dataset.py --num-esempi 10000 --output-file dataset.json
```

**Output**: `dataset.json`
```json
{
  "examples": [
    {
      "sequence": [
        {"word": "spiegami", "ruolo": "UTENTE", "semantico": "VERBO_AZIONE"},
        {"word": "la", "ruolo": "UTENTE", "semantico": "ARTICOLO"},
        ...
      ]
    }
  ]
}
```

### 2. Training Fase 1 (V9)
```bash
python minimal_9.py
```

**Processo**:
1. Carica GPT-2 italiano pre-allenato
2. Aggiunge embedding per ruolo e semantico
3. Fine-tuning per predire parole
4. Early stopping basato su validation loss

**Output**: 
- `model_v9_checkpoints/best_model.pt`
- `model_v9_checkpoints/meta_vocabs.json`

### 3. Training Fase 2 (V10)
```bash
python minimal_10.py
```

**Processo**:
1. Carica checkpoint V9
2. Aggiunge teste per ruolo e semantico
3. Fine-tuning per predizione autonoma
4. Genera esempi con meta-token predetti

**Output**:
- `model_v10_final/best_model.pt`
- `model_v10_final/meta_vocabs.json`

### 4. Inferenza
```bash
python inference_example.py --prompt "Cos'è un buco nero?"
```

**Output**: Testo + tabella con meta-token predetti

## Meta-Token: Concetti Chiave

### Ruoli Strutturali

| Ruolo | Descrizione | Esempio |
|-------|-------------|---------|
| `UTENTE` | Prompt dell'utente | "Spiegami la relatività" |
| `BOT_RAGIONAMENTO` | Processo di pensiero | "Analizzo la richiesta..." |
| `BOT_RISPOSTA` | Risposta finale | "La relatività è..." |
| `BOT_CHIARIMENTO` | Richiesta chiarimenti | "Intendi la relatività generale?" |

### Tipi Semantici (Esempi)

#### Grammaticali
- `ARTICOLO`: il, la, un, una
- `VERBO_AZIONE`: spiegami, dimmi, descrivi
- `VERBO_ESSERE`: è, sono, era
- `CONGIUNZIONE`: e, ma, però, quindi
- `PREPOSIZIONE`: di, a, in, con

#### Contenuto
- `SOSTANTIVO_ASTRATTO`: concetto, teoria, idea
- `SOSTANTIVO_CONCRETO`: oggetto, strumento
- `AGGETTIVO`: importante, complesso, facile
- `AVVERBIO`: molto, sempre, probabilmente

#### Domini Specialistici
- `CONCETTO_SCIENTIFICO`: relatività, quantistica, evoluzione
- `CONCETTO_TECNOLOGICO`: algoritmo, transformer, neural network
- `CONCETTO_MATEMATICO`: derivata, matrice, probabilità

#### Strutturali
- `PUNCT`: . , ? ! ; :
- `AFFERMAZIONE`: Certamente, Certo, Sicuramente
- `NEGAZIONE`: No, Non, Mai
- `PAROLA_CONTENUTO`: Fallback generico

## Parametri Configurabili

### Training
```python
max_iters = 10000        # Iterazioni totali
learning_rate = 2e-5     # Tasso apprendimento
batch_size = 4           # Dimensione batch
eval_interval = 500      # Frequenza valutazione
patience = 5             # Early stopping
warmup_steps = 500       # Warmup scheduler
```

### Generazione
```python
max_new_tokens = 100     # Token da generare
top_p = 0.95            # Nucleus sampling
repetition_penalty = 1.2 # Anti-ripetizione
block_size = 128        # Dimensione contesto
```

### Dataset
```python
num_esempi = 10000      # Esempi da generare
template_variety = True  # Usa tutti i template
dynamic_content = True   # Usa contenuti ricchi
```

## Best Practices

### Dataset
1. **Dimensione**: Minimo 1000, ottimale 10k+ esempi
2. **Bilanciamento**: Distribuisci esempi tra template
3. **Varietà**: Usa contenuti dinamici ricchi
4. **Validazione**: Controlla format JSON prima del training

### Training
1. **Device**: Usa GPU/MPS se disponibile
2. **Monitoring**: Osserva train/val loss
3. **Checkpointing**: Salva best model
4. **Early Stopping**: Evita overfitting

### Generazione
1. **Prompt**: Chiaro e contestualizzato
2. **Parametri**: Bilancia creatività (top_p) e coerenza
3. **Lunghezza**: Genera abbastanza token per output completo
4. **Post-processing**: Decodifica BPE correttamente

## Troubleshooting Comune

### Training non converge
- Riduci learning rate
- Aumenta warmup steps
- Verifica qualità dataset

### Generazione ripetitiva
- Aumenta repetition_penalty
- Riduci top_p
- Verifica overfitting

### Out of Memory
- Riduci batch_size
- Riduci block_size
- Usa modello più piccolo

### Output incoerente
- Aumenta dimensione dataset
- Migliora template
- Usa modello base più grande

## Risorse Aggiuntive

- [README.md](README.md): Documentazione teorica completa
- [SETUP.md](SETUP.md): Guida installazione e setup
- [QUICKSTART.md](QUICKSTART.md): Guida rapida
- [CONTRIBUTING.md](CONTRIBUTING.md): Come contribuire

## Prossimi Passi

1. Segui [SETUP.md](SETUP.md) per configurare l'ambiente
2. Genera un dataset con `crea_dataset.py`
3. Allena il modello V9 con `minimal_9.py`
4. Sperimenta con l'inferenza usando `inference_example.py`
5. (Opzionale) Allena V10 per predizione autonoma

---

Per domande o supporto, apri un issue su GitHub.
