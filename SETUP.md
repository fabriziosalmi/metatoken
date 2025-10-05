# Setup e Utilizzo del Progetto Meta-Token

## Prerequisiti

- Python 3.8 o superiore
- pip (gestore pacchetti Python)

## Installazione

1. Clonare il repository:
```bash
git clone https://github.com/fabriziosalmi/metatoken.git
cd metatoken
```

2. Installare le dipendenze:
```bash
pip install -r requirements.txt
```

## Generazione del Dataset

Il progetto include un generatore di dataset ibrido avanzato che crea esempi di training con meta-token.

### Struttura dei Dati di Configurazione

Il generatore utilizza file JSON nella cartella `generator_data/`:

- **vocabs.json**: Vocabolari organizzati per tipo semantico (verbi, articoli, sostantivi, ecc.)
- **rich_content.json**: Contenuti dinamici ricchi (fatti scientifici, spiegazioni tecniche, ecc.)
- **templates.json**: Template di conversazione che definiscono la struttura degli esempi
- **semantic_map.json**: Mappatura parola → tipo semantico per classificazione automatica

### Generare un Dataset

```bash
# Genera 1000 esempi (default)
python crea_dataset.py

# Genera un numero personalizzato di esempi
python crea_dataset.py --num-esempi 5000

# Specifica un file di output personalizzato
python crea_dataset.py --num-esempi 10000 --output-file my_dataset.json
```

Il file generato avrà il formato:
```json
{
  "examples": [
    {
      "id": "unique_id",
      "template_name": "nome_template",
      "sequence": [
        {
          "word": "parola",
          "ruolo": "UTENTE|BOT_RAGIONAMENTO|BOT_RISPOSTA|BOT_CHIARIMENTO",
          "semantico": "tipo_semantico"
        }
      ],
      "metadata": {
        "vocab_keys_used": [...],
        "token_count": 42,
        "has_dynamic_content": true
      }
    }
  ]
}
```

## Training del Modello

Il progetto include diverse versioni di script di training, dalla più semplice (from scratch) alla più avanzata (con transfer learning).

### Script Principali

- **minimal.py**: Versione base con training da zero
- **minimal_9.py**: Versione 9 con GPT-2 italiano e singola testa di predizione
- **minimal_10.py**: Versione 10 con architettura autonoma (3 teste di predizione)

### Training con minimal_9.py

Questo script allena un modello che predice solo le parole, usando i meta-token come contesto:

```bash
python minimal_9.py
```

Parametri configurabili nello script:
- `max_iters`: Numero di iterazioni di training (default: 10000)
- `learning_rate`: Tasso di apprendimento (default: 2e-5)
- `batch_size`: Dimensione del batch (default: 4)
- `patience`: Pazienza per early stopping (default: 5)

### Training con minimal_10.py

Questo script riprende il modello V9 e lo allena a predire autonomamente i meta-token:

```bash
python minimal_10.py
```

**Nota**: Richiede che il modello V9 sia stato allenato in precedenza (checkpoint in `model_v9_checkpoints/`).

## Struttura del Progetto

```
metatoken/
├── README.md                  # Documentazione del progetto
├── SETUP.md                   # Questa guida
├── requirements.txt           # Dipendenze Python
├── crea_dataset.py           # Generatore di dataset
├── generator_data/           # File di configurazione per il generatore
│   ├── vocabs.json
│   ├── rich_content.json
│   ├── templates.json
│   └── semantic_map.json
├── minimal.py                # Script base (training from scratch)
├── minimal_2.py - minimal_8.py  # Versioni intermedie
├── minimal_9.py              # V9: Transfer learning con GPT-2
├── minimal_10.py             # V10: Predizione autonoma dei meta-token
└── metatoken.ipynb           # Notebook Jupyter con esperimenti
```

## Meta-Token: Concetti Chiave

### Ruoli Strutturali

- **UTENTE**: Token provenienti dal prompt dell'utente
- **BOT_RAGIONAMENTO**: Token che rappresentano il processo di pensiero interno del bot
- **BOT_RISPOSTA**: Token della risposta finale al utente
- **BOT_CHIARIMENTO**: Token usati quando il bot chiede chiarimenti

### Tipi Semantici (Esempi)

- **VERBO_AZIONE**: spiegami, dimmi, descrivi
- **ARTICOLO**: il, lo, la, un, una
- **SOSTANTIVO_ASTRATTO**: concetto, idea, teoria
- **CONCETTO_SCIENTIFICO**: relatività, quantistica, evoluzione
- **CONCETTO_TECNOLOGICO**: algoritmo, transformer, neural network
- **PUNCT**: . , ? ! ; :
- **PAROLA_CONTENUTO**: Parole non classificate altrimenti

## Personalizzazione

### Aggiungere Nuovi Template

Modifica `generator_data/templates.json`:

```json
{
  "nome": "mio_template",
  "struttura": [
    ["UTENTE", "VERBO_AZIONE"],
    ["UTENTE", "ARTICOLO"],
    ["UTENTE", "CONCETTO_SCIENTIFICO"],
    ["BOT_RISPOSTA", "DYNAMIC_SCIENTIFIC_FACT"]
  ]
}
```

### Aggiungere Nuovo Vocabolario

Modifica `generator_data/vocabs.json`:

```json
"MIO_TIPO_SEMANTICO": [
  "parola1",
  "parola2",
  "parola3"
]
```

### Aggiungere Contenuti Dinamici

Modifica `generator_data/rich_content.json`:

```json
"DYNAMIC_MIO_CONTENUTO": [
  "Frase ricca di contenuto 1",
  "Frase ricca di contenuto 2"
]
```

## Troubleshooting

### Errore: FileNotFoundError per generator_data

Assicurati che la cartella `generator_data/` esista e contenga tutti i file JSON necessari.

### Errore: ModuleNotFoundError

Installa le dipendenze mancanti:
```bash
pip install -r requirements.txt
```

### Errore: CUDA out of memory

Riduci il `batch_size` negli script di training.

### Errore: Checkpoint V9 non trovato

Esegui prima `minimal_9.py` per creare il checkpoint necessario per `minimal_10.py`.

## Risorse Aggiuntive

- README principale per dettagli teorici e architetturali
- Notebook `metatoken.ipynb` per esperimenti interattivi
- Script intermedi (`minimal_2.py` - `minimal_8.py`) per vedere l'evoluzione del progetto

## Licenza

Consulta il file LICENSE nel repository.
