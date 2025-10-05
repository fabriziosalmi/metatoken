# Guida Rapida per Sviluppatori

## Quick Start

```bash
# 1. Installazione
pip install -r requirements.txt

# 2. Genera dataset
python crea_dataset.py --num-esempi 1000

# 3. Allena modello V9
python minimal_9.py

# 4. (Opzionale) Allena modello V10 autonomo
python minimal_10.py
```

## Comandi Principali

### Generazione Dataset

```bash
# Dataset piccolo per test
python crea_dataset.py --num-esempi 100

# Dataset medio per sviluppo
python crea_dataset.py --num-esempi 1000

# Dataset grande per produzione
python crea_dataset.py --num-esempi 10000
```

### Verifica Dataset

```python
import json
with open('dataset.json') as f:
    data = json.load(f)
    print(f"Esempi totali: {len(data['examples'])}")
    print(f"Primo esempio: {data['examples'][0]}")
```

## Struttura dei File di Configurazione

### vocabs.json
Dizionario di liste: `{tipo_semantico: [parole]}`

### rich_content.json
Dizionario di liste: `{DYNAMIC_placeholder: [testi]}`

### templates.json
Lista di template: `[{nome, struttura: [[ruolo, definizione]]}]`

### semantic_map.json
Dizionario parola→tipo: `{parola: tipo_semantico}`

## Architettura del Modello

### minimal_9.py - Transfer Learning Base
- Input: word + ruolo + semantico (embeddings sommati)
- Output: solo predizione della prossima parola
- Base: GPT-2 italiano pre-allenato

### minimal_10.py - Predizione Autonoma
- Input: word + ruolo + semantico
- Output: 3 predizioni (word, ruolo, semantico)
- Base: Checkpoint dal modello V9

## Formato dei Dati

### Esempio in dataset.json
```json
{
  "word": "spiegami",
  "ruolo": "UTENTE",
  "semantico": "VERBO_AZIONE"
}
```

### Conversione durante il training
1. Tokenizzazione BPE della parola → lista di sub-token IDs
2. Ogni sub-token eredita ruolo e semantico dalla parola
3. Padding delle sequenze a `block_size` (default: 128)

## Device Support

Il codice rileva automaticamente:
- **MPS** (Apple Silicon): `device = 'mps'`
- **CPU** (fallback): `device = 'cpu'`

Per GPU NVIDIA, modificare:
```python
device = 'cuda' if torch.cuda.is_available() else 'cpu'
```

## Parametri Chiave

### Training
- `max_iters`: Iterazioni totali
- `learning_rate`: Tasso di apprendimento
- `batch_size`: Dimensione batch
- `eval_interval`: Frequenza valutazione
- `patience`: Early stopping

### Generazione
- `max_new_tokens`: Token da generare
- `top_p`: Nucleus sampling threshold
- `repetition_penalty`: Penalità ripetizioni

## Debug

### Verificare i vocabolari dei meta-token
```python
import json
from pathlib import Path

raw_data = json.load(open('dataset.json'))['examples']
all_ruoli = set([t['ruolo'] for ex in raw_data for t in ex['sequence']])
all_semantici = set([t['semantico'] for ex in raw_data for t in ex['sequence']])

print(f"Ruoli: {sorted(all_ruoli)}")
print(f"Semantici: {len(all_semantici)} tipi")
```

### Monitorare il training
```python
# Durante il training, osserva:
# - Train loss deve diminuire
# - Val loss dovrebbe essere simile a train loss
# - Se val loss >> train loss → overfitting
# - Se entrambe alte → underfitting o problemi dati
```

### Test di generazione rapido
```python
# Modifica minimal_9.py o minimal_10.py
# Cambia il prompt iniziale:
start_text = "Spiegami i transformer"  # ← Personalizza qui
```

## Contribuire

1. Fork il repository
2. Crea un branch per la feature: `git checkout -b feature/nome`
3. Commit: `git commit -m 'Aggiungi feature'`
4. Push: `git push origin feature/nome`
5. Apri una Pull Request

## Best Practices

### Dataset
- Usa almeno 1000 esempi per training significativo
- Bilancia i template (usa statistiche del generatore)
- Valida il JSON generato prima del training

### Training
- Monitora val loss, non solo train loss
- Usa early stopping per evitare overfitting
- Salva checkpoints regolarmente

### Generazione
- Testa con prompt diversi
- Regola `top_p` per creatività (0.9-0.95 ottimale)
- Usa `repetition_penalty` per ridurre loops

## FAQ

**Q: Quanto dataset serve?**
A: Minimo 1000, ottimale 10k+ per risultati robusti.

**Q: Quanto tempo impiega il training?**
A: Dipende da dataset size e device. ~10-30 min per 10k iterazioni su CPU.

**Q: Posso usare altri modelli base?**
A: Sì, modifica `MODEL_NAME` in minimal_*.py con qualsiasi GPT-2 compatible.

**Q: Come migliorare la qualità delle risposte?**
A: 1) Più dati, 2) Template migliori, 3) Modello base più grande.

## Contatti

Per domande o problemi, apri un Issue su GitHub.
