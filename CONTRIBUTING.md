# Guida alla Contribuzione

Grazie per l'interesse nel contribuire al progetto Meta-Token! Questa guida ti aiuterà a iniziare.

## Come Contribuire

### Segnalare Bug

Se trovi un bug, apri un issue su GitHub con:
- Descrizione chiara del problema
- Passi per riprodurlo
- Comportamento atteso vs. comportamento osservato
- Informazioni sul tuo ambiente (OS, versione Python, ecc.)

### Proporre Nuove Funzionalità

Per proporre nuove funzionalità:
1. Apri un issue descrivendo la funzionalità
2. Spiega il caso d'uso e i benefici
3. Discuti l'implementazione con i maintainer
4. Procedi con una Pull Request una volta approvata

### Inviare Pull Request

1. **Fork** il repository
2. **Crea un branch** per la tua modifica:
   ```bash
   git checkout -b feature/nome-funzionalita
   ```
3. **Fai le tue modifiche** seguendo le linee guida qui sotto
4. **Testa** le tue modifiche
5. **Commit** con messaggi descrittivi:
   ```bash
   git commit -m "Aggiungi funzionalità X per risolvere Y"
   ```
6. **Push** al tuo fork:
   ```bash
   git push origin feature/nome-funzionalita
   ```
7. **Apri una Pull Request** su GitHub

## Linee Guida per il Codice

### Stile Python

- Segui [PEP 8](https://www.python.org/dev/peps/pep-0008/)
- Usa nomi di variabili descrittivi
- Commenta codice complesso
- Mantieni funzioni brevi e focalizzate

### Struttura del Codice

```python
# Esempio di struttura consigliata

def funzione_ben_documentata(param1: str, param2: int) -> dict:
    """
    Descrizione breve della funzione.
    
    Args:
        param1: Descrizione del primo parametro
        param2: Descrizione del secondo parametro
        
    Returns:
        Dizionario contenente i risultati
    """
    # Implementazione
    pass
```

### Testing

- Testa le tue modifiche prima di inviare una PR
- Verifica che gli script esistenti continuino a funzionare
- Se possibile, aggiungi test per nuove funzionalità

```bash
# Test manuale base
python crea_dataset.py --num-esempi 10 --output-file test.json
python minimal_9.py  # Verifica che non ci siano errori di importazione
```

## Aree di Contribuzione

### 1. Miglioramento Dataset Generator

**File**: `crea_dataset.py`, `generator_data/*.json`

Contribuzioni benvenute:
- Nuovi template di conversazione in `templates.json`
- Espansione dei vocabolari in `vocabs.json`
- Nuovi contenuti dinamici in `rich_content.json`
- Miglioramento delle euristiche semantiche

Esempio di contribuzione:
```json
// Aggiungere un nuovo template in templates.json
{
  "nome": "domanda_con_esempio",
  "struttura": [
    ["UTENTE", "VERBO_AZIONE"],
    ["UTENTE", "ARTICOLO"],
    ["UTENTE", "CONCETTO_SCIENTIFICO"],
    ["UTENTE", "con un esempio"],
    ["BOT_RISPOSTA", "DYNAMIC_SCIENTIFIC_FACT"]
  ]
}
```

### 2. Architettura del Modello

**File**: `minimal_*.py`

Contribuzioni benvenute:
- Ottimizzazioni delle performance
- Supporto per nuovi device (CUDA, ROCm)
- Implementazione di nuove tecniche di training
- Miglioramento della generazione del testo

### 3. Documentazione

**File**: `README.md`, `SETUP.md`, `QUICKSTART.md`

Contribuzioni benvenute:
- Correzione errori di battitura
- Chiarimento di sezioni confuse
- Aggiunta di esempi pratici
- Traduzione in altre lingue

### 4. Nuove Funzionalità

Idee per nuove funzionalità:
- Interfaccia web con Gradio/Streamlit
- Script di valutazione quantitativa
- Supporto per training distribuito
- Export del modello in formato ONNX
- Visualizzazione interattiva dei meta-token

## Checklist per Pull Request

Prima di inviare una PR, assicurati che:

- [ ] Il codice segue le linee guida di stile
- [ ] Hai testato le tue modifiche
- [ ] La documentazione è aggiornata (se necessario)
- [ ] I messaggi di commit sono descrittivi
- [ ] Non hai incluso file generati (models, datasets) nel commit
- [ ] Le tue modifiche sono compatibili con Python 3.8+

## Processo di Review

1. Un maintainer esaminerà la tua PR
2. Potrebbero essere richieste modifiche
3. Una volta approvata, la PR verrà mergiata
4. Il tuo contributo sarà accreditato nel progetto!

## Linee Guida per Issue

### Template per Bug Report

```markdown
**Descrizione del Bug**
Descrizione chiara e concisa del problema.

**Come Riprodurre**
Passi per riprodurre il comportamento:
1. Vai a '...'
2. Clicca su '....'
3. Vedi l'errore

**Comportamento Atteso**
Cosa ti aspettavi che accadesse.

**Screenshot**
Se applicabile, aggiungi screenshot.

**Ambiente:**
 - OS: [es. Ubuntu 20.04]
 - Python Version: [es. 3.9.7]
 - Versione PyTorch: [es. 2.0.0]
```

### Template per Feature Request

```markdown
**Descrizione della Funzionalità**
Descrizione chiara e concisa della funzionalità proposta.

**Motivazione**
Perché questa funzionalità sarebbe utile? Quale problema risolve?

**Implementazione Proposta**
(Opzionale) Come pensi che potrebbe essere implementata?

**Alternative Considerate**
(Opzionale) Quali alternative hai considerato?
```

## Domande?

Se hai domande:
- Apri un issue con l'etichetta "question"
- Consulta la documentazione esistente
- Cerca tra gli issue chiusi per risposte simili

## Codice di Condotta

Sii rispettoso, costruttivo e professionale in tutte le interazioni. Questo progetto è uno spazio inclusivo per tutti.

## Licenza

Contribuendo al progetto, accetti che i tuoi contributi saranno licenziati sotto la stessa licenza del progetto.

---

Grazie per contribuire al progetto Meta-Token! 🚀
