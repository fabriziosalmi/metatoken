# Progetto Meta-Token: Da un'Idea Bizzarra a un Modello di Linguaggio Strutturato

## Sommario Esecutivo

Questo documento descrive lo sviluppo iterativo di un modello di linguaggio innovativo, partendo da un'idea sperimentale fino a un prototipo funzionante basato su un'architettura SOTA (State-of-the-Art).

L'obiettivo del progetto era superare i limiti di interpretabilità e controllabilità dei LLM standard, insegnando a un modello a generare non solo testo, ma anche **meta-token** che descrivono il proprio **percorso di ragionamento**.

Partendo da un training "from scratch" su un Mac M4, il progetto ha attraversato diverse fasi di diagnosi e miglioramento, affrontando sfide come l'overfitting, i limiti di capacità del modello e i conflitti di dipendenze. La soluzione finale ha visto l'adozione di un'architettura ibrida, combinando un potente modello pre-allenato (Mistral 7B) con layer custom per i meta-token, e l'utilizzo di un generatore di dati sintetici avanzato per creare un dataset di fine-tuning di altissima qualità (10k+ esempi).

Il risultato è un modello autonomo in grado di comprendere un prompt, pianificare una strategia di risposta (es. `BOT_RAGIONAMENTO` -> `BOT_RISPOSTA`), e generare un output testuale coerente, dimostrando la validità e il potenziale dell'approccio della "meta-tokenizzazione".

---

## 1. L'Idea Fondamentale: La Meta-Tokenizzazione

I modelli di linguaggio standard operano come "scatole nere". Danno risposte, ma il loro processo di pensiero interno è opaco. Il **Progetto Meta-Token** nasce per affrontare questo problema.

**L'ipotesi centrale:** Se forniamo a un modello, durante il training, non solo il testo ma anche una descrizione esplicita della *funzione* di ogni token, il modello imparerà a seguire e, infine, a generare autonomamente queste strutture di ragionamento.

Abbiamo arricchito ogni token dell'input con due metadati:
1.  **Ruolo Strutturale:** La funzione del token nel flusso conversazionale.
    *   *Esempi:* `UTENTE`, `BOT_RAGIONAMENTO`, `BOT_DOMANDA`, `BOT_RISPOSTA`.
2.  **Tipo Semantico:** La categoria grammaticale o concettuale del token.
    *   *Esempi:* `CONCETTO_SCIENTIFICO`, `VERBO_AZIONE`, `PUNCT`, `ARTICOLO`.

> **Esempio Pratico:**
> La frase `Spiegami la relatività` non è più solo una sequenza di parole, ma una sequenza di tuple ricche di informazione:
> `[(Spiegami, UTENTE, VERBO_AZIONE), (la, UTENTE, ARTICOLO), (relatività, UTENTE, CONCETTO_SCIENTIFICO)]`

Questo trasforma il fine-tuning da un semplice compito di "next word prediction" a un compito più complesso di "next (word + metatokens) prediction".

---

## 2. Il Percorso Iterativo: Cronistoria dei Test e delle Scoperte

Il progetto si è evoluto attraverso una serie di versioni, ognuna delle quali ha risolto un problema specifico e ha rivelato il successivo collo di bottiglia.

### Fase 1: Proof of Concept e i Limiti del "From Scratch" (Versioni 1-4)

*   **Obiettivo:** Validare l'idea base allenando un piccolo Transformer da zero su un Mac M4 (16GB RAM).
*   **Test:**
    *   **V1-V2:** Creazione di un modello a caratteri con 3-5 esempi hardcoded.
    *   **Successo Iniziale:** Il modello riusciva a "overfittare" e imparare a memoria i pochi esempi, dimostrando che l'architettura con embedding e teste di predizione multiple era valida.
    *   **Primo Ostacolo (`RuntimeError`):** Abbiamo subito incontrato problemi di gestione delle dimensioni delle sequenze, risolti introducendo il padding.
    *   **V3-V4:** Passaggio a un tokenizer a parole e a un dataset più grande (30-50 esempi).
*   **Diagnosi Finale della Fase 1:** **Overfitting Estremo.** Un modello, anche se piccolo (0.7M - 1.6M parametri), allenato da zero su poche decine di esempi, impara solo a memorizzare. La `validation loss` esplodeva, e la generazione era un'accozzaglia di frammenti del training set.
    > **Lezione Appresa:** Allenare un modello di linguaggio da zero richiede una quantità di dati proibitiva. La conoscenza linguistica di base non può essere imparata da poche centinaia di esempi.

### Fase 2: Stabilizzazione e il Potere del Transfer Learning (Versioni 5-9)

*   **Obiettivo:** Abbandonare l'approccio "from scratch" e sfruttare la conoscenza di un modello pre-allenato.
*   **Test:**
    *   **V5 (La Svolta):** L'architettura viene semplificata. I meta-token vengono usati **solo in input** per arricchire il contesto. Il modello deve predire solo la prossima parola. Questo ha stabilizzato immediatamente il training.
    *   **V6 (Raffinamento Architettonico):** Introduzione di tecniche moderne come **Pre-LayerNorm** e **SwiGLU** per migliorare la stabilità.
    *   **V7 (Il Primo Salto di Qualità):** Introduzione di un vero modello pre-allenato sull'italiano (`gpt2-small-ita`). I risultati migliorano drasticamente, ma l'overfitting è ancora un problema.
    *   **V8 (Fine-Tuning di Precisione):** Introduzione di **Learning Rate Scheduler con Warmup** e **Early Stopping**. Questo ci ha permesso di controllare l'overfitting e trovare il punto di training ottimale. La generazione è diventata grammaticalmente corretta, ma semanticamente debole e ripetitiva.
    *   **V9 (La Soluzione ai Dati):** Creazione di un **generatore di dataset ibrido avanzato**, capace di produrre 10k+ esempi con alta varietà strutturale e lessicale. Il training su questo dataset ha portato a una `validation loss` eccezionale (`~0.1`), dimostrando una padronanza quasi perfetta del task linguistico.
*   **Diagnosi Finale della Fase 2:** Abbiamo un modello che sa "parlare" fluentemente e comprende la struttura del nostro task in modo supervisionato. Tuttavia, non è ancora autonomo.

### Fase 3: L'Apprendimento dell'Autonomia e il Paradosso Finale (Versioni 10-13)

*   **Obiettivo:** Tornare all'idea originale. Insegnare al modello V9, già esperto di lingua, a predire autonomamente i meta-token.
*   **Test:**
    *   **V10 (Il Ritorno delle 3 Teste):** L'architettura viene modificata per reintrodurre le teste di predizione per `ruolo` e `semantica`. Il modello V9 viene caricato e si esegue un secondo, breve ciclo di fine-tuning.
    *   **Il Trionfo della Struttura:** L'analisi dell'output finale ha mostrato che il modello era diventato **autonomo**. Cambiava stato da `UTENTE` a `BOT_RAGIONAMENTO` e `BOT_RISPOSTA` da solo. **L'idea era stata validata.**
    *   **Il Paradosso della Semantica:** Il testo generato era un delirio incoerente.
*   **Diagnosi Finale:** Abbiamo affrontato e risolto due problemi consecutivi:
    1.  **Conflitto di Lingua (V11):** Un primo test con `gpt2-medium` (inglese) ha prodotto un output di "sub-parole" ibride, rivelando l'importanza di usare un modello pre-allenato sulla lingua target.
    2.  **Errore di Decodifica (V12):** L'output del modello italiano `gpt2-medium-ita` sembrava ancora un delirio di sillabe. La diagnosi ha rivelato che il problema non era il modello, ma un **errore nel loop di visualizzazione finale**, che decodificava i sub-token BPE uno per uno invece che in gruppo.
    3.  **Il Limite di Capacità (La Diagnosi Definitiva):** Anche con la decodifica corretta, il modello `gpt2-medium-ita` mostrava ancora difficoltà a mantenere la coerenza semantica nel ciclo di feedback autonomo. La sua "capacità" non era sufficiente a gestire i tre compiti simultaneamente.

---

## 3. L'Architettura e la Strategia Finali (Versione 13 con Mistral 7B)

La soluzione definitiva ha richiesto un upgrade del "motore" del modello, mantenendo intatta tutta la nostra architettura logica.

### Il Modello Base: Mistral 7B con QLoRA

*   **Mistral 7B:** Un modello da 7 miliardi di parametri, molto più potente di GPT-2, con una comprensione semantica e una capacità di ragionamento superiori.
*   **Quantizzazione a 4-bit:** Per rendere il modello gestibile su una GPU Colab gratuita, i suoi pesi vengono "compressi" a 4-bit, riducendo drasticamente l'uso di VRAM.
*   **LoRA (Low-Rank Adaptation):** Invece di fare il fine-tuning di tutti i 7 miliardi di parametri, "congeliamo" il modello base e alleniamo solo una piccolissima frazione di pesi aggiuntivi (<1%), chiamati "adattatori". Questo rende il fine-tuning estremamente efficiente in termini di calcolo e memoria.

### L'Architettura Custom `MetaModel`

La nostra classe custom `MetaModel` orchestra l'interazione tra il modello base e la nostra logica di meta-token:
1.  **Input Embedding Ibrido:** L'embedding di una parola non è solo quello del modello base, ma la **somma** dell'embedding della parola, dell'embedding del suo ruolo e dell'embedding della sua semantica. `Embedding_Finale = Embedding_Parola + Embedding_Ruolo + Embedding_Semantica`. Questa è l'iniezione della nostra informazione strutturale.
2.  **Corpo del Transformer:** Gli embedding ibridi vengono processati dai potentissimi layer del modello Mistral 7B.
3.  **Output a Tre Teste:** L'output finale del Transformer viene passato a tre teste di predizione separate:
    *   La testa originale di Mistral per predire la **prossima parola**.
    *   Una nostra testa custom (`nn.Linear`) per predire il **prossimo ruolo**.
    *   Una nostra testa custom (`nn.Linear`) per predire la **prossima semantica**.

### La Strategia di Training e Generazione

*   **Training a Singolo Stadio:** Con un modello così potente, non è più necessario il training a due fasi. Alleniamo contemporaneamente gli adattatori LoRA e i nostri layer custom per i meta-token.
*   **Loss Combinata:** La funzione di costo è una somma pesata delle loss delle tre teste di predizione, dando più importanza alla correttezza della parola.
*   **Generazione Autonoma:** La funzione di generazione è un ciclo in cui, a ogni passo, il modello predice la tripletta `(parola, ruolo, semantica)` e la usa come input per il passo successivo.

---

## 4. Risultati e Analisi Finale

*   **Performance di Training:** Nei run finali, il modello ha raggiunto una **validation loss eccezionalmente bassa (`~0.1`)**, indicando una padronanza quasi perfetta del task supervisionato.
*   **Output Generato:** Il modello finale, basato su Mistral 7B, è in grado di produrre un **output coerente, grammaticalmente corretto e semanticamente pertinente**, seguendo autonomamente la struttura conversazionale definita dai meta-token.

> **Esempio di Output Atteso (Illustrativo):**
> **Prompt:** `Spiegami il concetto di buco nero`
> 
> | Parola | Ruolo Preditto | Semantica Predetta |
> |---|---|---|
> | Certamente | BOT_RISPOSTA | AFFERMAZIONE |
> | . | BOT_RISPOSTA | PUNCT |
> | Analizzo | BOT_RAGIONAMENTO| ANALISI_RICHIESTA|
> | la | BOT_RAGIONAMENTO | ARTICOLO |
> | richiesta | BOT_RAGIONAMENTO | SOSTANTIVO_ASTRATTO|
> | . | BOT_RAGIONAMENTO | PUNCT |
> | Un | BOT_RISPOSTA | ARTICOLO |
> | buco | BOT_RISPOSTA | SOSTANTIVO_ASTRATTO|
> | nero | BOT_RISPOSTA | AGGETTIVO |
> | è | BOT_RISPOSTA | VERBO_ESSERE |
> | una | BOT_RISPOSTA | ARTICOLO |
> | regione | BOT_RISPOSTA | SOSTANTIVO_ASTRATTO|
> | ... | ... | ... |

---

## 5. Lezioni Apprese e Prossimi Passi

Questo progetto è stato un viaggio attraverso le sfide più comuni e avanzate dello sviluppo di LLM.

### Lezioni Chiave:

1.  **I Dati sono il Re:** Il salto di qualità più grande è avvenuto passando da 50 a 10k esempi. Un dataset di alta qualità, vario e abbondante è più importante di quasi ogni altro fattore.
2.  **Il Transfer Learning non è opzionale:** Partire da un modello pre-allenato è l'unico modo fattibile per insegnare compiti complessi senza budget e dati su scala industriale.
3.  **La Capacità del Modello Conta:** Un modello più grande non è solo "un po' meglio", ma può abilitare capacità emergenti (come la coerenza in un task multi-obiettivo) che un modello più piccolo semplicemente non possiede.
4.  **L'Iterazione è la Chiave:** Ogni "fallimento" è stato in realtà una diagnosi che ha indicato la strada per il miglioramento successivo.

### Prossimi Passi Possibili:

1.  **Costruire un'Interfaccia:** Usare Gradio o Streamlit per creare una demo interattiva del modello.
2.  **Espandere i Meta-Token:** Introdurre meta-token ancora più granulari (es. per il sentiment, per la fonte dell'informazione) per un controllo ancora più fine.
3.  **Fine-Tuning su Dati Specifici di un Dominio:** Adattare il modello a un campo specifico (es. medico, legale) creando un dataset ad hoc.
4.  **Sperimentare con il Controllo in Generazione:** Implementare una logica che permetta all'utente di "forzare" un certo ruolo (`BOT_RAGIONAMENTO`) durante la generazione per guidare attivamente la risposta del modello.

Questo progetto, nato da un'idea "bizzarra", ha dimostrato con successo che è possibile creare modelli di linguaggio più strutturati, interpretabili e controllabili, aprendo la strada a future e affascinanti sperimentazioni.