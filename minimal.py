import torch
import torch.nn as nn
from torch.nn import functional as F

# --- 1. IPERPARAMETRI E CONFIGURAZIONE ---
# Li mettiamo tutti qui per poterli modificare facilmente

block_size = 32      # Massima lunghezza del contesto per le predizioni
batch_size = 4       # Quanti esempi processare in parallelo (lo simuliamo)
max_iters = 5000     # Quante volte aggiornare i pesi del modello
eval_interval = 500  # Ogni quanto stampare la loss
learning_rate = 3e-4 # Tasso di apprendimento (un valore tipico per modelli piccoli)
device = 'mps' if torch.backends.mps.is_available() else 'cpu' # Usa la GPU del Mac M4!
d_model = 64         # La dimensione dei vettori di embedding (molto piccolo)
n_head = 4           # Numero di "teste" di attenzione
n_layer = 4          # Numero di blocchi Transformer
# ----------------------------------------------------

print(f"Sto usando il device: {device}")

# --- 2. IL DATASET CUSTOM (IL CUORE INNOVATIVO) ---
# Dati hardcoded per la massima semplicità. Ogni esempio è una conversazione.
# Ogni "token" è una tupla: (carattere, ruolo_strutturale)

raw_data = [
    # Esempio 1: Saluto base
    [('C', 'UTENTE'), ('i', 'UTENTE'), ('a', 'UTENTE'), ('o', 'UTENTE'), ('.', 'UTENTE'),
     ('C', 'BOT_RISPOSTA'), ('i', 'BOT_RISPOSTA'), ('a', 'BOT_RISPOSTA'), ('o', 'BOT_RISPOSTA'), ('!', 'BOT_RISPOSTA')],
    
    # Esempio 2: Domanda e risposta breve
    [('C', 'UTENTE'), ('h', 'UTENTE'), ('e', 'UTENTE'), (' ', 'UTENTE'), ('o', 'UTENTE'), ('r', 'UTENTE'), ('a', 'UTENTE'), (' ', 'UTENTE'), ('è', 'UTENTE'), ('?', 'UTENTE'),
     ('È', 'BOT_RAGIONAMENTO'), (' ', 'BOT_RAGIONAMENTO'), ('o', 'BOT_RAGIONAMENTO'), ('r', 'BOT_RAGIONAMENTO'), ('a', 'BOT_RAGIONAMENTO'), (' ', 'BOT_RAGIONAMENTO'), ('d', 'BOT_RAGIONAMENTO'), ('i', 'BOT_RAGIONAMENTO'), (' ', 'BOT_RAGIONAMENTO'), ('c', 'BOT_RAGIONAMENTO'), ('o', 'BOT_RAGIONAMENTO'), ('d', 'BOT_RAGIONAMENTO'), ('a', 'BOT_RAGIONAMENTO'), ('r', 'BOT_RAGIONAMENTO'), ('e', 'BOT_RAGIONAMENTO'), ('.', 'BOT_RISPOSTA')],
    
    # Esempio 3: Domanda più complessa con ragionamento esplicito
    [('P', 'UTENTE'), ('e', 'UTENTE'), ('r', 'UTENTE'), ('c', 'UTENTE'), ('h', 'UTENTE'), ('é', 'UTENTE'), (' ', 'UTENTE'), ('i', 'UTENTE'), ('l', 'UTENTE'), (' ', 'UTENTE'), ('c', 'UTENTE'), ('i', 'UTENTE'), ('e', 'UTENTE'), ('l', 'UTENTE'), ('o', 'UTENTE'), (' ', 'UTENTE'), ('è', 'UTENTE'), (' ', 'UTENTE'), ('b', 'UTENTE'), ('l', 'UTENTE'), ('u', 'UTENTE'), ('?', 'UTENTE'),
     ('D', 'BOT_RAGIONAMENTO'), ('o', 'BOT_RAGIONAMENTO'), ('m', 'BOT_RAGIONAMENTO'), ('a', 'BOT_RAGIONAMENTO'), ('n', 'BOT_RAGIONAMENTO'), ('d', 'BOT_RAGIONAMENTO'), ('a', 'BOT_RAGIONAMENTO'), (' ', 'BOT_RAGIONAMENTO'), ('s', 'BOT_RAGIONAMENTO'), ('u', 'BOT_RAGIONAMENTO'), ('l', 'BOT_RAGIONAMENTO'), ('l', 'BOT_RAGIONAMENTO'), ('o', 'BOT_RAGIONAMENTO'), (' ', 'BOT_RAGIONAMENTO'), ('s', 'BOT_RAGIONAMENTO'), ('c', 'BOT_RAGIONAMENTO'), ('a', 'BOT_RAGIONAMENTO'), ('t', 'BOT_RAGIONAMENTO'), ('t', 'BOT_RAGIONAMENTO'), ('e', 'BOT_RAGIONAMENTO'), ('r', 'BOT_RAGIONAMENTO'), ('i', 'BOT_RAGIONAMENTO'), ('n', 'BOT_RAGIONamento'), ('g', 'BOT_RAGIONamento'), ('.', 'BOT_RAGIONAMENTO'),
     (' ', 'BOT_RISPOSTA'), ('L', 'BOT_RISPOSTA'), ('u', 'BOT_RISPOSTA'), ('c', 'BOT_RISPOSTA'), ('e', 'BOT_RISPOSTA'), (' ', 'BOT_RISPOSTA'), ('d', 'BOT_RISPOSTA'), ('i', 'BOT_RISPOSTA'), ('f', 'BOT_RISPOSTA'), ('f', 'BOT_RISPOSTA'), ('u', 'BOT_RISPOSTA'), ('s', 'BOT_RISPOSTA'), ('a', 'BOT_RISPOSTA'), ('.', 'BOT_RISPOSTA')]
]

# Uniamo tutti i dati per creare i vocabolari
full_data = [item for sublist in raw_data for item in sublist]

# Creazione vocabolari per caratteri e meta-token di ruolo
chars = sorted(list(set(c for c, r in full_data)))
ruoli = sorted(list(set(r for c, r in full_data)))
# Aggiungiamo un token di PADDING per uniformare le lunghezze
PAD_TOKEN = '<PAD>'
chars.append(PAD_TOKEN)
ruoli.append(PAD_TOKEN)

vocab_size = len(chars)
num_ruoli = len(ruoli)

# Mapping da/a interi
char_to_id = {ch: i for i, ch in enumerate(chars)}
id_to_char = {i: ch for i, ch in enumerate(chars)}
ruolo_to_id = {r: i for i, r in enumerate(ruoli)}
id_to_ruolo = {i: r for i, r in enumerate(ruoli)}

# Funzione per ottenere un batch di dati
def get_batch():
    # Continua a cercare finché non trova un esempio abbastanza lungo
    while True:
        data = raw_data[torch.randint(len(raw_data), (1,)).item()]
        if len(data) > block_size:
            break
            
    # Scegliamo un punto di inizio casuale nell'esempio
    # Ora siamo sicuri che len(data) - block_size non sarà negativo
    ix = torch.randint(len(data) - block_size, (batch_size,))
    
    # Creiamo i batch di input (x) e target (y)
    x_char, x_ruolo, y_char, y_ruolo = [], [], [], []
    for i in ix:
        chunk = data[i:i+block_size+1]
        x_char.append([char_to_id[c] for c, r in chunk[:-1]])
        x_ruolo.append([ruolo_to_id[r] for c, r in chunk[:-1]])
        y_char.append([char_to_id[c] for c, r in chunk[1:]])
        y_ruolo.append([ruolo_to_id[r] for c, r in chunk[1:]])

    x_char = torch.tensor(x_char, dtype=torch.long).to(device)
    x_ruolo = torch.tensor(x_ruolo, dtype=torch.long).to(device)
    y_char = torch.tensor(y_char, dtype=torch.long).to(device)
    y_ruolo = torch.tensor(y_ruolo, dtype=torch.long).to(device)
    
    return x_char, x_ruolo, y_char, y_ruolo

# --- 3. L'ARCHITETTURA DEL MODELLO CUSTOM ---

# Blocco base del Transformer (per non reinventare la ruota)
class Block(nn.Module):
    def __init__(self, n_embd, n_head):
        super().__init__()
        head_size = n_embd // n_head
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=0.1, batch_first=True)
        self.ffwd = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(0.1),
        )
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        # Usiamo una maschera causale per l'attenzione
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        y, _ = self.sa(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x

class MetaCharTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        
        # --- PARTE CUSTOM 1: EMBEDDING MULTIPLO ---
        # Un embedding per i caratteri, uno per i ruoli, uno per la posizione
        self.char_embedding_table = nn.Embedding(vocab_size, d_model)
        self.ruolo_embedding_table = nn.Embedding(num_ruoli, d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        
        # --- Parte Standard: Blocchi Transformer ---
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        # --- PARTE CUSTOM 2: TESTA DI PREDIZIONE MULTIPLA ---
        # Una testa per predire il prossimo carattere
        self.lm_head_char = nn.Linear(d_model, vocab_size) 
        # Una testa per predire il prossimo ruolo
        self.lm_head_ruolo = nn.Linear(d_model, num_ruoli)

    def forward(self, char_idx, ruolo_idx, targets_char=None, targets_ruolo=None):
        B, T = char_idx.shape
        
        # --- Logica Embedding Custom ---
        char_emb = self.char_embedding_table(char_idx)    # (B, T, d_model)
        ruolo_emb = self.ruolo_embedding_table(ruolo_idx)  # (B, T, d_model)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device)) # (T, d_model)
        
        # Sommiamo i tre embedding per combinarli in un unico vettore
        x = char_emb + ruolo_emb + pos_emb # (B, T, d_model)
        
        # --- Logica Transformer Standard ---
        x = self.blocks(x)
        x = self.ln_f(x)

        # --- Logica Output Head Custom ---
        logits_char = self.lm_head_char(x)   # (B, T, vocab_size)
        logits_ruolo = self.lm_head_ruolo(x) # (B, T, num_ruoli)

        loss = None
        if targets_char is not None and targets_ruolo is not None:
            # --- PARTE CUSTOM 3: LOSS COMBINATA ---
            B, T, C_char = logits_char.shape
            _, _, C_ruolo = logits_ruolo.shape
            
            logits_char = logits_char.view(B*T, C_char)
            targets_char = targets_char.view(B*T)
            
            logits_ruolo = logits_ruolo.view(B*T, C_ruolo)
            targets_ruolo = targets_ruolo.view(B*T)

            # Calcoliamo due loss separate, ignorando il padding
            pad_id = char_to_id[PAD_TOKEN] # Stesso ID per ruolo e char
            loss_char = F.cross_entropy(logits_char, targets_char, ignore_index=pad_id)
            loss_ruolo = F.cross_entropy(logits_ruolo, targets_ruolo, ignore_index=pad_id)
            
            # Le sommiamo (si potrebbe dare pesi diversi, es. 0.8 * loss_char + 0.2 * loss_ruolo)
            loss = loss_char + loss_ruolo

        return logits_char, logits_ruolo, loss

    def generate(self, char_idx, ruolo_idx, max_new_tokens):
        # char_idx e ruolo_idx sono (B, T) array di indici nel contesto corrente
        for _ in range(max_new_tokens):
            # Assicuriamoci che il contesto non superi block_size
            char_idx_cond = char_idx[:, -block_size:]
            ruolo_idx_cond = ruolo_idx[:, -block_size:]
            
            # Ottieni le predizioni
            logits_char, logits_ruolo, _ = self(char_idx_cond, ruolo_idx_cond)
            
            # Concentrati solo sull'ultimo step temporale
            logits_char = logits_char[:, -1, :] # (B, vocab_size)
            logits_ruolo = logits_ruolo[:, -1, :] # (B, num_ruoli)
            
            # Applica softmax per ottenere le probabilità
            probs_char = F.softmax(logits_char, dim=-1)
            probs_ruolo = F.softmax(logits_ruolo, dim=-1)
            
            # Campiona dalla distribuzione
            next_char_id = torch.multinomial(probs_char, num_samples=1)
            next_ruolo_id = torch.multinomial(probs_ruolo, num_samples=1)
            
            # Aggiungi i token campionati alla sequenza
            char_idx = torch.cat((char_idx, next_char_id), dim=1)
            ruolo_idx = torch.cat((ruolo_idx, next_ruolo_id), dim=1)
            
        return char_idx, ruolo_idx

# --- 4. TRAINING E GENERAZIONE ---

# Istanziamo il modello e l'ottimizzatore
model = MetaCharTransformer()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

# Training loop
for iter in range(max_iters):
    # Ogni tanto, valutiamo la loss
    if iter % eval_interval == 0:
        # Mettiamo il modello in modalità valutazione
        m.eval()
        losses = torch.zeros(200)
        for k in range(200):
            xc, xr, yc, yr = get_batch()
            _, _, loss = m(xc, xr, yc, yr)
            losses[k] = loss.item()
        print(f"step {iter}: loss {losses.mean():.4f}")
        # Rimettiamo il modello in modalità training
        m.train()

    # Prendiamo un batch di dati
    xb_char, xb_ruolo, yb_char, yb_ruolo = get_batch()

    # Calcoliamo la loss
    logits_char, logits_ruolo, loss = m(xb_char, xb_ruolo, yb_char, yb_ruolo)
    
    # Backpropagation
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generazione di testo dopo il training
print("\n--- GENERAZIONE DOPO IL TRAINING ---")
# Creiamo un contesto di partenza: un batch di 1, con un solo token
start_char = 'P'
start_ruolo = 'UTENTE'
context_char = torch.tensor([[char_to_id[start_char]]], dtype=torch.long, device=device)
context_ruolo = torch.tensor([[ruolo_to_id[start_ruolo]]], dtype=torch.long, device=device)

# Chiamiamo la funzione di generazione
generated_char_ids, generated_ruolo_ids = m.generate(context_char, context_ruolo, max_new_tokens=100)

# Decodifichiamo gli ID in testo leggibile
generated_chars = [id_to_char[i] for i in generated_char_ids[0].tolist()]
generated_ruoli = [id_to_ruolo[i] for i in generated_ruolo_ids[0].tolist()]

# Stampiamo il risultato in un formato chiaro
print("Output generato:\n")
for char, ruolo in zip(generated_chars, generated_ruoli):
    print(f"Carattere: '{char}' \t| Ruolo Preditto: [{ruolo}]")
