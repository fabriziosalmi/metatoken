import torch
import torch.nn as nn
from torch.nn import functional as F
import re

# --- 1. CONFIGURAZIONE E IPERPARAMETRI V2 ---
block_size = 32      
batch_size = 4       
max_iters = 5000     
eval_interval = 250  
learning_rate = 3e-4 
device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
eval_iters = 200     # Quanti batch usare per la stima delle metriche

# Parametri del modello potenziati
d_model = 128        
n_head = 8           
n_layer = 8          

# Parametri di generazione
top_p_val = 0.9      # Nucleus sampling: considera solo i token la cui probabilità cumulativa è > 0.9

# Pesi per la loss combinata (sperimenta con questi!)
loss_weights = {'word': 1.0, 'ruolo': 0.5, 'semantico': 0.5}
# ----------------------------------------------------

print(f"Sto usando il device: {device}")

# --- 2. TOKENIZER INNOVATIVO "SMART" ---
class SmartWordTokenizer:
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.special_tokens = ['<PAD>', '<UNK>']

    def _prepare_text(self, text):
        text = text.lower()
        # Isola la punteggiatura con spazi attorno
        text = re.sub(r'([.,!?\'"¿])', r' \1 ', text)
        # Rimuovi spazi multipli
        text = re.sub(r'\s+', ' ', text).strip()
        return text.split(' ')

    def fit(self, text_corpus):
        all_words = set(self.special_tokens)
        for text in text_corpus:
            words = self._prepare_text(text)
            for word in words:
                all_words.add(word)
        
        sorted_vocab = sorted(list(all_words))
        self.word_to_id = {word: i for i, word in enumerate(sorted_vocab)}
        self.id_to_word = {i: word for i, word in enumerate(sorted_vocab)}
        self.vocab_size = len(sorted_vocab)
        self.pad_id = self.word_to_id['<PAD>']
        self.unk_id = self.word_to_id['<UNK>']

    def encode(self, text):
        words = self._prepare_text(text)
        return [self.word_to_id.get(word, self.unk_id) for word in words]

    def decode(self, ids):
        words = [self.id_to_word.get(i, '<UNK>') for i in ids]
        # Unisci il testo in modo un po' più intelligente
        text = ' '.join(words)
        text = text.replace(' <PAD>', '').strip()
        text = re.sub(r'\s+([.,!?\'"¿])', r'\1', text) # Rimuovi spazio prima della punteggiatura
        return text

# --- 3. DATASET V2 CON META-TOKEN SEMANTICO ---
# Formato: (parola, ruolo_strutturale, tipo_semantico)
raw_data = [
    [('Ciao', 'UTENTE', 'SALUTO'), ('.', 'UTENTE', 'PUNCT'),
     ('Ciao', 'BOT_RISPOSTA', 'SALUTO'), ('!', 'BOT_RISPOSTA', 'PUNCT')],
    
    [('Che', 'UTENTE', 'DOMANDA'), ('ore', 'UTENTE', 'CONCETTO_TEMPO'), ('sono', 'UTENTE', 'DOMANDA'), ('?', 'UTENTE', 'PUNCT'),
     ('È', 'BOT_RAGIONAMENTO', 'INFO'), ('ora', 'BOT_RAGIONAMENTO', 'CONCETTO_TEMPO'), ('di', 'BOT_RAGIONAMENTO', 'INFO'), ('allenare', 'BOT_RAGIONAMENTO', 'AZIONE'), ('un', 'BOT_RAGIONAMENTO', 'INFO'), ('modello', 'BOT_RAGIONAMENTO', 'OGGETTO_TECNICO'), ('.', 'BOT_RISPOSTA', 'PUNCT')],
    
    [('Perché', 'UTENTE', 'DOMANDA'), ('il', 'UTENTE', 'INFO'), ('cielo', 'UTENTE', 'OGGETTO_NATURALE'), ('è', 'UTENTE', 'DOMANDA'), ('blu', 'UTENTE', 'ATTRIBUTO'), ('?', 'UTENTE', 'PUNCT'),
     ('Questo', 'BOT_RAGIONAMENTO', 'INFO'), ('riguarda', 'BOT_RAGIONAMENTO', 'RELAZIONE'), ('lo', 'BOT_RAGIONAMENTO', 'INFO'), ('scattering', 'BOT_RAGIONAMENTO', 'CONCETTO_FISICO'), ('.', 'BOT_RAGIONAMENTO', 'PUNCT'),
     ('La', 'BOT_RISPOSTA', 'INFO'), ('luce', 'BOT_RISPOSTA', 'OGGETTO_FISICO'), ('viene', 'BOT_RISPOSTA', 'RELAZIONE'), ('diffusa', 'BOT_RISPOSTA', 'AZIONE'), ('.', 'BOT_RISPOSTA', 'PUNCT')],

    [('Spiegami', 'UTENTE', 'AZIONE'), ('i', 'UTENTE', 'INFO'), ('transformer', 'UTENTE', 'OGGETTO_TECNICO'), ('.', 'UTENTE', 'PUNCT'),
     ('Certo', 'BOT_RISPOSTA', 'AFFERMAZIONE'), (',', 'BOT_RISPOSTA', 'PUNCT'), ('ma', 'BOT_RISPOSTA', 'CONGIUNZIONE'), ('potresti', 'BOT_CHIARIMENTO', 'DOMANDA'), ('essere', 'BOT_CHIARIMENTO', 'DOMANDA'), ('più', 'BOT_CHIARIMENTO', 'ATTRIBUTO'), ('specifico', 'BOT_CHIARIMENTO', 'ATTRIBUTO'), ('?', 'BOT_CHIARIMENTO', 'PUNCT')]
]

# Dividiamo i dati
train_data = raw_data[:3]
val_data = raw_data[3:]

# Creiamo i vocabolari
corpus = [' '.join(word for word, _, _ in seq) for seq in raw_data]
tokenizer = SmartWordTokenizer()
tokenizer.fit(corpus)

all_ruoli = set(['<PAD>'] + [r for seq in raw_data for _, r, _ in seq])
all_semantici = set(['<PAD>'] + [s for seq in raw_data for _, _, s in seq])

ruolo_to_id = {r: i for i, r in enumerate(sorted(list(all_ruoli)))}
id_to_ruolo = {i: r for i, r in enumerate(sorted(list(all_ruoli)))}
semantico_to_id = {s: i for i, s in enumerate(sorted(list(all_semantici)))}
id_to_semantico = {i: s for i, s in enumerate(sorted(list(all_semantici)))}

# Funzione per ottenere un batch
def get_batch(split):
    data = train_data if split == 'train' else val_data
    ix = torch.randint(len(data), (batch_size,))
    
    seqs = [data[i] for i in ix]
    
    x_word, x_ruolo, x_sem, y_word, y_ruolo, y_sem = [], [], [], [], [], []

    for seq in seqs:
        # Codifichiamo l'intera sequenza
        full_word_ids = tokenizer.encode(' '.join(w for w,r,s in seq))
        full_ruolo_ids = [ruolo_to_id[r] for w,r,s in seq]
        full_sem_ids = [semantico_to_id[s] for w,r,s in seq]

        # Prendiamo input e target
        input_w = full_word_ids[:-1]
        input_r = full_ruolo_ids[:-1]
        input_s = full_sem_ids[:-1]
        target_w = full_word_ids[1:]
        target_r = full_ruolo_ids[1:]
        target_s = full_sem_ids[1:]

        # Padding
        def pad(l, size, pad_id):
            return l + [pad_id] * (size - len(l))

        x_word.append(pad(input_w, block_size, tokenizer.pad_id))
        x_ruolo.append(pad(input_r, block_size, ruolo_to_id['<PAD>']))
        x_sem.append(pad(input_s, block_size, semantico_to_id['<PAD>']))
        y_word.append(pad(target_w, block_size, tokenizer.pad_id))
        y_ruolo.append(pad(target_r, block_size, ruolo_to_id['<PAD>']))
        y_sem.append(pad(target_s, block_size, semantico_to_id['<PAD>']))
        
    def to_tensor(l): return torch.tensor(l, dtype=torch.long).to(device)
    
    return to_tensor(x_word), to_tensor(x_ruolo), to_tensor(x_sem), to_tensor(y_word), to_tensor(y_ruolo), to_tensor(y_sem)


# --- 4. CALCOLO METRICHE ---
@torch.no_grad()
def estimate_metrics():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        acc_word = torch.zeros(eval_iters)
        acc_ruolo = torch.zeros(eval_iters)
        acc_sem = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            Xw, Xr, Xs, Yw, Yr, Ys = get_batch(split)
            logits_w, logits_r, logits_s, loss = model(Xw, Xr, Xs, Yw, Yr, Ys)
            losses[k] = loss.item()

            # Calcolo accuracy ignorando il padding
            mask = (Yw != tokenizer.pad_id)
            total_valid_tokens = mask.sum().item()

            if total_valid_tokens > 0:
                pred_w = logits_w.argmax(dim=-1)
                pred_r = logits_r.argmax(dim=-1)
                pred_s = logits_s.argmax(dim=-1)
                
                acc_word[k] = ((pred_w == Yw) & mask).sum().item() / total_valid_tokens
                acc_ruolo[k] = ((pred_r == Yr) & mask).sum().item() / total_valid_tokens
                acc_sem[k] = ((pred_s == Ys) & mask).sum().item() / total_valid_tokens

        out[split + '_loss'] = losses.mean()
        out[split + '_perplexity'] = torch.exp(losses.mean())
        out[split + '_acc_word'] = acc_word.mean()
        out[split + '_acc_ruolo'] = acc_ruolo.mean()
        out[split + '_acc_semantico'] = acc_sem.mean()

    model.train()
    return out


# --- 5. ARCHITETTURA DEL MODELLO V2 ---
class Block(nn.Module):
    # (Identico a prima, lo includo per completezza)
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=0.1, batch_first=True)
        self.ffwd = nn.Sequential(nn.Linear(n_embd, 4 * n_embd), nn.ReLU(), nn.Linear(4 * n_embd, n_embd), nn.Dropout(0.1))
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        y, _ = self.sa(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = x + y
        x = self.ln1(x)
        x = x + self.ffwd(x)
        x = self.ln2(x)
        return x

class MetaWordTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding_table = nn.Embedding(tokenizer.vocab_size, d_model)
        self.ruolo_embedding_table = nn.Embedding(len(ruolo_to_id), d_model)
        self.semantico_embedding_table = nn.Embedding(len(semantico_to_id), d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head_word = nn.Linear(d_model, tokenizer.vocab_size)
        self.lm_head_ruolo = nn.Linear(d_model, len(ruolo_to_id))
        self.lm_head_semantico = nn.Linear(d_model, len(semantico_to_id))

    def forward(self, word_idx, ruolo_idx, semantico_idx, targets_w=None, targets_r=None, targets_s=None):
        B, T = word_idx.shape
        
        word_emb = self.word_embedding_table(word_idx)
        ruolo_emb = self.ruolo_embedding_table(ruolo_idx)
        semantico_emb = self.semantico_embedding_table(semantico_idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = word_emb + ruolo_emb + semantico_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)

        logits_w = self.lm_head_word(x)
        logits_r = self.lm_head_ruolo(x)
        logits_s = self.lm_head_semantico(x)

        loss = None
        if targets_w is not None:
            loss_w = F.cross_entropy(logits_w.view(-1, logits_w.size(-1)), targets_w.view(-1), ignore_index=tokenizer.pad_id)
            loss_r = F.cross_entropy(logits_r.view(-1, logits_r.size(-1)), targets_r.view(-1), ignore_index=ruolo_to_id['<PAD>'])
            loss_s = F.cross_entropy(logits_s.view(-1, logits_s.size(-1)), targets_s.view(-1), ignore_index=semantico_to_id['<PAD>'])
            
            loss = loss_weights['word'] * loss_w + loss_weights['ruolo'] * loss_r + loss_weights['semantico'] * loss_s

        return logits_w, logits_r, logits_s, loss

    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p):
        for _ in range(max_new_tokens):
            w_idx_cond = word_idx[:, -block_size:]
            r_idx_cond = ruolo_idx[:, -block_size:]
            s_idx_cond = semantico_idx[:, -block_size:]
            
            logits_w, logits_r, logits_s, _ = self(w_idx_cond, r_idx_cond, s_idx_cond)
            
            def nucleus_sampling(logits):
                logits = logits[:, -1, :] # (B, C)
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / probs.sum(dim=-1, keepdim=True)
                
                return torch.multinomial(probs, num_samples=1)

            next_w_id = nucleus_sampling(logits_w)
            next_r_id = nucleus_sampling(logits_r)
            next_s_id = nucleus_sampling(logits_s)
            
            word_idx = torch.cat((word_idx, next_w_id), dim=1)
            ruolo_idx = torch.cat((ruolo_idx, next_r_id), dim=1)
            semantico_idx = torch.cat((semantico_idx, next_s_id), dim=1)
            
        return word_idx, ruolo_idx, semantico_idx

# --- 6. TRAINING E GENERAZIONE ---

model = MetaWordTransformer()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

for iter in range(max_iters):
    if iter % eval_interval == 0 or iter == max_iters - 1:
        metrics = estimate_metrics()
        print("-" * 50)
        print(f"step {iter}:")
        print(f"  train loss: {metrics['train_loss']:.4f}, train perplexity: {metrics['train_perplexity']:.2f}")
        print(f"  val loss:   {metrics['val_loss']:.4f}, val perplexity:   {metrics['val_perplexity']:.2f}")
        print(f"  val acc word: {metrics['val_acc_word']:.2%}, ruolo: {metrics['val_acc_ruolo']:.2%}, semantico: {metrics['val_acc_semantico']:.2%}")

    xb_w, xb_r, xb_s, yb_w, yb_r, yb_s = get_batch('train')
    _, _, _, loss = m(xb_w, xb_r, xb_s, yb_w, yb_r, yb_s)
    
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# Generazione finale
print("\n" + "="*20 + " GENERAZIONE FINALE " + "="*20)
start_text = "Spiegami i transformer"
start_ruolo = 'UTENTE'
start_semantico = 'AZIONE' # L'ultimo token della frase è "transformer", ma l'intento è un'azione

context_w = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)
# Per il contesto dei metatoken, creiamo una sequenza fittizia della stessa lunghezza
context_r = torch.full_like(context_w, ruolo_to_id[start_ruolo])
context_s = torch.full_like(context_w, semantico_to_id[start_semantico])

generated_w, generated_r, generated_s = m.generate(context_w, context_r, context_s, max_new_tokens=20, top_p=top_p_val)

# Decodifica e stampa
gen_words = tokenizer.decode(generated_w[0].tolist())
gen_ruoli = [id_to_ruolo[i.item()] for i in generated_r[0]]
gen_sem = [id_to_semantico[i.item()] for i in generated_s[0]]

print(f"Prompt: '{start_text}'\n")
print("Output generato:\n")
print(f"{'Parola':<20} | {'Ruolo Preditto':<20} | {'Semantica Predetta'}")
print("-" * 70)

# Allineiamo la stampa parola per parola
decoded_words = tokenizer._prepare_text(gen_words)
# Facciamo in modo che la lunghezza delle liste di metadati corrisponda
max_len = len(decoded_words)
for word, ruolo, sem in zip(decoded_words, gen_ruoli[:max_len], gen_sem[:max_len]):
    print(f"{word:<20} | {ruolo:<20} | {sem}")
