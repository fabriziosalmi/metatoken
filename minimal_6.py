import torch
import torch.nn as nn
from torch.nn import functional as F
import re
import json
import math # Necessario per SwiGLU

# --- 1. CONFIGURAZIONE E IPERPARAMETRI V6 ---
block_size = 48      # Increased to fit longest sequences (max 46 tokens)
batch_size = 4       
max_iters = 5000     
eval_interval = 250  
learning_rate = 5e-4 
device = 'mps' if torch.backends.mps.is_available() else 'cpu' 
eval_iters = 200

# Architettura stabile
d_model = 128        
n_head = 8
n_layer = 8
dropout = 0.2

top_p_val = 0.9      
DATASET_FILE = 'dataset.json'
# ----------------------------------------------------

print(f"Sto usando il device: {device}")

# --- 2. TOKENIZER (Invariato) ---
class SmartWordTokenizer:
    # ... (copia la classe SmartWordTokenizer dalla v5, è identica) ...
    def __init__(self):
        self.word_to_id = {}
        self.id_to_word = {}
        self.vocab_size = 0
        self.special_tokens = ['<PAD>', '<UNK>']

    def _prepare_text(self, text):
        text = text.lower()
        text = re.sub(r'([.,!?\'"¿])', r' \1 ', text)
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
        text = ' '.join(words)
        text = text.replace(' <PAD>', '').strip()
        text = re.sub(r'\s+([.,!?\'"¿])', r'\1', text)
        return text

# --- 3. GESTIONE DATI (Invariato) ---
# ... (copia tutta la sezione 3 dalla v5, è identica) ...
def load_data_from_json(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"ERRORE: Il file '{filepath}' non è stato trovato.")
        exit()

    processed_data = []
    for example in data['examples']:
        seq = [(token['word'], token['ruolo'], token['semantico']) for token in example['sequence']]
        processed_data.append(seq)
    return processed_data

print(f"Caricamento dati da '{DATASET_FILE}'...")
raw_data = load_data_from_json(DATASET_FILE)
print(f"Caricati {len(raw_data)} esempi.")

train_data = raw_data[:-1]
val_data = raw_data[-1:]

corpus = [' '.join(word for word, _, _ in seq) for seq in raw_data]
tokenizer = SmartWordTokenizer()
tokenizer.fit(corpus)

all_ruoli = set(['<PAD>'] + [r for seq in raw_data for _, r, _ in seq])
all_semantici = set(['<PAD>'] + [s for seq in raw_data for _, _, s in seq])

ruolo_to_id = {r: i for i, r in enumerate(sorted(list(all_ruoli)))}
id_to_ruolo = {i: r for i, r in enumerate(sorted(list(all_ruoli)))}
semantico_to_id = {s: i for i, s in enumerate(sorted(list(all_semantici)))}
id_to_semantico = {i: s for i, s in enumerate(sorted(list(all_semantici)))}

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if not data: return None
    
    ix = torch.randint(len(data), (batch_size,))
    seqs = [data[i] for i in ix]
    
    x_word, x_ruolo, x_sem, y_word, y_ruolo, y_sem = [], [], [], [], [], []

    for seq in seqs:
        full_word_ids = tokenizer.encode(' '.join(w for w,r,s in seq))
        full_ruolo_ids = [ruolo_to_id[r] for w,r,s in seq]
        full_sem_ids = [semantico_to_id[s] for w,r,s in seq]

        input_w, input_r, input_s = full_word_ids[:-1], full_ruolo_ids[:-1], full_sem_ids[:-1]
        target_w, target_r, target_s = full_word_ids[1:], full_ruolo_ids[1:], full_sem_ids[1:]

        def pad(l, size, pad_id): 
            if len(l) > size:
                return l[:size]  # Truncate if too long
            else:
                return l + [pad_id] * (size - len(l))  # Pad if too short

        x_word.append(pad(input_w, block_size, tokenizer.pad_id))
        x_ruolo.append(pad(input_r, block_size, ruolo_to_id['<PAD>']))
        x_sem.append(pad(input_s, block_size, semantico_to_id['<PAD>']))
        y_word.append(pad(target_w, block_size, tokenizer.pad_id))
        y_ruolo.append(pad(target_r, block_size, ruolo_to_id['<PAD>']))
        y_sem.append(pad(target_s, block_size, semantico_to_id['<PAD>']))
        
    def to_tensor(l): return torch.tensor(l, dtype=torch.long).to(device)
    
    return to_tensor(x_word), to_tensor(x_ruolo), to_tensor(x_sem), to_tensor(y_word), to_tensor(y_ruolo), to_tensor(y_sem)


# --- 4. CALCOLO METRICHE (Invariato dalla v5) ---
@torch.no_grad()
def estimate_metrics():
    # ... (copia la funzione estimate_metrics dalla v5, è identica) ...
    out = {}
    model.eval()
    for split in ['train', 'val']:
        batch_data = get_batch(split)
        if batch_data is None: continue

        losses = torch.zeros(eval_iters)
        acc_word = torch.zeros(eval_iters)
        
        for k in range(eval_iters):
            Xw, Xr, Xs, Yw, _, _ = get_batch(split)
            logits_w, loss = model(Xw, Xr, Xs, Yw)
            losses[k] = loss.item()

            mask = (Yw != tokenizer.pad_id)
            total_valid_tokens = mask.sum().item()

            if total_valid_tokens > 0:
                pred_w = logits_w.argmax(dim=-1)
                acc_word[k] = ((pred_w == Yw) & mask).sum().item() / total_valid_tokens

        out[split + '_loss'] = losses.mean()
        out[split + '_perplexity'] = torch.exp(losses.mean())
        out[split + '_acc_word'] = acc_word.mean()

    model.train()
    return out


# --- 5. ARCHITETTURA DEL MODELLO V6 (Pre-LN e SwiGLU) ---

class SwiGLU(nn.Module):
    """Implementazione di Swish Gated Linear Unit"""
    def __init__(self, in_dim, hidden_dim=None, out_dim=None):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        
        # L'hidden dim per SwiGLU è solitamente 2/3 di quello di un FFN standard
        # e deve essere multiplo di un valore (es. 256) per efficienza, ma qui non importa
        hidden_dim = int(2 * hidden_dim / 3)
        
        self.w1 = nn.Linear(in_dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, out_dim, bias=False)
        self.w3 = nn.Linear(in_dim, hidden_dim, bias=False)

    def forward(self, x):
        return self.w2(F.silu(self.w1(x)) * self.w3(x))

class Block(nn.Module):
    """Blocco Transformer con Pre-LayerNorm e SwiGLU"""
    def __init__(self, n_embd, n_head):
        super().__init__()
        self.ln1 = nn.LayerNorm(n_embd)
        self.sa = nn.MultiheadAttention(n_embd, n_head, dropout=0.1, batch_first=True)
        self.ln2 = nn.LayerNorm(n_embd)
        self.ffwd = SwiGLU(n_embd, n_embd * 4) # Usiamo 4*n_embd come hidden dim per l'FFN

    def forward(self, x):
        # Prima la normalizzazione, poi il sottolayer, poi la connessione residua
        attn_norm = self.ln1(x)
        attn_mask = torch.triu(torch.ones(x.size(1), x.size(1), device=x.device), diagonal=1).bool()
        y, _ = self.sa(attn_norm, attn_norm, attn_norm, attn_mask=attn_mask, need_weights=False)
        x = x + y
        
        ffwd_norm = self.ln2(x)
        y = self.ffwd(ffwd_norm)
        x = x + y
        
        return x

class MetaWordTransformer(nn.Module):
    def __init__(self):
        super().__init__()
        self.word_embedding_table = nn.Embedding(tokenizer.vocab_size, d_model)
        self.ruolo_embedding_table = nn.Embedding(len(ruolo_to_id), d_model)
        self.semantico_embedding_table = nn.Embedding(len(semantico_to_id), d_model)
        self.position_embedding_table = nn.Embedding(block_size, d_model)
        
        self.emb_dropout = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(d_model, n_head=n_head) for _ in range(n_layer)])
        
        # La normalizzazione finale prima della testa di classificazione
        self.ln_f = nn.LayerNorm(d_model)

        self.lm_head_word = nn.Linear(d_model, tokenizer.vocab_size)

    def forward(self, word_idx, ruolo_idx, semantico_idx, targets_w=None):
        B, T = word_idx.shape
        
        word_emb = self.word_embedding_table(word_idx)
        ruolo_emb = self.ruolo_embedding_table(ruolo_idx)
        semantico_emb = self.semantico_embedding_table(semantico_idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        
        x = word_emb + ruolo_emb + semantico_emb + pos_emb
        x = self.emb_dropout(x)
        x = self.blocks(x)
        x = self.ln_f(x)

        logits_w = self.lm_head_word(x)

        loss = None
        if targets_w is not None:
            loss = F.cross_entropy(logits_w.view(-1, logits_w.size(-1)), targets_w.view(-1), ignore_index=tokenizer.pad_id)
            
        return logits_w, loss

    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p):
        # ... (copia la funzione generate dalla v5, è identica) ...
        for _ in range(max_new_tokens):
            w_idx_cond = word_idx[:, -block_size:]
            r_idx_cond = ruolo_idx[:, -block_size:]
            s_idx_cond = semantico_idx[:, -block_size:]

            logits_w, _ = self(w_idx_cond, r_idx_cond, s_idx_cond)
            
            logits_w = logits_w[:, -1, :]
            probs_w = F.softmax(logits_w, dim=-1)
            
            sorted_probs, sorted_indices = torch.sort(probs_w, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs_w[indices_to_remove] = 0
            probs_w = probs_w / probs_w.sum(dim=-1, keepdim=True)
            
            next_w_id = torch.multinomial(probs_w, num_samples=1)
            
            word_idx = torch.cat((word_idx, next_w_id), dim=1)
            ruolo_idx = torch.cat((ruolo_idx, ruolo_idx[:,-1:]), dim=1)
            semantico_idx = torch.cat((semantico_idx, semantico_idx[:,-1:]), dim=1)
            
        return word_idx

# --- 6. TRAINING E GENERAZIONE ---
model = MetaWordTransformer()
m = model.to(device)
optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate, weight_decay=1e-1) # Torniamo a un wd più standard
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max_iters)

print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")

for iter in range(max_iters):
    # ... (copia il training loop dalla v5, è identico) ...
    if iter % eval_interval == 0 or iter == max_iters - 1:
        metrics = estimate_metrics()
        print("-" * 50)
        print(f"step {iter}: lr: {scheduler.get_last_lr()[0]:.6f}")
        if 'train_loss' in metrics: print(f"  train loss: {metrics['train_loss']:.4f}, train perplexity: {metrics['train_perplexity']:.2f}")
        if 'val_loss' in metrics: print(f"  val loss:   {metrics['val_loss']:.4f}, val perplexity:   {metrics['val_perplexity']:.2f}")
        if 'val_acc_word' in metrics: print(f"  val acc word: {metrics['val_acc_word']:.2%}")

    batch_train = get_batch('train')
    if batch_train:
        xb_w, xb_r, xb_s, yb_w, _, _ = batch_train
        _, loss = m(xb_w, xb_r, xb_s, yb_w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

# Generazione finale
# ... (copia la sezione di generazione finale dalla v5, è identica) ...
print("\n" + "="*20 + " GENERAZIONE FINALE " + "="*20)
start_text = "Spiegami i transformer"
start_ruolo, start_semantico = 'UTENTE', 'AZIONE'

context_w = torch.tensor([tokenizer.encode(start_text)], dtype=torch.long, device=device)
context_r = torch.full_like(context_w, ruolo_to_id[start_ruolo])
context_s = torch.full_like(context_w, semantico_to_id[start_semantico])

generated_w = m.generate(context_w, context_r, context_s, max_new_tokens=30, top_p=top_p_val)
gen_words = tokenizer.decode(generated_w[0].tolist())

print(f"Prompt: '{start_text}'\n")
print("Output generato:\n")
print(gen_words)