import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from pathlib import Path # Gestione moderna dei percorsi
from tqdm import tqdm # Progress bar professionale

# Workaround per il conflitto torchvision/transformers
import sys
import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # Fallback per operazioni MPS non supportate

# Blocca completamente torchvision per evitare conflitti
import importlib.util
sys.modules['torchvision'] = None
sys.modules['torchvision.transforms'] = None

# Ora importa transformers con un approccio più sicuro
try:
    from transformers import GPT2Tokenizer, GPT2Config, GPT2LMHeadModel
    # Usa GPT2LMHeadModel invece di GPT2Model per evitare problemi di import
    print("Importato GPT2LMHeadModel con successo")
except ImportError as e:
    print(f"Errore nell'importazione: {e}")
    sys.exit(1)

# --- 1. CONFIGURAZIONE E IPERPARAMETRI V7.1 ---
# Usa un modello GPT-2 valido - cambiamo a un modello italiano esistente o standard
MODEL_NAME = 'GroNLP/gpt2-small-italian'  # Modello italiano valido
# Alternativa: MODEL_NAME = 'gpt2'  # Modello inglese standard

# Parametri di training
max_iters = 2000
eval_interval = 100
learning_rate = 3e-5
batch_size = 4
eval_iters = 100

# Parametri del modello
block_size = 128
dropout = 0.2

# Parametri di generazione
top_p_val = 0.9
max_new_tokens_gen = 60

# File e cartelle
DATASET_FILE = Path('dataset.json')
SAVE_DIR = Path('model_v7_checkpoints')
SAVE_DIR.mkdir(exist_ok=True) # Crea la cartella se non esiste

# Configurazione device
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# ----------------------------------------------------

print(f"Sto usando il device: {device}")
print(f"Modello pre-allenato: {MODEL_NAME}")

# --- 2. GESTIONE DATI (con Tokenizer Pre-allenato) ---
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Caricamento dati da '{DATASET_FILE}'...")
raw_data = json.load(open(DATASET_FILE, 'r', encoding='utf-8'))['examples']
print(f"Caricati {len(raw_data)} esempi.")

train_data = raw_data[:-1]
val_data = raw_data[-1:]

# Creazione vocabolari per i meta-token
all_ruoli = set(['<PAD>'] + [token['ruolo'] for seq in raw_data for token in seq['sequence']])
all_semantici = set(['<PAD>'] + [token['semantico'] for seq in raw_data for token in seq['sequence']])

ruolo_to_id = {r: i for i, r in enumerate(sorted(list(all_ruoli)))}
semantico_to_id = {s: i for i, s in enumerate(sorted(list(all_semantici)))}

def get_batch(split):
    data = train_data if split == 'train' else val_data
    if not data: return None
    
    ix = torch.randint(len(data), (batch_size,))
    seqs_json = [data[i]['sequence'] for i in ix]
    
    input_ids, ruolo_ids, semantico_ids = [], [], []
    
    for seq in seqs_json:
        seq_input_ids, seq_ruolo_ids, seq_sem_ids = [], [], []
        for token_data in seq:
            word = token_data['word']
            word_tokens = tokenizer.encode(" " + word)
            seq_input_ids.extend(word_tokens)
            seq_ruolo_ids.extend([ruolo_to_id[token_data['ruolo']]] * len(word_tokens))
            seq_sem_ids.extend([semantico_to_id[token_data['semantico']]] * len(word_tokens))
        
        input_ids.append(torch.tensor(seq_input_ids))
        ruolo_ids.append(torch.tensor(seq_ruolo_ids))
        semantico_ids.append(torch.tensor(seq_sem_ids))

    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_ruoli = torch.nn.utils.rnn.pad_sequence(ruolo_ids, batch_first=True, padding_value=ruolo_to_id['<PAD>'])
    padded_semantici = torch.nn.utils.rnn.pad_sequence(semantico_ids, batch_first=True, padding_value=semantico_to_id['<PAD>'])

    padded_inputs = padded_inputs[:, :block_size]
    padded_ruoli = padded_ruoli[:, :block_size]
    padded_semantici = padded_semantici[:, :block_size]

    targets = padded_inputs.clone()
    
    xb_w, xb_r, xb_s = padded_inputs.to(device), padded_ruoli.to(device), padded_semantici.to(device)
    yb_w = targets.to(device)
    
    return xb_w, xb_r, xb_s, yb_w

# --- 3. ARCHITETTURA DEL MODELLO V7.1 (Transfer Learning) ---
class MetaGPT2(nn.Module):
    def __init__(self, model_name, num_ruoli, num_semantici):
        super().__init__()
        # Usa GPT2LMHeadModel invece di GPT2Model
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(model_name)
        config = self.gpt2_model.config
        
        d_model = config.n_embd
        
        self.ruolo_embedding_table = nn.Embedding(num_ruoli, d_model)
        self.semantico_embedding_table = nn.Embedding(num_semantici, d_model)
        
        # Il lm_head è già incluso in GPT2LMHeadModel
        
    def forward(self, word_idx, ruolo_idx, semantico_idx, targets_w=None):
        word_embeddings = self.gpt2_model.transformer.wte(word_idx)
        ruolo_embeddings = self.ruolo_embedding_table(ruolo_idx)
        semantico_embeddings = self.semantico_embedding_table(semantico_idx)
        
        inputs_embeds = word_embeddings + ruolo_embeddings + semantico_embeddings
        
        attention_mask = (word_idx != tokenizer.pad_token_id).float()
        
        outputs = self.gpt2_model(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=targets_w if targets_w is not None else None)
        
        logits_w = outputs.logits
        loss = outputs.loss if targets_w is not None else None
            
        return logits_w, loss

    @torch.no_grad()
    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p):
        self.eval()
        for _ in range(max_new_tokens):
            w_idx_cond, r_idx_cond, s_idx_cond = word_idx[:, -block_size:], ruolo_idx[:, -block_size:], semantico_idx[:, -block_size:]
            logits_w, _ = self(w_idx_cond, r_idx_cond, s_idx_cond)
            logits_w = logits_w[:, -1, :]
            probs_w = F.softmax(logits_w, dim=-1)
            
            # Nucleus sampling
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
        
        self.train()
        return word_idx

@torch.no_grad()
def estimate_loss(model):
    """Calcola la loss media su train e val set."""
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb_w, xb_r, xb_s, yb_w = get_batch(split)
            _, loss = model(xb_w, xb_r, xb_s, yb_w)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. FUNZIONE PRINCIPALE DI ESECUZIONE ---
def main():
    model = MetaGPT2(MODEL_NAME, num_ruoli=len(ruolo_to_id), num_semantici=len(semantico_to_id))
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)

    print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")
    print(f"Inizio fine-tuning per {max_iters} iterazioni...")

    best_val_loss = float('inf')
    progress_bar = tqdm(range(max_iters))

    for iter in progress_bar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m)
            val_loss = losses['val']
            progress_bar.set_description(f"Iter {iter}: Train loss {losses['train']:.4f}, Val loss {val_loss:.4f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                print(f"  -> Nuovo minimo di val loss! Salvo il modello in '{SAVE_DIR}'")
                torch.save(m.state_dict(), SAVE_DIR / 'best_model.pt')

        try:
            xb_w, xb_r, xb_s, yb_w = get_batch('train')
            _, loss = m(xb_w, xb_r, xb_s, yb_w)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
        except RuntimeError as e:
            if "MPS" in str(e):
                print("\nERRORE: un'operazione non è supportata da MPS. Prova a rieseguire su CPU.")
                print(e)
                break
            else:
                raise e

    # Salviamo i vocabolari dei meta-token per poterli riutilizzare
    meta_vocabs = {
        'ruolo_to_id': ruolo_to_id,
        'semantico_to_id': semantico_to_id
    }
    with open(SAVE_DIR / 'meta_vocabs.json', 'w') as f:
        json.dump(meta_vocabs, f)

    # --- Generazione Finale con il Modello Migliore ---
    print("\n" + "="*20 + " GENERAZIONE FINALE " + "="*20)
    print("Caricamento del modello migliore per la generazione...")
    
    m.load_state_dict(torch.load(SAVE_DIR / 'best_model.pt'))

    start_text = "Spiegami i transformer"
    start_ruolo = 'UTENTE'
    start_semantico = 'AZIONE'

    prompt_tokens = tokenizer.encode(" " + start_text)
    context_w = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    context_r = torch.full_like(context_w, ruolo_to_id[start_ruolo])
    context_s = torch.full_like(context_w, semantico_to_id[start_semantico])

    generated_w = m.generate(context_w, context_r, context_s, max_new_tokens=max_new_tokens_gen, top_p=top_p_val)
    gen_words = tokenizer.decode(generated_w[0].tolist())

    print(f"Prompt: '{start_text}'\n")
    print("Output generato:\n")
    print(gen_words)

if __name__ == "__main__":
    main()