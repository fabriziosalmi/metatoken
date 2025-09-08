import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# --- 1. CONFIGURAZIONE E IPERPARAMETRI V9 ---
MODEL_NAME = 'GroNLP/gpt2-small-italian'

# Parametri di training per un dataset grande
max_iters = 10000    # ### MODIFICA V9: Training più lungo ###
eval_interval = 500  # ### MODIFICA V9: Valutiamo meno spesso ###
learning_rate = 2e-5
batch_size = 4       # Mantieni 4 per VRAM, se hai più memoria puoi aumentarlo a 8
eval_iters = 200
warmup_steps = 500   # ### MODIFICA V9: Più warmup ###
patience = 5         # ### MODIFICA V9: Più pazienza ###

# Parametri di generazione migliorati
top_p_val = 0.95
max_new_tokens_gen = 100
repetition_penalty = 1.2 # ### MODIFICA V9: Penalizza le ripetizioni ###

# (Il resto rimane invariato)
block_size = 128
DATASET_FILE = Path('dataset.json')
SAVE_DIR = Path('model_v9_checkpoints') # ### MODIFICA V9: Nuova cartella ###
SAVE_DIR.mkdir(exist_ok=True)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# ----------------------------------------------------

print(f"Sto usando il device: {device}")
print(f"Modello pre-allenato: {MODEL_NAME}")

# --- 2. GESTIONE DATI (con Tokenizer Pre-allenato) ---
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Caricamento dati da '{DATASET_FILE}'...")
# ### MODIFICA V9: Corretto per leggere la chiave 'examples' ###
raw_data_json = json.load(open(DATASET_FILE, 'r', encoding='utf-8'))
raw_data = raw_data_json['examples']
print(f"Caricati {len(raw_data)} esempi.")

train_data = raw_data[:-1]
val_data = raw_data[-1:]

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
            word_tokens = tokenizer.encode(" " + token_data['word'])
            seq_input_ids.extend(word_tokens)
            seq_ruolo_ids.extend([ruolo_to_id.get(token_data['ruolo'], 0)] * len(word_tokens))
            seq_sem_ids.extend([semantico_to_id.get(token_data['semantico'], 0)] * len(word_tokens))
        input_ids.append(torch.tensor(seq_input_ids))
        ruolo_ids.append(torch.tensor(seq_ruolo_ids))
        semantico_ids.append(torch.tensor(seq_sem_ids))
    padded_inputs = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)
    padded_ruoli = torch.nn.utils.rnn.pad_sequence(ruolo_ids, batch_first=True, padding_value=ruolo_to_id['<PAD>'])
    padded_semantici = torch.nn.utils.rnn.pad_sequence(semantico_ids, batch_first=True, padding_value=semantico_to_id['<PAD>'])
    padded_inputs, padded_ruoli, padded_semantici = padded_inputs[:,:block_size], padded_ruoli[:,:block_size], padded_semantici[:,:block_size]
    targets = padded_inputs.clone()
    return padded_inputs.to(device), padded_ruoli.to(device), padded_semantici.to(device), targets.to(device)

# --- 3. ARCHITETTURA DEL MODELLO V9 ---
class MetaGPT2(nn.Module):
    def __init__(self, model_name, num_ruoli, num_semantici):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        d_model = self.gpt2.config.n_embd
        self.ruolo_embedding_table = nn.Embedding(num_ruoli, d_model)
        self.semantico_embedding_table = nn.Embedding(num_semantici, d_model)

    def forward(self, word_idx, ruolo_idx, semantico_idx, labels=None):
        word_embeddings = self.gpt2.transformer.wte(word_idx)
        ruolo_embeddings = self.ruolo_embedding_table(ruolo_idx)
        semantico_embeddings = self.semantico_embedding_table(semantico_idx)
        inputs_embeds = word_embeddings + ruolo_embeddings + semantico_embeddings
        attention_mask = (word_idx != tokenizer.pad_token_id).float()
        outputs = self.gpt2(inputs_embeds=inputs_embeds, attention_mask=attention_mask, labels=labels)
        return outputs.logits, outputs.loss

    @torch.no_grad()
    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            w_idx_cond, r_idx_cond, s_idx_cond = word_idx[:, -block_size:], ruolo_idx[:, -block_size:], semantico_idx[:, -block_size:]
            logits, _ = self(w_idx_cond, r_idx_cond, s_idx_cond)
            logits = logits[:, -1, :]

            # ### MODIFICA V9: Applica repetition_penalty ###
            if repetition_penalty > 1.0:
                for i in range(word_idx.shape[0]):
                    unique_tokens_in_sequence = torch.unique(word_idx[i])
                    logits[i, unique_tokens_in_sequence] /= repetition_penalty

            probs = F.softmax(logits, dim=-1)
            
            # Nucleus sampling
            sorted_probs, sorted_indices = torch.sort(probs, descending=True)
            cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = 0
            indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
            probs[indices_to_remove] = 0
            probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
            
            next_w_id = torch.multinomial(probs, num_samples=1)
            
            word_idx = torch.cat((word_idx, next_w_id), dim=1)
            ruolo_idx = torch.cat((ruolo_idx, ruolo_idx[:,-1:]), dim=1)
            semantico_idx = torch.cat((semantico_idx, semantico_idx[:,-1:]), dim=1)
        self.train()
        return word_idx

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb_w, xb_r, xb_s, yb_w = get_batch(split)
            _, loss = model(xb_w, xb_r, xb_s, labels=yb_w)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. FUNZIONE PRINCIPALE DI ESECUZIONE ---
def main():
    model = MetaGPT2(MODEL_NAME, num_ruoli=len(ruolo_to_id), num_semantici=len(semantico_to_id))
    m = model.to(device)
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_iters)

    print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")
    print(f"Inizio fine-tuning per {max_iters} iterazioni...")

    best_val_loss = float('inf')
    patience_counter = 0
    progress_bar = tqdm(range(max_iters))

    for iter in progress_bar:
        if iter % eval_interval == 0 or iter == max_iters - 1:
            losses = estimate_loss(m)
            val_loss = losses['val']
            lr = scheduler.get_last_lr()[0]
            progress_bar.set_description(f"Iter {iter}: Train loss {losses['train']:.4f}, Val loss {val_loss:.4f}, LR {lr:.6f}")
            
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                print(f"  -> Nuovo minimo di val loss! Salvo il modello in '{SAVE_DIR}'")
                torch.save(m.state_dict(), SAVE_DIR / 'best_model.pt')
            else:
                patience_counter += 1
            
            if patience_counter >= patience:
                print(f"Nessun miglioramento per {patience} valutazioni. Interrompo il training (Early Stopping).")
                break

        xb_w, xb_r, xb_s, yb_w = get_batch('train')
        _, loss = m(xb_w, xb_r, xb_s, labels=yb_w)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    meta_vocabs = {'ruolo_to_id': ruolo_to_id, 'semantico_to_id': semantico_to_id}
    with open(SAVE_DIR / 'meta_vocabs.json', 'w') as f: json.dump(meta_vocabs, f)

    print("\n" + "="*20 + " GENERAZIONE FINALE " + "="*20)
    print("Caricamento del modello migliore per la generazione...")
    m.load_state_dict(torch.load(SAVE_DIR / 'best_model.pt'))
    
    start_text = "Spiegami i transformer"
    # Assicurati che questi meta-token esistano nel tuo dizionario!
    # Controlla il tuo `semantic_map.json` e il tuo `templates.json` per trovare i valori corretti
    # Esempio basato sul generatore V2: 'Spiegami' diventerebbe 'PAROLA_CONTENUTO'
    start_ruolo = 'UTENTE'
    start_semantico = 'PAROLA_CONTENUTO' 
    
    prompt_tokens = tokenizer.encode(" " + start_text)
    context_w = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    context_r = torch.full_like(context_w, ruolo_to_id.get(start_ruolo, 0))
    context_s = torch.full_like(context_w, semantico_to_id.get(start_semantico, 0))
    
    generated_w = m.generate(context_w, context_r, context_s, 
                             max_new_tokens=max_new_tokens_gen, 
                             top_p=top_p_val,
                             repetition_penalty=repetition_penalty) # Passiamo il nuovo parametro
    
    gen_words = tokenizer.decode(generated_w[0].tolist())
    print(f"Prompt: '{start_text}'\n")
    print("Output generato:\n")
    print(gen_words)

if __name__ == "__main__":
    main()