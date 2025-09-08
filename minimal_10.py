import torch
import torch.nn as nn
from torch.nn import functional as F
import json
from pathlib import Path
from tqdm import tqdm
from transformers import GPT2Tokenizer, GPT2LMHeadModel, get_linear_schedule_with_warmup

# --- 1. CONFIGURAZIONE E IPERPARAMETRI V10 (Final Fine-Tuning) ---
MODEL_NAME = 'GroNLP/gpt2-small-italian'
V9_CHECKPOINT_PATH = Path('model_v9_checkpoints/best_model.pt')

max_iters = 2000
eval_interval = 100
learning_rate = 1e-5
batch_size = 4
eval_iters = 100
warmup_steps = 100
patience = 4
loss_weights = {'word': 1.0, 'ruolo': 0.5, 'semantico': 0.5}

block_size = 128
top_p_val = 0.95
max_new_tokens_gen = 100
repetition_penalty = 1.2
DATASET_FILE = Path('dataset.json')
SAVE_DIR = Path('model_v10_final')
SAVE_DIR.mkdir(exist_ok=True)
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
# ----------------------------------------------------

print(f"Sto usando il device: {device}")

# --- 2. GESTIONE DATI (Adattata per 3 target) ---
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token

print(f"Caricamento dati da '{DATASET_FILE}'...")
raw_data_json = json.load(open(DATASET_FILE, 'r', encoding='utf-8'))
raw_data = raw_data_json['examples']
print(f"Caricati {len(raw_data)} esempi.")

train_data = raw_data[:-1]
val_data = raw_data[-1:]

all_ruoli = set(['<PAD>'] + [token['ruolo'] for seq in raw_data for token in seq['sequence']])
all_semantici = set(['<PAD>'] + [token['semantico'] for seq in raw_data for token in seq['sequence']])
ruolo_to_id = {r: i for i, r in enumerate(sorted(list(all_ruoli)))}
id_to_ruolo = {i: r for i, r in enumerate(sorted(list(all_ruoli)))}
semantico_to_id = {s: i for i, s in enumerate(sorted(list(all_semantici)))}
id_to_semantico = {i: s for i, s in enumerate(sorted(list(all_semantici)))}

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
    
    targets_w, targets_r, targets_s = padded_inputs.clone(), padded_ruoli.clone(), padded_semantici.clone()
    
    return padded_inputs.to(device), padded_ruoli.to(device), padded_semantici.to(device), targets_w.to(device), targets_r.to(device), targets_s.to(device)

# --- 3. ARCHITETTURA DEL MODELLO V10 ---
class MetaGPT2Autonomous(nn.Module):
    def __init__(self, model_name, num_ruoli, num_semantici):
        super().__init__()
        self.gpt2 = GPT2LMHeadModel.from_pretrained(model_name)
        d_model = self.gpt2.config.n_embd
        
        self.ruolo_embedding_table = nn.Embedding(num_ruoli, d_model)
        self.semantico_embedding_table = nn.Embedding(num_semantici, d_model)
        
        self.lm_head_ruolo = nn.Linear(d_model, num_ruoli)
        self.lm_head_semantico = nn.Linear(d_model, num_semantici)

    def forward(self, word_idx, ruolo_idx, semantico_idx, labels_w=None, labels_r=None, labels_s=None):
        word_embeddings = self.gpt2.transformer.wte(word_idx)
        ruolo_embeddings = self.ruolo_embedding_table(ruolo_idx)
        semantico_embeddings = self.semantico_embedding_table(semantico_idx)
        inputs_embeds = word_embeddings + ruolo_embeddings + semantico_embeddings
        attention_mask = (word_idx != tokenizer.pad_token_id).float()
        
        transformer_outputs = self.gpt2.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state

        logits_w = self.gpt2.lm_head(hidden_states)
        logits_r = self.lm_head_ruolo(hidden_states)
        logits_s = self.lm_head_semantico(hidden_states)
        
        loss = None
        if labels_w is not None and labels_r is not None and labels_s is not None:
            shift_logits_w, shift_labels_w = logits_w[..., :-1, :].contiguous(), labels_w[..., 1:].contiguous()
            shift_logits_r, shift_labels_r = logits_r[..., :-1, :].contiguous(), labels_r[..., 1:].contiguous()
            shift_logits_s, shift_labels_s = logits_s[..., :-1, :].contiguous(), labels_s[..., 1:].contiguous()
            
            loss_w = F.cross_entropy(shift_logits_w.view(-1, shift_logits_w.size(-1)), shift_labels_w.view(-1), ignore_index=tokenizer.pad_token_id)
            loss_r = F.cross_entropy(shift_logits_r.view(-1, shift_logits_r.size(-1)), shift_labels_r.view(-1), ignore_index=ruolo_to_id['<PAD>'])
            loss_s = F.cross_entropy(shift_logits_s.view(-1, shift_logits_s.size(-1)), shift_labels_s.view(-1), ignore_index=semantico_to_id['<PAD>'])
            
            loss = loss_weights['word'] * loss_w + loss_weights['ruolo'] * loss_r + loss_weights['semantico'] * loss_s
            
        return logits_w, logits_r, logits_s, loss

    @torch.no_grad()
    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p, repetition_penalty=1.0):
        self.eval()
        for _ in range(max_new_tokens):
            w_idx_cond, r_idx_cond, s_idx_cond = word_idx[:, -block_size:], ruolo_idx[:, -block_size:], semantico_idx[:, -block_size:]
            
            logits_w, logits_r, logits_s, _ = self(w_idx_cond, r_idx_cond, s_idx_cond)
            
            def sample_from_logits(logits, is_word=False):
                logits = logits[:, -1, :]
                if is_word and repetition_penalty > 1.0:
                    for i in range(word_idx.shape[0]):
                        unique_tokens = torch.unique(word_idx[i])
                        logits[i, unique_tokens] /= repetition_penalty
                probs = F.softmax(logits, dim=-1)
                sorted_probs, sorted_indices = torch.sort(probs, descending=True)
                cumulative_probs = torch.cumsum(sorted_probs, dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                indices_to_remove = sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)
                probs[indices_to_remove] = 0
                probs = probs / (probs.sum(dim=-1, keepdim=True) + 1e-9)
                return torch.multinomial(probs, num_samples=1)
            
            next_w_id = sample_from_logits(logits_w, is_word=True)
            next_r_id = sample_from_logits(logits_r)
            next_s_id = sample_from_logits(logits_s)
            
            word_idx = torch.cat((word_idx, next_w_id), dim=1)
            ruolo_idx = torch.cat((ruolo_idx, next_r_id), dim=1)
            semantico_idx = torch.cat((semantico_idx, next_s_id), dim=1)
        
        self.train()
        return word_idx, ruolo_idx, semantico_idx

@torch.no_grad()
def estimate_loss(model):
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb_w, xb_r, xb_s, yb_w, yb_r, yb_s = get_batch(split)
            _, _, _, loss = model(xb_w, xb_r, xb_s, yb_w, yb_r, yb_s)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

# --- 4. FUNZIONE PRINCIPALE DI ESECUZIONE ---
def main():
    model = MetaGPT2Autonomous(MODEL_NAME, num_ruoli=len(ruolo_to_id), num_semantici=len(semantico_to_id))
    
    print(f"Caricamento pesi dal checkpoint V9: {V9_CHECKPOINT_PATH}")
    if not V9_CHECKPOINT_PATH.exists():
        print(f"ERRORE: Checkpoint V9 non trovato in '{V9_CHECKPOINT_PATH}'. Esegui prima lo script V9.")
        return
        
    # ### MODIFICA CHIAVE V10: Carichiamo i pesi su CPU per evitare il bug MPS ###
    checkpoint = torch.load(V9_CHECKPOINT_PATH, map_location='cpu')
    model.load_state_dict(checkpoint, strict=False)
    
    m = model.to(device) # Spostiamo il modello su MPS solo DOPO aver caricato i pesi
    
    optimizer = torch.optim.AdamW(m.parameters(), lr=learning_rate)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=max_iters)

    print(f"Numero di parametri: {sum(p.numel() for p in m.parameters())/1e6:.2f} M")
    print(f"Inizio fine-tuning finale per {max_iters} iterazioni...")

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
                print(f"Nessun miglioramento. Interrompo il training (Early Stopping).")
                break

        xb_w, xb_r, xb_s, yb_w, yb_r, yb_s = get_batch('train')
        _, _, _, loss = m(xb_w, xb_r, xb_s, yb_w, yb_r, yb_s)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

    meta_vocabs = {'ruolo_to_id': ruolo_to_id, 'semantico_to_id': semantico_to_id, 'id_to_ruolo': id_to_ruolo, 'id_to_semantico': id_to_semantico}
    with open(SAVE_DIR / 'meta_vocabs.json', 'w') as f: json.dump(meta_vocabs, f, indent=2)

    print("\n" + "="*20 + " GENERAZIONE FINALE " + "="*20)
    print("Caricamento del modello migliore per la generazione...")
    # Carichiamo di nuovo su CPU per sicurezza, poi spostiamo
    final_model = MetaGPT2Autonomous(MODEL_NAME, num_ruoli=len(ruolo_to_id), num_semantici=len(semantico_to_id))
    final_model.load_state_dict(torch.load(SAVE_DIR / 'best_model.pt', map_location='cpu'))
    final_model.to(device)
    
    start_text = "Spiegami i transformer"
    start_ruolo = 'UTENTE'
    start_semantico = 'PAROLA_CONTENUTO'
    
    prompt_tokens = tokenizer.encode(" " + start_text)
    context_w = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    context_r = torch.full_like(context_w, ruolo_to_id.get(start_ruolo, 0))
    context_s = torch.full_like(context_w, semantico_to_id.get(start_semantico, 0))
    
    generated_w, generated_r, generated_s = final_model.generate(context_w, context_r, context_s, 
                             max_new_tokens=max_new_tokens_gen, 
                             top_p=top_p_val,
                             repetition_penalty=repetition_penalty)
    
    print(f"Prompt: '{start_text}'\n")
    print("Output generato (con meta-token predetti):\n")
    print(f"{'Parola':<25} | {'Ruolo Preditto':<25} | {'Semantica Predetta'}")
    print("-" * 80)
    
    full_sequence_ids = generated_w[0].tolist()
    
    for i in range(len(prompt_tokens), len(full_sequence_ids)):
        word = tokenizer.decode([full_sequence_ids[i]])
        ruolo = id_to_ruolo.get(generated_r[0, i].item(), "N/A")
        semantico = id_to_semantico.get(generated_s[0, i].item(), "N/A")
        print(f"{word:<25} | {ruolo:<25} | {semantico}")

if __name__ == "__main__":
    main()