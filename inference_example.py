#!/usr/bin/env python3
"""
Script di esempio per l'inferenza con il modello Meta-Token.

Questo script dimostra come caricare un modello allenato e utilizzarlo
per generare testo con meta-token predetti autonomamente.

Uso:
    python inference_example.py --model-path model_v10_final --prompt "Spiegami la relatività"
"""

import argparse
import json
import torch
from pathlib import Path
from transformers import GPT2Tokenizer, GPT2LMHeadModel
import torch.nn as nn
from torch.nn import functional as F


class MetaGPT2Autonomous(nn.Module):
    """Architettura del modello con predizione autonoma dei meta-token."""
    
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
        attention_mask = (word_idx != self.gpt2.config.pad_token_id).float()
        
        transformer_outputs = self.gpt2.transformer(inputs_embeds=inputs_embeds, attention_mask=attention_mask)
        hidden_states = transformer_outputs.last_hidden_state

        logits_w = self.gpt2.lm_head(hidden_states)
        logits_r = self.lm_head_ruolo(hidden_states)
        logits_s = self.lm_head_semantico(hidden_states)
        
        return logits_w, logits_r, logits_s, None

    @torch.no_grad()
    def generate(self, word_idx, ruolo_idx, semantico_idx, max_new_tokens, top_p, repetition_penalty=1.0, block_size=128):
        self.eval()
        for _ in range(max_new_tokens):
            w_idx_cond = word_idx[:, -block_size:]
            r_idx_cond = ruolo_idx[:, -block_size:]
            s_idx_cond = semantico_idx[:, -block_size:]
            
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
        
        return word_idx, ruolo_idx, semantico_idx


def load_model(model_path: Path, device: str = 'cpu'):
    """
    Carica un modello allenato.
    
    Args:
        model_path: Percorso alla cartella del modello
        device: Device su cui caricare il modello ('cpu', 'cuda', 'mps')
        
    Returns:
        Tupla (model, tokenizer, vocabs)
    """
    print(f"Caricamento modello da {model_path}...")
    
    # Carica i vocabolari dei meta-token
    vocab_path = model_path / 'meta_vocabs.json'
    if not vocab_path.exists():
        raise FileNotFoundError(f"File vocabolari non trovato: {vocab_path}")
    
    with open(vocab_path, 'r') as f:
        vocabs = json.load(f)
    
    # Inizializza il modello
    model_name = 'GroNLP/gpt2-small-italian'
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    
    num_ruoli = len(vocabs['ruolo_to_id'])
    num_semantici = len(vocabs['semantico_to_id'])
    
    model = MetaGPT2Autonomous(model_name, num_ruoli, num_semantici)
    
    # Carica i pesi
    checkpoint_path = model_path / 'best_model.pt'
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint del modello non trovato: {checkpoint_path}")
    
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.to(device)
    model.eval()
    
    print(f"Modello caricato con successo su {device}")
    return model, tokenizer, vocabs


def generate_text(model, tokenizer, vocabs, prompt: str, max_tokens: int = 50, 
                  top_p: float = 0.95, repetition_penalty: float = 1.2, 
                  device: str = 'cpu'):
    """
    Genera testo con meta-token dal prompt.
    
    Args:
        model: Modello allenato
        tokenizer: Tokenizer GPT-2
        vocabs: Dizionari dei meta-token
        prompt: Testo iniziale
        max_tokens: Numero massimo di token da generare
        top_p: Soglia per nucleus sampling
        repetition_penalty: Penalità per ripetizioni
        device: Device di computazione
        
    Returns:
        Lista di tuple (parola, ruolo, semantico)
    """
    # Prepara l'input
    start_ruolo = 'UTENTE'
    start_semantico = 'PAROLA_CONTENUTO'
    
    prompt_tokens = tokenizer.encode(" " + prompt)
    context_w = torch.tensor([prompt_tokens], dtype=torch.long, device=device)
    context_r = torch.full_like(context_w, vocabs['ruolo_to_id'].get(start_ruolo, 0))
    context_s = torch.full_like(context_w, vocabs['semantico_to_id'].get(start_semantico, 0))
    
    # Genera
    print(f"\nGenerazione in corso per il prompt: '{prompt}'")
    generated_w, generated_r, generated_s = model.generate(
        context_w, context_r, context_s,
        max_new_tokens=max_tokens,
        top_p=top_p,
        repetition_penalty=repetition_penalty
    )
    
    # Decodifica i risultati
    results = []
    full_sequence_ids = generated_w[0].tolist()
    
    for i in range(len(full_sequence_ids)):
        word = tokenizer.decode([full_sequence_ids[i]])
        ruolo = vocabs['id_to_ruolo'].get(str(generated_r[0, i].item()), "N/A")
        semantico = vocabs['id_to_semantico'].get(str(generated_s[0, i].item()), "N/A")
        results.append((word, ruolo, semantico))
    
    return results


def print_results(results, prompt_length: int = 0):
    """Stampa i risultati in formato tabellare."""
    print("\n" + "="*80)
    print(f"{'Parola':<25} | {'Ruolo Preditto':<25} | {'Semantica Predetta'}")
    print("="*80)
    
    for i, (word, ruolo, semantico) in enumerate(results):
        marker = "[PROMPT]" if i < prompt_length else "[GEN]"
        print(f"{word:<25} | {ruolo:<25} | {semantico} {marker if i < 5 else ''}")


def main():
    parser = argparse.ArgumentParser(description="Inferenza con modello Meta-Token")
    parser.add_argument('--model-path', type=str, default='model_v10_final',
                       help='Percorso alla cartella del modello')
    parser.add_argument('--prompt', type=str, default='Spiegami i transformer',
                       help='Prompt iniziale')
    parser.add_argument('--max-tokens', type=int, default=50,
                       help='Numero massimo di token da generare')
    parser.add_argument('--top-p', type=float, default=0.95,
                       help='Soglia per nucleus sampling')
    parser.add_argument('--repetition-penalty', type=float, default=1.2,
                       help='Penalità per ripetizioni')
    parser.add_argument('--device', type=str, default='auto',
                       help='Device di computazione (auto, cpu, cuda, mps)')
    
    args = parser.parse_args()
    
    # Determina il device
    if args.device == 'auto':
        if torch.cuda.is_available():
            device = 'cuda'
        elif torch.backends.mps.is_available():
            device = 'mps'
        else:
            device = 'cpu'
    else:
        device = args.device
    
    print(f"Usando device: {device}")
    
    # Carica il modello
    model_path = Path(args.model_path)
    model, tokenizer, vocabs = load_model(model_path, device)
    
    # Genera testo
    results = generate_text(
        model, tokenizer, vocabs,
        prompt=args.prompt,
        max_tokens=args.max_tokens,
        top_p=args.top_p,
        repetition_penalty=args.repetition_penalty,
        device=device
    )
    
    # Stampa risultati
    prompt_tokens = len(tokenizer.encode(" " + args.prompt))
    print_results(results, prompt_tokens)
    
    # Stampa solo il testo generato
    print("\n" + "="*80)
    print("TESTO GENERATO:")
    print("="*80)
    generated_text = tokenizer.decode([r[0] for r in results], skip_special_tokens=True)
    print(generated_text)


if __name__ == "__main__":
    main()
