import json
import random
import uuid
import argparse
import re
from pathlib import Path
from typing import List, Dict, Any, Tuple
from tqdm import tqdm

class DatasetGenerator:
    """
    Generatore di dati ibrido avanzato per il fine-tuning di LLM.
    Carica dati da file JSON esterni per separare la logica dalla configurazione.
    """
    def __init__(self, data_path: Path = Path("generator_data")):
        self.data_path = data_path
        self._load_all_data()

    def _load_json_data(self, filename: str) -> Any:
        """Carica un file JSON dalla cartella dati."""
        file_path = self.data_path / filename
        if not file_path.exists():
            raise FileNotFoundError(f"File di dati mancante: {file_path}. Assicurati che esista.")
        with open(file_path, 'r', encoding='utf-8') as f:
            return json.load(f)

    def _load_all_data(self):
        """Carica tutti i dati necessari (vocabolari, contenuti, templates)."""
        print("Caricamento dati per il generatore...")
        self.vocab: Dict[str, List[str]] = self._load_json_data("vocabs.json")
        self.rich_content: Dict[str, List[str]] = self._load_json_data("rich_content.json")
        self.templates: List[Dict[str, Any]] = self._load_json_data("templates.json")
        self.semantic_map: Dict[str, str] = self._load_json_data("semantic_map.json")
        print("Dati caricati con successo.")

    def _get_semantic_type_for_word(self, word: str) -> str:
        """Restituisce il tipo semantico per una parola con euristica migliorata."""
        word_lower = word.lower()
        if word_lower in self.semantic_map:
            return self.semantic_map[word_lower]
        
        # Le tue ottime euristiche
        if re.match(r".*(are|ere|ire)$", word_lower): return "VERBO_INFINITO"
        if re.match(r".*(ando|endo)$", word_lower): return "VERBO_GERUNDIO"
        if re.match(r".*(ato|ito|uto)$", word_lower): return "VERBO_PARTICIPIO"
        if re.match(r".*(oso|osa|ico|ica|ivo|iva)$", word_lower): return "AGGETTIVO"
        if re.match(r".*(zione|sione|tà|ità)$", word_lower): return "SOSTANTIVO_ASTRATTO"
        if re.match(r".*(ore|tore|ista)$", word_lower): return "SOSTANTIVO_PERSONA"
        if re.match(r".*mente$", word_lower): return "AVVERBIO"
        
        return "PAROLA_CONTENUTO"

    def _resolve_dynamic_content(self, placeholder: str) -> str:
        """Sceglie casualmente un contenuto ricco dal placeholder."""
        if placeholder in self.rich_content:
            return random.choice(self.rich_content[placeholder])
        print(f"Attenzione: placeholder dinamico '{placeholder}' non trovato. Restituisco il placeholder stesso.")
        return placeholder

    def generate_example(self) -> Dict[str, Any]:
        """Genera un singolo esempio JSON con contenuto ibrido ricco."""
        template = random.choice(self.templates)
        sequence: List[Dict[str, str]] = []
        used_vocab_keys: set = set()

        for ruolo, definition in template['struttura']:
            # Caso 1: Lista di alternative
            if isinstance(definition, list):
                chosen_phrase = random.choice(definition)
                for word in chosen_phrase.split():
                    sequence.append({"word": word, "ruolo": ruolo, "semantico": self._get_semantic_type_for_word(word)})
            
            # Caso 2: Chiave del vocabolario
            elif definition in self.vocab:
                chosen_word = random.choice(self.vocab[definition])
                used_vocab_keys.add(definition)
                sequence.append({"word": chosen_word, "ruolo": ruolo, "semantico": definition})
            
            # Caso 3: Contenuto dinamico
            elif definition.startswith("DYNAMIC_"):
                dynamic_content = self._resolve_dynamic_content(definition)
                # Split preservando gli spazi per una tokenizzazione più pulita
                words = re.split(r'(\s+)', dynamic_content)
                for word in words:
                    if word.strip():
                        sequence.append({"word": word, "ruolo": ruolo, "semantico": "CONTENUTO_DINAMICO"})
            
            # Caso 4: Parola o punteggiatura fissa
            else:
                for word in definition.split():
                    sequence.append({"word": word, "ruolo": ruolo, "semantico": self._get_semantic_type_for_word(word)})

        return {
            "id": f"hybrid_v2_{uuid.uuid4()}",
            "template_name": template["nome"],
            "sequence": sequence,
            "metadata": {
                "vocab_keys_used": list(used_vocab_keys),
                "token_count": len(sequence),
                "has_dynamic_content": any(tok["semantico"] == "CONTENUTO_DINAMICO" for tok in sequence)
            }
        }

def main(n_examples: int, output_file: str, data_path: str):
    """Funzione principale per generare il dataset e salvarlo."""
    try:
        generator = DatasetGenerator(data_path=Path(data_path))
    except FileNotFoundError as e:
        print(e)
        print("Assicurati di aver creato la cartella 'generator_data' con i file JSON necessari.")
        return

    dataset: List[Dict[str, Any]] = []
    
    print(f"Generazione di {n_examples} esempi ibridi avanzati...")
    
    for i in tqdm(range(n_examples), desc="Generazione Esempi"):
        dataset.append(generator.generate_example())
            
    # Statistiche
    template_usage = {}
    for example in dataset:
        template_name = example.get("template_name", "unknown")
        template_usage[template_name] = template_usage.get(template_name, 0) + 1
    
    print(f"\nStatistiche diversità template:")
    for template, count in sorted(template_usage.items()):
        print(f"  {template}: {count} esempi ({count/len(dataset)*100:.1f}%)")
            
    output_path = Path(output_file)
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump({"examples": dataset}, f, indent=2, ensure_ascii=False) # Formato corretto per il trainer
        
    print(f"\nFatto! Il file '{output_path}' è stato creato con {len(dataset)} esempi.")
    print("Il dataset ora è nel formato corretto per essere letto dallo script di training.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generatore di dataset ibrido V2.")
    parser.add_argument("--num-esempi", type=int, default=1000, help="Numero di esempi da generare.")
    parser.add_argument("--output-file", type=str, default="dataset.json", help="File JSON di output.")
    parser.add_argument("--data-path", type=str, default="generator_data", help="Cartella contenente i file di configurazione JSON.")
    args = parser.parse_args()
    
    main(args.num_esempi, args.output_file, args.data_path)