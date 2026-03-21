import os
os.environ["PYTORCH_ATTENTION_BACKEND"] = "math"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

from transformers import AutoModelForCausalLM, AutoTokenizer
import torch


class QwenWrapper:
    """
    Wrapper Qwen3-4B avec accès aux représentations internes.

    Gelé définitivement — aucun poids ne sera jamais modifié.
    Fournit :
      - get_layer_representations() → hidden states aux couches voulues
      - generate() → génération de texte standard
    """

    def __init__(self, model_name: str = "Qwen/Qwen3-4B"):
        # T4 (compute 7.5) ne supporte pas bfloat16 → float16
        # Ampere+ (compute 8.0+) et AMD ROCm → bfloat16
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            use_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            use_dtype = torch.float32
        print(f"Chargement {model_name} en {use_dtype}...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            dtype=use_dtype,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager"
        )
        self.model.eval()

        # Geler TOUS les poids — non négociable
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = next(self.model.parameters()).device

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            print(f"Modèle chargé sur {self.device} | VRAM totale : {vram:.1f} GB")
        else:
            print(f"Modèle chargé sur {self.device} (CPU — pas de GPU détecté)")

    def get_layer_representations(
        self,
        prompt: str,
        layers: list = None
    ) -> list:
        """
        Extrait les représentations internes aux couches demandées.

        layers : liste d'indices (0=embeddings, 1-36=layers)
        Retourne : liste de tenseurs (1, seq_len, 2560)
        """
        if layers is None:
            layers = [8, 16, 24, 36]

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=2048
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states : tuple de 37 tenseurs (1, seq_len, 2560)
        hidden = outputs.hidden_states
        return [hidden[i] for i in layers]

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Génération de texte standard."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt"
        ).to(self.device)

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                pad_token_id=self.tokenizer.eos_token_id,
                **kwargs
            )

        # Retourner uniquement les nouveaux tokens
        new_tokens = out[0][inputs['input_ids'].shape[1]:]
        return self.tokenizer.decode(new_tokens, skip_special_tokens=True)
