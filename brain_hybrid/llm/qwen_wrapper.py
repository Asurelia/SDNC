import os
os.environ["PYTORCH_ATTENTION_BACKEND"] = "math"
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"

from transformers import AutoProcessor
import importlib

def _get_model_class():
    """Auto-detect Qwen VL model class available in transformers."""
    try:
        mod = importlib.import_module("transformers")
        # Qwen3-VL (latest)
        if hasattr(mod, "Qwen3VLForConditionalGeneration"):
            return mod.Qwen3VLForConditionalGeneration
        # Qwen2.5-VL (fallback)
        if hasattr(mod, "Qwen2_5_VLForConditionalGeneration"):
            return mod.Qwen2_5_VLForConditionalGeneration
        # Generic fallback
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
    except Exception:
        from transformers import AutoModelForVision2Seq
        return AutoModelForVision2Seq
import torch


def _can_use_bnb() -> bool:
    """Check if bitsandbytes is available (Linux + CUDA/ROCm only)."""
    try:
        import bitsandbytes
        return True
    except ImportError:
        return False


class QwenWrapper:
    """
    Wrapper Qwen avec accès aux représentations internes.

    Supporte :
      - Modèles VL (Vision-Language) : texte + image
      - Modèles text-only (causal LM) : texte seul (ex: Qwen3.5-4B)

    Auto-détection basée sur le nom du modèle (présence de "VL"/"vl").
    Gelé définitivement — aucun poids ne sera jamais modifié.
    """

    def __init__(self, model_name: str = "Qwen/Qwen2.5-VL-7B-Instruct",
                 use_quantization: bool = True):
        # Détection automatique : VL vs text-only
        self._is_vl = "VL" in model_name or "vl" in model_name

        # Dtype : bfloat16 sur Ampere+/ROCm, float16 sur T4
        if torch.cuda.is_available():
            cap = torch.cuda.get_device_capability(0)
            use_dtype = torch.bfloat16 if cap[0] >= 8 else torch.float16
        else:
            use_dtype = torch.float32

        # Quantization Q8 si disponible (Linux CUDA/ROCm)
        load_kwargs = {
            "device_map": "auto",
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }

        if use_quantization and _can_use_bnb():
            from transformers import BitsAndBytesConfig
            skip_modules = ["lm_head", "visual"] if self._is_vl else ["lm_head"]
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_8bit=True,
                llm_int8_skip_modules=skip_modules,
            )
            load_kwargs["torch_dtype"] = use_dtype
            print(f"Chargement {model_name} en Q8 ({use_dtype})...")
        else:
            load_kwargs["dtype"] = use_dtype
            if use_quantization:
                print(f"bitsandbytes indisponible — chargement {model_name} en {use_dtype}...")
            else:
                print(f"Chargement {model_name} en {use_dtype}...")

        if self._is_vl:
            # Mode VL : AutoProcessor + modèle VL
            self.processor = AutoProcessor.from_pretrained(
                model_name,
                trust_remote_code=True,
                min_pixels=256 * 28 * 28,
                max_pixels=1024 * 28 * 28,
            )
            model_class = _get_model_class()
            print(f"  Classe modèle : {model_class.__name__}")
            self.model = model_class.from_pretrained(
                model_name,
                **load_kwargs
            )
        else:
            # Mode text-only : AutoTokenizer + AutoModelForCausalLM
            from transformers import AutoTokenizer, AutoModelForCausalLM
            self.processor = AutoTokenizer.from_pretrained(
                model_name,
                trust_remote_code=True,
            )
            print(f"  Mode text-only (causal LM)")
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                **load_kwargs
            )

        self.model.eval()

        # Geler TOUS les poids — non négociable
        for p in self.model.parameters():
            p.requires_grad = False

        self.device = next(self.model.parameters()).device

        if torch.cuda.is_available():
            vram = torch.cuda.get_device_properties(0).total_memory / 1e9
            vram_used = torch.cuda.memory_allocated(0) / 1e9
            print(f"Modèle chargé sur {self.device} | VRAM : {vram_used:.1f} / {vram:.1f} GB")
        else:
            print(f"Modèle chargé sur {self.device} (CPU)")

    def _prepare_inputs(self, prompt: str, image=None):
        """
        Prépare les inputs pour le modèle.

        prompt : texte
        image  : PIL.Image, path str, ou None (texte seul)
        """
        if not self._is_vl:
            # Mode text-only — tokenisation directe
            inputs = self.processor(
                prompt,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048,
            )
            return {k: v.to(self.device) for k, v in inputs.items()}

        # Mode VL — messages structurés
        if image is not None:
            if isinstance(image, str):
                from PIL import Image
                image = Image.open(image).convert("RGB")
            messages = [{
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": prompt},
                ]
            }]
        else:
            messages = [{
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                ]
            }]

        text = self.processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=False,
        )

        if image is not None:
            inputs = self.processor(
                text=[text],
                images=[image],
                return_tensors="pt",
                padding=True,
            )
        else:
            inputs = self.processor(
                text=[text],
                return_tensors="pt",
                padding=True,
            )

        return {k: v.to(self.device) for k, v in inputs.items()}

    def get_layer_representations(
        self,
        prompt: str,
        layers: list = None,
        image=None,
    ) -> list:
        """
        Extrait les représentations internes aux couches demandées.

        layers : liste d'indices (0=embeddings, 1-36=layers)
        image  : PIL.Image, path str, ou None
        Retourne : liste de tenseurs (1, seq_len, 4096)
        """
        if layers is None:
            layers = [9, 18, 27, 36]

        inputs = self._prepare_inputs(prompt, image)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        # hidden_states : tuple de 29 tenseurs (1, seq_len, 3584)
        hidden = outputs.hidden_states
        return [hidden[i] for i in layers]

    def generate(
        self,
        prompt: str,
        image=None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """Génération de texte (optionnellement avec image)."""
        inputs = self._prepare_inputs(prompt, image)
        input_len = inputs["input_ids"].shape[1]

        with torch.no_grad():
            out = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=True,
                temperature=0.7,
                **kwargs
            )

        new_tokens = out[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)

    def generate_with_brain(
        self,
        prompt: str,
        brain_modules,
        injection_gates,
        hook_manager,
        hippocampal_prefix=None,
        image=None,
        max_new_tokens: int = 256,
        **kwargs
    ) -> str:
        """
        Génération avec injection CfC via forward hooks.

        Les hooks modifient les hidden states du Qwen pendant la génération
        pour que le cerveau CfC influence les réponses.
        """
        inputs = self._prepare_inputs(prompt, image)
        input_len = inputs["input_ids"].shape[1]

        # Indices des couches (0-indexed) pour les hooks
        layer_indices = [l - 1 for l in [9, 18, 27, 36]]

        # Enregistrer les hooks
        hook_manager.register_hooks(
            self.model, brain_modules, injection_gates, layer_indices
        )

        try:
            with torch.no_grad():
                out = self.model.generate(
                    **inputs,
                    max_new_tokens=max_new_tokens,
                    do_sample=True,
                    temperature=0.7,
                    **kwargs
                )
        finally:
            # Toujours supprimer les hooks, même en cas d'erreur
            hook_manager.remove_hooks()

        new_tokens = out[0][input_len:]
        return self.processor.decode(new_tokens, skip_special_tokens=True)
