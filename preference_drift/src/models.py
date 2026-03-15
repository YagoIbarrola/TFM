"""Thin wrapper around HuggingFace causal LMs for log-prob scoring and embedding extraction."""

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


class ModelWrapper:
    """Wraps a HuggingFace causal LM for log-prob and embedding extraction."""

    def __init__(self, model_name: str, device: str = "auto",
                 dtype: str = "float32", revision: str = "main"):
        self.model_name = model_name
        self.device = self._resolve_device(device)
        self.dtype = getattr(torch, dtype)

        self.tokenizer = AutoTokenizer.from_pretrained(model_name, revision=revision)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name, revision=revision, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    def get_sentence_logprob(self, text: str, normalize: bool = True) -> float:
        """
        Compute log-probability of a sentence under the model.

        log P(sentence) = sum_{t=1}^{T} log P(token_t | token_{<t})

        If normalize=True, divides by token count for fair length comparison.
        """
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        input_ids = inputs["input_ids"]

        with torch.no_grad():
            outputs = self.model(**inputs)
            logits = outputs.logits

        # Shift: logits[t] predicts token[t+1]
        shift_logits = logits[:, :-1, :]
        shift_labels = input_ids[:, 1:]

        log_probs = F.log_softmax(shift_logits, dim=-1)
        token_log_probs = log_probs.gather(2, shift_labels.unsqueeze(-1)).squeeze(-1)

        total_log_prob = token_log_probs.sum().item()
        num_tokens = shift_labels.shape[1]

        if normalize and num_tokens > 0:
            return total_log_prob / num_tokens
        return total_log_prob

    def get_embeddings(self, text: str, layer: int = -1) -> torch.Tensor:
        """Extract hidden states from a specific layer. Returns (seq_len, hidden_dim)."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs, output_hidden_states=True)

        hidden = outputs.hidden_states[layer]  # (1, seq_len, hidden_dim)
        return hidden.squeeze(0).cpu()

    def get_word_embedding(self, word: str, template: str = "This is about {word}.") -> torch.Tensor:
        """
        Get contextual embedding for a word by embedding it in a template sentence
        and extracting hidden states at the word's token positions.
        """
        sentence = template.format(word=word)

        full_ids = self.tokenizer.encode(sentence)
        word_ids = self.tokenizer.encode(word, add_special_tokens=False)

        start_idx = self._find_sublist(full_ids, word_ids)
        if start_idx is None:
            embeddings = self.get_embeddings(word, layer=-1)
            return embeddings.mean(dim=0)

        embeddings = self.get_embeddings(sentence, layer=-1)
        word_embeddings = embeddings[start_idx:start_idx + len(word_ids)]
        return word_embeddings.mean(dim=0)

    def get_trainable_copy(self):
        """Return model and tokenizer for fine-tuning (model set to train mode)."""
        self.model.train()
        return self.model, self.tokenizer

    def reload_from_checkpoint(self, checkpoint_path: str):
        """Load fine-tuned weights from a checkpoint directory."""
        self.model = AutoModelForCausalLM.from_pretrained(
            checkpoint_path, torch_dtype=self.dtype
        ).to(self.device)
        self.model.eval()

    @staticmethod
    def _resolve_device(device: str) -> str:
        if device == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        return device

    @staticmethod
    def _find_sublist(full_list, sub_list):
        for i in range(len(full_list) - len(sub_list) + 1):
            if full_list[i:i + len(sub_list)] == sub_list:
                return i
        return None
