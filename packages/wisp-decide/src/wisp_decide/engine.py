"""Load the adapter and score the offered options from the next-token logits."""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .prompt import BASE_MODEL, BASE_REVISION, GENERIC_POLICY, SYMBOLS, SYSTEM, TASKS, build_prompt, validate

REVIEW_THRESHOLD = 0.7


@dataclass(frozen=True)
class Decision:
    decision: str
    probabilities: dict[str, float]
    confidence: float
    margin: float
    needs_review: bool
    calibration: str = "uncalibrated"

    def as_dict(self) -> dict:
        return asdict(self)


def softmax(values: list[float]) -> list[float]:
    top = max(values)
    exps = [math.exp(v - top) for v in values]
    total = sum(exps)
    return [e / total for e in exps]


class Wisp:
    """Typed decisions. `Wisp.decide` never generates text: the answer is one of your options."""

    def __init__(self, model=None, tokenizer=None) -> None:
        self.model, self.tokenizer = model, tokenizer
        self._codes = None

    def prompt_ids(self, prompt: str):
        ids = self.tokenizer.apply_chat_template(
            [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            add_generation_prompt=True, tokenize=True, return_tensors="pt", enable_thinking=False)
        ids = getattr(ids, "input_ids", ids)
        if ids.shape[-1] > 4096:
            raise ValueError("request exceeds the 4096-token context limit")
        return ids

    def option_logits(self, prompt: str, n: int) -> list[float]:
        """Logits of the first n option symbols at the answer position. Overridden in tests."""
        import torch
        if self._codes is None:
            self._codes = [self.tokenizer.encode(c, add_special_tokens=False) for c in SYMBOLS]
            if any(len(c) != 1 for c in self._codes):
                raise ValueError("option symbols must each be one token")
        ids = self.prompt_ids(prompt).to(self.model.device)
        self.last_tokens = int(ids.shape[-1])
        with torch.inference_mode():
            logits = self.model(input_ids=ids, logits_to_keep=1, use_cache=False).logits[0, -1].float()
        return [logits[self._codes[i][0]].item() for i in range(n)]

    def decide(self, text: str, question: str, options: list[str], task: str | None = None,
               kind: str | None = None, policy: str | None = None,
               glosses: dict[str, str] | None = None) -> Decision:
        validate(text, question, options)
        if task is not None and task not in TASKS:
            raise ValueError(f"task must be one of {', '.join(TASKS)}")
        default_kind, default_policy = TASKS.get(task, ("USER MESSAGE", GENERIC_POLICY))
        prompt = build_prompt(text, question, options, kind or default_kind, policy or default_policy, glosses)
        probs = softmax(self.option_logits(prompt, len(options)))
        ranked = sorted(probs, reverse=True)
        return Decision(decision=options[probs.index(ranked[0])], probabilities=dict(zip(options, probs)),
                        confidence=ranked[0], margin=ranked[0] - ranked[1], needs_review=ranked[0] < REVIEW_THRESHOLD)


def load(adapter: str = "notebox/Wisp-v0.1.2-4B", quantize: bool = True, device_map: str = "auto") -> Wisp:
    """Load Qwen3.5-4B (4-bit by default) plus the adapter from the Hub or a local folder."""
    import torch
    from peft import PeftModel
    from transformers import AutoTokenizer, BitsAndBytesConfig, Qwen3_5ForCausalLM

    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, revision=BASE_REVISION)
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                               bnb_4bit_use_double_quant=True) if quantize else None
    base = Qwen3_5ForCausalLM.from_pretrained(BASE_MODEL, revision=BASE_REVISION, quantization_config=quant,
                                              device_map=device_map)
    return Wisp(PeftModel.from_pretrained(base, adapter).eval(), tokenizer)
