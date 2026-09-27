"""One Qwen3-VL, one observation prefill, several decisions, and conditional text.

The contract the rest of the system codes against:

    handle = backend.prefill(prefix_text, image)      # one visual encode + prefill
    answers = backend.decide(handle, branches)        # batched fan-out, logit readout
    text    = backend.generate(handle, suffix)        # continues the SAME prefix KV
    backend.release(handle)

The prefix cache is immutable. Every branch reads it and writes only into its own
copy, so `decide` can run before `generate` without the branches contaminating each
other or the generation. That is the property `scripts/check_cache_equivalence.py`
tests numerically against a full recompute.
"""

import time
from dataclasses import dataclass, field

import torch
from transformers import AutoProcessor, Qwen3VLForConditionalGeneration
from transformers.cache_utils import DynamicCache


def _layers(cache):
    """(keys, values) per layer, across the DynamicCache APIs we may meet."""
    if hasattr(cache, "layers"):
        return [(layer.keys, layer.values) for layer in cache.layers]
    if hasattr(cache, "key_cache"):
        return list(zip(cache.key_cache, cache.value_cache))
    raise TypeError(f"Unsupported cache type {type(cache)}")


def _cache_from(pairs):
    cache = DynamicCache()
    for index, (keys, values) in enumerate(pairs):
        cache.update(keys, values, index)
    return cache


def fork_cache(cache, batch):
    """An independent copy of the prefix KV, widened to `batch` branches.

    `repeat` rather than `expand`: the branches write into their copy, and the
    original must survive unchanged for the next branch and for generation.
    """
    return _cache_from(
        (keys.repeat(batch, 1, 1, 1), values.repeat(batch, 1, 1, 1)) for keys, values in _layers(cache)
    )


def cache_length(cache):
    return _layers(cache)[0][0].shape[-2]


@dataclass
class Prefix:
    """An observation that has been encoded once and can be continued many times."""

    cache: object
    length: int
    next_position: int
    input_ids: torch.Tensor
    image_token_count: int
    # The processor output is kept so the recompute control can rebuild the *same*
    # context, image included. Without it a "recompute" would silently compare
    # against a text-only context whose image placeholders were never filled.
    inputs: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)


class Qwen3VLBackend:
    def __init__(self, model_path, *, device="cuda", dtype=torch.bfloat16, adapter=None, attn=None):
        self.model_path = str(model_path)
        self.device = device
        load = dict(dtype=dtype)
        if attn:
            load["attn_implementation"] = attn
        # Plain load then .to(device): a single-GPU backend does not need accelerate,
        # and keeping the dependency out means the serving env can run these checks.
        self.model = Qwen3VLForConditionalGeneration.from_pretrained(self.model_path, **load)
        self.model.to(device)
        self.model.eval()
        if adapter:
            from peft import PeftModel

            self.model = PeftModel.from_pretrained(self.model, str(adapter))
            self.model.eval()
        self.processor = AutoProcessor.from_pretrained(self.model_path)
        self.tokenizer = self.processor.tokenizer
        self.image_token_id = getattr(self.model.config, "image_token_id", None)

    # ---------------------------------------------------------------- prefill

    @torch.inference_mode()
    def prefill(self, prefix_text, image=None):
        """Encode the shared observation once. Returns a reusable, immutable Prefix."""
        started = time.perf_counter()
        if image is not None:
            inputs = self.processor(text=[prefix_text], images=[image], return_tensors="pt")
        else:
            inputs = self.processor(text=[prefix_text], return_tensors="pt")
        inputs = {k: (v.to(self.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        input_ids = inputs["input_ids"]
        prepared = time.perf_counter()

        cache = DynamicCache()
        outputs = self.model(**inputs, past_key_values=cache, use_cache=True)
        torch.cuda.synchronize(self.device) if self.device.startswith("cuda") else None
        done = time.perf_counter()

        # Where the next token sits on the rope timeline. For Qwen3-VL the three
        # mrope dimensions advance together on text, so one scalar is enough to
        # continue any text suffix after the image.
        position_ids = outputs.get("position_ids") if isinstance(outputs, dict) else None
        if position_ids is None:
            position_ids = self._rope_index(inputs)
        next_position = int(position_ids.max().item()) + 1

        image_tokens = 0
        if self.image_token_id is not None:
            image_tokens = int((input_ids[0] == self.image_token_id).sum().item())
        return Prefix(
            cache=outputs.past_key_values,
            length=input_ids.shape[1],
            next_position=next_position,
            input_ids=input_ids,
            image_token_count=image_tokens,
            inputs={k: v for k, v in inputs.items() if k != "attention_mask"},
            timings={
                "processor_ms": (prepared - started) * 1000,
                "prefill_ms": (done - prepared) * 1000,
                "prefill_tokens": input_ids.shape[1],
            },
        )

    def _rope_index(self, inputs):
        """mrope positions for the prefill, tolerant of the signature changes across
        transformers versions (`mm_token_type_ids` became required in 5.x)."""
        get_rope = getattr(self.model, "get_rope_index", None) or getattr(
            getattr(self.model, "model", None), "get_rope_index", None
        )
        if get_rope is None:
            length = inputs["input_ids"].shape[1]
            return torch.arange(length, device=self.device).view(1, 1, -1).expand(3, 1, -1)
        kwargs = dict(
            input_ids=inputs["input_ids"],
            image_grid_thw=inputs.get("image_grid_thw"),
            video_grid_thw=inputs.get("video_grid_thw"),
            attention_mask=inputs.get("attention_mask"),
        )
        import inspect

        if "mm_token_type_ids" in inspect.signature(get_rope).parameters:
            mm = inputs.get("mm_token_type_ids")
            if mm is None:
                # 1 marks a multimodal token; the processor supplies this when it can.
                mm = torch.zeros_like(inputs["input_ids"], dtype=torch.int32)
                if self.image_token_id is not None:
                    mm[inputs["input_ids"] == self.image_token_id] = 1
            kwargs["mm_token_type_ids"] = mm
        position_ids, _ = get_rope(**kwargs)
        return position_ids

    def _suffix_positions(self, prefix, lengths, padded):
        """mrope positions for right-padded suffixes continuing the prefix."""
        batch = len(lengths)
        offsets = torch.arange(padded, device=self.device).unsqueeze(0).expand(batch, -1)
        positions = prefix.next_position + offsets
        return positions.unsqueeze(0).expand(3, -1, -1).contiguous()

    # ----------------------------------------------------------------- decide

    @torch.inference_mode()
    def decide(self, prefix, branches, *, parallel=True):
        """Read one restricted next-token distribution per branch.

        branches: list of {"suffix": str, "candidate_token_ids": [int], "key": str}
        Returns one dict per branch with the argmax and the normalised candidate
        probabilities. Nothing outside the candidate set can be produced.
        """
        if not branches:
            return []
        if not parallel:
            results = []
            for branch in branches:
                results.extend(self._decide_batch(prefix, [branch]))
            return results
        return self._decide_batch(prefix, branches)

    def _decide_batch(self, prefix, branches):
        started = time.perf_counter()
        encoded = [self.tokenizer.encode(b["suffix"], add_special_tokens=False) for b in branches]
        lengths = [len(e) for e in encoded]
        padded = max(lengths)
        pad_id = self.tokenizer.pad_token_id or 0

        suffix_ids = torch.full((len(branches), padded), pad_id, dtype=torch.long, device=self.device)
        for row, ids in enumerate(encoded):
            suffix_ids[row, : len(ids)] = torch.tensor(ids, device=self.device)

        attention = torch.zeros((len(branches), prefix.length + padded), dtype=torch.long, device=self.device)
        attention[:, : prefix.length] = 1
        for row, length in enumerate(lengths):
            attention[row, prefix.length : prefix.length + length] = 1

        cache = fork_cache(prefix.cache, len(branches))
        cache_position = torch.arange(prefix.length, prefix.length + padded, device=self.device)
        outputs = self.model(
            input_ids=suffix_ids,
            attention_mask=attention,
            position_ids=self._suffix_positions(prefix, lengths, padded),
            past_key_values=cache,
            cache_position=cache_position,
            use_cache=True,
        )
        logits = outputs.logits
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        elapsed = (time.perf_counter() - started) * 1000

        answers = []
        for row, branch in enumerate(branches):
            last = logits[row, lengths[row] - 1].float()
            ids = torch.tensor(branch["candidate_token_ids"], device=logits.device)
            restricted = last.index_select(0, ids)
            probabilities = torch.softmax(restricted, dim=-1)
            best = int(probabilities.argmax().item())
            answers.append(
                {
                    "key": branch.get("key"),
                    "choice_position": best,
                    "probabilities": [float(p) for p in probabilities],
                    "confidence": float(probabilities[best]),
                    "suffix_tokens": lengths[row],
                    "branch_ms": elapsed / len(branches),
                    "batch_ms": elapsed,
                    "batched_with": len(branches),
                }
            )
        return answers

    # --------------------------------------------------------------- generate

    @torch.inference_mode()
    def generate(self, prefix, suffix_text, *, max_new_tokens=64, stop_strings=("\n",), reuse=True):
        """Continue the shared prefix with a text branch.

        reuse=False rebuilds the whole context and re-prefills instead, which is the
        control arm for the KV-reuse ablation. Both paths must agree numerically.
        """
        started = time.perf_counter()
        suffix_ids = self.tokenizer.encode(suffix_text, add_special_tokens=False)
        suffix = torch.tensor([suffix_ids], device=self.device)

        if reuse:
            cache = fork_cache(prefix.cache, 1)
            attention = torch.ones((1, prefix.length + len(suffix_ids)), dtype=torch.long, device=self.device)
            position_ids = self._suffix_positions(prefix, [len(suffix_ids)], len(suffix_ids))
            cache_position = torch.arange(
                prefix.length, prefix.length + len(suffix_ids), device=self.device
            )
            outputs = self.model(
                input_ids=suffix,
                attention_mask=attention,
                position_ids=position_ids,
                past_key_values=cache,
                cache_position=cache_position,
                use_cache=True,
            )
            reused_tokens = prefix.length
            position = prefix.next_position + len(suffix_ids)
            total = prefix.length + len(suffix_ids)
        else:
            # Rebuild the identical context, image tensors included, and prefill it
            # from scratch. Anything less is not a control for KV reuse.
            full = torch.cat([prefix.input_ids, suffix], dim=1)
            attention = torch.ones_like(full)
            rebuilt = {k: v for k, v in prefix.inputs.items() if k != "input_ids"}
            if "mm_token_type_ids" in rebuilt:
                pad = torch.zeros(
                    (1, len(suffix_ids)), dtype=rebuilt["mm_token_type_ids"].dtype, device=self.device
                )
                rebuilt["mm_token_type_ids"] = torch.cat([rebuilt["mm_token_type_ids"], pad], dim=1)
            cache = DynamicCache()
            outputs = self.model(
                input_ids=full,
                attention_mask=attention,
                past_key_values=cache,
                use_cache=True,
                **rebuilt,
            )
            reused_tokens = 0
            position = prefix.next_position + len(suffix_ids)
            total = full.shape[1]

        first_token_at = time.perf_counter()
        cache = outputs.past_key_values
        logits = outputs.logits[:, -1]
        produced = []
        text = ""
        for _ in range(max_new_tokens):
            token = int(logits.argmax(dim=-1).item())
            if token in (self.tokenizer.eos_token_id, self.tokenizer.convert_tokens_to_ids("<|im_end|>")):
                break
            produced.append(token)
            text = self.tokenizer.decode(produced, skip_special_tokens=True)
            if any(stop in text for stop in stop_strings if stop):
                for stop in stop_strings:
                    if stop and stop in text:
                        text = text.split(stop)[0]
                break
            step = torch.tensor([[token]], device=self.device)
            outputs = self.model(
                input_ids=step,
                attention_mask=torch.ones((1, total + 1), dtype=torch.long, device=self.device),
                position_ids=torch.full((3, 1, 1), position, dtype=torch.long, device=self.device),
                past_key_values=cache,
                cache_position=torch.tensor([total], device=self.device),
                use_cache=True,
            )
            cache = outputs.past_key_values
            logits = outputs.logits[:, -1]
            position += 1
            total += 1
        if self.device.startswith("cuda"):
            torch.cuda.synchronize(self.device)
        finished = time.perf_counter()
        return {
            "text": text.strip(),
            "tokens": len(produced),
            "token_ids": produced,
            "reused_prefix_tokens": reused_tokens,
            "recomputed_tokens": total - len(produced) - reused_tokens,
            "ttft_ms": (first_token_at - started) * 1000,
            "total_ms": (finished - started) * 1000,
        }

    # ---------------------------------------------------------------- release

    def release(self, prefix):
        prefix.cache = None
        if self.device.startswith("cuda"):
            torch.cuda.empty_cache()
