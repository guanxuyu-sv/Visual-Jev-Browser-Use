"""Verify the training loss is aligned, before spending GPU hours on it.

Reference: the standard HF formulation -- full-sequence logits, labels shifted by
one, everything but the answer masked out. If the fast path (logits_to_keep plus
a hand-rolled slice) disagrees with that, the supervision is on the wrong tokens.
"""
import argparse, sys
from pathlib import Path
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vjb.model.backend import Qwen3VLBackend  # noqa: E402


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--model", required=True)
    args = ap.parse_args()
    be = Qwen3VLBackend(args.model)
    tok = be.tokenizer
    stop = tok.convert_tokens_to_ids("<|im_end|>")

    prompt = "<|im_start|>user\nWrite a city name.<|im_end|>\n<|im_start|>assistant\n"
    answer = "Bloggs"
    answer_ids = tok.encode(answer, add_special_tokens=False) + [stop]
    full = prompt + answer

    inputs = tok(full, return_tensors="pt").to(be.device)
    with torch.inference_mode():
        # reference: whole-sequence logits, shift by one, mask all but the answer
        ref_out = be.model(**inputs, use_cache=False)
        ref_logits = ref_out.logits[0]
        n = len(answer_ids)
        start = inputs["input_ids"].shape[1] - (n - 1)     # first answer token index
        ref_slice = ref_logits[start - 1 : start - 1 + n]   # predicts a1..an, STOP
        ref_loss = F.cross_entropy(ref_slice.float(), torch.tensor(answer_ids, device=ref_logits.device))

        # fast path, exactly as training computes it
        fast_out = be.model(**inputs, use_cache=False, logits_to_keep=n)
        fast_loss = F.cross_entropy(
            fast_out.logits[0].float(), torch.tensor(answer_ids, device=ref_logits.device)
        )

    delta = abs(float(ref_loss) - float(fast_loss))
    top = ref_slice.argmax(-1).tolist()
    print(f"answer tokens      : {answer_ids} -> {[tok.decode([t]) for t in answer_ids]}")
    print(f"reference loss     : {float(ref_loss):.6f}")
    print(f"training-path loss : {float(fast_loss):.6f}")
    print(f"|difference|       : {delta:.2e}")
    print(f"argmax at those pos: {[tok.decode([t]) for t in top]}")
    ok = delta < 1e-4
    print(f"\n[{'PASS' if ok else 'FAIL'}] supervision is aligned with the answer tokens")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
