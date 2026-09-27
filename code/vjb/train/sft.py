"""Joint SFT: decisions read from a shared prefix, text generated from the same one.

    L = λ_op·L_operation + λ_target·L_target + λ_text·1[TYPE_TEXT]·L_text

Training mirrors inference rather than approximating it: the observation prefix
is forwarded once per example and every branch continues that same forward, with
gradients flowing back through the shared prefill. A branch never sees another
branch's tokens, which is the property the inference-time cache relies on.

Loss bookkeeping follows the plan:

  * decision losses are full-vocabulary next-token CE on the single answer token,
    which is the quantity the logit readout later renormalises within candidates
  * the text loss covers only the generated content tokens
  * the two are normalised separately -- decisions per decision, text per text
    token -- so a long field value cannot drown out the decisions
"""

import argparse
import json
import math
import random
import sys
import time
from pathlib import Path

import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from vjb.model.backend import Qwen3VLBackend  # noqa: E402
from vjb.model.symbols import verify_pool  # noqa: E402
from vjb.train.dataset import StepSequences, load_jsonl  # noqa: E402


class Trainer:
    def __init__(self, backend, *, lambda_op=1.0, lambda_target=1.0, lambda_text=0.5, accumulate=1):
        self.backend = backend
        self.model = backend.model
        self.tokenizer = backend.tokenizer
        self.accumulate = accumulate
        self.weights = {"operation": lambda_op, "target": lambda_target, "text": lambda_text}
        stop = backend.tokenizer.convert_tokens_to_ids("<|im_end|>")
        self.stop_token = stop if stop is not None else backend.tokenizer.eos_token_id

    def _sequence_logits(self, text, image, keep):
        """Forward one full prefix+suffix sequence; return only the last `keep` logits.

        Each branch is forwarded independently rather than continuing one shared
        prefill. The loss is identical either way -- the token sequence a branch
        sees is the same, and the shared prefill is only a way of not computing
        the common part twice -- but an independent forward can use gradient
        checkpointing, which sharing cannot, because checkpointing and a retained
        KV cache are mutually exclusive. We pay roughly 2-3x the prefill compute
        for a model that fits.

        `keep` bounds how many positions produce vocabulary logits. Without it a
        300-token suffix materialises 300 x |V| floats and their graph, which is
        what put a 4B model over 31 GiB.
        """
        if image is not None:
            inputs = self.backend.processor(text=[text], images=[image], return_tensors="pt")
        else:
            inputs = self.backend.processor(text=[text], return_tensors="pt")
        inputs = {k: (v.to(self.backend.device) if hasattr(v, "to") else v) for k, v in inputs.items()}
        outputs = self.model(**inputs, use_cache=False, logits_to_keep=keep)
        return outputs.logits[0]

    def example_loss(self, sequences, image):
        """One example: every branch it legitimately supervises, each backwarded
        as it is built so no branch's graph outlives its own backward."""
        report = {}
        total = 0.0
        pieces = 0
        for sequence in sequences:
            use_image = sequence["use_image"]
            branch_image = image if use_image else None
            if sequence["kind"] in ("text", "compact"):
                answer_ids = self.tokenizer.encode(sequence["answer_text"], add_special_tokens=False)
                if not answer_ids:
                    continue
                # Teach the model where the answer ends. Without a terminator in
                # the target it never learns to stop, and generation runs to the
                # token cap: arm A filled a destination field with
                # "Zurich City, Switzerland, Zurich, Zurich, Zurich, ..." and
                # failed every DOM-sufficient task for that reason alone.
                answer_ids = answer_ids + [self.stop_token]
                text = sequence["prefix"] + sequence["suffix"] + sequence["answer_text"]
                # Alignment. The input is S + [a1..an]; the terminator is a target,
                # never an input. Predicting [a1..an, STOP] needs the logits at
                # positions |S|-1 .. |S|+n-1, which are exactly the last n+1 --
                # that is len(answer_ids) once STOP has been appended. Keeping one
                # more and dropping the last, as this did, shifts every target by
                # one position: the model is taught to emit a1 where the last
                # suffix token belongs, and generation then loses the start of the
                # answer ("Bloggs" came out as "gs").
                logits = self._sequence_logits(text, branch_image, keep=len(answer_ids))
                loss = F.cross_entropy(
                    logits.float(), torch.tensor(answer_ids, device=logits.device)
                )
                weight = self.weights["text"] if sequence["kind"] == "text" else 1.0
                report[sequence["kind"]] = float(loss)
                report[sequence["kind"] + "_tokens"] = len(answer_ids)
            else:
                answer_id = self.backend.policy_pool.token_id(sequence["answer_symbol"])
                text = sequence["prefix"] + sequence["suffix"]
                logits = self._sequence_logits(text, branch_image, keep=1)
                loss = F.cross_entropy(
                    logits.float(), torch.tensor([answer_id], device=logits.device)
                )
                weight = self.weights["operation"] if sequence["kind"] == "operation" else self.weights["target"]
                report[sequence["kind"]] = float(loss)
            (weight * loss / self.accumulate).backward()
            total += float(weight * loss)
            pieces += 1
        if not pieces:
            return None, report
        return total, report


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train", required=True)
    parser.add_argument("--validation", default=None)
    parser.add_argument("--out", required=True)
    parser.add_argument("--steps", type=int, default=2000)
    parser.add_argument("--accumulate", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--rank", type=int, default=16)
    parser.add_argument("--alpha", type=int, default=32)
    parser.add_argument("--lambda-text", type=float, default=0.5)
    parser.add_argument("--image-dropout", type=float, default=0.15)
    parser.add_argument("--no-image", action="store_true", help="train the DOM-only arm")
    parser.add_argument("--output-format", choices=["branch", "compact"], default="branch",
                        help="branch = arms C/D (logit readout); compact = arms A/B (one AR line)")
    parser.add_argument("--arm", default=None, help="label recorded in the run config")
    parser.add_argument("--visual-tokens", type=int, default=1024,
                        help="cap on visual tokens per observation (plan 4.5 budget)")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=25)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    backend = Qwen3VLBackend(args.model)
    backend.policy_pool = verify_pool(backend.tokenizer, "Answer:")

    from peft import LoraConfig, get_peft_model

    # The vision tower stays frozen in this first stage, as the plan starts from
    # the old setup; only the language tower gets adapters.
    config = LoraConfig(
        r=args.rank,
        lora_alpha=args.alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    )
    backend.model = get_peft_model(backend.model, config)
    for name, parameter in backend.model.named_parameters():
        if "visual" in name or "vision" in name:
            parameter.requires_grad = False
    backend.model.gradient_checkpointing_enable()
    backend.model.enable_input_require_grads()
    trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in backend.model.parameters())
    print(f"trainable {trainable/1e6:.1f}M / {total/1e9:.2f}B ({trainable/total:.2%})")
    backend.model.train()

    builder = StepSequences(
        backend.policy_pool,
        image_dropout=0.0 if args.no_image else args.image_dropout,
        seed=args.seed,
        visual_token_budget=args.visual_tokens,
    )
    trainer = Trainer(backend, lambda_text=args.lambda_text, accumulate=args.accumulate)

    examples = load_jsonl(args.train)
    # Keep only steps that reduce to one viewport. The filter is deterministic and
    # arm-independent, so all four arms are trained on exactly this list.
    before = len(examples)
    cache = Path(args.train).with_suffix(".viewport_ok.json")
    if cache.exists():
        allowed = set(json.loads(cache.read_text()))
    else:
        allowed = {e["step_id"] for e in examples if builder.feasible(e)}
        cache.write_text(json.dumps(sorted(allowed)))
    examples = [e for e in examples if e["step_id"] in allowed]
    random.Random(args.seed).shuffle(examples)
    print(f"{len(examples)}/{before} training examples survive the viewport filter "
          f"({len(examples)/before:.0%})")

    optimizer = torch.optim.AdamW(
        [p for p in backend.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=args.steps, pct_start=0.03
    )

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "train_log.jsonl"
    log = open(log_path, "w")

    started = time.perf_counter()
    cursor = 0
    skipped = 0
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated, reports = 0.0, []
        for _ in range(args.accumulate):
            example = examples[cursor % len(examples)]
            cursor += 1
            # Restrict the step to one viewport first, so the element table and
            # the screenshot describe the same thing the executor would show.
            # Every arm goes through this, including the DOM-only ones: the four
            # arms must see the same examples with the same candidate lists, or a
            # difference between them is a difference of training data.
            view, image = builder.viewport(example)
            if image is None:  # pre-filtered, so this should not happen
                skipped += 1
                continue
            if args.no_image:
                image = None
            build = builder.build_compact if args.output_format == "compact" else builder.build
            sequences, problem = build(view, with_image=not args.no_image)
            if problem or not sequences:
                skipped += 1
                continue
            if not sequences[0]["use_image"]:
                image = None
            loss, report = trainer.example_loss(sequences, image)
            if loss is None:
                skipped += 1
                continue
            accumulated += loss / args.accumulate
            reports.append(report)
        torch.nn.utils.clip_grad_norm_(
            [p for p in backend.model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()
        scheduler.step()

        if step % args.log_every == 0 or step == 1:
            means = {}
            for key in ("operation", "click_target", "type_text_target", "select_target", "text", "compact"):
                values = [r[key] for r in reports if key in r]
                if values:
                    means[key] = sum(values) / len(values)
            record = {
                "step": step,
                "loss": accumulated,
                "lr": scheduler.get_last_lr()[0],
                "components": means,
                "skipped": skipped,
                "elapsed_s": round(time.perf_counter() - started),
                "peak_gib": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            }
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(
                f"step {step:5d}  loss {accumulated:6.3f}  "
                + "  ".join(f"{k}={v:.3f}" for k, v in means.items())
                + f"  {record['elapsed_s']}s  {record['peak_gib']}GiB",
                flush=True,
            )
        if step % 500 == 0 or step == args.steps:
            backend.model.save_pretrained(out / f"step{step}")

    backend.model.save_pretrained(out / "final")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.close()
    print(f"\nsaved {out}/final  (skipped {skipped} examples)")


if __name__ == "__main__":
    main()
