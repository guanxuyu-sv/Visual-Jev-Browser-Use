"""Continue training an existing adapter so it learns when to stop.

The corpus teaches every operation except the two that end a run. DONE examples
are few -- the collector only produces one per episode the policy actually
solves -- so they are oversampled against a slice of the original corpus rather
than appended to it: training on 80 DONE examples alone would teach the model to
answer DONE everywhere, and appending them to 3327 others would teach it nothing.

The mix is the parameter that matters and it is reported, not hidden: `--done-share`
is the fraction of each batch drawn from the DONE pool. The original examples in
the other fraction are there to hold the rest of the behaviour in place, which
the L1 check after training is what actually verifies.
"""

import argparse
import json
import random
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from vjb.model.backend import Qwen3VLBackend  # noqa: E402
from vjb.model.symbols import verify_pool  # noqa: E402
from vjb.train.dataset import StepSequences, load_jsonl  # noqa: E402
from vjb.train.sft import Trainer  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter", required=True, help="adapter to continue from")
    parser.add_argument("--train", required=True, help="original corpus")
    parser.add_argument("--done", required=True, help="collected DONE examples")
    parser.add_argument("--out", required=True)
    parser.add_argument("--arm", required=True, choices=["A", "B", "C", "D"])
    parser.add_argument("--steps", type=int, default=400)
    parser.add_argument("--accumulate", type=int, default=4)
    parser.add_argument("--lr", type=float, default=3e-5)
    parser.add_argument("--done-share", type=float, default=0.35)
    parser.add_argument("--lambda-text", type=float, default=0.5)
    parser.add_argument("--image-dropout", type=float, default=0.15)
    parser.add_argument("--visual-tokens", type=int, default=1024)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=50)
    args = parser.parse_args()

    no_image = args.arm in ("A", "C")
    output_format = "compact" if args.arm in ("A", "B") else "branch"

    torch.manual_seed(args.seed)
    random.seed(args.seed)

    backend = Qwen3VLBackend(args.model, adapter=args.adapter)
    backend.policy_pool = verify_pool(backend.tokenizer, "Answer:")
    # Continue training the adapter that was loaded, rather than starting a new one.
    for name, parameter in backend.model.named_parameters():
        parameter.requires_grad = "lora_" in name and "visual" not in name and "vision" not in name
    backend.model.gradient_checkpointing_enable()
    backend.model.enable_input_require_grads()
    trainable = sum(p.numel() for p in backend.model.parameters() if p.requires_grad)
    print(f"continuing from {args.adapter}: {trainable/1e6:.1f}M trainable")
    backend.model.train()

    builder = StepSequences(
        backend.policy_pool,
        image_dropout=0.0 if no_image else args.image_dropout,
        seed=args.seed,
        visual_token_budget=args.visual_tokens,
    )
    trainer = Trainer(backend, lambda_text=args.lambda_text, accumulate=args.accumulate)

    originals = load_jsonl(args.train)
    cache = Path(args.train).with_suffix(".viewport_ok.json")
    if cache.exists():
        allowed = set(json.loads(cache.read_text()))
        originals = [e for e in originals if e["step_id"] in allowed]
    done_pool = [e for e in load_jsonl(args.done) if builder.feasible(e)]
    random.Random(args.seed).shuffle(originals)
    random.Random(args.seed).shuffle(done_pool)
    print(f"{len(originals)} original examples, {len(done_pool)} DONE examples "
          f"(share {args.done_share:.0%} per batch)")
    if not done_pool:
        raise SystemExit("no usable DONE examples; nothing to continue on")

    optimizer = torch.optim.AdamW(
        [p for p in backend.model.parameters() if p.requires_grad], lr=args.lr, weight_decay=0.0
    )
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log = open(out / "continue_log.jsonl", "w")

    rng = random.Random(args.seed)
    started = time.perf_counter()
    cursors = {"orig": 0, "done": 0}
    for step in range(1, args.steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accumulated, reports = 0.0, []
        for _ in range(args.accumulate):
            from_done = rng.random() < args.done_share
            pool, key = (done_pool, "done") if from_done else (originals, "orig")
            example = pool[cursors[key] % len(pool)]
            cursors[key] += 1
            view, image = builder.viewport(example)
            if image is None:
                continue
            if no_image:
                image = None
            build = builder.build_compact if output_format == "compact" else builder.build
            sequences, problem = build(view, with_image=not no_image)
            if problem or not sequences:
                continue
            if not sequences[0]["use_image"]:
                image = None
            loss, report = trainer.example_loss(sequences, image)
            if loss is None:
                continue
            accumulated += loss / args.accumulate
            reports.append({**report, "_from": key})
        torch.nn.utils.clip_grad_norm_(
            [p for p in backend.model.parameters() if p.requires_grad], 1.0
        )
        optimizer.step()

        if step % args.log_every == 0 or step == 1:
            done_n = sum(1 for r in reports if r["_from"] == "done")
            record = {"step": step, "loss": accumulated, "done_in_batch": done_n,
                      "elapsed_s": round(time.perf_counter() - started)}
            log.write(json.dumps(record) + "\n")
            log.flush()
            print(f"step {step:4d}  loss {accumulated:6.3f}  done_in_batch={done_n}/{args.accumulate}  "
                  f"{record['elapsed_s']}s", flush=True)

    backend.model.save_pretrained(out / "final")
    (out / "config.json").write_text(json.dumps(vars(args), indent=2))
    log.close()
    print(f"saved {out}/final")


if __name__ == "__main__":
    main()
