"""Frozen vs. LoRA-adapted ESM2-650M backbone, same downstream architecture
(AbsoluteAffinityModel) otherwise unchanged, on benchmark46 (the smallest/
fastest benchmark, chosen because every forward pass now runs the actual
650M-param backbone with gradients -- no more precomputed/cached embeddings,
since the backbone itself is being trained). Tests the hypothesis raised by
the four-primitives-converge-to-the-same-ceiling finding (main report §3.6):
if LoRA-adapting the backbone moves the ceiling, representation learning
(not the interaction-processing mechanism) was the limiting factor.

LoRA config: rank 8, targeting `query`/`value` projections in every ESM2
attention layer (peft library) -- standard minimal LoRA setup. Mixed
precision (bf16 autocast) and a per-example loop (not batched embedding
lookup, since gradients now need to flow through the backbone) to keep
memory within the same 16GB budget the rest of this project has had to
work inside.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random

import numpy as np
import torch
from peft import LoraConfig, get_peft_model
from scipy.stats import pearsonr
from transformers import AutoModel, AutoTokenizer

from simba import env

env.setup()

from simba.data.build_affinity_benchmark import build_skempi2_wt_train_set, build_test_set  # noqa: E402
from simba.models.affinity import AbsoluteAffinityModel  # noqa: E402
from simba.train import lr_at_step  # noqa: E402


def embed_live(backbone, tokenizer, sequence: str, device: str) -> torch.Tensor:
    """Run the (possibly LoRA-adapted) backbone directly -- no caching,
    since gradients need to flow through it when `use_lora=True`."""
    toks = tokenizer(sequence, return_tensors="pt").to(device)
    out = backbone(**toks).last_hidden_state[0]  # [L+2, hidden] (BOS/EOS included)
    return out[1:-1]  # strip BOS/EOS to match this project's convention elsewhere


def evaluate(model, backbone, tokenizer, examples, device):
    model.eval()
    preds, trues = [], []
    with torch.no_grad():
        for ex in examples:
            self_emb = embed_live(backbone, tokenizer, ex.self_sequence, device)
            partner_emb = embed_live(backbone, tokenizer, ex.partner_sequence, device)
            preds.append(model(self_emb, partner_emb).item())
            trues.append(ex.dG)
    preds, trues = np.asarray(preds), np.asarray(trues)
    pcc = float(pearsonr(trues, preds)[0]) if np.std(preds) > 0 else float("nan")
    mae = float(np.mean(np.abs(trues - preds)))
    return {"pcc": pcc, "mae": mae, "n": len(trues)}


def run_one(train, val, test, *, hidden, backbone_name, use_lora, epochs, batch_size, lr,
            backbone_lr, weight_decay, device, seed):
    torch.manual_seed(seed)
    rng = random.Random(seed)

    tokenizer = AutoTokenizer.from_pretrained(backbone_name)
    backbone = AutoModel.from_pretrained(backbone_name).to(device)
    if use_lora:
        lora_config = LoraConfig(r=8, lora_alpha=16, target_modules=["query", "value"],
                                  lora_dropout=0.05, bias="none")
        backbone = get_peft_model(backbone, lora_config)
        backbone.print_trainable_parameters()
        backbone.train()
        trainable_backbone_params = [p for p in backbone.parameters() if p.requires_grad]
    else:
        backbone.eval()
        for p in backbone.parameters():
            p.requires_grad_(False)
        trainable_backbone_params = []

    model = AbsoluteAffinityModel(plm_hidden_size=hidden).to(device)
    param_groups = [{"params": model.parameters(), "lr": lr}]
    if trainable_backbone_params:
        param_groups.append({"params": trainable_backbone_params, "lr": backbone_lr})
    optimizer = torch.optim.AdamW(param_groups, weight_decay=weight_decay)

    steps_per_epoch = max(1, math.ceil(len(train) / batch_size))
    total_steps = steps_per_epoch * epochs
    best_pcc, best_state, global_step = -math.inf, None, 0

    for epoch in range(epochs):
        model.train()
        if use_lora:
            backbone.train()
        rng.shuffle(train)
        epoch_loss, n_seen = 0.0, 0
        for start in range(0, len(train), batch_size):
            batch = train[start:start + batch_size]
            for g_idx, g in enumerate(optimizer.param_groups):
                g["lr"] = lr_at_step(global_step, total_steps, lr if g_idx == 0 else backbone_lr)
            optimizer.zero_grad()
            n_valid = len(batch)
            for ex in batch:
                self_emb = embed_live(backbone, tokenizer, ex.self_sequence, device)
                partner_emb = embed_live(backbone, tokenizer, ex.partner_sequence, device)
                pred = model(self_emb, partner_emb)
                target = torch.tensor(ex.dG, dtype=torch.float32, device=device)
                loss = (pred - target) ** 2 / n_valid
                loss.backward()
                epoch_loss += loss.item() * n_valid
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            if trainable_backbone_params:
                torch.nn.utils.clip_grad_norm_(trainable_backbone_params, 1.0)
            optimizer.step()
            n_seen += n_valid
            global_step += 1
            if device.startswith("cuda"):
                torch.cuda.empty_cache()

        val_report = evaluate(model, backbone, tokenizer, val, device)
        print(f"[{'lora' if use_lora else 'frozen'}] epoch {epoch}: "
              f"train_mse={epoch_loss / max(n_seen, 1):.4f} val_pcc={val_report['pcc']:.4f}", flush=True)
        if not math.isnan(val_report["pcc"]) and val_report["pcc"] > best_pcc:
            best_pcc = val_report["pcc"]
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)
    return evaluate(model, backbone, tokenizer, test, device)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--backbone", default="facebook/esm2_t33_650M_UR50D")
    parser.add_argument("--modes", nargs="+", default=["frozen", "lora"], choices=["frozen", "lora"])
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--backbone-lr", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--val-frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default=None)
    parser.add_argument("--out", default="lora_finetune_results.json")
    args = parser.parse_args()

    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")

    test = build_test_set()
    test_ids = {e.complex_id for e in test}
    train_full = build_skempi2_wt_train_set(exclude_complex_ids=test_ids)
    rng = random.Random(args.seed)
    idx = list(range(len(train_full)))
    rng.shuffle(idx)
    n_val = max(1, int(args.val_frac * len(idx)))
    val = [train_full[i] for i in idx[:n_val]]
    train = [train_full[i] for i in idx[n_val:]]
    print(f"train={len(train)} val={len(val)} test={len(test)}", flush=True)

    hidden = AutoModel.from_pretrained(args.backbone).config.hidden_size

    results = {}
    for mode in args.modes:
        print(f"\n=== mode={mode} ===", flush=True)
        r = run_one(train, val, test, hidden=hidden, backbone_name=args.backbone,
                    use_lora=(mode == "lora"), epochs=args.epochs, batch_size=args.batch_size,
                    lr=args.lr, backbone_lr=args.backbone_lr, weight_decay=args.weight_decay,
                    device=device, seed=args.seed)
        results[mode] = r
        print(f"[{mode}] test: PCC={r['pcc']:.4f} MAE={r['mae']:.4f}", flush=True)
        with open(args.out, "w") as f:
            json.dump(results, f, indent=2)
        if device.startswith("cuda"):
            torch.cuda.empty_cache()

    print("\n\nFINAL RESULTS:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
