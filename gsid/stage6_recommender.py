"""Stage 6: T5 (from scratch) generating the next item's SID; trie-constrained beam search.

Tasks: REC (history SIDs -> next SID, Graph SID included in input/output) and,
when rec.alignment, explicit T<->V translation of an item's own SID (no Graph).
Metrics: HR@1/5/10, NDCG@5/10 on all / warm / cold targets
(cold = target never appears in the train split).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from transformers import T5Config, T5ForConditionalGeneration

from .stage5_sid import EOS, PAD, TASK_REC, TASK_T2V, TASK_V2T

KS = (1, 5, 10)


def rec_example(sample, tokens, max_history):
    history = sample.history[-max_history:]
    source = [TASK_REC, *np.concatenate([tokens[h] for h in history]).tolist(), EOS] if history else [TASK_REC, EOS]
    return source, [*tokens[sample.target].tolist(), EOS]


def alignment_examples(tokens, order):
    t = slice(order.index("T")*3, order.index("T")*3+3)
    v = slice(order.index("V")*3, order.index("V")*3+3)
    pairs = []
    for row in tokens:
        pairs.append(([TASK_T2V, *row[t].tolist(), EOS], [*row[v].tolist(), EOS]))
        pairs.append(([TASK_V2T, *row[v].tolist(), EOS], [*row[t].tolist(), EOS]))
    return pairs


def collate(pairs, device):
    width = max(len(s) for s, _ in pairs)
    target_width = max(len(t) for _, t in pairs)
    source = torch.full((len(pairs), width), PAD, dtype=torch.long)
    labels = torch.full((len(pairs), target_width), -100, dtype=torch.long)
    for i, (s, t) in enumerate(pairs):
        source[i, :len(s)] = torch.as_tensor(s)
        labels[i, :len(t)] = torch.as_tensor(t)
    return source.to(device), (source != PAD).to(device), labels.to(device)


class SIDTrie:
    def __init__(self, tokens):
        self.length = tokens.shape[1]
        self.item = {tuple(row): i for i, row in enumerate(tokens.tolist())}
        nxt = {}
        for row in tokens.tolist():
            for depth in range(self.length):
                nxt.setdefault(tuple(row[:depth]), set()).add(row[depth])
        self.next = {k: sorted(v) for k, v in nxt.items()}

    def allowed(self, _batch, sequence):
        prefix = tuple(sequence.tolist()[1:])  # drop decoder start token
        if len(prefix) < self.length:
            return self.next.get(prefix, [EOS])
        return [EOS] if len(prefix) == self.length else [PAD]


@torch.no_grad()
def evaluate(model, samples, tokens, trie, cfg, max_history, train_items, device):
    model.eval()
    top = min(max(KS), cfg.beam)
    ranks = []
    for start in range(0, len(samples), cfg.eval_batch_size):
        batch = samples[start:start+cfg.eval_batch_size]
        source, mask, _ = collate([rec_example(s, tokens, max_history) for s in batch], device)
        out = model.generate(input_ids=source, attention_mask=mask, num_beams=cfg.beam, num_return_sequences=top,
                             max_new_tokens=trie.length+1, prefix_allowed_tokens_fn=trie.allowed, do_sample=False)
        out = out[:, 1:1+trie.length].cpu().tolist()
        for i, sample in enumerate(batch):
            predicted = [trie.item.get(tuple(seq), -1) for seq in out[i*top:(i+1)*top]]
            ranks.append(predicted.index(sample.target) if sample.target in predicted else math.inf)
    model.train()
    ranks = np.asarray(ranks, dtype=float)
    warm = np.asarray([bool(train_items[s.target]) for s in samples])
    result = {}
    for name, select in (("all", np.ones_like(warm)), ("warm", warm), ("cold", ~warm)):
        r = ranks[select]
        row = dict(count=int(select.sum()))
        for k in KS:
            row[f"HR@{k}"] = float((r < k).mean()) if len(r) else None
        for k in (5, 10):
            row[f"NDCG@{k}"] = float(np.where(r < k, 1/np.log2(r+2), 0).mean()) if len(r) else None
        result[name] = row
    return result


def run(data, cfg, sid_dir: Path, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    rcfg = cfg.rec
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    tokens = np.load(sid_dir / "sid.npz")["tokens"]
    vocab = json.loads((sid_dir / "vocab.json").read_text())
    order = vocab["order"]
    train_pairs = [rec_example(s, tokens, cfg.data.max_history) for s in data.train]
    if rcfg.alignment:
        train_pairs += alignment_examples(tokens, order)
    model = T5ForConditionalGeneration(T5Config(
        vocab_size=vocab["vocab_size"], d_model=rcfg.d_model, d_ff=rcfg.d_ff, d_kv=rcfg.d_kv,
        num_layers=rcfg.layers, num_decoder_layers=rcfg.layers, num_heads=rcfg.heads,
        dropout_rate=rcfg.dropout, pad_token_id=PAD, eos_token_id=EOS, decoder_start_token_id=PAD,
        feed_forward_proj="relu", tie_word_embeddings=True)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=rcfg.lr)
    steps_per_epoch = math.ceil(len(train_pairs) / rcfg.batch_size)
    total = steps_per_epoch * rcfg.epochs
    schedule = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s+1)/rcfg.warmup_steps)
                                                 * 0.5*(1+math.cos(math.pi*min(s/total, 1.0))))
    trie = SIDTrie(tokens)
    valid = data.valid
    if rcfg.max_eval_users and len(valid) > rcfg.max_eval_users:
        valid = [valid[i] for i in np.sort(rng.choice(len(valid), rcfg.max_eval_users, replace=False))]
    best, best_epoch, waited = -1.0, 0, 0
    log = (out / "train_log.jsonl").open("w")
    for epoch in range(rcfg.epochs):
        order_idx = rng.permutation(len(train_pairs))
        running = 0.0
        for step in range(steps_per_epoch):
            source, mask, labels = collate([train_pairs[i] for i in order_idx[step*rcfg.batch_size:(step+1)*rcfg.batch_size]], device)
            loss = model(input_ids=source, attention_mask=mask, labels=labels).loss
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            schedule.step()
            running += loss.item()
        entry = dict(epoch=epoch+1, loss=running/steps_per_epoch)
        if (epoch+1) % rcfg.eval_every == 0 or epoch+1 == rcfg.epochs:
            metrics = evaluate(model, valid, tokens, trie, rcfg, cfg.data.max_history, data.train_items, device)
            entry["valid"] = metrics
            score = metrics["all"]["NDCG@10"]
            print(f"[stage6] epoch {epoch+1} loss={entry['loss']:.4f} valid NDCG@10={score:.4f} "
                  f"HR@10={metrics['all']['HR@10']:.4f}", flush=True)
            if score > best:
                best, best_epoch, waited = score, epoch+1, 0
                torch.save(model.state_dict(), out / "best.pt")
            else:
                waited += 1
        log.write(json.dumps(entry) + "\n")
        log.flush()
        if waited >= rcfg.patience:
            print(f"[stage6] early stop at epoch {epoch+1}; best epoch {best_epoch}", flush=True)
            break
    log.close()
    model.load_state_dict(torch.load(out / "best.pt", map_location=device))
    result = dict(best_epoch=best_epoch, best_valid_ndcg10=best,
                  test=evaluate(model, data.test, tokens, trie, rcfg, cfg.data.max_history, data.train_items, device))
    (out / "metrics.json").write_text(json.dumps(result, indent=2))
    print(f"[stage6] test {json.dumps(result['test'])}", flush=True)
    return result
