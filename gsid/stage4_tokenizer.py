"""Stage 4: T / V / G RQ-VAE tokenizers with adaptive hierarchy guidance.

T, V: MLP encoder -> RQ -> MLP decoder reconstructing the (standardized) input.
G: GNN([x_T || x_V], EdgeDrop(Ã)) -> RQ -> decoder reconstructing the fixed
   neighborhood context Â[x_T || x_V] (self-loop normalized, full graph), so the
   Graph SID has to carry structure rather than copy its own input.
L = Σ_m L_RQ^m + λ(e) · hierarchy guidance (hierarchy/adaptive_distillation.py).
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from hierarchy.adaptive_distillation import LossEMA, guidance_lambda, hierarchy_guidance, st_softmax_blocks
from hierarchy.sampling import cluster_aware_batches, edge_drop


def mlp(dims):
    layers = []
    for i in range(len(dims)-1):
        layers.append(nn.Linear(dims[i], dims[i+1]))
        if i < len(dims)-2:
            layers.append(nn.ReLU())
    return nn.Sequential(*layers)


@torch.no_grad()
def kmeans(x, k, iters=25, generator=None):
    index = torch.randint(len(x), (k,), generator=generator) if len(x) < k else torch.randperm(len(x), generator=generator)[:k]
    centers = x[index.to(x.device)].clone()
    for _ in range(iters):
        assign = torch.cdist(x, centers).argmin(-1)
        total = torch.zeros_like(centers).index_add_(0, assign, x)
        count = torch.bincount(assign, minlength=k).float()[:, None]
        centers = torch.where(count > 0, total / count.clamp_min(1), centers)
    return centers


class ResidualQuantizer(nn.Module):
    def __init__(self, levels, size, dim, beta):
        super().__init__()
        self.codebooks = nn.Parameter(0.1*torch.randn(levels, size, dim))
        self.beta = beta

    def books(self):
        return list(self.codebooks.unbind(0))

    def forward(self, z):
        residual, quant, loss, codes = z, 0, 0, []
        for book in self.books():
            code = torch.cdist(residual, book).argmin(-1)
            e = book[code]
            loss = loss + F.mse_loss(e, residual.detach()) + self.beta*F.mse_loss(residual, e.detach())
            quant = quant + e
            residual = residual - e.detach()
            codes.append(code)
        return z + (quant - z).detach(), torch.stack(codes, 1), loss

    @torch.no_grad()
    def kmeans_init(self, z, generator=None):
        residual = z
        for level in range(self.codebooks.shape[0]):
            centers = kmeans(residual, self.codebooks.shape[1], generator=generator)
            self.codebooks.data[level] = centers
            residual = residual - centers[torch.cdist(residual, centers).argmin(-1)]


class GraphEncoder(nn.Module):
    def __init__(self, in_dim, hidden, layers, tail, latent):
        super().__init__()
        self.inp = nn.Linear(in_dim, hidden)
        self.layers = layers
        self.out = mlp([hidden, *tail, latent])

    def forward(self, x, adjacency):
        h = F.relu(self.inp(x))
        hs = [h]
        for _ in range(self.layers):
            h = torch.sparse.mm(adjacency, h)
            hs.append(h)
        return self.out(torch.stack(hs).mean(0))


class Branch(nn.Module):
    def __init__(self, encoder, out_dim, cfg):
        super().__init__()
        self.encoder = encoder
        self.rq = ResidualQuantizer(cfg.levels, cfg.codebook_size, cfg.latent_dim, cfg.beta)
        self.decoder = mlp([cfg.latent_dim, *reversed(cfg.hidden), out_dim])


def build_model(dims, cfg, branches):
    model = nn.ModuleDict()
    for m in branches:
        if m == "G":
            encoder = GraphEncoder(dims["G"], cfg.gnn_hidden, cfg.gnn_layers, cfg.hidden[1:], cfg.latent_dim)
        else:
            encoder = mlp([dims[m], *cfg.hidden, cfg.latent_dim])
        model[m] = Branch(encoder, dims[m], cfg)
    return model


def torch_adjacency(src, dst, weight, n, device):
    """D^-1/2 (W + I) D^-1/2 from undirected edges listed once."""
    loops = torch.arange(n, device=device)
    rows = torch.cat([src, dst, loops])
    cols = torch.cat([dst, src, loops])
    values = torch.cat([weight, weight, torch.ones(n, device=device)])
    degree = torch.zeros(n, device=device).index_add_(0, rows, values)
    inv = degree.rsqrt()
    return torch.sparse_coo_tensor(torch.stack([rows, cols]), inv[rows]*values*inv[cols], (n, n)).coalesce()


def standardize(x):
    return (x - x.mean(0, keepdim=True)) / (x.std(0, keepdim=True) + 1e-6)


def prepare_inputs(data, graph, device):
    x = {"T": standardize(torch.as_tensor(data.text, device=device)),
         "V": standardize(torch.as_tensor(data.image, device=device))}
    x["G_in"] = torch.cat([x["T"], x["V"]], -1)
    coo = graph.tocoo()
    upper = coo.row < coo.col
    edges = torch.as_tensor(np.vstack([coo.row[upper], coo.col[upper]]), dtype=torch.long, device=device)
    weight = torch.as_tensor(coo.data[upper], dtype=torch.float32, device=device)
    full = torch_adjacency(edges[0], edges[1], weight, data.n_items, device)
    x["G"] = torch.sparse.mm(full, x["G_in"])  # fixed reconstruction target of the Graph branch
    return x, edges, weight, full


def input_dims(x):
    """Encoder input = decoder output width; G reconstructs Â[x_T||x_V], same width as its input."""
    return {"T": x["T"].shape[1], "V": x["V"].shape[1], "G": x["G_in"].shape[1]}


def encode(model, m, x, ids, adjacency):
    if m == "G":
        return model[m].encoder(x["G_in"], adjacency)[ids]
    return model[m].encoder(x[m][ids])


@torch.no_grad()
def encode_codes(model, x, adjacency, n, batch=4096):
    """Hard codes [n, L] per branch on the full (undropped) graph."""
    model.eval()
    codes = {}
    for m in model:
        if m == "G":
            codes[m] = model[m].rq(model[m].encoder(x["G_in"], adjacency))[1].cpu().numpy()
        else:
            codes[m] = torch.cat([model[m].rq(model[m].encoder(x[m][i:i+batch]))[1]
                                  for i in range(0, n, batch)]).cpu().numpy()
    model.train()
    return codes


def code_stats(codes):
    stats = {}
    for m, c in codes.items():
        stats[m] = dict(unique_full=len(np.unique(c, axis=0))/len(c),
                        used_per_level=[int(len(np.unique(c[:, l]))) for l in range(c.shape[1])])
    joint = np.concatenate([codes[m] for m in sorted(codes)], 1)
    stats["joint_unique"] = len(np.unique(joint, axis=0))/len(joint)
    return stats


def run(data, graph, pseudo, gates, cfg, out: Path, device):
    out.mkdir(parents=True, exist_ok=True)
    tcfg, gcfg = cfg.tokenizer, cfg.guidance
    torch.manual_seed(cfg.seed)
    rng = np.random.default_rng(cfg.seed)
    generator = torch.Generator().manual_seed(cfg.seed)
    branches = ("T", "V", "G") if cfg.sid.use_graph else ("T", "V")
    x, edges, weight, full_adj = prepare_inputs(data, graph, device)
    n = data.n_items
    model = build_model(input_dims(x), tcfg, branches).to(device)
    with torch.no_grad():
        for m in branches:
            all_ids = torch.arange(n, device=device)
            z = torch.cat([encode(model, m, x, all_ids[i:i+8192], full_adj) for i in range(0, n, 8192)])
            model[m].rq.kmeans_init(z, generator)
    optimizer = torch.optim.AdamW(model.parameters(), lr=tcfg.lr, weight_decay=tcfg.weight_decay)
    labels_np = pseudo["labels"]
    labels = torch.as_tensor(labels_np, device=device)
    conf = torch.as_tensor(gates["C"], device=device)
    compat = {"T": torch.as_tensor(gates["M_T"], device=device), "V": torch.as_tensor(gates["M_V"], device=device)}
    tracker = LossEMA(n, branches, tcfg.levels, momentum=tcfg.ema_momentum, device=device)
    labeled_size = max(1, int(tcfg.batch_size * tcfg.labeled_fraction))
    uniform_size = max(1, tcfg.batch_size - labeled_size)
    has_labels = bool((labels_np[:, 0] >= 0).any()) and gcfg.mode != "none"
    fixed = tuple(gcfg.fixed_levels) if gcfg.mode == "fixed" else None
    log = (out / "train_log.jsonl").open("w")

    def labeled_batches():
        while True:
            yield from cluster_aware_batches(labels_np, labeled_size, per_fine=tcfg.per_fine,
                                             fines_per_local=tcfg.fines_per_local,
                                             locals_per_global=tcfg.locals_per_global, rng=rng)

    stream = labeled_batches() if has_labels else None
    for epoch in range(tcfg.epochs):
        lam = guidance_lambda(epoch, start=tcfg.warmup_start, ramp=tcfg.warmup_ramp, maximum=tcfg.lambda_max)
        perm = rng.permutation(n)
        sums = {}
        steps = math.ceil(n / uniform_size)
        for step in range(steps):
            parts = [perm[step*uniform_size:(step+1)*uniform_size]]
            if stream is not None:
                parts.append(next(stream))
            ids = torch.as_tensor(np.unique(np.concatenate(parts)), device=device)
            adjacency = full_adj
            if "G" in branches and tcfg.edge_drop:
                kept, w, _ = edge_drop(edges, weight, n, edge_p=tcfg.edge_p, node_p=tcfg.node_p)
                adjacency = torch_adjacency(kept[0], kept[1], w, n, device)
            loss, blocks, record = 0, {}, {}
            for m in branches:
                z = encode(model, m, x, ids, adjacency)
                zq, codes, cb_loss = model[m].rq(z)
                recon = F.mse_loss(model[m].decoder(zq), x[m][ids])
                loss = loss + recon + cb_loss
                record[f"recon_{m}"] = recon.item()
                record[f"codebook_{m}"] = cb_loss.item()
                if has_labels:
                    blocks[m] = st_softmax_blocks(z, model[m].rq.books(), tau_q=tcfg.tau_q, codes=codes)[0]
            if has_labels:
                guidance, info = hierarchy_guidance(
                    tracker, ids, labels[ids], blocks, conf[ids], {k: v[ids] for k, v in compat.items()},
                    temperature=tcfg.tau, components=tuple(gcfg.components), fixed_levels=fixed)
                loss = loss + lam * guidance
                record["guidance"] = guidance.item()
                valid = info["valid"]
                for level in range(tcfg.levels):
                    v = valid[:, level]
                    record[f"valid_frac_l{level+1}"] = float(v.float().mean())
                    for m in branches:
                        if v.any():
                            record[f"w_{m}_l{level+1}"] = float(info["weights"][m][v, level].mean())
                            record[f"loss_{m}_l{level+1}"] = float(info["loss"][m][v, level].mean())
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            for key, value in record.items():
                sums[key] = sums.get(key, 0.0) + value
        entry = dict(epoch=epoch+1, lam=lam, **{k: v/steps for k, v in sums.items()})
        if (epoch+1) % tcfg.log_every == 0 or epoch+1 == tcfg.epochs:
            entry["codes"] = code_stats(encode_codes(model, x, full_adj, n))
            print(f"[stage4] epoch {epoch+1}/{tcfg.epochs} lam={lam:.3f} "
                  + " ".join(f"{k}={entry[k]:.4f}" for k in sorted(entry) if k.startswith(("recon", "guidance")))
                  + f" joint_unique={entry['codes']['joint_unique']:.3f}", flush=True)
        log.write(json.dumps(entry) + "\n")
        log.flush()
    log.close()
    torch.save(dict(state=model.state_dict(), branches=branches), out / "tokenizer.pt")
    stats = code_stats(encode_codes(model, x, full_adj, n))
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    return stats
