import math
import numpy as np
import torch
from torch.optim import Optimizer


def get_batch(dataset, batch_size, context_length, device):
    n = len(dataset)
    max_start = n - context_length - 1

    starts = np.random.randint(0, max_start + 1, size=batch_size)

    x_batch = np.empty((batch_size, context_length), dtype=np.int64)
    y_batch = np.empty((batch_size, context_length), dtype=np.int64)

    for i, st in enumerate(starts):
        x_batch[i] = dataset[st : st + context_length]
        y_batch[i] = dataset[st + 1 : st + context_length + 1]

    x = torch.from_numpy(x_batch).to(device=device)
    y = torch.from_numpy(y_batch).to(device=device)

    return x, y


def cross_entropy(inputs, targets):
    logsumexp = torch.logsumexp(inputs, dim=-1, keepdim=True)
    log_probs = inputs - logsumexp

    batch_size = inputs.size(0)
    idx = torch.arange(batch_size, device=inputs.device)
    correct_logp = log_probs[idx, targets]

    loss = -correct_logp.mean()
    return loss


def gradient_clipping(parameters, max_l2_norm):
    grads = []
    for p in parameters:
        if p.grad is None or not p.requires_grad:
            continue
        grads.append(p.grad)

    if not grads:
        return

    total_norm_sq = 0.0
    for g in grads:
        total_norm_sq += g.detach().float().pow(2).sum().item()

    total_norm = math.sqrt(total_norm_sq)

    if total_norm == 0 or total_norm <= max_l2_norm or max_l2_norm <= 0:
        return

    scale = max_l2_norm / total_norm

    for g in grads:
        g.mul_(scale)


class AdamW(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, weight_decay=0.0):
        if lr < 0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if eps < 0:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if weight_decay < 0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay)
        super().__init__(params, defaults)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group["lr"]
            betas = group["betas"]
            eps = group["eps"]
            weight_decay = group["weight_decay"]
            b1, b2 = betas

            for p in group["params"]:
                if p.grad is None:
                    continue
                grad = p.grad.detach()
                if grad.is_sparse:
                    raise RuntimeError("AdamW does not support sparse gradients")

                state = self.state[p]

                if len(state) == 0:
                    state["step"] = 0
                    state["exp_avg"] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state["exp_avg_sq"] = torch.zeros_like(p, memory_format=torch.preserve_format)

                exp_avg = state["exp_avg"]
                exp_avg_sq = state["exp_avg_sq"]
                step = state["step"] + 1
                state["step"] = step

                exp_avg.mul_(b1).add_(grad, alpha=1 - b1)
                exp_avg_sq.mul_(b2).addcmul_(grad, grad, value=1 - b2)

                bias_correction1 = 1 - b1 ** step
                bias_correction2 = 1 - b2 ** step
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)
                step_size = lr / bias_correction1

                if weight_decay != 0:
                    p.data.add_(p.data, alpha=-lr * weight_decay)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

        return loss


def get_lr_cosine_schedule(it, max_learning_rate, min_learning_rate, warmup_iters, cosine_cycle_iters):
    if it < warmup_iters:
        lr = max_learning_rate * it / warmup_iters
    elif it < cosine_cycle_iters:
        progress = (it - warmup_iters) / (cosine_cycle_iters - warmup_iters)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        lr = min_learning_rate + (max_learning_rate - min_learning_rate) * cosine
    else:
        lr = min_learning_rate

    return lr


def save_checkpoint(model, optimizer, iteration, out):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "iteration": iteration,
    }
    torch.save(checkpoint, out)


def load_checkpoint(src, model, optimizer):
    checkpoint = torch.load(src, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    return checkpoint["iteration"]
