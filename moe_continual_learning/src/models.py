"""Model definitions for the continual learning experiments.

Three families of models are defined:

* ``BaselineMLP``: a plain multi-layer perceptron (naive fine-tuning and
  EWC baselines use this architecture).
* ``WideMLP``: a wider MLP whose parameter count roughly matches the MoE
  model, used to separate the effect of additional capacity from the
  effect of the MoE structure.
* ``MoENet``: a Mixture of Experts network with a shared trunk followed
  by a MoE layer with a top-k gating ("router") network.

All models support multiple output heads so that SplitMNIST can be evaluated
in the task-incremental protocol.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiHeadMixin:
    """Adds lazily created per-task output heads to a network."""

    def _init_heads(self):
        self.heads = nn.ModuleDict()

    def _head_for(self, task_id, n_classes, device):
        key = str(task_id)
        if key not in self.heads:
            head = nn.Linear(self.feature_dim, n_classes).to(device)
            self.heads[key] = head
        return self.heads[key]


class BaselineMLP(nn.Module, MultiHeadMixin):
    """Plain two-hidden-layer perceptron used as the naive baseline."""

    def __init__(self, input_dim=784, hidden_dim=400):
        super().__init__()
        self.feature_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_heads()

    def features(self, x):
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))

    def forward(self, x, task_id=0, n_classes=10):
        head = self._head_for(task_id, n_classes, x.device)
        return head(self.features(x))


class WideMLP(nn.Module, MultiHeadMixin):
    """Wider MLP with roughly as many parameters as the MoE model."""

    def __init__(self, input_dim=784, hidden_dim=1100):
        super().__init__()
        self.feature_dim = hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self._init_heads()

    def features(self, x):
        h = F.relu(self.fc1(x))
        return F.relu(self.fc2(h))

    def forward(self, x, task_id=0, n_classes=10):
        head = self._head_for(task_id, n_classes, x.device)
        return head(self.features(x))


class MoELayer(nn.Module):
    """Mixture-of-Experts layer with top-k gating.

    The *router* (gating network) produces a probability distribution over
    the ``n_experts`` experts for every input sample.  Only the ``top_k``
    experts with the highest gate values are evaluated and their outputs are
    combined with the (re-normalised) gate values as weights.  An auxiliary
    load-balancing loss encourages the router to use all experts uniformly
    over a batch, which prevents the router from collapsing onto a single
    expert early in training.
    """

    def __init__(self, dim, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, dim), nn.ReLU()) for _ in range(n_experts)]
        )
        self.router = nn.Linear(dim, n_experts)

    def forward(self, x):
        """Returns (mixed expert output, full gate probability matrix)."""
        logits = self.router(x)                       # [B, E]
        gates = F.softmax(logits, dim=-1)             # [B, E]
        topv, topi = gates.topk(self.top_k, dim=-1)   # [B, k]
        topv = topv / topv.sum(dim=-1, keepdim=True)  # re-normalise

        out = torch.zeros_like(x)
        for slot in range(self.top_k):
            idx = topi[:, slot]                       # expert id per sample
            w = topv[:, slot].unsqueeze(1)            # mixing weight
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out, gates

    def load_balance_loss(self, gates):
        """Switch-style auxiliary loss: n_experts * sum(f_e * P_e).

        f_e = fraction of samples routed to expert e (top-1 assignment),
        P_e = mean gate probability of expert e over the batch.
        """
        f = torch.zeros(self.n_experts, device=gates.device)
        top1 = gates.argmax(dim=-1)
        for e in range(self.n_experts):
            f[e] = (top1 == e).float().mean()
        P = gates.mean(dim=0)
        return self.n_experts * (f * P).sum()


class MoENet(nn.Module, MultiHeadMixin):
    """MoE network: shared trunk, one MoE layer, per-task output heads."""

    def __init__(self, input_dim=784, hidden_dim=400, n_experts=8, top_k=2):
        super().__init__()
        self.feature_dim = hidden_dim
        self.trunk = nn.Linear(input_dim, hidden_dim)
        self.moe = MoELayer(hidden_dim, n_experts=n_experts, top_k=top_k)
        self._init_heads()

    def features(self, x):
        h = F.relu(self.trunk(x))
        h, gates = self.moe(h)
        return h, gates

    def forward(self, x, task_id=0, n_classes=10, return_gates=False):
        head = self._head_for(task_id, n_classes, x.device)
        h, gates = self.features(x)
        logits = head(h)
        if return_gates:
            return logits, gates
        return logits


class MoEInputLayer(nn.Module):
    """MoE layer that routes the *raw input* (not a hidden representation).

    Identical gating mechanics to ``MoELayer``, but the router and the
    experts operate directly on the input features.  Designed for
    domain-incremental benchmarks such as PermutedMNIST, where the
    task-specific information lives in the input mapping itself and a
    shared input trunk would otherwise be overwritten by every new task.
    """

    def __init__(self, in_dim=784, out_dim=256, n_experts=8, top_k=2):
        super().__init__()
        self.n_experts = n_experts
        self.top_k = top_k
        self.experts = nn.ModuleList(
            [nn.Sequential(nn.Linear(in_dim, out_dim), nn.ReLU())
             for _ in range(n_experts)]
        )
        self.router = nn.Linear(in_dim, n_experts)
        self.out_dim = out_dim

    def forward(self, x):
        """Returns (mixed expert output, full gate probability matrix)."""
        logits = self.router(x)
        gates = F.softmax(logits, dim=-1)
        topv, topi = gates.topk(self.top_k, dim=-1)
        topv = topv / topv.sum(dim=-1, keepdim=True)

        out = torch.zeros(x.size(0), self.out_dim, device=x.device)
        for slot in range(self.top_k):
            idx = topi[:, slot]
            w = topv[:, slot].unsqueeze(1)
            for e in range(self.n_experts):
                mask = idx == e
                if mask.any():
                    out[mask] += w[mask] * self.experts[e](x[mask])
        return out, gates

    def load_balance_loss(self, gates):
        """Same Switch-style auxiliary loss as in ``MoELayer``."""
        f = torch.zeros(self.n_experts, device=gates.device)
        top1 = gates.argmax(dim=-1)
        for e in range(self.n_experts):
            f[e] = (top1 == e).float().mean()
        P = gates.mean(dim=0)
        return self.n_experts * (f * P).sum()


class MoEInputNet(nn.Module, MultiHeadMixin):
    """MoE variant with input-level routing (ablation/comparison model).

    Structure: input-level MoE (8 experts, top-2) -> shared hidden layer ->
    per-task heads.  Roughly matches the parameter count of ``MoENet``.
    """

    def __init__(self, input_dim=784, expert_dim=256, hidden_dim=256,
                 n_experts=8, top_k=2):
        super().__init__()
        self.feature_dim = hidden_dim
        self.moe = MoEInputLayer(input_dim, expert_dim, n_experts, top_k)
        self.fc = nn.Linear(expert_dim, hidden_dim)
        self._init_heads()

    def forward(self, x, task_id=0, n_classes=10, return_gates=False):
        head = self._head_for(task_id, n_classes, x.device)
        h, gates = self.moe(x)
        h = F.relu(self.fc(h))
        logits = head(h)
        if return_gates:
            return logits, gates
        return logits


def count_parameters(model):
    """Total number of trainable parameters of a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
