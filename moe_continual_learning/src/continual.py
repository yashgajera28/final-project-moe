"""Continual learning training strategies and evaluation metrics.

Strategies:

* ``NaiveTrainer``: trains sequentially on one task after another without
  any protection against forgetting (standard fine tuning).
* ``EWCTrainer``: Elastic Weight Consolidation. After finishing a task,
  the diagonal Fisher information of the parameters is estimated and a
  quadratic penalty pulls important weights back towards their old values
  while learning later tasks.
* ``MoETrainer``: trains a MoENet sequentially with an additional
  load balancing loss on the router.

Metrics:
The accuracy matrix R (entry R[i, j] = accuracy on task j after training on
task i) is the basis for average accuracy, average forgetting and backward
transfer (BWT).
"""

import numpy as np
import torch
import torch.nn.functional as F


# Evaluation helpers

@torch.no_grad()
def evaluate(model, loader, task_id, n_classes, device):
    """Accuracy of a model on one task's test set."""
    model.eval()
    correct, total = 0, 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        logits = model(x, task_id=task_id, n_classes=n_classes)
        correct += int((logits.argmax(dim=-1) == y).sum())
        total += len(y)
    return correct / total


def accuracy_matrix_at_end(accs):
    """Computes the standard continual learning metrics.

    ``accs`` is a list of lists; accs[i][j] is the accuracy on task j after
    training task i (j <= i).
    """
    n = len(accs)
    R = np.full((n, n), np.nan)
    for i in range(n):
        for j in range(i + 1):
            R[i, j] = accs[i][j]

    avg_acc = float(np.nanmean(R[-1, :]))
    # Forgetting of task j: best previous accuracy minus final accuracy.
    forgetting = []
    for j in range(n - 1):
        best = np.nanmax(R[:-1, j])
        forgetting.append(best - R[-1, j])
    avg_forgetting = float(np.mean(forgetting))
    bwt = float(np.nanmean(R[-1, :n - 1] - np.diag(R)[:n - 1]))
    return R, avg_acc, avg_forgetting, bwt


# Trainers

class NaiveTrainer:
    """Sequential fine-tuning without any continual learning mechanism."""

    name = "Naive MLP"

    def __init__(self, model, lr=1e-3, epochs=5, device="cpu"):
        self.model = model.to(device)
        self.lr = lr
        self.epochs = epochs
        self.device = device
        self.accs = []          # per-stage accuracy rows

    def _loss(self, logits, y):
        return F.cross_entropy(logits, y)

    def train_task(self, task_id, train_loader, n_classes):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits = self.model(x, task_id=task_id, n_classes=n_classes)
                loss = self._loss(logits, y)
                opt.zero_grad()
                loss.backward()
                opt.step()

    def run(self, tasks):
        """Trains on all tasks sequentially, evaluating after every stage."""
        for t, (tr, _, n_classes) in enumerate(tasks):
            self.train_task(t, tr, n_classes)
            row = [evaluate(self.model, tasks[j][1], j, tasks[j][2], self.device)
                   for j in range(t + 1)]
            self.accs.append(row)
        return self.accs


class EWCTrainer(NaiveTrainer):
    """Elastic Weight Consolidation (Kirkpatrick et al., 2017)."""

    name = "MLP + EWC"

    def __init__(self, model, lr=1e-3, epochs=5, device="cpu", ewc_lambda=500.0):
        super().__init__(model, lr, epochs, device)
        self.ewc_lambda = ewc_lambda
        self._fisher = None     # diagonal Fisher per parameter
        self._star = None       # parameter values after the previous task

    def _loss(self, logits, y):
        loss = F.cross_entropy(logits, y)
        if self._fisher is not None:
            penalty = 0.0
            for p, f, s in zip(self.model.parameters(), self._fisher, self._star):
                penalty = penalty + (f * (p - s) ** 2).sum()
            loss = loss + 0.5 * self.ewc_lambda * penalty
        return loss

    @torch.no_grad()
    def _copy_params(self):
        return [p.clone() for p in self.model.parameters()]

    def _estimate_fisher(self, loader, task_id, n_classes, n_batches=50):
        """Empirical diagonal Fisher over (at most) ``n_batches`` batches."""
        fisher = [torch.zeros_like(p) for p in self.model.parameters()]
        self.model.eval()
        for i, (x, y) in enumerate(loader):
            if i >= n_batches:
                break
            x, y = x.to(self.device), y.to(self.device)
            self.model.zero_grad()
            logits = self.model(x, task_id=task_id, n_classes=n_classes)
            F.cross_entropy(logits, y).backward()
            for f, p in zip(fisher, self.model.parameters()):
                # Heads of other tasks do not participate in this task's
                # computation graph, so their gradient stays None.
                if p.grad is not None:
                    f += p.grad.detach() ** 2
        fisher = [f / min(n_batches, len(loader)) for f in fisher]
        if self._fisher is None:
            self._fisher = fisher
        else:  # accumulate penalties over tasks (standard practice)
            self._fisher = [fold + fnew for fold, fnew in zip(self._fisher, fisher)]

    def train_task(self, task_id, train_loader, n_classes):
        super().train_task(task_id, train_loader, n_classes)
        self._estimate_fisher(train_loader, task_id, n_classes)
        self._star = self._copy_params()


class MoETrainer(NaiveTrainer):
    """Sequential training of a MoENet with a load-balancing auxiliary loss."""

    name = "MoE"

    def __init__(self, model, lr=1e-3, epochs=5, device="cpu", lb_coeff=0.01):
        super().__init__(model, lr, epochs, device)
        self.lb_coeff = lb_coeff

    def train_task(self, task_id, train_loader, n_classes):
        self.model.train()
        opt = torch.optim.Adam(self.model.parameters(), lr=self.lr)
        for _ in range(self.epochs):
            for x, y in train_loader:
                x, y = x.to(self.device), y.to(self.device)
                logits, gates = self.model(x, task_id=task_id, n_classes=n_classes,
                                           return_gates=True)
                loss = (F.cross_entropy(logits, y)
                        + self.lb_coeff * self.model.moe.load_balance_loss(gates))
                opt.zero_grad()
                loss.backward()
                opt.step()

    @torch.no_grad()
    def router_profile(self, tasks):
        """Mean gate probability of every expert for every task's test data.

        Returns a [n_tasks, n_experts] matrix, the central artefact for the
        router analysis in the report.
        """
        self.model.eval()
        n_experts = self.model.moe.n_experts
        profile = np.zeros((len(tasks), n_experts))
        for t, (_, te, n_classes) in enumerate(tasks):
            total = np.zeros(n_experts)
            count = 0
            for x, _ in te:
                x = x.to(self.device)
                _, gates = self.model(x, task_id=t, n_classes=n_classes,
                                      return_gates=True)
                total += gates.sum(dim=0).cpu().numpy()
                count += len(x)
            profile[t] = total / count
        return profile
