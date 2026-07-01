import torch
from torch.optim import Optimizer

def project_orthogonal(G, steps=5):
    """
    Orthogonalize a 2D matrix G using Newton-Schulz iteration.
    """
    transposed = False
    if G.shape[0] < G.shape[1]:
        G = G.t()
        transposed = True

    # Compute the L2 operator norm (maximum singular value)
    norm = torch.linalg.norm(G, ord=2) + 1e-7
    X = G / norm

    # Newton-Schulz iteration
    for _ in range(steps):
        XT = X.t()
        # Compute X @ (XT @ X) efficiently
        X = 1.5 * X - 0.5 * X @ (XT @ X)

    if transposed:
        X = X.t()

    return X


class MuSGD(Optimizer):
    """MuSGD optimizer (SGD + Muon hybrid) for stable and quick convergence."""

    def __init__(self, params, lr=1e-3, momentum=0.9, weight_decay=0.0, nesterov=False):
        if lr < 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if momentum < 0.0:
            raise ValueError(f"Invalid momentum value: {momentum}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")

        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, nesterov=nesterov)
        super().__init__(params, defaults)

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            lr = group['lr']
            momentum = group['momentum']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad

                # Apply weight decay directly
                if weight_decay != 0:
                    p.add_(p, alpha=-lr * weight_decay)

                # Determine if 2D or higher (Muon orthogonalization applies to weights, not bias/norms)
                if p.ndim >= 2:
                    # Orthogonalize gradient using Newton-Schulz
                    orig_shape = grad.shape
                    grad_2d = grad.view(orig_shape[0], -1)
                    d_p = project_orthogonal(grad_2d).view(orig_shape)
                else:
                    d_p = grad

                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(d_p, alpha=1.0)

                    if nesterov:
                        d_p = d_p.add(buf, alpha=momentum)
                    else:
                        d_p = buf

                p.add_(d_p, alpha=-lr)

        return loss
