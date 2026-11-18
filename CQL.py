import torch, torch.nn as nn, torch.nn.functional as F

class QNet(nn.Module):
    def __init__(self, d_in, n_actions=2, h=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_in, h), nn.ReLU(),
            nn.Linear(h, h), nn.ReLU(),
            nn.Linear(h, n_actions),
        )
    def forward(self, x):
        return self.net(x)

def train_cql(data, n_actions=2, alpha=1.0, gamma=0.99, lr=3e-4,
              steps=50_000, batch=512, tau=0.005, reward_scale=28.0):
    # data: dict met s, a, r, s2, done (numpy); eerst op trainvisits normaliseren
    S  = torch.tensor(data["s"],  dtype=torch.float32)
    A  = torch.tensor(data["a"],  dtype=torch.long)
    R  = torch.tensor(data["r"],  dtype=torch.float32) * reward_scale  # return ~ VFD in dagen
    S2 = torch.tensor(data["s2"], dtype=torch.float32)
    Dn = torch.tensor(data["done"], dtype=torch.float32)

    q, q_tgt = QNet(S.shape[1], n_actions), QNet(S.shape[1], n_actions)
    q_tgt.load_state_dict(q.state_dict())
    opt = torch.optim.Adam(q.parameters(), lr=lr)

    for step in range(steps):
        i = torch.randint(0, len(S), (batch,))
        s, a, r, s2, d = S[i], A[i], R[i], S2[i], Dn[i]

        qs = q(s)
        q_sa = qs.gather(1, a[:, None]).squeeze(1)
        with torch.no_grad():
            target = r + gamma * (1 - d) * q_tgt(s2).max(1).values

        td  = F.mse_loss(q_sa, target)
        cql = (torch.logsumexp(qs, dim=1) - q_sa).mean()   # conservatieve straf
        loss = td + alpha * cql

        opt.zero_grad(); loss.backward(); opt.step()
        with torch.no_grad():                               # zachte target-update
            for p, pt in zip(q.parameters(), q_tgt.parameters()):
                pt.mul_(1 - tau).add_(tau * p)
    return q

# Beleid: a* = argmax_a Q(s, a)
# policy = lambda s: q(torch.tensor(s, dtype=torch.float32)).argmax(-1)