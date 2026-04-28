"""
Compare loss curves for plain ReLU vs squared ReLU.
Same architecture, same init seed, same Adam, same data.
Only the MLP activation differs.
"""

import os
import math
import json
import random
import matplotlib.pyplot as plt


if not os.path.exists("input.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")


class Value:
    __slots__ = ("data", "grad", "children", "local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.children = children
        self.local_grads = local_grads
        self.grad = 0

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), ())
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), ())
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def __truediv__(self, other):
        return self * (other**-1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        v = math.exp(self.data)
        return Value(v, (self,), (v,))

    def relu(self):
        return (
            Value(self.data, (self,), (1,))
            if self.data > 0
            else Value(0, (self,), (0,))
        )

    def backward(self):
        topo, visited = [], set()

        def build(v):
            if v not in visited:
                visited.add(v)
                for c in v.children:
                    build(c)
                topo.append(v)

        build(self)
        self.grad = 1
        for v in reversed(topo):
            for c, lg in zip(v.children, v.local_grads):
                c.grad += v.grad * lg


# Smaller model for a fast comparison run
n_embd = 16
block_size = 8
n_head = 4
n_layer = 1
head_dim = n_embd // n_head
num_steps = 300


def train(use_squared_relu, label):
    """Train the model and return loss history."""
    # Reset RNG so both runs start from identical init and see identical data order
    random.seed(42)

    docs = [l.strip() for l in open("input.txt").read().strip().split("\n") if l.strip()]
    random.shuffle(docs)

    uchars = sorted(set("".join(docs)))
    vocab_size = len(uchars) + 1
    BOS = len(uchars)

    matrix = lambda nout, nin: [
        [Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)
    ]

    state_dict = {
        "wte": matrix(vocab_size, n_embd),
        "wpe": matrix(block_size, n_embd),
        "lm_head": matrix(vocab_size, n_embd),
    }
    for l in range(n_layer):
        state_dict[f"attn_wq_{l}"] = matrix(n_embd, n_embd)
        state_dict[f"attn_wk_{l}"] = matrix(n_embd, n_embd)
        state_dict[f"attn_wv_{l}"] = matrix(n_embd, n_embd)
        state_dict[f"attn_wo_{l}"] = matrix(n_embd, n_embd)
        state_dict[f"mlp_fc1_{l}"] = matrix(4 * n_embd, n_embd)
        state_dict[f"mlp_fc2_{l}"] = matrix(n_embd, 4 * n_embd)
    params = [p for m in state_dict.values() for v in m for p in v]

    def linear(x, m):
        return [sum(xi * mi for xi, mi in zip(x, mi)) for mi in m]

    def rmsnorm(x):
        m = sum([v * v for v in x]) / len(x)
        ms = (m + 1e-5) ** 0.5
        return [xi / ms for xi in x]

    def softmax(logits):
        l_max = max([l.data for l in logits])
        l_e = [(l - l_max).exp() for l in logits]
        total = sum(l_e)
        return [l / total for l in l_e]

    def gpt(token_id, pos_id, keys, values):
        token = state_dict["wte"][token_id]
        pos = state_dict["wpe"][pos_id]
        x = [t + p for t, p in zip(token, pos)]

        for l in range(n_layer):
            x_residual = x
            x = rmsnorm(x)
            q = linear(x, state_dict[f"attn_wq_{l}"])
            k = linear(x, state_dict[f"attn_wk_{l}"])
            v = linear(x, state_dict[f"attn_wv_{l}"])
            keys[l].append(k)
            values[l].append(v)

            attn = [[0 for _ in range(head_dim)] for _ in range(n_head)]
            for head in range(n_head):
                hs = head * head_dim
                he = (head + 1) * head_dim
                score_scale = math.sqrt(head_dim)
                attn_scores = [
                    sum(q_i * k_i for q_i, k_i in zip(q[hs:he], key[hs:he])) / score_scale
                    for key in keys[l]
                ]
                attn_weights = softmax(attn_scores)
                for weight, value_vect in zip(attn_weights, values[l]):
                    for i, param in enumerate(value_vect[hs:he]):
                        attn[head][i] = attn[head][i] + param * weight

            attn_concat = [p for head in attn for p in head]
            attn_output = linear(attn_concat, state_dict[f"attn_wo_{l}"])
            x = [res + ao for res, ao in zip(x_residual, attn_output)]

            x_residual = x
            x = rmsnorm(x)
            pre = linear(x, state_dict[f"mlp_fc1_{l}"])
            # THE ONLY DIFFERENCE BETWEEN THE TWO RUNS
            if use_squared_relu:
                horizontal = [h.relu() ** 2 for h in pre]
            else:
                horizontal = [h.relu() for h in pre]
            mlp = linear(horizontal, state_dict[f"mlp_fc2_{l}"])
            x = [res + mo for res, mo in zip(x_residual, mlp)]

        x = rmsnorm(x)
        return linear(x, state_dict["lm_head"])

    # Adam
    lr0, beta1, beta2, eps = 1e-2, 0.9, 0.95, 1e-8
    m1 = [0.0] * len(params)
    m2 = [0.0] * len(params)

    losses = []
    for step in range(num_steps):
        doc = docs[step % len(docs)]
        tokens = [BOS] + [uchars.index(t) for t in doc] + [BOS]
        n = min(block_size, len(tokens) - 1)

        keys = [[] for _ in range(n_layer)]
        values = [[] for _ in range(n_layer)]
        step_losses = []
        for i in range(n):
            token_id, target_id = tokens[i], tokens[i + 1]
            logits = gpt(token_id, i, keys, values)
            probs = softmax(logits)
            step_losses.append(-probs[target_id].log())
        loss = sum(step_losses) / len(step_losses)
        loss.backward()

        lr = lr0 * (1 - step / num_steps)
        for i, p in enumerate(params):
            m1[i] = beta1 * m1[i] + (1 - beta1) * p.grad
            m2[i] = beta2 * m2[i] + (1 - beta2) * p.grad ** 2
            m1_hat = m1[i] / (1 - beta1 ** (step + 1))
            m2_hat = m2[i] / (1 - beta2 ** (step + 1))
            p.data -= lr * m1_hat / (m2_hat ** 0.5 + eps)
            p.grad = 0

        losses.append(loss.data)
        if step % 25 == 0 or step == num_steps - 1:
            print(f"[{label}] step {step+1:4d}/{num_steps} | loss {loss.data:.4f}")

    return losses


print("=== Run 1: plain ReLU ===")
losses_relu = train(use_squared_relu=False, label="ReLU ")

print("\n=== Run 2: squared ReLU ===")
losses_relu2 = train(use_squared_relu=True, label="ReLU2")

# Save raw loss arrays so we can re-plot without re-training
with open("revision/relu_comparison_losses.json", "w") as f:
    json.dump({"relu": losses_relu, "relu2": losses_relu2}, f)


def moving_avg(xs, k=20):
    """Simple trailing moving average of window size k."""
    out = []
    for i in range(len(xs)):
        lo = max(0, i - k + 1)
        out.append(sum(xs[lo : i + 1]) / (i - lo + 1))
    return out


smooth_relu = moving_avg(losses_relu, k=20)
smooth_relu2 = moving_avg(losses_relu2, k=20)

# Two-panel plot: raw on top, smoothed below
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

ax1.plot(losses_relu, label="plain ReLU", color="tab:blue", alpha=0.5)
ax1.plot(losses_relu2, label="squared ReLU (x²)", color="tab:orange", alpha=0.5)
ax1.set_ylabel("loss (raw, per step)")
ax1.set_title(
    f"Loss curve: plain ReLU vs squared ReLU\n"
    f"(same seed, same init, same Adam, only activation differs)"
)
ax1.legend()
ax1.grid(True, alpha=0.3)

ax2.plot(smooth_relu, label="plain ReLU (smoothed)", color="tab:blue", linewidth=2)
ax2.plot(smooth_relu2, label="squared ReLU (smoothed)", color="tab:orange", linewidth=2)
ax2.set_xlabel("training step")
ax2.set_ylabel("loss (20-step moving avg)")
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
out_path = "revision/relu_comparison.png"
plt.savefig(out_path, dpi=120)
print(f"\nSaved plot to {out_path}")
print(f"Saved raw data to revision/relu_comparison_losses.json")
print(f"Final losses (raw):       ReLU={losses_relu[-1]:.4f}  ReLU²={losses_relu2[-1]:.4f}")
print(f"Final losses (smoothed):  ReLU={smooth_relu[-1]:.4f}  ReLU²={smooth_relu2[-1]:.4f}")
