import os  # os.path.exists
import math  # math.log, math.exp
import random  # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42)

if not os.path.exists("input.txt"):
    import urllib.request

    names_url = "https://raw.githubusercontent.com/karpathy/makemore/refs/heads/master/names.txt"
    urllib.request.urlretrieve(names_url, "input.txt")
docs = [l.strip() for l in open("input.txt").read().strip().split("\n") if l.strip()]
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1
BOS = len(uchars)


class Value:
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

    def __rtruediv__(self, other):
        return (self**-1) * other

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def exp(self):
        value = math.exp(self.data)
        return Value(value, (self,), (value,))

    def relu(self):
        return (
            Value(self.data, (self,), (1,))
            if self.data > 0
            else Value(0, (self,), (0,))
        )

    def backward(self):
        topo = []
        visited = set()

        def buildTopo(value):
            if value not in visited:
                visited.add(value)
                for c in value.children:
                    buildTopo(c)
                topo.append(value)

        buildTopo(self)
        self.grad = 1

        for v in reversed(topo):
            for c, lg in zip(v.children, v.local_grads):
                c.grad += v.grad * lg


n_embd = 8
block_size = 8
n_head = 4
n_layer = 2
head_dim = n_embd // n_head
matrix = lambda nout, nin, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(nin)] for _ in range(nout)
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
    state_dict[f"attn_wo_{l}"] = matrix(n_embd, n_embd, 0)
    state_dict[f"mlp_fc1_{l}"] = matrix(4 * n_embd, n_embd)
    state_dict[f"mlp_fc2_{l}"] = matrix(n_embd, 4 * n_embd, 0)

params = [p for matrix in state_dict.values() for vector in matrix for p in vector]


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
            he = hs + head_dim

            # Compute attention scores: q_h @ k_h / sqrt(head_dim) for each key
            score_scale = math.sqrt(head_dim)
            attn_scores = [
                sum(q_i * k_i for q_i, k_i in zip(q[hs:he], key[hs:he])) / score_scale
                for key in keys[l]
            ]

            # Apply softmax to get attention weights
            attn_weights = softmax(attn_scores)

            # Compute attention
            for weight, value_vect in zip(attn_weights, values[l]):
                for i, param in enumerate(value_vect[hs:he]):
                    attn[head][i] = attn[head][i] + param * weight

        # Output projection and residual connection
        attn_concat = [p for head in attn for p in head]
        attn_output = linear(attn_concat, state_dict[f"attn_wo_{l}"])
        x = [res + attn_out for res, attn_out in zip(x_residual, attn_output)]

        # mlp
        x_residual = x
        x = rmsnorm(x)
        horizontal_pre = linear(x, state_dict[f"mlp_fc1_{l}"])
        horizontal = [l_pre.relu() for l_pre in horizontal_pre]
        mlp = linear(horizontal, state_dict[f"mlp_fc2_{l}"])
        x = [res + mlp_out for res, mlp_out in zip(x_residual, mlp)]

    x = rmsnorm(x)
    logits = linear(x, state_dict["lm_head"])

    return logits


default_lr, beta1, beta2, adam_eps = 1e-2, 0.9, 0.95, 1e-8
m, v = [0.0] * len(params), [0.0] * len(params)
steps = 1000

for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(t) for t in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    losses = []
    for i in range(n):
        token_id, target_id = tokens[i], tokens[i + 1]
        logits = gpt(token_id, i, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = sum(losses) / len(losses)

    loss.backward()

    # SGD with Adam - REITERATE
    lr = default_lr * (1 - step / steps)
    for i, p in enumerate(params):
        m[i] = beta1 * m[i] + (1 - beta1) * p.grad
        v[i] = beta2 * v[i] + (1 - beta2) * p.grad**2
        m_hat = m[i] / (1 - beta1 ** (step + 1))
        v_hat = v[i] / (1 - beta2 ** (step + 1))
        p.data -= lr * m_hat / (v_hat**0.5 + adam_eps)
        p.grad = 0

    if step < 5 or step % 200 == 0:
        print(f"step {step+1:4d} / {steps:4d} | loss {loss.data:.4f}")

temperature = 0.5
steps = 20
for step in range(steps):
    token = BOS
    output = []

    keys = [[] for _ in range(n_layer)]
    values = [[] for _ in range(n_layer)]
    for n in range(block_size):
        logits = gpt(token, n, keys, values)
        probs = softmax([l / temperature for l in logits])
        token = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token == BOS:
            break
        output.append(uchars[token])

    print(f"sample {step+1:2d}: {''.join(output)}")
