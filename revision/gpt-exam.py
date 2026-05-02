import math  # math.log, math.exp
import random  # random.seed, random.choices, random.gauss, random.shuffle

random.seed(42)

docs = [line.strip() for line in open("input.txt") if line.strip()]
random.shuffle(docs)

uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1
BOS = len(uchars)


class Value:
    __slots__ = ("data", "grad", "children", "local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.children = children
        self.local_grads = local_grads
        self.grad = 0

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * self.data ** (other - 1),))

    def __truediv__(self, other):
        return self * other**-1

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + -other

    def exp(self):
        data = math.exp(self.data)
        return Value(data, (self,), (data,))

    def relu(self):
        return (
            Value(self.data, (self,), (1,))
            if self.data > 0
            else Value(0, (self,), (0,))
        )

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

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


n_embd = 16
block_size = 16
n_layer = 2
n_head = 4
head_dim = n_embd // n_head
matrix = lambda n_row, n_column, std=0.08: [
    [Value(random.gauss(0, std)) for _ in range(n_column)] for _ in range(n_row)
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

params = [v for m in state_dict.values() for r in m for v in r]


def linear(x, matrix):
    return [sum([xi * pi for xi, pi in zip(x, mr)]) for mr in matrix]


def rmsnorm(x):
    ms = sum([xi * xi for xi in x]) / len(x)
    m = (ms + 1e-5) ** 0.5
    return [xi / m for xi in x]


def softmax(logits):
    l_max = max([l.data for l in logits])
    exps = [(l - l_max).exp() for l in logits]
    total = sum(exps)
    return [e / total for e in exps]


def gpt(token_id, pos_id, keys, values):
    t_embd = state_dict["wte"][token_id]
    p_embd = state_dict["wpe"][pos_id]
    x = [ti + pi for ti, pi in zip(t_embd, p_embd)]

    for l in range(n_layer):
        # attention
        x_residual = x
        x = rmsnorm(x)

        q = linear(x, state_dict[f"attn_wq_{l}"])
        k = linear(x, state_dict[f"attn_wk_{l}"])
        v = linear(x, state_dict[f"attn_wv_{l}"])

        keys[l].append(k)
        values[l].append(v)
        attn = []

        for h in range(n_head):
            hs = h * head_dim
            he = (h + 1) * head_dim

            lookup = [
                sum(qi * ki for qi, ki in zip(q[hs:he], k[hs:he])) / head_dim**0.5
                for k in keys[l]
            ]
            attn_weights = softmax(lookup)

            payload = [0 for _ in range(head_dim)]
            for i, v_vector in enumerate(values[l]):
                weighted = [vi * attn_weights[i] for vi in v_vector[hs:he]]
                payload = [pi + wi for pi, wi in zip(payload, weighted)]
            attn.extend(payload)

        attn_out = linear(attn, state_dict[f"attn_wo_{l}"])
        x = [xi + aoi for xi, aoi in zip(x_residual, attn_out)]

        # mlp
        x_residual = x
        x = rmsnorm(x)

        h_pre = linear(x, state_dict[f"mlp_fc1_{l}"])
        h = [li.relu() for li in h_pre]
        mlp = linear(h, state_dict[f"mlp_fc2_{l}"])
        x = [xi + mlpi for xi, mlpi in zip(x_residual, mlp)]

    x = rmsnorm(x)
    logits = linear(x, state_dict["lm_head"])
    return logits


m1_params, m2_params = [0 for _ in range(len(params))], [0 for _ in range(len(params))]
learning_rate, beta1, beta2, adam_eps = 0.01, 0.85, 0.99, 1e-8
steps = 1000
for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    losses = []
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for i in range(n):
        token_id, target_id = tokens[i], tokens[i + 1]
        logits = gpt(token_id, i, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = sum(losses) / len(losses)

    loss.backward()

    lr = learning_rate * (1 - step / steps)
    for i, p in enumerate(params):
        m1_params[i] = m1_params[i] * beta1 + (1 - beta1) * p.grad
        m2_params[i] = m2_params[i] * beta2 + (1 - beta2) * p.grad**2
        m1_hat = m1_params[i] / (1 - beta1 ** (step + 1))
        m2_hat = m2_params[i] / (1 - beta2 ** (step + 1))

        p.data -= lr * m1_hat / (m2_hat**0.5 + adam_eps)
        p.grad = 0

    if step % 50 == 0 or step == steps - 1 or step <= 5:
        print(f"step {step+1:4d}/{steps} | loss {loss.data:.4f}")


temperature = 0.5
steps = 20
for s in range(steps):
    token_id = BOS
    output = []
    keys, values = [[] for _ in range(n_layer)], [[] for _ in range(n_layer)]
    for i in range(block_size):
        logits = gpt(token_id, i, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        output.append(uchars[token_id])
    print(f"sample {s+1:2d}: {''.join(output)}")
