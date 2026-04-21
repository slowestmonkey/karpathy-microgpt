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
BOS = len(uchars)
vocab_size = len(uchars) + 1


class Value:
    __slots__ = ("data", "grad", "children", "local_grads")

    def __init__(self, data, children=(), local_grads=()):
        self.data = data
        self.children = children
        self.local_grads = local_grads
        self.grad = 0

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), ())
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, (), ())
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * (self.data ** (other - 1)),))

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return (-self) + other

    def __rmul__(self, other):
        return self * other

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def relu(self):
        return (
            Value(self.data, (self,), (1,))
            if self.data > 0.0
            else Value(0.0, (self,), (0.0,))
        )

    def exp(self):
        data = math.exp(self.data)
        return Value(data, (self,), (data,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for c in value.children:
                    build_topo(c)
                topo.append(value)

        build_topo(self)
        self.grad = 1

        for v in reversed(topo):
            for c, local_grad in zip(v.children, v.local_grads):
                c.grad += v.grad * local_grad


n_embd = 16
block_size = 16
matrix = lambda nout, nin: [
    [Value(random.gauss(0, 0.08)) for _ in range(nin)] for _ in range(nout)
]
state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "wpe": matrix(block_size, n_embd),
    "attn_wq": matrix(n_embd, n_embd),
    "attn_wk": matrix(n_embd, n_embd),
    "attn_wv": matrix(n_embd, n_embd),
    "attn_wo": matrix(n_embd, n_embd),
    "mlp_fc1": matrix(4 * n_embd, n_embd),
    "mlp_fc2": matrix(n_embd, 4 * n_embd),
    "lm_head": matrix(vocab_size, n_embd),
}
params = [p for mat in state_dict.values() for row in mat for p in row]


def linear(x, m):
    return [sum(xv * rv for xv, rv in zip(x, row)) for row in m]


def rmsnorm(x):
    ms = sum([xi * xi for xi in x]) / len(x)
    scale = (ms + 1e-5) ** -0.5
    return [xi * scale for xi in x]


def softmax(logits):
    l_max = max([l.data for l in logits])
    softed = [(l - l_max).exp() for l in logits]
    total = sum(softed)
    return [sl / total for sl in softed]


def gpt(token_id, pos_id, keys, values):
    x = state_dict["wte"][token_id]
    p = state_dict["wpe"][pos_id]
    x = [xv + pv for xv, pv in zip(x, p)]

    # 1 head attention
    x = rmsnorm(x)
    x_residual = list(x)
    xq = linear(x, state_dict["attn_wq"])
    xk = linear(x, state_dict["attn_wk"])
    xv = linear(x, state_dict["attn_wv"])

    keys.append(xk)
    values.append(xv)

    attn_logits = [
        sum(q * k for q, k in zip(xq, key)) / math.sqrt(n_embd) for key in keys
    ]
    attn_weights = softmax(attn_logits)
    x_attn = [0.0] * n_embd
    for i in range(len(values)):
        payload = [attn_weights[i] * v for v in values[i]]
        x_attn = [x + p for x, p in zip(x_attn, payload)]

    xo = linear(x_attn, state_dict["attn_wo"])
    x = [x + x_out for x, x_out in zip(x_residual, xo)]

    # mlp
    x = rmsnorm(x)
    x_residual = list(x)
    h_pre = linear(x, state_dict["mlp_fc1"])
    h_pre = [l.relu() for l in h_pre]
    x_mlp = linear(h_pre, state_dict["mlp_fc2"])
    x = [x_res + x for x_res, x in zip(x_residual, x_mlp)]

    logits = linear(x, state_dict["lm_head"])

    return logits


default_lr = 0.1
steps = 1000
for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = min(block_size, len(tokens) - 1)

    losses, keys, values = [], [], []
    for i in range(n):
        token_id, target_id = tokens[i], tokens[i + 1]
        logits = gpt(token_id, i, keys, values)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = sum(losses) / len(losses)

    loss.backward()

    lr = default_lr * (1 - step / steps)
    for p in params:
        p.data -= p.grad * lr
        p.grad = 0

    if step < 5 or step % 200 == 0:
        print(f"step {step+1:4d} / {steps:4d} | loss {loss.data:.4f}")

steps = 20
temperature = 0.5
token_id = BOS
for step in range(steps):
    keys, values = [], []
    output = []

    for i in range(block_size):
        logits = gpt(token_id, i, keys, values)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        output.append(uchars[token_id])

    print(f"sample {step+1:2d}: {''.join(output)}")
