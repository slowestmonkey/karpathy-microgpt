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
        self.grad = 0
        self.data = data
        self.children = children
        self.local_grads = local_grads

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data + other.data, (self, other), (1, 1))

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        return Value(self.data * other.data, (self, other), (other.data, self.data))

    def __pow__(self, other):
        return Value(self.data**other, (self,), (other * (self.data ** (other - 1)),))

    def __sub__(self, other):
        return self + (-other)

    def __rsub__(self, other):
        return -self + other

    def __neg__(self):
        return self * -1

    def __radd__(self, other):
        return self + other

    def __truediv__(self, other):
        return self * (other**-1)

    def __rtruediv__(self, other):
        return other * (self**-1)

    def relu(self):
        return (
            Value(max(0.0, self.data), (self,), (1.0,))
            if self.data > 0
            else Value(0.0, (self,), (0.0,))
        )

    def exp(self):
        value = math.exp(self.data)
        return Value(value, (self,), (value,))

    def log(self):
        return Value(math.log(self.data), (self,), (1 / self.data,))

    def backward(self):
        topo = []
        visited = set()

        def build_topo(value):
            if value not in visited:
                visited.add(value)
                for child in value.children:
                    build_topo(child)
                topo.append(value)

        build_topo(self)
        topo.reverse()

        self.grad = 1
        for v in topo:
            for c, local_grad in zip(v.children, v.local_grads):
                c.grad += v.grad * local_grad


n_embd = 16
matrix = lambda nout, nin: [
    [Value(random.gauss(0, 1 / vocab_size)) for _ in range(nin)] for _ in range(nout)
]
state_dict = {
    "wte": matrix(vocab_size, n_embd),
    "mlp_fc1": matrix(4 * n_embd, n_embd),
    "mlp_fc2": matrix(vocab_size, 4 * n_embd),
}
params = [p for mat in state_dict.values() for row in mat for p in row]


def linear(x, matrix):
    return [sum(a * b for a, b in zip(x, row)) for row in matrix]


def softmax(logits):
    l_max = max([l.data for l in logits])
    softed = [(l - l_max).exp() for l in logits]
    total = sum(softed)
    return [s / total for s in softed]


def mlp(token_id):
    x = state_dict["wte"][token_id]
    h_pre = linear(x, state_dict["mlp_fc1"])
    h = [v.relu() for v in h_pre]
    return linear(h, state_dict["mlp_fc2"])


default_lr = 2
steps = 1000
for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    n = len(tokens) - 1

    losses = []
    for i in range(n):
        token_id, target_id = tokens[i], tokens[i + 1]
        logits = mlp(token_id)
        probs = softmax(logits)
        loss_t = -probs[target_id].log()
        losses.append(loss_t)
    loss = sum(losses) / len(losses)

    loss.backward()

    lr = default_lr * (1 - step / steps)
    for p in params:
        p.data -= lr * p.grad
        p.grad = 0.0
        # p.local_grads = ()
        # p.children = ()

        # if step < 5 or step % 200 == 0:  # print a bit less often
    print(f"step {step+1:4d} / {steps:4d} | loss {loss.data:.4f} | lr {lr:.4f}")

temperature = 0.5
steps = 20
for step in range(steps):
    token_id = BOS
    output = []
    for i in range(10):
        logits = mlp(token_id)
        probs = softmax([l / temperature for l in logits])
        token_id = random.choices(range(vocab_size), weights=[p.data for p in probs])[0]
        if token_id == BOS:
            break
        output.append(uchars[token_id])
    print(f"sample {step+1:2d}: {''.join(output)}")
