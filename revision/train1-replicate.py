import math  # math.log
import random  # random.seed, random.choices, random.shuffle

random.seed(42)

# Dataset: load and tokenize a list of names
docs = [
    l.strip() for l in open("input.txt").read().strip().split("\n") if l.strip()
]  # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

uchars = sorted(set("".join(docs)))
vocab_size = len(uchars) + 1
BOS = len(uchars)

matrix = lambda nout, nin: [
    [random.gauss(0, 1 / vocab_size) for _ in range(nin)] for _ in range(nout)
]
n_embd = 16
state = {
    "wte": matrix(vocab_size, n_embd),
    "mlp_fc1": matrix(4 * n_embd, n_embd),
    "mlp_fc2": matrix(vocab_size, 4 * n_embd),
}
params = [(row, j) for mat in state.values() for row in mat for j in range(len(row))]


def linear(x, matrix):
    return [sum(xi * ri for xi, ri in zip(x, r)) for r in matrix]


def softmax(logits):
    max_l = max(logits)
    soft = [math.exp(l - max_l) for l in logits]
    total = sum(soft)
    return [li / total for li in soft]


def mlp(token_id):
    x = list(state["wte"][token_id])
    h_pre = linear(x, state["mlp_fc1"])
    h = [max(0, h_pre[i]) for i in range(len(h_pre))]
    logits = linear(h, state["mlp_fc2"])
    return logits


def num_grad(tokens):
    n = len(tokens) - 1
    losses = []
    grads = [0.0] * len(params)

    for i in range(n):
        token_id, target_id = tokens[i], tokens[i + 1]
        logits_b = mlp(token_id)
        probs_b = softmax(logits_b)
        loss_b = -math.log(probs_b[target_id])
        eps = 1e-5
        losses.append(loss_b)

        for index, (neuron, row) in enumerate(params):
            before = neuron[row]
            neuron[row] += eps
            logits = mlp(token_id)
            probs = softmax(logits)
            loss = -math.log(probs[target_id])
            grad_nudged = (loss - loss_b) / eps
            grads[index] += grad_nudged / n
            neuron[row] = before

    loss = sum(losses) / len(losses)

    return loss, grads


def analytical_grad(tokens):
    n = len(tokens) - 1
    losses = []
    grad = {k: [[0.0] * len(row) for row in mat] for k, mat in state.items()}

    for token_index in range(n):
        token_id, target_id = tokens[token_index], tokens[token_index + 1]

        # forward
        x = state["wte"][token_id]
        h_pre = linear(x, state["mlp_fc1"])
        h = [max(0, h_pre[i]) for i in range(len(h_pre))]
        logits = linear(h, state["mlp_fc2"])
        probs = softmax(logits)
        loss_t = -math.log(probs[target_id])
        losses.append(loss_t)

        # backward
        dlogits = [p / n for p in probs]
        dlogits[target_id] -= 1.0 / n

        dh = [0.0] * len(h)
        for i in range(len(logits)):
            for j in range(len(h)):
                grad["mlp_fc2"][i][j] += dlogits[i] * h[j]
                dh[j] += dlogits[i] * state["mlp_fc2"][i][j]

        dh_pre = [dh[j] if h_pre[j] > 0 else 0.0 for j in range(len(dh))]

        dx = [0.0] * len(x)
        for i in range(len(dh_pre)):
            for j in range(len(x)):
                grad["mlp_fc1"][i][j] += dh_pre[i] * x[j]
                dx[j] += dh_pre[i] * state["mlp_fc1"][i][j]

        for j in range(len(dx)):
            grad["wte"][token_id][j] += dx[j]

    loss = sum(losses) / len(losses)
    grad = [val for mat in grad.values() for row in mat for val in row]

    return loss, grad


# check
doc = "mark"
tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
loss, grad = analytical_grad(tokens)
loss_n, grad_n = num_grad(tokens)
grad_diff = max(abs(gn - gh) for gn, gh in zip(grad_n, grad))
print(
    f"gradient check | loss_n {loss_n:.6f} | loss_a {loss:.6f} | max diff {grad_diff:.8f}"
)

steps = 1000
learning_rate = 1
for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [uchars.index(c) for c in doc] + [BOS]
    loss, grad = analytical_grad(tokens)

    lr = learning_rate * (1 - step / steps)
    for index, (row, j) in enumerate(params):
        row[j] -= lr * grad[index]

    if step < 5 or step % 200 == 0:  # print a bit less often
        print(f"step {step+1:4d} / {steps:4d} | loss {loss:.4f} | lr {lr:.4f}")


steps = 20
temperature = 0.5 
for step in range(steps):
    token_id = BOS
    n = 20
    sample = []

    for i in range(n):
        logits = mlp(token_id)
        token_id = random.choices(
            range(vocab_size), softmax([l / temperature for l in logits])
        )[0]
        if token_id == BOS:
            break
        sample.append(uchars[token_id])

    print(f"sample {step+1:2d}: {''.join(sample)}")
