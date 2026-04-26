"""
Visualize the autograd graph for softmax([x]) — the length-1 softmax that
returns 1.0 regardless of x and produces zero gradient back to x.

Run:  python revision/debug_softmax_graph.py
Render the dot file (optional, needs graphviz installed):
      dot -Tpng revision/softmax_graph.dot -o revision/softmax_graph.png
"""

import math
import os


class Value:
    __slots__ = ("data", "grad", "children", "local_grads", "_id", "_op")
    _next_id = 0

    def __init__(self, data, children=(), local_grads=(), op="leaf"):
        self.data = data
        self.children = children
        self.local_grads = local_grads
        self.grad = 0
        self._id = Value._next_id
        self._op = op
        Value._next_id += 1

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other, op=f"const({other})")
        return Value(self.data * other.data, (self, other), (other.data, self.data), op="*")

    def __rmul__(self, other):
        return self * other

    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other, op=f"const({other})")
        return Value(self.data + other.data, (self, other), (1, 1), op="+")

    def __radd__(self, other):
        return self + other

    def __pow__(self, other):
        return Value(self.data ** other, (self,), (other * self.data ** (other - 1),), op=f"**{other}")

    def __truediv__(self, other):
        return self * (other ** -1)

    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def exp(self):
        e = math.exp(self.data)
        return Value(e, (self,), (e,), op="exp")

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


def softmax(logits):
    l_max = max(l.data for l in logits)
    l_e = [(l - l_max).exp() for l in logits]
    total = sum(l_e)
    return [l / total for l in l_e]


def topo_order(root):
    order, seen = [], set()

    def visit(v):
        if v not in seen:
            seen.add(v)
            for c in v.children:
                visit(c)
            order.append(v)

    visit(root)
    return order


def print_text_graph(root):
    order = topo_order(root)
    print(f"{'id':>4} | {'op':<12} | {'data':>10} | {'grad':>10} | children (local_grad)")
    print("-" * 78)
    for v in order:
        kids = ", ".join(f"#{c._id}({lg:+.3f})" for c, lg in zip(v.children, v.local_grads))
        print(f"#{v._id:<3} | {v._op:<12} | {v.data:>10.4f} | {v.grad:>10.4f} | {kids}")


def write_dot(root, path):
    order = topo_order(root)
    with open(path, "w") as f:
        f.write("digraph G {\n")
        f.write("  rankdir=BT;\n")
        f.write('  node [shape=box, fontname="monospace"];\n')
        for v in order:
            label = f"#{v._id} {v._op}\\ndata={v.data:.3f}\\ngrad={v.grad:.3f}"
            f.write(f'  n{v._id} [label="{label}"];\n')
            for c, lg in zip(v.children, v.local_grads):
                f.write(f'  n{c._id} -> n{v._id} [label="lg={lg:+.3f}"];\n')
        f.write("}\n")


if __name__ == "__main__":
    x = Value(2.5, op="x (input)")
    result = softmax([x])[0]
    result.backward()

    print(f"\nresult.data = {result.data}   (constant 1.0 for length-1 softmax)")
    print(f"x.grad      = {x.grad}   (zero — two paths cancel)\n")
    print("autograd graph (topo order, leaves first):\n")
    print_text_graph(result)

    dot_path = os.path.join(os.path.dirname(__file__), "softmax_graph.dot")
    write_dot(result, dot_path)
    print(f"\nwrote {dot_path}")
    print("render with:  dot -Tpng revision/softmax_graph.dot -o revision/softmax_graph.png")
