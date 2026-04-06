
import math     # math.log
import random   # random.seed, random.choices, random.shuffle
random.seed(42)

# Dataset: load and tokenize a list of names
docs = [l.strip() for l in open('input.txt').read().strip().split('\n') if l.strip()] # list[str] of documents
random.shuffle(docs)
print(f"num docs: {len(docs)}")

vocab = sorted(set("".join(docs)))
vocab.append('BOS')
BOS = len(vocab) - 1
matrix = [[0] * len(vocab) for _ in range(len(vocab))]

def bigram(tokenId: int):
    probs = matrix[tokenId]
    total = sum(probs) + len(probs)
    return [(p + 1) / total for p in probs]

steps = 100000
for step in range(steps):
    doc = docs[step % len(docs)]
    tokens = [BOS] + [vocab.index(x) for x in doc] + [BOS]
    n = len(tokens) - 1
    losses = []
   
    for pos_id in range(n):
        [current_id, target_id] = [tokens[pos_id], tokens[pos_id + 1]]
        probs = bigram(current_id)
        loss = -math.log(probs[target_id]) # This is still the most unclear part
        losses.append(loss)
        
    total_loss = sum(losses) / len(losses)
    print(f"step {step}, loss: {total_loss:.4f}")
    
    for pos_id in range(n):
        [current_id, target_id] = [tokens[pos_id], tokens[pos_id + 1]]
        matrix[current_id][target_id] += 1
        
for step in range(20):
    output = ''
    token_id = BOS
    for _ in range(10):
       token_id = random.choices(range(len(vocab)), bigram(token_id))[0]
       if token_id == BOS:
           break
       output += vocab[token_id]
    print(f"step {step}: {output}")
       


