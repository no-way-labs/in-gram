"""Insert In-gram after the first block of a small model, run the parallel pass, then decode token by token.

    python examples/quickstart.py
"""
import torch
import torch.nn as nn

from ingram import InGram, InGramDecoder

torch.manual_seed(0)
vocab, d_model, L = 1000, 64, 512

embed = nn.Embedding(vocab, d_model)
block1 = nn.GRU(d_model, d_model, batch_first=True)      # stands in for any causal block (GDN in the paper)
ingram = InGram(d_model)                                   # cue lengths 1, 2, 4, 8, 16, 32; slots of width 64

# a context that repeats itself: the second half copies the first
half = torch.randint(0, vocab, (1, L // 2))
tokens = torch.cat([half, half], dim=1)

# training-style pass: the schedule of writes and reads comes from the token ids alone
schedule = ingram.schedule(tokens)
x = block1(embed(tokens))[0]                               # hidden states after block 1
y = ingram(x, schedule)                                    # x + gated read; feed y to the next block
print("slots written:", int(schedule["n_slots"][0]), "for", L, "tokens (bound:", 6 * L, ")")

# decoding: one read and one write per cue length for each new token
decoder = InGramDecoder(ingram)
with torch.no_grad():
    stepped = torch.stack([decoder.step(x[0, t], int(tokens[0, t])) for t in range(L)])
print("largest difference between the parallel pass and token-by-token decoding:",
      float((stepped - y[0].detach()).abs().max()))
print("session state:", decoder.n_slots, "slots,", decoder.state_bytes() / 1e6, "MB in fp32")
