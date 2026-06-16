#!/usr/bin/env python3
# Tensor-parallel GPT-2 on the Trainium backend (multi-core).
#
# Shards ONE GPT-2 across N NeuronCores and confirms the output matches the single-core run.
# Runs on the NKI simulator by default (fast, no hardware; tinygrad's MULTI/ALLREDUCE lowering is
# backend-independent, so correctness transfers); TRAINIUM_HW=1 runs the same plan on real cores,
# one tinygrad device (TRAINIUM:i) per physical NeuronCore.
#
#   PYTHONPATH=. DEV=TRAINIUM JIT=0 DEVS=2 python examples/gpt2_tp.py --prompt "Hello, I am" --count 8
#
# Strategy (Megatron-style; tinygrad nn.Linear weight layout is (out_features, in_features)):
#   The MLP and lm_head are the big matmuls. Everything else (embeddings, attention, layernorms) runs
#   on a single device -- activations stay on ONE device at the block boundaries and are only sharded
#   *inside* the wrapped MLP / lm_head. That sidesteps the model's internal single-device tensors (the
#   positional arange, the causal mask, the embedding one-hot), which would otherwise collide with the
#   MULTI activations under a whole-model shard.
#     mlp.c_fc    weight axis 0 (+bias axis 0)   column-parallel: split the 4*dim hidden across cores
#     mlp.c_proj  weight axis 1 (bias replicated) row-parallel: partial sums -> ALLREDUCE(ADD)
#     lm_head     weight axis 0                   column/vocab-parallel: split the 50257 vocab
#   Inside the wrapper: x (1 device) -> .shard(DEVS) replicate -> sharded matmuls -> the result comes
#   back replicated (after the allreduce) -> .to(orig device). The matmuls run on distinct cores.
#
# No backend code is required for correctness: the cross-core copies route through host memory and the
# naive ALLREDUCE lowers to ordinary COPY+ADD kernels -- both already-supported Trainium kernels.
import sys, math, argparse, importlib.util, os
import numpy as np
from tinygrad import Tensor, dtypes, Variable, Device
from tinygrad.dtype import least_upper_dtype
from tinygrad.nn.state import get_state_dict

gpt2ex = importlib.util.module_from_spec(importlib.util.spec_from_file_location(
  "gpt2ex", os.path.join(os.path.dirname(__file__), "gpt2.py")))
gpt2ex.__spec__.loader.exec_module(gpt2ex)

# ---- backend-compat shims (same as the single-core run): no fused flash-attn, host argmax ----
def _sdpa(self, key, value, attn_mask=None, dropout_p=0.0, is_causal=False, enable_gqa=False):
  if enable_gqa:
    key = key.repeat_interleave(int(self.shape[-3] // key.shape[-3]), dim=-3)
    value = value.repeat_interleave(int(self.shape[-3] // value.shape[-3]), dim=-3)
  q = self
  qk = (q.matmul(key.transpose(-2, -1), dtype=least_upper_dtype(q.dtype, key.dtype, dtypes.float32))
        / math.sqrt(q.shape[-1])).contiguous()
  if is_causal:
    if attn_mask is not None: raise RuntimeError("cannot set attn_mask when is_causal=True")
    attn_mask = qk.const_like(1).cast(dtypes.bool).tril()
  if attn_mask is not None:
    if attn_mask.dtype == dtypes.bool: attn_mask = attn_mask.where(0, -float("inf"))
    qk = qk + attn_mask.contiguous()
  return (qk.cast(self.dtype).softmax(-1).dropout(dropout_p).contiguous()) @ value
Tensor.scaled_dot_product_attention = _sdpa

REAL_VOCAB = 50257   # GPT-2 vocab (prime). lm_head is padded to a multiple of NDEV for even sharding;
                     # the padded rows are zero-weight -> 0 logits, sliced off here before the argmax.
def _host_argmax(self, axis=None, keepdim=False):
  a = np.asarray(self.numpy())
  if a.shape[-1] > REAL_VOCAB and axis in (-1, a.ndim - 1): a = a[..., :REAL_VOCAB]
  r = a.argmax(axis=axis)
  if keepdim and axis is not None: r = np.expand_dims(r, axis)
  return Tensor(np.ascontiguousarray(r))
Tensor.argmax = _host_argmax

def _dev0(d): return d if isinstance(d, str) else d[0]

# Wrap FeedForward: when its weights are sharded (MULTI), replicate the activation in, run the two
# sharded matmuls (the allreduce happens inside c_proj), bring the replicated result back to one device.
_orig_ff = gpt2ex.FeedForward.__call__
def _ff_tp(self, x):
  if isinstance(self.c_fc.weight.device, tuple):
    return _orig_ff(self, x.shard(self.c_fc.weight.device)).to(_dev0(x.device))
  return _orig_ff(self, x)
gpt2ex.FeedForward.__call__ = _ff_tp

class _TPLinear:   # wraps the lm_head Linear: shard activation in, vocab-sharded matmul, gather out
  def __init__(self, lin): self.lin = lin
  def __getattr__(self, n): return getattr(self.lin, n)
  def __call__(self, x):
    if isinstance(self.lin.weight.device, tuple):
      return self.lin(x.shard(self.lin.weight.device)).to(_dev0(x.device))
    return self.lin(x)

def apply_shard_plan(model, devs):
  """Shard the MLP + lm_head weights across devs (in place); leave everything else single-device."""
  n = len(devs)
  if model.lm_head.weight.shape[0] % n != 0:   # pad the prime vocab up to a multiple of n
    V, D = model.lm_head.weight.shape
    model.lm_head.weight = model.lm_head.weight.cat(
      Tensor.zeros((-V) % n, D, dtype=model.lm_head.weight.dtype), dim=0).contiguous().realize()
  sd = get_state_dict(model)
  plan = {}
  for k, t in sd.items():
    if   k.endswith("mlp.c_fc.weight"):   ax = 0   # column-parallel (split hidden)
    elif k.endswith("mlp.c_fc.bias"):     ax = 0
    elif k.endswith("mlp.c_proj.weight"): ax = 1   # row-parallel (split contraction -> allreduce)
    elif k == "lm_head.weight":           ax = 0   # vocab-parallel
    else: continue
    t.shard_(devs, ax); plan[k] = ax
  for k, t in sd.items():   # the row-parallel bias must be replicated so it aligns after the allreduce
    if k.endswith("mlp.c_proj.bias"): t.shard_(devs, None)
  model.lm_head = _TPLinear(model.lm_head)
  return plan

def greedy(model, toks, count):
  toks = list(toks)
  for _ in range(count):
    sp = Variable("start_pos", 0, gpt2ex.MAX_CONTEXT - 1).bind(0)
    out = model.model(Tensor([toks]), sp, 0.0)     # host argmax shim -> next token id
    toks.append(int(np.asarray(out.numpy()).reshape(-1)[0]))
  return toks

if __name__ == "__main__":
  ap = argparse.ArgumentParser()
  ap.add_argument("--model_size", default="gpt2")
  ap.add_argument("--prompt", default="Hello, I am")
  ap.add_argument("--count", type=int, default=8)
  args = ap.parse_args()
  NDEV = int(os.getenv("DEVS", "2"))
  DEV = Device.DEFAULT
  DEVS = tuple(f"{DEV}:{i}" for i in range(NDEV))
  print(f"device={DEV}  shards={DEVS}")

  gpt2 = gpt2ex.GPT2.build(args.model_size)
  prompt_toks = gpt2.tokenizer.encode(args.prompt)
  print(f"prompt: {args.prompt!r}")

  ref = greedy(gpt2, prompt_toks, args.count)        # single-core reference
  print("REF (1 core):", repr(gpt2.tokenizer.decode(ref)))

  plan = apply_shard_plan(gpt2.model, DEVS)          # shard MLP + lm_head across DEVS
  print(f"sharded {len(plan)} tensors across {NDEV} cores: "
        f"{sorted(set(k.split('.')[-2]+'.'+k.split('.')[-1] for k in plan))}")
  tp = greedy(gpt2, prompt_toks, args.count)         # tensor-parallel run
  print("TP  (%d core):" % NDEV, repr(gpt2.tokenizer.decode(tp)))

  match = ref == tp
  print(f"\nTOKEN MATCH: {match}")
  if not match:
    print("  ref:", ref); print("  tp :", tp); sys.exit(1)
  print("PASS: tensor-parallel output == single-core output")
