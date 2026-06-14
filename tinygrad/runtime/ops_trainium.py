# Trainium backend (Phase 1a: NKI CPU simulator path, no hardware).
# See scratch/ml/theory/tinygrad-notes/backend_design.md
import json, os
import numpy as np
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.nki import NKIRenderer

_ALU = {"ADD":lambda a,b:a+b, "MUL":lambda a,b:a*b, "SUB":lambda a,b:a-b, "MAX":np.maximum,
        "FLOORDIV":lambda a,b:a//b, "FLOORMOD":lambda a,b:a%b,
        "CDIV":lambda a,b:(np.abs(a)//np.abs(b))*np.sign(a)*np.sign(b), "CMOD":lambda a,b:a-(_ALU["CDIV"](a,b))*b,
        "CMPLT":lambda a,b:a<b, "CMPNE":lambda a,b:a!=b, "CMPEQ":lambda a,b:a==b,
        "AND":lambda a,b:a&b, "OR":lambda a,b:a|b, "XOR":lambda a,b:a^b}

def _eval_index(node, shape, bufs):
  # evaluate a serialized (data-dependent) index expression over the iteration grid -> int array
  t = node["t"]
  if t == "const": return np.int64(node["v"])
  if t == "rng":
    sh = [1]*len(shape); sh[node["ax"]] = shape[node["ax"]]
    return np.arange(shape[node["ax"]], dtype=np.int64).reshape(sh) + np.zeros(shape, dtype=np.int64)
  if t == "cast": return _eval_index(node["s"][0], shape, bufs).astype(np.int64)
  if t == "where":
    c, a, b = (_eval_index(x, shape, bufs) for x in node["s"])
    return np.where(c, a, b)
  if t == "load":   # gated load: out-of-bounds (Invalid) index -> 0
    sub = _eval_index(node["s"][0], shape, bufs).astype(np.int64)
    buf = np.frombuffer(bufs[node["slot"]], dtype=np.dtype(node["dtype"]))
    valid = (sub >= 0) & (sub < len(buf))
    return np.where(valid, buf[np.clip(sub, 0, len(buf)-1)], 0).astype(np.int64)
  if t == "alu": return _ALU[node["op"]](*[_eval_index(x, shape, bufs) for x in node["s"]])
  raise NotImplementedError(f"NKI gather eval: {t}")

class TrainiumCompiler(Compiler):
  # SIM path: pass the rendered NKI source through as bytes. (HW path later: neuron-cc -> NEFF.)
  def compile(self, src:str) -> bytes: return src.encode()

class TrainiumAllocator(Allocator['TrainiumDevice']):
  def _alloc(self, size, options): return memoryview(bytearray(size))
  def _copyin(self, dest, src:memoryview): dest[:] = src
  def _copyout(self, dest:memoryview, src): dest[:] = src

class TrainiumProgram:
  def __init__(self, name:str, lib:bytes, *aux, runtimevars=None, prg=None, **kwargs):
    self.name, self.src = name, lib.decode()
    self.meta = json.loads(self.src.splitlines()[0].split("TRAINIUM_META", 1)[1])
    if os.getenv("NKI_SRC"): print(self.src)
    ns:dict = {}
    exec(compile(self.src, f"<nki:{name}>", "exec"), ns)   # defines `kernel`
    self.kernel = ns["kernel"]

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    m = self.meta
    # Build each input INDEX as a strided view over the canonical iteration space: full size on
    # PARTITION (kept) axes, natural size on FREE axes (1 where stride is 0). A 0 stride broadcasts,
    # a nonzero stride + offset reads contiguous/transpose/slice -- all uniformly. Then reshape (P, F).
    cs, split, nd = m["canonical_sizes"], m["split"], len(m["canonical_sizes"])
    P = int(np.prod(cs[:split])) if split else 1
    in_arrs = []
    for inp in m["inputs"]:
      d = np.dtype(inp["dtype"])
      flat = np.frombuffer(bufs[inp["param_slot"]], dtype=d)
      if inp["kind"] == "gather":   # data-dependent index: eval offsets over the grid, then gather
        idx = _eval_index(inp["index"], cs, bufs)
        valid = (idx >= 0) & (idx < len(flat))      # gated load: Invalid/OOB index -> 0 (e.g. cat/pad)
        view = np.where(valid, flat[np.clip(idx, 0, len(flat)-1)], 0)
        shape = cs
      else:                         # affine: a strided view (full partition, natural free)
        st = inp["strides"]
        shape = [cs[c] if (c < split or st[c] != 0) else 1 for c in range(nd)]
        view = np.lib.stride_tricks.as_strided(flat[inp["offset"]:], shape=shape, strides=[st[c]*d.itemsize for c in range(nd)])
      in_arrs.append(np.ascontiguousarray(view).reshape(P, int(np.prod(shape[split:])) or 1))
    if os.getenv("NKI_TRACE"): print(f"[trainium] {self.name} via nki.simulate, in={[a.shape for a in in_arrs]}")
    import nki
    # NKI partition dim (axis 0) is capped at 128 -> tile the kernel call into <=128-row chunks
    PMAX, rows = 128, in_arrs[0].shape[0]
    if rows <= PMAX:
      out = np.asarray(nki.simulate(self.kernel)(*in_arrs))
    else:
      out = np.concatenate([np.asarray(nki.simulate(self.kernel)(*[a[i:i+PMAX] for a in in_arrs]))
                            for i in range(0, rows, PMAX)], axis=0)
    bufs[m["out_slot"]][:] = np.ascontiguousarray(out, dtype=np.dtype(m["out_dtype"])).tobytes()
    return None

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
