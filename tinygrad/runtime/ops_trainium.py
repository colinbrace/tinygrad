# Trainium backend (Phase 1a: NKI CPU simulator path, no hardware).
# See scratch/ml/theory/tinygrad-notes/backend_design.md
import json, os
import numpy as np
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.nki import NKIRenderer

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
      st = inp["strides"]
      shape = [cs[c] if (c < split or st[c] != 0) else 1 for c in range(nd)]
      bytestrides = [st[c] * d.itemsize for c in range(nd)]
      view = np.lib.stride_tricks.as_strided(flat[inp["offset"]:], shape=shape, strides=bytestrides)
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
