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
    m, dt = self.meta, self.meta["np_dtypes"]
    npd = lambda s: np.dtype(dt[str(s)])
    # broadcast each input over the PARTITION (kept) axes to a common P; leave FREE axes at the
    # input's natural size (1 if absent) and let NKI's nl.* ops broadcast them. This lets a
    # post-reduce operand (e.g. a bias spanning only kept axes) align (P,1) with the reduced (P,1).
    cs, split, nd = m["canonical_sizes"], m["split"], len(m["canonical_sizes"])
    P = int(np.prod(cs[:split])) if split else 1
    in_arrs = []
    for inp in m["inputs"]:
      arr = np.frombuffer(bufs[inp["slot"]], dtype=npd(inp["slot"])).reshape(inp["logical_shape"] or (1,))
      ci = inp["canon_idx"]
      arr = np.transpose(arr, sorted(range(len(ci)), key=lambda i: ci[i]))   # logical axes -> canonical order
      present = set(ci)
      arr = arr.reshape([cs[c] if c in present else 1 for c in range(nd)])    # size-1 for missing axes
      target = [cs[c] if (c < split or c in present) else 1 for c in range(nd)]  # full partition, natural free
      arr = np.ascontiguousarray(np.broadcast_to(arr, target)).reshape(P, int(np.prod(target[split:])) or 1)
      in_arrs.append(arr)
    if os.getenv("NKI_TRACE"): print(f"[trainium] {self.name} via nki.simulate, in={[a.shape for a in in_arrs]}")
    import nki
    # NKI partition dim (axis 0) is capped at 128 -> tile the kernel call into <=128-row chunks
    PMAX, rows = 128, in_arrs[0].shape[0]
    if rows <= PMAX:
      out = np.asarray(nki.simulate(self.kernel)(*in_arrs))
    else:
      out = np.concatenate([np.asarray(nki.simulate(self.kernel)(*[a[i:i+PMAX] for a in in_arrs]))
                            for i in range(0, rows, PMAX)], axis=0)
    bufs[m["out_slot"]][:] = np.ascontiguousarray(out, dtype=npd(m["out_slot"])).tobytes()
    return None

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
