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
    if m["kind"] == "elementwise":
      # flat buffers -> (1, N) tiles
      in_arrs = [np.frombuffer(bufs[s], dtype=npd(s)).reshape(1, -1) for s in m["in_slots"]]
    else:  # reduce: arrange the input as (kept=partition, reduced=free) so nl.sum reduces axis 1
      buf, shape = bufs[m["in_slot"]], m["in_shape"]
      arr = np.frombuffer(buf, dtype=npd(m["in_slot"])).reshape(shape if shape else (1,))
      perm = m["kept_pos"] + m["reduce_pos"]
      arr = np.ascontiguousarray(np.transpose(arr, perm))
      P = int(np.prod([shape[p] for p in m["kept_pos"]])) if m["kept_pos"] else 1
      F = int(np.prod([shape[p] for p in m["reduce_pos"]]))
      in_arrs = [arr.reshape(P, F)]
    if os.getenv("NKI_TRACE"): print(f"[trainium] {self.name} ({m['kind']}) via nki.simulate, in={[a.shape for a in in_arrs]}")
    import nki
    out = np.asarray(nki.simulate(self.kernel)(*in_arrs))
    bufs[m["out_slot"]][:] = np.ascontiguousarray(out, dtype=npd(m["out_slot"])).tobytes()
    return None

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
