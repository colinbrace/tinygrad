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
    dt = self.meta["np_dtypes"]
    # bufs arrive in global/slot order; reshape each flat buffer to a (1, N) NKI tile
    in_arrs = [np.frombuffer(bufs[s], dtype=np.dtype(dt[str(s)])).reshape(1, -1) for s in self.meta["in_slots"]]
    if os.getenv("NKI_TRACE"): print(f"[trainium] running {self.name} via nki.simulate, in shapes={[a.shape for a in in_arrs]}")
    import nki
    out = np.asarray(nki.simulate(self.kernel)(*in_arrs))
    os_slot = self.meta["out_slot"]
    bufs[os_slot][:] = np.ascontiguousarray(out, dtype=np.dtype(dt[str(os_slot)])).tobytes()
    return None

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
