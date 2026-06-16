# Trainium backend. Two execution paths behind DEV=TRAINIUM, sharing the NKIRenderer:
#   - default: NKI CPU simulator (no hardware)            -- nki.simulate(kernel)(*args)
#   - TRAINIUM_HW=1: real NeuronCore (Phase 1b)           -- kernel(*args) JIT-compiles to NEFF + runs
# See scratch/ml/theory/tinygrad-notes/backend_design.md
import json, os
import numpy as np
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.nki import NKIRenderer

# Phase 1b: when set, run kernels on the real device instead of the CPU simulator. The nki.jit
# callable compiles to a NEFF and executes on /dev/neuron* when called directly with numpy arrays.
_HW = os.getenv("TRAINIUM_HW") == "1"

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
    self.kernel = self._build_kernel(name, self.src)

  @staticmethod
  def _build_kernel(name:str, src:str):
    # The sim just needs the callable, so exec'ing the source is enough. The HW compiler frontend,
    # however, looks the entry function up by source location (AST/linecache), so an exec'd kernel
    # fails with "entry function not found" -- it must live in a real importable .py file.
    if not _HW:
      ns:dict = {}
      exec(compile(src, f"<nki:{name}>", "exec"), ns)   # defines `kernel`
      return ns["kernel"]
    import importlib.util, hashlib, tempfile
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    d = os.path.join(tempfile.gettempdir(), "tinygrad_nki"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"k_{h}.py")
    if not os.path.exists(path):
      with open(path, "w") as f: f.write(src)
    spec = importlib.util.spec_from_file_location(f"tinygrad_nki_k_{h}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.kernel

  def _run(self, *np_args):
    # one place the two paths diverge: real device vs CPU simulator. numpy in -> numpy out either way.
    if _HW: return np.asarray(self.kernel(*np_args))
    import nki
    return np.asarray(nki.simulate(self.kernel)(*np_args))

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    m = self.meta
    if m.get("kind") == "matmul":   # nl.matmul fast path: build A as (K,M), B as (K,N) strided views
      def view(spec, shape):
        d = np.dtype(spec["dtype"]); flat = np.frombuffer(bufs[spec["param_slot"]], dtype=d)
        return np.ascontiguousarray(np.lib.stride_tricks.as_strided(
          flat[spec["offset"]:], shape=shape, strides=[s*d.itemsize for s in spec["strides"]]))
      args = [view(m["A"], (m["K"], m["M"])), view(m["B"], (m["K"], m["N"]))]
      args += [view(p, (m["M"], m["N"])) for p in m["post"]]   # post-ops (bias etc.) broadcast to (M,N)
      out = self._run(*args)
      bufs[m["out_slot"]][:] = np.ascontiguousarray(out, dtype=np.dtype(m["out_dtype"])).tobytes()
      return None
    # Build each input INDEX as a strided view over the canonical iteration space: full size on
    # PARTITION (kept) axes, natural size on FREE axes (1 where stride is 0). A 0 stride broadcasts,
    # a nonzero stride + offset reads contiguous/transpose/slice -- all uniformly. Then reshape (P, F).
    cs, split, nd = m["canonical_sizes"], m["split"], len(m["canonical_sizes"])
    P = int(np.prod(cs[:split])) if split else 1
    in_arrs = []
    for inp in m["inputs"]:
      if inp["kind"] == "iota":   # a RANGE used as a value -> coordinate of canonical axis over the grid
        ax = inp["axis"]
        sh = [1]*nd; sh[ax] = cs[ax]
        coord = np.arange(cs[ax]).reshape(sh) + np.zeros(cs, dtype=np.int64)   # broadcast to full grid
        in_arrs.append(coord.reshape(P, int(np.prod(cs[split:])) or 1).astype(np.float32))
        continue
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
    if os.getenv("NKI_TRACE"): print(f"[trainium] {self.name} {'on HW' if _HW else 'via nki.simulate'}, in={[a.shape for a in in_arrs]}")
    # NKI partition dim (axis 0) is capped at 128 -> tile the kernel call into <=128-row chunks
    PMAX, rows = 128, in_arrs[0].shape[0]
    if rows <= PMAX:
      out = self._run(*in_arrs)
    else:
      out = np.concatenate([self._run(*[a[i:i+PMAX] for a in in_arrs]) for i in range(0, rows, PMAX)], axis=0)
    bufs[m["out_slot"]][:] = np.ascontiguousarray(out, dtype=np.dtype(m["out_dtype"])).tobytes()
    return None

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
