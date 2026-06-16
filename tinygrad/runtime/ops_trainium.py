# Trainium backend. Two execution paths behind DEV=TRAINIUM, sharing the NKIRenderer:
#   - default: NKI CPU simulator (no hardware)   -- nki.simulate(kernel)(*args)
#   - TRAINIUM_HW=1: real NeuronCore (Phase 1b). By default uses a PERSISTENT EXECUTABLE: compile each
#     kernel+signature to a NEFF once, load it onto the device once, then launch many (warm launch is
#     ~0.2ms vs ~1.2s for the high-level kernel(*args) path, which reloads the NEFF every call).
#     TRAINIUM_SIMPLE_LAUNCH=1 forces the simpler high-level path (more stable, much slower).
# See scratch/ml/theory/tinygrad-notes/backend_design.md
import json, os, tempfile, shutil, hashlib
import numpy as np
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.nki import NKIRenderer

_HW = os.getenv("TRAINIUM_HW") == "1"
_SIMPLE_LAUNCH = os.getenv("TRAINIUM_SIMPLE_LAUNCH") == "1"   # opt out of the persistent-executable path

# built kernel callables keyed by source hash; see TrainiumProgram._build_kernel for why this matters
_KERNEL_CACHE:dict = {}
# in-process CompiledKernel (loaded NEFF) keyed by (src_hash, arg signature); see _compiled_for
_COMPILED_CACHE:dict = {}
# cross-process on-disk NEFF cache root: a fresh process reuses an existing kernel.neff (skips neuron-cc)
_NEFF_CACHE_DIR = os.getenv("TRAINIUM_NEFF_CACHE", os.path.join(tempfile.gettempdir(), "tinygrad_nki_neff"))

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
    self._srchash = hashlib.sha256(self.src.encode()).hexdigest()[:16]
    if os.getenv("NKI_SRC"): print(self.src)
    # a constfill is folded in the runtime (no device kernel), so don't build one
    self.kernel = None if self.meta.get("kind") == "constfill" else self._build_kernel(name, self.src)

  @staticmethod
  def _build_kernel(name:str, src:str):
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    # Cache the kernel CALLABLE by source hash. nki.jit's NEFF compile-cache lives ON the function
    # object (func._nki_compile_cache), so a stable func per unique source lets every realization
    # of an identical kernel -- across Program rebuilds and the >128-row tiling loop -- reuse the
    # same compiled NEFF instead of re-running neuron-cc (~11s each). Keyed by source, not (name,
    # shapes): the source already encodes the kernel, and nki keys its own cache by arg shapes.
    if (cached := _KERNEL_CACHE.get(h)) is not None: return cached
    # The sim just needs the callable, so exec'ing the source is enough. The HW compiler frontend,
    # however, looks the entry function up by source location (AST/linecache), so an exec'd kernel
    # fails with "entry function not found" -- it must live in a real importable .py file.
    if not _HW:
      ns:dict = {}
      exec(compile(src, f"<nki:{name}>", "exec"), ns)   # defines `kernel`
      kernel = ns["kernel"]
    else:
      import importlib.util
      d = os.path.join(tempfile.gettempdir(), "tinygrad_nki"); os.makedirs(d, exist_ok=True)
      path = os.path.join(d, f"k_{h}.py")
      if not os.path.exists(path):
        with open(path, "w") as f: f.write(src)
      spec = importlib.util.spec_from_file_location(f"tinygrad_nki_k_{h}", path)
      mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
      kernel = mod.kernel
    _KERNEL_CACHE[h] = kernel
    return kernel

  def _compiled_for(self, np_args):
    # Build (once) and cache a CompiledKernel specialized to these args' shapes/dtypes. The loaded NEFF
    # lives on CompiledKernel._model (lazy, loaded on first .run), so caching this object turns the per
    # call cost from "reload NEFF + re-init runtime" (~1.2s) into "launch" (~0.2ms).
    # Two cache levels: in-process (_COMPILED_CACHE) and on-disk by (src_hash, sig). On a disk hit a fresh
    # process recomputes only the cheap `bir` (needed for run()) and reuses the cached kernel.neff,
    # skipping neuron-cc (~1.2s -> ~0.1s). neuron-cc requires a CLEAN output dir, so a miss compiles into
    # a freshly-emptied per-(kernel,signature) dir.
    sig = tuple((tuple(a.shape), a.dtype.str) for a in np_args)
    key = (self._srchash, sig)
    if (hit := _COMPILED_CACHE.get(key)) is not None: return hit
    from dataclasses import replace
    from nki.framework.compiled import StandaloneKernel
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.ncc_driver import compile_bir_to_neff, CompiledKernel
    sk = self.kernel._to_subclass(StandaloneKernel)
    inputs = dict(sk._bind_args(np_args, {}))                          # {param_name: ndarray}
    frontend = sk._frontend_cls(enable_backend_opt=sk._enable_backend_opt)
    sighash = hashlib.sha256(repr(sig).encode()).hexdigest()[:16]      # stable per-signature dir (no hash collisions)
    cdir = os.path.join(_NEFF_CACHE_DIR, f"{self._srchash}_{sighash}"); neff = os.path.join(cdir, "kernel.neff")
    if os.path.exists(neff):    # cross-process disk hit: skip neuron-cc, recompute bir into a throwaway dir
      tmp = tempfile.mkdtemp(prefix="tinygrad_nki_bir_")
      copts = replace(sk._compile_opts(), artifacts_dir=tmp, output_path=os.path.join(tmp, "kernel.neff"))
      bir = compile_to_bir(sk, frontend=frontend, inputs=inputs, compile_opts=copts)
      compiled = CompiledKernel(neff_path=neff, target=copts.target, lnc=copts.lnc, artifacts_dir=tmp, bir=bir)
    else:                       # miss: full compile into a clean persistent dir
      shutil.rmtree(cdir, ignore_errors=True); os.makedirs(cdir)
      copts = replace(sk._compile_opts(), artifacts_dir=cdir, output_path=neff)
      bir = compile_to_bir(sk, frontend=frontend, inputs=inputs, compile_opts=copts)
      compiled = compile_bir_to_neff(copts, bir, input_arrays=[],
                                     argument_names=[s.name for s in bir.descriptor.input_specs],
                                     output_arg_names=[s.name for s in bir.descriptor.output_specs])
    _COMPILED_CACHE[key] = (hit := (compiled, list(inputs.keys())))
    return hit

  def _run(self, *np_args):
    # one place the two paths diverge: real device vs CPU simulator. numpy in -> numpy out either way.
    if not _HW:
      import nki
      return np.asarray(nki.simulate(self.kernel)(*np_args))
    if _SIMPLE_LAUNCH: return np.asarray(self.kernel(*np_args))    # high-level path: reloads NEFF per call
    compiled, names = self._compiled_for(np_args)                  # persistent executable: load once, launch many
    res = compiled.run(**dict(zip(names, np_args)))
    return np.asarray(next(iter(res.outputs.values())))

  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    m = self.meta
    if m.get("kind") == "constfill":   # pure-constant tensor (Tensor.full/zeros/ones): fill in numpy, no device
      d = np.dtype(m["out_dtype"]); n = len(bufs[m["out_slot"]]) // d.itemsize
      bufs[m["out_slot"]][:] = np.full(n, m["value"], dtype=d).tobytes()
      return None
    if m.get("kind") == "matmul":   # nc_matmul fast path: build A as (Bp,K,M), B as (Bp,K,N) strided views
      batch = m["batch"]; Bp = int(np.prod(batch)) if batch else 1; Ksizes = m["Ksizes"]
      def view(spec, shape):
        # strides come from the renderer (0 on a batch axis = broadcast that operand). The contraction may
        # span >1 axis (Ksizes), e.g. a reshape merging heads*head_dim -- materialize then flatten below.
        d = np.dtype(spec["dtype"]); flat = np.frombuffer(bufs[spec["param_slot"]], dtype=d)
        return np.ascontiguousarray(np.lib.stride_tricks.as_strided(
          flat[spec["offset"]:], shape=shape, strides=[s*d.itemsize for s in spec["strides"]]))
      args = [view(m["A"], (*batch, *Ksizes, m["M"])).reshape(Bp, m["K"], m["M"]),
              view(m["B"], (*batch, *Ksizes, m["N"])).reshape(Bp, m["K"], m["N"])]
      args += [view(p, (*batch, m["M"], m["N"])).reshape(Bp, m["M"], m["N"]) for p in m["post"]]  # -> (Bp,M,N)
      out = self._run(*args)                                    # kernel returns (Bp, M, N)
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
    PMAX, FMAX, FFLAT = 128, 4096, 8192   # partition cap 128; free chunk for SBUF; (1,N) is safe up to FFLAT
    if m.get("flat"):        # split==0 AND no reduce (a full reduce is also split==0 but outputs (P,1))
      # pure flat elementwise. A single (1,N) tile is fine (and matches the proven small-tensor path) until
      # N is large enough to overflow the 192KB/partition SBUF; only then spread N across <=128 partitions
      # (and free-chunk if still wide). Elementwise is position-preserving, so any reshape covering N works
      # as long as we flatten the result back. (Spreading only when N>FFLAT also avoids F==1 tiles, which
      # fail MLIR verification for fp16.)
      N = max(a.shape[1] for a in in_arrs)
      if N <= FFLAT:
        out = self._run(*in_arrs)
      else:
        Pn = min(PMAX, N); F = -(-N // Pn); pad = Pn*F - N
        def spread(a):
          if a.shape[1] == 1: return np.broadcast_to(a, (Pn, F)).copy()    # scalar/broadcast operand
          flat = a.reshape(-1)
          if pad: flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
          return flat.reshape(Pn, F)
        sa = [spread(a) for a in in_arrs]
        if F <= FMAX: out = self._run(*sa)
        else: out = np.concatenate([self._run(*[c[:, j:j+FMAX] for c in sa]) for j in range(0, F, FMAX)], axis=1)
      out = np.asarray(out).reshape(-1)[:N]
    else:
      rows = in_arrs[0].shape[0]
      if rows <= PMAX:
        out = self._run(*in_arrs)
      else:
        out = np.concatenate([self._run(*[a[i:i+PMAX] for a in in_arrs]) for i in range(0, rows, PMAX)], axis=0)
    self._writeback(bufs, m, out)
    return None

  @staticmethod
  def _writeback(bufs, m, out):
    # place the (kept-axis) result into the output buffer. Fast path: a normal full contiguous output is
    # written densely. Otherwise (assign-into-a-slice, e.g. the KV cache) scatter via the out strides/
    # offset and keep the rest of the buffer intact.
    d = np.dtype(m["out_dtype"]); slot = m["out_slot"]; nbuf = len(bufs[slot]) // d.itemsize
    val = np.ascontiguousarray(out, dtype=d)
    kept, off, ostr = m["out_kept"], m["out_off"], m["out_strides"]
    contig = ostr == [int(np.prod(kept[i+1:])) for i in range(len(kept))]
    if off == 0 and val.size == nbuf and contig:
      bufs[slot][:] = val.tobytes(); return
    full = np.frombuffer(bufs[slot], dtype=d).copy()
    np.lib.stride_tricks.as_strided(full[off:], shape=kept, strides=[s*d.itemsize for s in ostr])[...] = val.reshape(kept)
    bufs[slot][:] = full.tobytes()

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer], TrainiumProgram)
