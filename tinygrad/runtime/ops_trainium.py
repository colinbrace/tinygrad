# Trainium backend. One device (DEV=TRAINIUM) renders NKI via NKIRenderer, then executes it through one
# of two pluggable backends chosen once at construction (NOT a separate device -- the rendered source is
# identical, only the execution call differs):
#   - SIM  (default):       nki.simulate(kernel)(*args)        -- CPU simulator, no hardware
#   - HW   (TRAINIUM_HW=1):  a PERSISTENT EXECUTABLE -- compile each kernel+signature to a NEFF once, load
#     it onto the NeuronCore once, then launch many (warm launch ~0.2ms vs ~1.2s for the high-level
#     kernel(*args) path, which reloads the NEFF every call). TRAINIUM_SIMPLE_LAUNCH=1 forces that simpler
#     (more stable, much slower) high-level path.
#
# TrainiumProgram.__call__ dispatches on the kernel KIND (set by the renderer's meta): constfill / matmul /
# generic (elementwise+reduce+gather). _run() delegates to the bound sim-or-hw executor.
# See scratch/ml/theory/tinygrad-notes/backend_design.md
import json, os, tempfile, shutil, hashlib
import numpy as np
from tinygrad.device import Compiled, Allocator, Compiler
from tinygrad.renderer.nki import NKIRenderer

_HW = os.getenv("TRAINIUM_HW") == "1"
_SIMPLE_LAUNCH = os.getenv("TRAINIUM_SIMPLE_LAUNCH") == "1"   # opt out of the persistent-executable path
_NO_RESIDENT = os.getenv("TRAINIUM_NO_RESIDENT") == "1"       # opt out of on-core weight residency (Goal 3)

# [Goal 3] residency stats, for verification: from_numpy (host->device DMA) count vs cache reuse count.
# A weight should DMA once and then only ever be reused, no matter how many forward passes run.
_RESIDENT_STATS = {"dma": 0, "reuse": 0}

# built kernel callables keyed by source hash; see TrainiumProgram._build_kernel for why this matters
_KERNEL_CACHE:dict = {}
# in-process CompiledKernel (loaded NEFF) keyed by (src_hash, arg signature); see _compiled_for
_COMPILED_CACHE:dict = {}
# cross-process on-disk NEFF cache root: a fresh process reuses an existing kernel.neff (skips neuron-cc)
_NEFF_CACHE_DIR = os.getenv("TRAINIUM_NEFF_CACHE", os.path.join(tempfile.gettempdir(), "tinygrad_nki_neff"))
# loaded SpikeModel per (src_hash, arg sig, core_id) for multi-core (per-NeuronCore) execution
_MODEL_CACHE:dict = {}

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
    buf = np.frombuffer(bufs[node["slot"]].host, dtype=np.dtype(node["dtype"]))
    valid = (sub >= 0) & (sub < len(buf))
    return np.where(valid, buf[np.clip(sub, 0, len(buf)-1)], 0).astype(np.int64)
  if t == "alu": return _ALU[node["op"]](*[_eval_index(x, shape, bufs) for x in node["s"]])
  raise NotImplementedError(f"NKI gather eval: {t}")

class TrainiumCompiler(Compiler):
  # SIM path: pass the rendered NKI source through as bytes. (HW path later: neuron-cc -> NEFF.)
  def compile(self, src:str) -> bytes: return src.encode()

class TrainiumBuffer:
  # [Goal 3] The allocator opaque, enriched from a bare memoryview to {host bytes + version + per-core
  # device-tensor cache}. `ver` bumps on EVERY host-side write (copyin / kernel writeback / constfill);
  # `dev` caches on-NeuronCore SpikeTensors keyed by (core, param_name, view-signature) together with the
  # `ver` they were DMA'd at. A read-only weight is copyin'd once -> ver never moves again -> its device
  # tensor is DMA'd once and reused across every forward pass (on-core residency). An activation/output is
  # rewritten each pass -> ver bumps -> its stale cache entry is rejected and it is correctly re-DMA'd.
  __slots__ = ("host", "ver", "dev", "dmas")
  def __init__(self, size:int): self.host = memoryview(bytearray(size)); self.ver = 0; self.dev = {}; self.dmas = 0
  def __len__(self): return len(self.host)
  def write(self, mv): self.host[:] = mv; self.ver += 1   # any host->buffer write invalidates resident copies

class TrainiumAllocator(Allocator['TrainiumDevice']):
  def _alloc(self, size, options): return TrainiumBuffer(size)
  def _copyin(self, dest, src:memoryview): dest.write(src)
  def _copyout(self, dest:memoryview, src): dest[:] = src.host
  def _as_buffer(self, src) -> memoryview: return src.host   # zero-copy view (DISK copies / as_memoryview)

class _Operand:
  # A lazily-materialized kernel argument. shape/dtype are known up front (enough for the compile signature
  # and the per-core model/residency cache keys), but the expensive CONTIGUOUS numpy view is built only when
  # we actually need to DMA it -- and NOT when the operand is already resident on the core (its SpikeTensor
  # is cached). That skip is the saved work: a resident weight's ~MB view is no longer rebuilt every launch.
  __slots__ = ("shape", "dtype", "src", "_build", "_arr")
  def __init__(self, shape, dtype, src, build=None, arr=None):
    self.shape, self.dtype, self.src, self._build, self._arr = tuple(shape), np.dtype(dtype), src, build, arr
  def array(self):
    if self._arr is None: self._arr = self._build()
    return self._arr

def _eager(arrs, srcs=None):
  # wrap already-materialized arrays as _Operands (the tiled/spread elementwise paths must build their views
  # anyway). residency still applies via `srcs`; build() is a no-op since the array is already present.
  srcs = srcs if srcs is not None else [None]*len(arrs)
  return [_Operand(a.shape, a.dtype, s, arr=a) for a, s in zip(arrs, srcs)]

class TrainiumProgram:
  def __init__(self, name:str, lib:bytes, *aux, runtimevars=None, prg=None, core_id:int=0, **kwargs):
    self.name, self.src, self.core_id = name, lib.decode(), core_id   # core_id: which NeuronCore (multi-core)
    self.meta = json.loads(self.src.splitlines()[0].split("TRAINIUM_META", 1)[1])
    self._srchash = hashlib.sha256(self.src.encode()).hexdigest()[:16]
    if os.getenv("NKI_SRC"): print(self.src)
    # a constfill is folded in the runtime (no device kernel), so don't build one
    self.kernel = None if self.meta.get("kind") == "constfill" else self._build_kernel(name, self.src)
    self._execute = self._run_hw if _HW else self._run_sim   # the pluggable execution backend

  # ---- kernel build (the rendered NKI source -> a callable) ----
  @staticmethod
  def _build_kernel(name:str, src:str):
    # Cache the kernel CALLABLE by source hash. nki.jit's NEFF compile-cache lives ON the function
    # object (func._nki_compile_cache), so a stable func per unique source lets every realization
    # of an identical kernel -- across Program rebuilds and the >128-row tiling loop -- reuse the
    # same compiled NEFF instead of re-running neuron-cc (~11s each). Keyed by source, not (name,
    # shapes): the source already encodes the kernel, and nki keys its own cache by arg shapes.
    h = hashlib.sha256(src.encode()).hexdigest()[:16]
    if (cached := _KERNEL_CACHE.get(h)) is not None: return cached
    kernel = TrainiumProgram._kernel_from_file(h, src) if _HW else TrainiumProgram._kernel_from_exec(name, src)
    _KERNEL_CACHE[h] = kernel
    return kernel

  @staticmethod
  def _kernel_from_exec(name:str, src:str):
    # SIM: just exec the source to get the callable (the simulator doesn't care where it's defined).
    ns:dict = {}
    exec(compile(src, f"<nki:{name}>", "exec"), ns)   # defines `kernel`
    return ns["kernel"]

  @staticmethod
  def _kernel_from_file(h:str, src:str):
    # HW: the compiler frontend looks the entry function up by source LOCATION (AST/linecache), so an
    # exec'd kernel fails with "entry function not found" -- it must live in a real importable .py file.
    import importlib.util
    d = os.path.join(tempfile.gettempdir(), "tinygrad_nki"); os.makedirs(d, exist_ok=True)
    path = os.path.join(d, f"k_{h}.py")
    if not os.path.exists(path):
      with open(path, "w") as f: f.write(src)
    spec = importlib.util.spec_from_file_location(f"tinygrad_nki_k_{h}", path)
    mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
    return mod.kernel

  # ---- execution backends (bound to self._execute at construction) ----
  def _run(self, operands):
    # operands: list[_Operand]. numpy in -> numpy out. On HW, a resident operand (op.src unchanged since its
    # last DMA) skips BOTH its host->device DMA and its view materialization; sim / legacy paths build all.
    return self._execute(operands)

  def _run_sim(self, operands):
    # CPU simulator: no hardware, no residency -> materialize every operand.
    import nki
    return np.asarray(nki.simulate(self.kernel)(*[op.array() for op in operands]))

  def _run_hw(self, operands):
    # real NeuronCore. Default: the persistent executable (load the NEFF once, launch many).
    if _SIMPLE_LAUNCH: return np.asarray(self.kernel(*[op.array() for op in operands]))   # reloads NEFF per call
    compiled, names = self._compiled_for(operands)                 # persistent executable: load once, launch many
    if _NO_RESIDENT and self.core_id == 0:                         # legacy path: well-tested high-level run, no residency
      res = compiled.run(**{n: op.array() for n, op in zip(names, operands)})
      return np.asarray(next(iter(res.outputs.values())))
    return self._run_on_core(compiled, names, operands, residency=not _NO_RESIDENT)

  def _compiled_for(self, operands):
    # Build (once) and cache a CompiledKernel specialized to these args' shapes/dtypes. The loaded NEFF
    # lives on CompiledKernel._model (lazy, loaded on first .run), so caching this object turns the per
    # call cost from "reload NEFF + re-init runtime" (~1.2s) into "launch" (~0.2ms).
    # Two cache levels: in-process (_COMPILED_CACHE) and on-disk by (src_hash, sig). On a disk hit a fresh
    # process recomputes only the cheap `bir` (needed for run()) and reuses the cached kernel.neff,
    # skipping neuron-cc (~1.2s -> ~0.1s). neuron-cc requires a CLEAN output dir, so a miss compiles into
    # a freshly-emptied per-(kernel,signature) dir.
    sig = tuple((op.shape, op.dtype.str) for op in operands)
    key = (self._srchash, sig)
    if (hit := _COMPILED_CACHE.get(key)) is not None: return hit      # warm: never materializes the operands
    from dataclasses import replace
    from nki.framework.compiled import StandaloneKernel
    from nki.compiler.driver import compile_to_bir
    from nki.compiler.ncc_driver import compile_bir_to_neff, CompiledKernel
    sk = self.kernel._to_subclass(StandaloneKernel)
    inputs = dict(sk._bind_args(tuple(op.array() for op in operands), {}))   # cold compile only: needs arrays to trace
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

  def _run_on_core(self, compiled, names, operands, residency=True):
    # Run a cached NEFF on a SPECIFIC NeuronCore. The high-level CompiledKernel.run() / _ensure_loaded()
    # hardcode core 0 (SpikeModel.load_from_neff(neff_path) with the default core_id=0), so for core N we
    # load our own SpikeModel with core_id=N and place every I/O SpikeTensor on core N. The loaded model
    # is cached per (kernel, signature, core) and reused, so repeat launches are just DMA + execute.
    # This mirrors CompiledKernel.run() (from_numpy inputs, prepare_outputs, model(in, outputs=)), threading
    # core_id throughout, PLUS [Goal 3] on-core residency: when an input's source buffer is unchanged
    # (same ver) since its last DMA we reuse the resident SpikeTensor instead of re-DMA'ing it. Kernels
    # never write their inputs (the renderer emits separate output tensors), so reusing an input is safe.
    from nki.runtime import SpikeModel, SpikeTensor
    key = (self._srchash, tuple((op.shape, op.dtype.str) for op in operands), self.core_id)
    if (model := _MODEL_CACHE.get(key)) is None:
      model = _MODEL_CACHE[key] = SpikeModel.load_from_neff(compiled.neff_path, core_id=self.core_id)
    si = {}
    for n, op in zip(names, operands):
      s = op.src if residency else None
      ck = (self.core_id, n, s[1]) if s is not None else None      # cache key: core + param name + view sig
      if s is not None and (hit := s[0].dev.get(ck)) is not None and hit[1] == s[0].ver:
        si[n] = hit[0]; _RESIDENT_STATS["reuse"] += 1; continue    # resident: skip BOTH the view build AND the DMA
      si[n] = SpikeTensor.from_numpy(np.ascontiguousarray(op.array()), n, core_id=self.core_id); _RESIDENT_STATS["dma"] += 1
      if s is not None: s[0].dev[ck] = (si[n], s[0].ver); s[0].dmas += 1   # remember this resident copy; count DMAs of this buffer
    so = {n: SpikeTensor.from_numpy(v, n, core_id=self.core_id) for n, v in compiled.prepare_outputs().items()}
    model(si, outputs=so)
    return np.asarray(next(iter(so.values())).numpy())

  # ---- kernel dispatch (by KIND) ----
  def __call__(self, *bufs, global_size=(1,1,1), local_size=(1,1,1), vals=(), wait=False, **kwargs):
    kind = self.meta.get("kind")
    if kind == "constfill": return self._run_constfill(bufs)
    if kind == "matmul":    return self._run_matmul(bufs)
    return self._run_elementwise(bufs)    # generic: elementwise / reduce / gather / flat

  def _run_constfill(self, bufs):
    # pure-constant tensor (Tensor.full/zeros/ones, e.g. gpt2 mask / KV cache): fill in numpy, no device.
    m = self.meta
    d = np.dtype(m["out_dtype"]); n = len(bufs[m["out_slot"]]) // d.itemsize
    bufs[m["out_slot"]].write(np.full(n, m["value"], dtype=d).tobytes())
    return None

  def _run_matmul(self, bufs):
    # nc_matmul fast path: build A as (Bp,K,M), B as (Bp,K,N) strided views; the kernel returns (Bp,M,N).
    m = self.meta
    batch = m["batch"]; Bp = int(np.prod(batch)) if batch else 1; Ksizes = m["Ksizes"]
    def operand(spec, view_shape, role, final_shape):
      # strides come from the renderer (0 on a batch axis = broadcast). The contraction may span >1 axis
      # (Ksizes), e.g. a reshape merging heads*head_dim. LAZY: the contiguous view is built only if this
      # operand isn't already resident -- a matmul operand is overwhelmingly a read-only weight, so once it
      # is on the core, rebuilding its ~MB view every launch is pure waste. src = (buffer, view-sig).
      d = np.dtype(spec["dtype"])
      src = (bufs[spec["param_slot"]], (role, spec["offset"], tuple(spec["strides"]), tuple(view_shape)))
      def build():
        flat = np.frombuffer(bufs[spec["param_slot"]].host, dtype=d)
        return np.ascontiguousarray(np.lib.stride_tricks.as_strided(
          flat[spec["offset"]:], shape=view_shape, strides=[s*d.itemsize for s in spec["strides"]])).reshape(final_shape)
      return _Operand(final_shape, d, src, build=build)
    ops = [operand(m["A"], (*batch, *Ksizes, m["M"]), "A", (Bp, m["K"], m["M"])),
           operand(m["B"], (*batch, *Ksizes, m["N"]), "B", (Bp, m["K"], m["N"]))]
    ops += [operand(p, (*batch, m["M"], m["N"]), f"P{i}", (Bp, m["M"], m["N"])) for i, p in enumerate(m["post"])]
    out = self._run(ops)                                         # kernel returns (Bp, M, N)
    bufs[m["out_slot"]].write(np.ascontiguousarray(out, dtype=np.dtype(m["out_dtype"])).tobytes())
    return None

  def _run_elementwise(self, bufs):
    # generic iteration-space kernel (elementwise / reduce / gather). Build each input INDEX as a strided
    # view over the canonical iteration space: full size on PARTITION (kept) axes, natural size on FREE
    # axes (1 where stride is 0). A 0 stride broadcasts, a nonzero stride + offset reads contiguous/
    # transpose/slice -- all uniformly. Then reshape (P, F), tile partition <=128, and write back.
    m = self.meta
    cs, split, nd = m["canonical_sizes"], m["split"], len(m["canonical_sizes"])
    P = int(np.prod(cs[:split])) if split else 1
    in_arrs = []; srcs = []   # srcs[i] = (buffer, view-sig) for residency, or None (synthetic/non-affine)
    for inp in m["inputs"]:
      if inp["kind"] == "iota":   # a RANGE used as a value -> coordinate of canonical axis over the grid
        ax = inp["axis"]
        sh = [1]*nd; sh[ax] = cs[ax]
        coord = np.arange(cs[ax]).reshape(sh) + np.zeros(cs, dtype=np.int64)   # broadcast to full grid
        in_arrs.append(coord.reshape(P, int(np.prod(cs[split:])) or 1).astype(np.float32)); srcs.append(None)
        continue
      d = np.dtype(inp["dtype"])
      flat = np.frombuffer(bufs[inp["param_slot"]].host, dtype=d)
      if inp["kind"] == "gather":   # data-dependent index: eval offsets over the grid, then gather
        idx = _eval_index(inp["index"], cs, bufs)
        valid = (idx >= 0) & (idx < len(flat))      # gated load: Invalid/OOB index -> 0 (e.g. cat/pad)
        view = np.where(valid, flat[np.clip(idx, 0, len(flat)-1)], 0)
        shape = cs; srcs.append(None)               # data-dependent -> not safely resident
      else:                         # affine: a strided view (full partition, natural free)
        st = inp["strides"]
        shape = [cs[c] if (c < split or st[c] != 0) else 1 for c in range(nd)]
        view = np.lib.stride_tricks.as_strided(flat[inp["offset"]:], shape=shape, strides=[st[c]*d.itemsize for c in range(nd)])
        srcs.append((bufs[inp["param_slot"]], ("ew", inp["offset"], tuple(st), tuple(shape))))
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
        out = self._run(_eager(in_arrs, srcs))    # full (1,N) views -> residency keys stay valid
      else:
        Pn = min(PMAX, N); F = -(-N // Pn); pad = Pn*F - N
        def spread(a):
          if a.shape[1] == 1: return np.broadcast_to(a, (Pn, F)).copy()    # scalar/broadcast operand
          flat = a.reshape(-1)
          if pad: flat = np.concatenate([flat, np.zeros(pad, flat.dtype)])
          return flat.reshape(Pn, F)
        sa = [spread(a) for a in in_arrs]
        if F <= FMAX: out = self._run(_eager(sa))
        else: out = np.concatenate([self._run(_eager([c[:, j:j+FMAX] for c in sa])) for j in range(0, F, FMAX)], axis=1)
      out = np.asarray(out).reshape(-1)[:N]
    else:
      rows = in_arrs[0].shape[0]
      if rows <= PMAX:
        out = self._run(_eager(in_arrs, srcs))    # un-tiled: full views -> residency keys stay valid
      else:                                       # tiled over partitions: args are row-slices, so re-DMA each
        out = np.concatenate([self._run(_eager([a[i:i+PMAX] for a in in_arrs])) for i in range(0, rows, PMAX)], axis=0)
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
      bufs[slot].write(val.tobytes()); return     # write() bumps ver so any resident copy of this buffer is dropped
    full = np.frombuffer(bufs[slot].host, dtype=d).copy()
    np.lib.stride_tricks.as_strided(full[off:], shape=kept, strides=[s*d.itemsize for s in ostr])[...] = val.reshape(kept)
    bufs[slot].write(full.tobytes())

class TrainiumDevice(Compiled):
  def __init__(self, device:str):
    import functools
    # multi-core: "TRAINIUM" / "TRAINIUM:0" -> core 0, "TRAINIUM:N" -> core N. tinygrad shards across
    # TRAINIUM:0..3; each device runs its kernels on its own NeuronCore (load_from_neff core_id).
    core_id = int(device.split(":")[1]) if ":" in device else 0
    super().__init__(device, TrainiumAllocator(self), [NKIRenderer],
                     functools.partial(TrainiumProgram, core_id=core_id))
