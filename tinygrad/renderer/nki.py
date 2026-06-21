# NKI renderer: tinygrad UOp graph -> AWS NKI (neuronxcc.nki.language) source.
# "Option B": consumes the HIGH-LEVEL form (Ops.REDUCE / ALU / INDEX over ranges,
# before tinygrad lowers reduce into a scalar accumulator loop -- see the
# render_high_level hook in codegen/__init__.py). NKI is a tile-op machine, so we
# map high-level ops directly: ALU -> nl.*, Ops.REDUCE -> nl.sum/nl.max.
#
# render() parses the UOp graph once, then dispatches on the kernel's SHAPE to one of three emitters:
#   constfill  -- a pure-constant store (no inputs, no reduce): filled in the runtime, no kernel
#   matmul     -- a sum-of-products of two inputs: a tiled nl.matmul (nc_matmul) fast path
#   generic    -- everything else: a unified iteration-space elementwise / reduce / gather kernel
# See scratch/ml/theory/tinygrad-notes/backend_design.md.
import os, json, math
import numpy as np
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp, AxisType
from tinygrad.dtype import _to_np_dtype, dtypes

NL_BINOP = {Ops.ADD:"nl.add", Ops.MUL:"nl.multiply", Ops.SUB:"nl.subtract",
            Ops.MAX:"nl.maximum", Ops.FDIV:"nl.divide", Ops.CDIV:"nl.divide",
            Ops.CMPLT:"nl.less", Ops.CMPNE:"nl.not_equal", Ops.CMPEQ:"nl.equal"}
NL_UNOP  = {Ops.NEG:"nl.negative", Ops.SQRT:"nl.sqrt", Ops.RECIPROCAL:"nl.reciprocal",
            Ops.SIN:"nl.sin"}
NL_REDUCE = {Ops.ADD:"nl.sum", Ops.MAX:"nl.max", Ops.MUL:"nl.prod"}
NL_BOOL = {Ops.AND:"and", Ops.OR:"or", Ops.XOR:"xor"}   # logical_* for bool, bitwise_* for int (see _emit)
LN2 = 0.6931471805599453

def _affine(u:UOp) -> tuple[dict, int]:
  # parse an integer index expr into ({RANGE uop: stride}, constant offset). strides index the FLAT
  # buffer, so this captures contiguous / transpose / slice (offset) / broadcast (stride 0) uniformly.
  if u.op is Ops.RANGE: return {u: 1}, 0
  if u.op is Ops.CONST: return {}, int(u.arg)
  if u.op is Ops.ADD:
    d:dict = {}; off = 0
    for s in u.src:
      ds, o = _affine(s)
      for k,v in ds.items(): d[k] = d.get(k, 0) + v
      off += o
    return d, off
  if u.op is Ops.MUL:
    (da, oa), (db, ob) = _affine(u.src[0]), _affine(u.src[1])
    if not da: return {k: v*oa for k,v in db.items()}, oa*ob
    if not db: return {k: v*ob for k,v in da.items()}, oa*ob
  raise NotImplementedError(f"NKI: non-affine index op {u.op}")

def _rsize(r:UOp) -> int: return int(r.src[0].arg)
def _is_reduce(r:UOp) -> bool: return r.arg[1] is AxisType.REDUCE

def _depends_on(u:UOp, target:UOp) -> bool:
  # does `target` appear anywhere in u's source DAG? (used to order nested reduces inner-first)
  seen, stack = set(), [u]
  while stack:
    x = stack.pop()
    if x in seen: continue
    seen.add(x)
    for s in x.src:
      if s is target: return True
      stack.append(s)
  return False

def _has_range(u:UOp, seen:set|None=None) -> bool:
  # does the expression contain a RANGE (loop coordinate)? a constfill has none.
  seen = set() if seen is None else seen
  if u in seen: return False
  seen.add(u)
  return u.op is Ops.RANGE or any(_has_range(s, seen) for s in u.src)

# ALU ops that can appear inside a data-dependent (gather) index expression
ALU_SER = {Ops.ADD:"ADD", Ops.MUL:"MUL", Ops.SUB:"SUB", Ops.MAX:"MAX", Ops.CMPLT:"CMPLT",
           Ops.CMPNE:"CMPNE", Ops.CMPEQ:"CMPEQ", Ops.AND:"AND", Ops.OR:"OR", Ops.XOR:"XOR",
           Ops.FLOORDIV:"FLOORDIV", Ops.FLOORMOD:"FLOORMOD", Ops.CDIV:"CDIV", Ops.CMOD:"CMOD"}

def _ser_index(u:UOp, canonical:list, npname) -> dict:
  # serialize a (possibly data-dependent) index expression so the runtime can evaluate it over the
  # iteration grid -> flat offsets -> gather. RANGE -> a grid coordinate; INDEX(param,...) -> a load.
  if u.op is Ops.RANGE: return {"t":"rng", "ax":canonical.index(u)}
  if u.op is Ops.CONST:
    try: v = int(u.arg)
    except (TypeError, ValueError): v = -1   # Invalid sentinel (OOB positions get clipped)
    return {"t":"const", "v":v}
  if u.op is Ops.CAST: return {"t":"cast", "s":[_ser_index(u.src[0], canonical, npname)]}
  if u.op is Ops.WHERE: return {"t":"where", "s":[_ser_index(x, canonical, npname) for x in u.src]}
  if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM:
    return {"t":"load", "slot":u.src[0].arg.slot, "dtype":npname(u.src[0].dtype), "s":[_ser_index(u.src[1], canonical, npname)]}
  if u.op in ALU_SER: return {"t":"alu", "op":ALU_SER[u.op], "s":[_ser_index(x, canonical, npname) for x in u.src]}
  raise NotImplementedError(f"NKI gather index: unhandled {u.op}")

class NKIRenderer(Renderer):
  suffix = "NKI"
  has_local = False
  has_shared = False
  supports_float4 = False
  code_for_op = {op: None for op in (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL)}
  disable_opts = True          # tile machine: skip tinygrad's scalar-loop opts (UPCAST/...)
  render_high_level = True     # consume Ops.REDUCE / ALU / INDEX before scalar lowering

  def _npname(self, dt) -> str:
    name = np.dtype(_to_np_dtype(dt.scalar())).name
    return "float32" if name == "float64" else name   # NKI/Trainium has no float64 -> compute in float32

  # shared expression emitter; `leaf(u)` resolves INDEX/REDUCE leaves to tile var names
  def _emit(self, u:UOp, leaf, ref:str) -> str:
    r = lambda x: self._emit(x, leaf, ref)
    lf = leaf(u)
    if lf is not None: return lf
    if u.op is Ops.CONST:
      if dtypes.is_float(u.dtype):
        v = float(u.arg)
        return repr(v) if math.isfinite(v) else f"float('{v}')"   # inf/-inf/nan aren't valid bare literals
      if dtypes.is_int(u.dtype):   return repr(int(u.arg))
      return repr(bool(u.arg))
    if u.op is Ops.CAST:   # only a real dtype change needs an nl.copy; same-dtype CAST is a no-op
      if u.dtype.scalar() != u.src[0].dtype.scalar():
        src = r(u.src[0])
        # float->int must truncate toward zero (tinygrad/C/numpy semantics). nl.copy(dtype=int)
        # rounds-to-nearest-even on real hardware (the simulator happens to truncate), so trunc first.
        if dtypes.is_float(u.src[0].dtype) and dtypes.is_int(u.dtype): src = f"nl.trunc({src})"
        return f"nl.copy({src}, dtype=nl.{self._npname(u.dtype)})"
      return r(u.src[0])
    if u.op is Ops.WHERE:
      tile = lambda s: f"nl.full({ref}.shape, {r(s)}, dtype={ref}.dtype)" if s.op is Ops.CONST else r(s)
      return f"nl.where({r(u.src[0])}, {tile(u.src[1])}, {tile(u.src[2])})"
    if u.op is Ops.MULACC: return f"nl.add(nl.multiply({r(u.src[0])}, {r(u.src[1])}), {r(u.src[2])})"
    if u.op is Ops.EXP2: return f"nl.exp(nl.multiply({r(u.src[0])}, {LN2!r}))"
    if u.op is Ops.LOG2: return f"nl.multiply(nl.log({r(u.src[0])}), {1.0/LN2!r})"
    if u.op is Ops.FLOORMOD: return f"nl.mod({r(u.src[0])}, {r(u.src[1])})"
    if u.op is Ops.FLOORDIV: return f"nl.floor(nl.divide({r(u.src[0])}, {r(u.src[1])}))"
    if u.op in NL_BOOL:   # boolean masks (xent one-hot) use logical_*; integer bitwise uses bitwise_*
      pre = "logical" if u.dtype == dtypes.bool else "bitwise"
      return f"nl.{pre}_{NL_BOOL[u.op]}({r(u.src[0])}, {r(u.src[1])})"
    if u.op in NL_BINOP: return f"{NL_BINOP[u.op]}({r(u.src[0])}, {r(u.src[1])})"
    if u.op in NL_UNOP:  return f"{NL_UNOP[u.op]}({r(u.src[0])})"
    raise NotImplementedError(f"NKI renderer: unhandled op {u.op}")

  def _source(self, sig:str, lines:list, meta:dict) -> str:
    return (f"# TRAINIUM_META {json.dumps(meta)}\n"
            "import nki\nimport nki.language as nl\n\n"
            f"@nki.jit\ndef kernel({sig}):\n" + "\n".join(lines) + "\n")

  def _match_matmul(self, store, ordered, kept_ranges, out_slot, out_dtype, in_index):
    # detect out[*b,m,n] = f(sum_k a[*b,m,k]*b[*b,k,n], <post-ops>): a sum-reduce of a product of two
    # indexed inputs over the trailing two kept axes (M,N), with any number of leading BATCH axes,
    # optionally fused with elementwise post-ops on (*b,M,N). Routes to a tiled nc_matmul, avoiding the
    # generic path's O(B*M*N*K) broadcast-materialize. (2D matmul is the batch==[] special case, Bp=1.)
    if len(ordered) != 1 or len(kept_ranges) < 2: return None
    rd = ordered[0]
    if rd.arg[0] is not Ops.ADD: return None
    strip = lambda u: strip(u.src[0]) if u.op is Ops.CAST else u
    body = strip(rd.src[0])
    if body.op is not Ops.MUL: return None
    xa, xb = strip(body.src[0]), strip(body.src[1])
    if not all(x.op is Ops.INDEX and x.src[0].op is Ops.PARAM for x in (xa, xb)): return None
    Kr = list(rd.src[1:]); *batch_rs, Mr, Nr = kept_ranges  # contraction = ALL reduce ranges (may be >1,
    (ca, oa), (cb, ob) = _affine(xa.src[1]), _affine(xb.src[1])   # e.g. a reshape merges (heads, head_dim));
    # orient so A spans (rows Mr, K...) and B spans (K..., cols Nr); swap operands if needed   # trailing
    if Mr in cb and Nr in ca: (xa, ca, oa), (xb, cb, ob) = (xb, cb, ob), (xa, ca, oa)          # two kept = M,N
    if not (Mr in ca and Nr in cb and all(k in ca and k in cb for k in Kr)): return None
    Msz, Nsz = _rsize(Mr), _rsize(Nr)
    Ksizes = [_rsize(k) for k in Kr]; Ksz = int(np.prod(Ksizes))   # flattened contraction width
    batch = [_rsize(r) for r in batch_rs]
    bs = lambda c: [c.get(r, 0) for r in batch_rs]           # per-batch-axis stride (0 = broadcast that operand)
    ks = lambda c: [c[k] for k in Kr]                        # per-contraction-axis stride
    # post-reduce operands (bias etc.): every other indexed input, affine over the kept axes only, as (*b,M,N)
    post = [u for u in in_index if u is not xa and u is not xb]
    post_meta = []
    for u in post:
      try: c, o = _affine(u.src[1])
      except NotImplementedError: return None                # data-dependent post-op -> generic path
      if any(k in c for k in Kr): return None                # a post-op may not span the contraction
      post_meta.append({"param_slot":u.src[0].arg.slot, "dtype":self._npname(u.src[0].dtype),
                        "strides":bs(c)+[c.get(Mr, 0), c.get(Nr, 0)], "offset":o})
    tpost = {u:i for i,u in enumerate(post)}
    def leaf(u):
      if u is rd: return "res_s"                             # the (tile of the) matmul result
      if u in tpost: return f"t{tpost[u]}"
      return None
    final = self._emit(store.src[1], leaf, "res_s")
    # A is passed as (Bp, K, M), B as (Bp, K, N) (K = flattened Ksizes). nc_matmul(res,A_kt,B_k)=A_kt.T@B_k.
    meta = {"kind":"matmul", "out_slot":out_slot, "out_dtype":out_dtype, "batch":batch, "M":Msz, "N":Nsz, "K":Ksz, "Ksizes":Ksizes,
            "A":{"param_slot":xa.src[0].arg.slot, "dtype":self._npname(xa.src[0].dtype), "strides":bs(ca)+ks(ca)+[ca[Mr]], "offset":oa},
            "B":{"param_slot":xb.src[0].arg.slot, "dtype":self._npname(xb.src[0].dtype), "strides":bs(cb)+ks(cb)+[cb[Nr]], "offset":ob},
            "post":post_meta}
    # Tiled Tensor-Engine matmul: loop leading batch (Bp=prod(batch), 1 if 2D); per batch, tile the output
    # into <=128 (M) x <=512 (N) blocks; for each, accumulate the contraction over <=128-wide K-blocks into
    # one PSUM tile (successive nc_matmul accumulate), copy to SBUF, apply post-ops, store. min(...) handles
    # ragged tails. Inputs are 3D (Bp,K,*) and output 3D (Bp,M,N); for 2D the Bp=1 axis flattens away.
    Bp = int(np.prod(batch)) if batch else 1
    i4, i8, i12, i16, i20 = "    ", "        ", "            ", "                ", "                    "
    lines = [f"{i4}out = nl.ndarray(({Bp}, {Msz}, {Nsz}), dtype=nl.{out_dtype}, buffer=nl.shared_hbm)",
             f"{i4}for bb in range({Bp}):",
             f"{i8}for mi in range(0, {Msz}, 128):",
             f"{i12}m1 = min(mi + 128, {Msz})",
             f"{i12}for ni in range(0, {Nsz}, 512):",
             f"{i16}n1 = min(ni + 512, {Nsz})",
             f"{i16}res = nl.ndarray((m1 - mi, n1 - ni), dtype=nl.float32, buffer=nl.psum)",
             f"{i16}for ki in range(0, {Ksz}, 128):",
             f"{i20}k1 = min(ki + 128, {Ksz})",
             f"{i20}a_t = nl.load(in0[bb, ki:k1, mi:m1])",
             f"{i20}b_t = nl.load(in1[bb, ki:k1, ni:n1])",
             f"{i20}nisa.nc_matmul(res, a_t, b_t)",
             f"{i16}res_s = nl.ndarray((m1 - mi, n1 - ni), dtype=nl.{out_dtype}, buffer=nl.sbuf)",
             f"{i16}nisa.tensor_copy(res_s, res)"]
    lines += [f"{i16}t{i} = nl.load(in{i+2}[bb, mi:m1, ni:n1])" for i in range(len(post))]
    lines += [f"{i16}nl.store(out[bb, mi:m1, ni:n1], value=nl.broadcast_to({final}, res_s.shape))",
              f"{i4}return out"]
    sig = ", ".join(["in0", "in1"] + [f"in{i+2}" for i in range(len(post))])
    return (f"# TRAINIUM_META {json.dumps(meta)}\nimport nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n"
            f"@nki.jit\ndef kernel({sig}):\n" + "\n".join(lines) + "\n")

  # ---- graph parsing (render() dispatches on the parsed shape) ----
  def _dump(self, uops:list[UOp]):
    idx = {u:i for i,u in enumerate(uops)}
    print("\n".join(f"{i:3} {str(u.op):16} {str(u.dtype):16} src={[idx.get(s,'?') for s in u.src]} arg={u.arg!r}"
                     for i,u in enumerate(uops)))

  @staticmethod
  def _ordered_reduces(reduces:list[UOp]) -> list[UOp]:
    # topological order: a reduce whose body contains another reduce is emitted LATER (inner reduces first).
    ordered, rem = [], list(reduces)
    while rem:
      nxt = next((rd for rd in rem if not any(_depends_on(rd.src[0], o) for o in rem if o is not rd)), rem[0])
      ordered.append(nxt); rem.remove(nxt)
    return ordered

  @staticmethod
  def _input_indices(uops:list[UOp], out_index:UOp, out_slot:int) -> list[UOp]:
    # Each distinct input INDEX node is its own tile -- so the same buffer read with two access patterns
    # (e.g. a @ a.T) becomes two tiles. The runtime passes a strided view per INDEX (as_strided reads
    # contiguous / transpose / slice / broadcast uniformly). May be empty: a coord-only kernel (arange).
    in_index:list[UOp] = []
    for u in uops:
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.slot != out_slot and u is not out_index and u not in in_index:
        in_index.append(u)
    return in_index

  def _render_constfill(self, store, out_slot, out_dtype, in_index, reduces) -> str|None:
    # pure-constant store (no input, no reduce, no coord): e.g. Tensor.full/zeros/ones (gpt2 mask,
    # KV cache). Const-fold in the runtime -- fill the output buffer directly, no kernel / device needed.
    if in_index or reduces or _has_range(store.src[1]): return None
    def _ceval(u):
      if u.op is Ops.CONST: return u.arg
      if u.op is Ops.CAST: return _ceval(u.src[0])
      raise NotImplementedError(f"NKI constfill: {u.op}")
    meta = {"kind":"constfill", "out_slot":out_slot, "out_dtype":out_dtype, "value":float(_ceval(store.src[1]))}
    return f"# TRAINIUM_META {json.dumps(meta)}\n# constfill (filled in the runtime; no kernel)\n"

  def render(self, uops:list[UOp]) -> str:
    if os.getenv("NKI_DUMP"): self._dump(uops)
    # --- parse the graph: one output STORE, its slot/dtype, the reduces (ordered), the input tiles ---
    params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    stores = [u for u in uops if u.op is Ops.STORE]
    if len(stores) != 1: raise NotImplementedError(f"NKI renderer: expected 1 store, got {len(stores)}")
    store = stores[0]
    out_index = store.src[0]
    out_slot = out_index.src[0].arg.slot
    out_dtype = self._npname(next(p for p in params if p.arg.slot == out_slot).dtype)
    reduces = [u for u in uops if u.op is Ops.REDUCE]
    for rd in reduces:
      if rd.arg[0] not in NL_REDUCE: raise NotImplementedError(f"NKI: reduce op {rd.arg[0]}")
    ordered = self._ordered_reduces(reduces)
    in_index = self._input_indices(uops, out_index, out_slot)

    # --- dispatch on kernel SHAPE: constfill, matmul fast path, or the generic iteration-space kernel ---
    if (cf := self._render_constfill(store, out_slot, out_dtype, in_index, reduces)) is not None: return cf
    kept_ranges = [r for r,_ in sorted(_affine(out_index.src[1])[0].items(), key=lambda kv: -kv[1])]
    if (mm := self._match_matmul(store, ordered, kept_ranges, out_slot, out_dtype, in_index)) is not None: return mm
    return self._render_generic(store, out_index, out_slot, out_dtype, in_index, ordered, kept_ranges)

  def _render_generic(self, store, out_index, out_slot, out_dtype, in_index, ordered, kept_ranges) -> str:
    # Unified iteration space: kept axes (LOOP, in OUTPUT order) + the reduced axis (if any).
    # partition|free split at `split` (runtime tiles partition <=128); reduce -> nl.sum/max over free.
    # A reduce may span >1 range -- conv (sum over cin,kh,kw) and pooling (max over ph,pw) reduce over a
    # whole window. For a SINGLE reduce we flatten all its ranges into the free dim and reduce the whole
    # free (correct: it IS one reduce over all those axes). For MULTIPLE reduces, mixing their ranges in one
    # free dim would cross-reduce them, so keep one-free-axis-per-reduce and guard multi-range there.
    if len(ordered) == 1:
      canonical = kept_ranges + list(ordered[0].src[1:])      # one reduce: all its ranges form the free dim
    else:
      if any(len(rd.src) > 2 for rd in ordered): raise NotImplementedError("NKI: multi-range reduce with multiple reduces")
      canonical = kept_ranges + [rd.src[1] for rd in ordered]   # partition=kept; each reduce gets a free axis
    split = len(kept_ranges) if ordered else max(0, len(kept_ranges) - 1)
    csize = [_rsize(r) for r in canonical]
    tid = {u:i for i,u in enumerate(in_index)}   # input INDEX -> tile var index (may be empty: coord-only)
    inputs_meta = []
    for u in in_index:
      base = {"param_slot":u.src[0].arg.slot, "dtype":self._npname(u.src[0].dtype)}
      try:                                   # affine index -> a strided view (fast path)
        coeffs, offset = _affine(u.src[1])
        if any(r not in canonical for r,c in coeffs.items() if c != 0): raise NotImplementedError("axis not in iteration space")
        inputs_meta.append({**base, "kind":"strided", "strides":[coeffs.get(r, 0) for r in canonical], "offset":offset})
      except NotImplementedError:            # data-dependent index (gather) -> serialize for runtime eval
        inputs_meta.append({**base, "kind":"gather", "index":_ser_index(u.src[1], canonical, self._npname)})

    ref = "t0" if in_index else "c0"   # coord-only kernels (arange) reference the first coord tile
    reduce_vars:dict = {}
    coords:dict = {}            # RANGE used as a value -> coordinate tile var (arange / iota / triangular masks)
    def leaf(u):
      if u in reduce_vars: return reduce_vars[u]
      if u in tid: return f"t{tid[u]}"
      if u.op is Ops.RANGE:
        if u not in coords:
          if u not in canonical: raise NotImplementedError("NKI: range-as-value not in iteration space")
          coords[u] = f"c{len(coords)}"
        return coords[u]
      return None
    # each reduce's inputs are (P, R_k) tiles (their index spans reduce range R_k); reduce the
    # trailing free axis -> (P,1). Independent reduces (e.g. attention numerator/denominator) are
    # separate statements; nested ones are emitted inner-first via the topological order above.
    body = []
    for i, rd in enumerate(ordered):
      body.append(f"    r{i} = {NL_REDUCE[rd.arg[0]]}({self._emit(rd.src[0], leaf, ref)}, axis=[1], keepdims=True)")
      reduce_vars[rd] = f"r{i}"
    out_free = 1 if ordered else (int(np.prod(csize[split:])) if split < len(csize) else 1)
    final = self._emit(store.src[1], leaf, ref)
    # coordinate-as-value (arange etc.): fine in an elementwise kernel, and fine inside a reduce when the
    # coord runs over a REDUCE (free) axis -- it's just another (P,F) tile reduced over F (e.g. argmax =
    # max over an index arange). A coord over a KEPT axis combined with a reduce is the fused-flash-
    # attention case (coord and reduce on different axes) which the 2D tile model can't express -> raise.
    if coords and ordered and any(canonical.index(c) < split for c in coords):
      raise NotImplementedError("NKI: range-as-value over a kept axis combined with reduce not supported")
    if not in_index and not coords: raise NotImplementedError("NKI renderer: no input access")
    # output ndarray takes the OUTPUT param's dtype (not an input's), else stores truncate (e.g. int<-float);
    # broadcast the value up to the output shape (e.g. expand: a (P,1) value into a (P,F) output).
    # A pure flat elementwise (split==0, no reduce) puts everything in the free dim. Hardcoding out_free
    # there forces a single (1,N) tile that overflows the 192KB/partition SBUF for large N (the sim
    # doesn't model capacity, so it only fails on real HW). Emit a DYNAMIC free size so the runtime can
    # reshape the flat data into (P<=128, F) -- spreading it across partitions -- and free-chunk if wide.
    # (A full reduce is also split==0 but its out is (P,1), so it must keep the hardcoded out_free.)
    flat = split == 0 and not ordered
    of = f"{ref}.shape[1]" if flat else str(out_free)
    body += [f"    out = nl.ndarray(({ref}.shape[0], {of}), dtype=nl.{out_dtype}, buffer=nl.shared_hbm)",
             f"    nl.store(out, value=nl.broadcast_to({final}, out.shape))", "    return out"]
    nin = len(in_index)
    loads = [f"    t{i} = nl.load(in{i})" for i in range(nin)]
    loads += [f"    {cv} = nl.load(in{nin+j})" for j,(r,cv) in enumerate(coords.items())]
    inputs_meta += [{"kind":"iota", "axis":canonical.index(r)} for r in coords]
    # output address: where the (kept-axis) result lands in the out buffer. For a normal full output this
    # is contiguous from 0; for assign-into-a-slice (e.g. the KV cache) it has an offset + strides, so the
    # runtime scatters the result into the slice and preserves the rest of the buffer.
    oc, oo = _affine(out_index.src[1])
    meta = {"out_slot":out_slot, "out_dtype":out_dtype, "canonical_sizes":csize, "split":split, "flat":flat,
            "inputs":inputs_meta, "out_kept":[_rsize(r) for r in kept_ranges],
            "out_off":oo, "out_strides":[oc.get(r, 0) for r in kept_ranges]}
    return self._source(", ".join(f"in{i}" for i in range(nin + len(coords))), loads + body, meta)
