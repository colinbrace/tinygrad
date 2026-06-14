# NKI renderer: tinygrad UOp graph -> AWS NKI (neuronxcc.nki.language) source.
# "Option B": consumes the HIGH-LEVEL form (Ops.REDUCE / ALU / INDEX over ranges,
# before tinygrad lowers reduce into a scalar accumulator loop -- see the
# render_high_level hook in codegen/__init__.py). NKI is a tile-op machine, so we
# map high-level ops directly: ALU -> nl.*, Ops.REDUCE -> nl.sum/nl.max.
# See scratch/ml/theory/tinygrad-notes/backend_design.md.
import os, json
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

  def _npname(self, dt) -> str: return np.dtype(_to_np_dtype(dt.scalar())).name

  # shared expression emitter; `leaf(u)` resolves INDEX/REDUCE leaves to tile var names
  def _emit(self, u:UOp, leaf, ref:str) -> str:
    r = lambda x: self._emit(x, leaf, ref)
    lf = leaf(u)
    if lf is not None: return lf
    if u.op is Ops.CONST:
      if dtypes.is_float(u.dtype): return repr(float(u.arg))
      if dtypes.is_int(u.dtype):   return repr(int(u.arg))
      return repr(bool(u.arg))
    if u.op is Ops.CAST:   # only a real dtype change needs an nl.copy; same-dtype CAST is a no-op
      if u.dtype.scalar() != u.src[0].dtype.scalar(): return f"nl.copy({r(u.src[0])}, dtype=nl.{self._npname(u.dtype)})"
      return r(u.src[0])
    if u.op is Ops.WHERE:
      tile = lambda s: f"nl.full({ref}.shape, {r(s)}, dtype={ref}.dtype)" if s.op is Ops.CONST else r(s)
      return f"nl.where({r(u.src[0])}, {tile(u.src[1])}, {tile(u.src[2])})"
    if u.op is Ops.MULACC: return f"nl.add(nl.multiply({r(u.src[0])}, {r(u.src[1])}), {r(u.src[2])})"
    if u.op is Ops.EXP2: return f"nl.exp(nl.multiply({r(u.src[0])}, {LN2!r}))"
    if u.op is Ops.LOG2: return f"nl.multiply(nl.log({r(u.src[0])}), {1.0/LN2!r})"
    if u.op in NL_BINOP: return f"{NL_BINOP[u.op]}({r(u.src[0])}, {r(u.src[1])})"
    if u.op in NL_UNOP:  return f"{NL_UNOP[u.op]}({r(u.src[0])})"
    raise NotImplementedError(f"NKI renderer: unhandled op {u.op}")

  def _source(self, sig:str, lines:list, meta:dict) -> str:
    return (f"# TRAINIUM_META {json.dumps(meta)}\n"
            "import nki\nimport nki.language as nl\n\n"
            f"@nki.jit\ndef kernel({sig}):\n" + "\n".join(lines) + "\n")

  def _match_matmul(self, store, ordered, kept_ranges, out_slot, out_dtype, in_index):
    # detect out[m,n] = f(sum_k a[m,k]*b[k,n], <elementwise post-ops>): a 2D sum-reduce of a product of
    # two indexed inputs, optionally fused with elementwise post-ops (bias/scale/activation) on (M,N).
    if len(ordered) != 1 or len(kept_ranges) != 2: return None
    rd = ordered[0]
    if rd.arg[0] is not Ops.ADD: return None
    strip = lambda u: strip(u.src[0]) if u.op is Ops.CAST else u
    body = strip(rd.src[0])
    if body.op is not Ops.MUL: return None
    xa, xb = strip(body.src[0]), strip(body.src[1])
    if not all(x.op is Ops.INDEX and x.src[0].op is Ops.PARAM for x in (xa, xb)): return None
    K, (Mr, Nr) = rd.src[1], kept_ranges
    (ca, oa), (cb, ob) = _affine(xa.src[1]), _affine(xb.src[1])
    # orient so A spans (rows Mr, K) and B spans (K, cols Nr); swap operands if needed
    if Mr in cb and Nr in ca: (xa, ca, oa), (xb, cb, ob) = (xb, cb, ob), (xa, ca, oa)
    if not (Mr in ca and K in ca and Nr in cb and K in cb): return None
    Msz, Nsz, Ksz = _rsize(Mr), _rsize(Nr), _rsize(K)
    if Msz > 128 or Ksz > 128 or Nsz > 512: return None     # exceeds nl.matmul tile limits -> generic path
    # post-reduce operands (bias etc.): every other indexed input, affine over the kept axes only, as (M,N)
    post = [u for u in in_index if u is not xa and u is not xb]
    post_meta = []
    for u in post:
      try: c, o = _affine(u.src[1])
      except NotImplementedError: return None                # data-dependent post-op -> generic path
      if K in c: return None                                 # a post-op may not span the contraction
      post_meta.append({"param_slot":u.src[0].arg.slot, "dtype":self._npname(u.src[0].dtype),
                        "strides":[c.get(Mr, 0), c.get(Nr, 0)], "offset":o})
    tpost = {u:i for i,u in enumerate(post)}
    def leaf(u):
      if u is rd: return "res"                               # the matmul result tile
      if u in tpost: return f"t{tpost[u]}"
      return None
    final = self._emit(store.src[1], leaf, "res")
    # A is passed transposed as (K, M); nl.matmul(A, B, transpose_x=True) = A.T @ B = a @ b
    meta = {"kind":"matmul", "out_slot":out_slot, "out_dtype":out_dtype, "M":Msz, "N":Nsz, "K":Ksz,
            "A":{"param_slot":xa.src[0].arg.slot, "dtype":self._npname(xa.src[0].dtype), "strides":[ca[K], ca[Mr]], "offset":oa},
            "B":{"param_slot":xb.src[0].arg.slot, "dtype":self._npname(xb.src[0].dtype), "strides":[cb[K], cb[Nr]], "offset":ob},
            "post":post_meta}
    lines = ["    A = nl.load(in0)", "    B = nl.load(in1)",
             "    p = nl.matmul(A, B, transpose_x=True)",
             f"    res = nl.ndarray(({Msz}, {Nsz}), dtype=nl.{out_dtype}, buffer=nl.sbuf)",
             "    nisa.tensor_copy(res, p)"]
    lines += [f"    t{i} = nl.load(in{i+2})" for i in range(len(post))]
    lines += [f"    out = nl.ndarray(({Msz}, {Nsz}), dtype=nl.{out_dtype}, buffer=nl.shared_hbm)",
              f"    nl.store(out, value=nl.broadcast_to({final}, out.shape))", "    return out"]
    sig = ", ".join(["in0", "in1"] + [f"in{i+2}" for i in range(len(post))])
    return (f"# TRAINIUM_META {json.dumps(meta)}\nimport nki\nimport nki.language as nl\nimport nki.isa as nisa\n\n"
            f"@nki.jit\ndef kernel({sig}):\n" + "\n".join(lines) + "\n")

  def render(self, uops:list[UOp]) -> str:
    if os.getenv("NKI_DUMP"):
      idx = {u:i for i,u in enumerate(uops)}
      print("\n".join(f"{i:3} {str(u.op):16} {str(u.dtype):16} src={[idx.get(s,'?') for s in u.src]} arg={u.arg!r}"
                       for i,u in enumerate(uops)))
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
    # topological order: a reduce whose body contains another reduce is emitted later
    def _has(u, target):
      seen, stack = set(), [u]
      while stack:
        x = stack.pop()
        if x in seen: continue
        seen.add(x)
        for s in x.src:
          if s is target: return True
          stack.append(s)
      return False
    ordered, rem = [], list(reduces)
    while rem:
      nxt = next((rd for rd in rem if not any(_has(rd.src[0], o) for o in rem if o is not rd)), rem[0])
      ordered.append(nxt); rem.remove(nxt)

    # Each distinct input INDEX node is its own tile -- so the same buffer read with two access
    # patterns (e.g. a @ a.T) becomes two tiles. We pass a strided view per INDEX (the runtime
    # as_strided reads contiguous / transpose / slice / broadcast uniformly).
    in_index = []
    for u in uops:
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.slot != out_slot and u is not out_index and u not in in_index:
        in_index.append(u)
    tid = {u:i for i,u in enumerate(in_index)}   # may be empty: a coord-only kernel (e.g. arange)

    kept_ranges = [r for r,_ in sorted(_affine(out_index.src[1])[0].items(), key=lambda kv: -kv[1])]
    # matmul fast path: a sum-reduce of a product of two inputs -> nl.matmul (real tile matmul,
    # not the O(M*N*K) broadcast-materialize). Within NKI limits only; else fall through to generic.
    if (mm := self._match_matmul(store, ordered, kept_ranges, out_slot, out_dtype, in_index)) is not None: return mm

    # Unified iteration space: kept axes (LOOP, in OUTPUT order) + the reduced axis (if any).
    # partition|free split at `split` (runtime tiles partition <=128); reduce -> nl.sum/max over free.
    canonical = kept_ranges + [rd.src[1] for rd in ordered]   # partition=kept; each reduce gets a free axis
    split = len(kept_ranges) if ordered else max(0, len(kept_ranges) - 1)
    csize = [_rsize(r) for r in canonical]
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
    # coordinate-as-value (arange etc.) is supported for elementwise kernels; coords interacting with
    # a reduce need per-reduce placement (fused attention) -- not handled, so raise instead of silently wrong.
    if coords and ordered: raise NotImplementedError("NKI: range-as-value combined with reduce not supported")
    if not in_index and not coords: raise NotImplementedError("NKI renderer: no input access")
    # output ndarray takes the OUTPUT param's dtype (not an input's), else stores truncate (e.g. int<-float);
    # broadcast the value up to the output shape (e.g. expand: a (P,1) value into a (P,F) output).
    body += [f"    out = nl.ndarray(({ref}.shape[0], {out_free}), dtype=nl.{out_dtype}, buffer=nl.shared_hbm)",
             f"    nl.store(out, value=nl.broadcast_to({final}, out.shape))", "    return out"]
    nin = len(in_index)
    loads = [f"    t{i} = nl.load(in{i})" for i in range(nin)]
    loads += [f"    {cv} = nl.load(in{nin+j})" for j,(r,cv) in enumerate(coords.items())]
    inputs_meta += [{"kind":"iota", "axis":canonical.index(r)} for r in coords]
    meta = {"out_slot":out_slot, "out_dtype":out_dtype, "canonical_sizes":csize, "split":split, "inputs":inputs_meta}
    return self._source(", ".join(f"in{i}" for i in range(nin + len(coords))), loads + body, meta)
