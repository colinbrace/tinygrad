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
    if len(reduces) > 1: raise NotImplementedError("NKI: only single reduce supported")
    red = reduces[0] if reduces else None
    if red is not None and red.arg[0] not in NL_REDUCE: raise NotImplementedError(f"NKI: reduce op {red.arg[0]}")

    # Each distinct input INDEX node is its own tile -- so the same buffer read with two access
    # patterns (e.g. a @ a.T) becomes two tiles. We pass a strided view per INDEX (the runtime
    # as_strided reads contiguous / transpose / slice / broadcast uniformly).
    in_index = []
    for u in uops:
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and u.src[0].arg.slot != out_slot and u is not out_index and u not in in_index:
        in_index.append(u)
    if not in_index: raise NotImplementedError("NKI renderer: no input access")
    tid = {u:i for i,u in enumerate(in_index)}

    # Unified iteration space: kept axes (LOOP, in OUTPUT order) + the reduced axis (if any).
    # partition|free split at `split` (runtime tiles partition <=128); reduce -> nl.sum/max over free.
    kept_ranges = [r for r,_ in sorted(_affine(out_index.src[1])[0].items(), key=lambda kv: -kv[1])]
    canonical = kept_ranges + ([red.src[1]] if red is not None else [])
    split = len(kept_ranges) if red is not None else max(0, len(kept_ranges) - 1)
    csize = [_rsize(r) for r in canonical]
    inputs_meta = []
    for u in in_index:
      coeffs, offset = _affine(u.src[1])
      if any(r not in canonical for r,c in coeffs.items() if c != 0): raise NotImplementedError("NKI: input axis not in iteration space")
      inputs_meta.append({"param_slot":u.src[0].arg.slot, "dtype":self._npname(u.src[0].dtype),
                          "strides":[coeffs.get(r, 0) for r in canonical], "offset":offset})

    ref = "t0"
    lines = [f"    t{i} = nl.load(in{i})" for i in range(len(in_index))]
    def leaf(u):
      if u is red: return "r"
      if u in tid: return f"t{tid[u]}"
      return None
    if red is not None:
      lines.append(f"    r = {NL_REDUCE[red.arg[0]]}({self._emit(red.src[0], leaf, ref)}, axis=[1], keepdims=True)")
      out_free = 1
    else:
      out_free = int(np.prod(csize[split:])) if split < len(csize) else 1
    final = self._emit(store.src[1], leaf, ref)
    # output ndarray takes the OUTPUT param's dtype (not an input's), else stores truncate (e.g. int<-float);
    # broadcast the value up to the output shape (e.g. expand: a (P,1) value into a (P,F) output).
    lines += [f"    out = nl.ndarray(({ref}.shape[0], {out_free}), dtype=nl.{out_dtype}, buffer=nl.shared_hbm)",
              f"    nl.store(out, value=nl.broadcast_to({final}, out.shape))", "    return out"]
    meta = {"out_slot":out_slot, "out_dtype":out_dtype, "canonical_sizes":csize, "split":split, "inputs":inputs_meta}
    return self._source(", ".join(f"in{i}" for i in range(len(in_index))), lines, meta)
