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
NL_REDUCE = {Ops.ADD:"nl.sum", Ops.MAX:"nl.max"}
LN2 = 0.6931471805599453

def _affine(u:UOp) -> dict:
  # parse an integer index expression into {RANGE uop: coefficient}; CONST offsets dropped
  if u.op is Ops.RANGE: return {u: 1}
  if u.op is Ops.CONST: return {}
  if u.op is Ops.ADD:
    d:dict = {}
    for s in u.src:
      for k,v in _affine(s).items(): d[k] = d.get(k, 0) + v
    return d
  if u.op is Ops.MUL:
    a, b = u.src
    da, db = _affine(a), _affine(b)
    if not da: return {k: v*int(a.arg) for k,v in db.items()}
    if not db: return {k: v*int(b.arg) for k,v in da.items()}
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
    if u.op is Ops.CAST: return r(u.src[0])
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
    out_slot = store.src[0].src[0].arg.slot
    in_slots = [p.arg.slot for p in params if p.arg.slot != out_slot]
    if not in_slots: raise NotImplementedError("NKI renderer: no input params")
    dtmeta = {str(p.arg.slot): self._npname(p.dtype) for p in params}
    reduces = [u for u in uops if u.op is Ops.REDUCE]
    if len(reduces) > 1: raise NotImplementedError("NKI: only single reduce supported")
    red = reduces[0] if reduces else None
    if red is not None and red.arg[0] not in NL_REDUCE: raise NotImplementedError(f"NKI: reduce op {red.arg[0]}")

    # Unified iteration space: kept axes (LOOP, in OUTPUT order) + the reduced axis (if any).
    # Each input is broadcast to this space; partition|free split at `split` (the runtime tiles
    # partition <=128). Reduce -> nl.sum/max over the free (trailing) dim; elementwise -> keep it.
    kept_ranges = [r for r,_ in sorted(_affine(store.src[0].src[1]).items(), key=lambda kv: -kv[1])]
    canonical = kept_ranges + ([red.src[1]] if red is not None else [])
    split = len(kept_ranges) if red is not None else max(0, len(kept_ranges) - 1)
    csize = [_rsize(r) for r in canonical]
    # one INDEX per input param; record its logical shape + map to canonical axes (for broadcasting)
    idx_of:dict = {}
    for u in uops:
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM and (s := u.src[0].arg.slot) in in_slots and s not in idx_of:
        idx_of[s] = u
    inputs_meta = []
    for s in in_slots:
      if s not in idx_of: raise NotImplementedError(f"NKI: input slot {s} not directly indexed")
      axes = sorted(_affine(idx_of[s].src[1]).items(), key=lambda kv: -kv[1])
      if any(r not in canonical for r,_ in axes): raise NotImplementedError("NKI: input axis not in iteration space")
      inputs_meta.append({"slot":s, "logical_shape":[_rsize(r) for r,_ in axes],
                          "canon_idx":[canonical.index(r) for r,_ in axes]})

    ref = f"t{in_slots[0]}"
    lines = [f"    t{s} = nl.load(in{s})" for s in in_slots]
    def leaf(u):
      if u is red: return "r"
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM: return f"t{u.src[0].arg.slot}"
      return None
    if red is not None:
      lines.append(f"    r = {NL_REDUCE[red.arg[0]]}({self._emit(red.src[0], leaf, ref)}, axis=[1], keepdims=True)")
      out_free = 1
    else:
      out_free = int(np.prod(csize[split:])) if split < len(csize) else 1
    out_shape = f"({ref}.shape[0], {out_free})"
    final = self._emit(store.src[1], leaf, ref)
    lines += [f"    out{out_slot} = nl.ndarray({out_shape}, dtype={ref}.dtype, buffer=nl.shared_hbm)",
              f"    nl.store(out{out_slot}, value={final})", f"    return out{out_slot}"]
    meta = {"out_slot":out_slot, "np_dtypes":dtmeta, "canonical_sizes":csize, "split":split, "inputs":inputs_meta}
    return self._source(", ".join(f"in{s}" for s in in_slots), lines, meta)
