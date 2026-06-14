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
    if reduces: return self._render_reduce(uops, params, store, out_slot, in_slots, dtmeta, reduces)
    return self._render_elementwise(store, out_slot, in_slots, dtmeta)

  def _render_elementwise(self, store, out_slot, in_slots, dtmeta) -> str:
    ref = in_slots[0]
    loaded:dict[int,str] = {ref: f"t{ref}"}
    lines = [f"    t{ref} = nl.load(in{ref})"]
    def leaf(u):
      if u.op is Ops.INDEX:
        if u.src[0].op is not Ops.PARAM: raise NotImplementedError(f"NKI: INDEX of {u.src[0].op}")
        slot = u.src[0].arg.slot
        if slot not in loaded:
          loaded[slot] = f"t{slot}"; lines.append(f"    t{slot} = nl.load(in{slot})")
        return loaded[slot]
      return None
    expr = self._emit(store.src[1], leaf, f"t{ref}")
    lines += [f"    out{out_slot} = nl.ndarray(in{ref}.shape, dtype=in{ref}.dtype, buffer=nl.shared_hbm)",
              f"    nl.store(out{out_slot}, value={expr})", f"    return out{out_slot}"]
    meta = {"kind":"elementwise", "out_slot":out_slot, "in_slots":in_slots, "np_dtypes":dtmeta}
    return self._source(", ".join(f"in{s}" for s in in_slots), lines, meta)

  def _render_reduce(self, uops, params, store, out_slot, in_slots, dtmeta, reduces) -> str:
    if len(reduces) != 1: raise NotImplementedError("NKI: only single reduce supported")
    if len(in_slots) != 1: raise NotImplementedError("NKI: reduce with !=1 input not supported")
    red, in_slot = reduces[0], in_slots[0]
    rop = red.arg[0]
    if rop not in NL_REDUCE: raise NotImplementedError(f"NKI: reduce op {rop}")
    # locate the input INDEX feeding the reduce, parse its affine index
    def find_index(u):
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM: return u
      for s in u.src:
        if (r := find_index(s)) is not None: return r
      return None
    in_index = find_index(red.src[0])
    in_axes = sorted(_affine(in_index.src[1]).items(), key=lambda kv: -kv[1])   # [(range, coeff)] outer->inner
    in_shape = [_rsize(r) for r,_ in in_axes]
    pos = {r:i for i,(r,_) in enumerate(in_axes)}
    # kept axes in OUTPUT order (so partition rows line up with the output buffer)
    out_axes = sorted(_affine(store.src[0].src[1]).items(), key=lambda kv: -kv[1])
    kept_pos = [pos[r] for r,_ in out_axes if r in pos]
    reduce_pos = [i for i,(r,_) in enumerate(in_axes) if _is_reduce(r)]
    if len(reduce_pos) + len(kept_pos) != len(in_axes): raise NotImplementedError("NKI: reduce axis bookkeeping mismatch")

    lines = [f"    t{in_slot} = nl.load(in{in_slot})"]
    def leaf(u):
      if u is red: return "r"
      if u.op is Ops.INDEX and u.src[0].op is Ops.PARAM: return f"t{u.src[0].arg.slot}"
      return None
    reduced_expr = self._emit(red.src[0], leaf, f"t{in_slot}")
    lines.append(f"    r = {NL_REDUCE[rop]}({reduced_expr}, axis=[1], keepdims=True)")
    final = self._emit(store.src[1], leaf, "r")
    lines += [f"    out{out_slot} = nl.ndarray((t{in_slot}.shape[0], 1), dtype=t{in_slot}.dtype, buffer=nl.shared_hbm)",
              f"    nl.store(out{out_slot}, value={final})", f"    return out{out_slot}"]
    meta = {"kind":"reduce", "out_slot":out_slot, "in_slot":in_slot, "np_dtypes":dtmeta,
            "in_shape":in_shape, "kept_pos":kept_pos, "reduce_pos":reduce_pos}
    return self._source(f"in{in_slot}", lines, meta)
