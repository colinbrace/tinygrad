# NKI renderer: tinygrad UOp graph -> AWS NKI (neuronxcc.nki.language) source.
# Phase 1a: simulator-first, ELEMENTWISE only. Promotes the single elementwise
# RANGE to a whole (1,N) tile and emits nl.* tile ops. Requires NOOPT=1 for now
# (the default optimizer UPCASTs small kernels into STACK/unrolled form, which
# this renderer does not yet handle -- see backend_design.md open question #4).
import os, json
import numpy as np
from tinygrad.renderer import Renderer
from tinygrad.uop.ops import Ops, UOp
from tinygrad.dtype import _to_np_dtype, dtypes

# tinygrad ALU Op -> nki.language function
NL_BINOP = {Ops.ADD:"nl.add", Ops.MUL:"nl.multiply", Ops.SUB:"nl.subtract",
            Ops.MAX:"nl.maximum", Ops.FDIV:"nl.divide", Ops.CDIV:"nl.divide",
            Ops.CMPLT:"nl.less", Ops.CMPNE:"nl.not_equal", Ops.CMPEQ:"nl.equal"}
NL_UNOP  = {Ops.NEG:"nl.negative", Ops.SQRT:"nl.sqrt", Ops.RECIPROCAL:"nl.reciprocal",
            Ops.SIN:"nl.sin"}
LN2 = 0.6931471805599453   # for EXP2/LOG2, which nl lacks directly

class NKIRenderer(Renderer):
  suffix = "NKI"
  has_local = False
  has_shared = False
  supports_float4 = False
  # declaring these as "supported" stops tinygrad expanding them into bit-trick
  # polynomials (codegen reads tuple(code_for_op.keys())); our emit() renders them.
  # values are never called -- our render() doesn't use the cstyle code_for_op path.
  code_for_op = {op: None for op in (Ops.EXP2, Ops.LOG2, Ops.SIN, Ops.SQRT, Ops.RECIPROCAL)}

  def _npname(self, dt) -> str: return np.dtype(_to_np_dtype(dt.scalar())).name

  def render(self, uops:list[UOp]) -> str:
    if os.getenv("NKI_DUMP"):
      idx = {u:i for i,u in enumerate(uops)}
      print("\n".join(f"{i:3} {str(u.op):22} {str(u.dtype):16} src={[idx.get(s,'?') for s in u.src]} arg={u.arg!r}"
                       for i,u in enumerate(uops)))

    params = sorted((u for u in uops if u.op is Ops.PARAM), key=lambda u: u.arg.slot)
    stores = [u for u in uops if u.op is Ops.STORE]
    if len(stores) != 1: raise NotImplementedError(f"NKI renderer: expected 1 store, got {len(stores)}")
    store = stores[0]
    out_param = store.src[0].src[0]                  # STORE -> INDEX -> PARAM
    out_slot = out_param.arg.slot
    in_slots = [p.arg.slot for p in params if p.arg.slot != out_slot]
    if not in_slots: raise NotImplementedError("NKI renderer: no input params")

    ref = in_slots[0]           # reference input tile, for scalar->tile broadcast
    loaded:dict[int,str] = {}   # input slot -> tile var name
    lines:list[str] = [f"    t{ref} = nl.load(in{ref})"]
    loaded[ref] = f"t{ref}"
    def emit(u:UOp) -> str:
      if u.op is Ops.LOAD:
        slot = u.src[0].src[0].arg.slot              # LOAD -> INDEX -> PARAM
        if slot not in loaded:
          loaded[slot] = f"t{slot}"
          lines.append(f"    t{slot} = nl.load(in{slot})")
        return loaded[slot]
      if u.op is Ops.CONST:
        if dtypes.is_float(u.dtype): return repr(float(u.arg))
        if dtypes.is_int(u.dtype):   return repr(int(u.arg))
        return repr(bool(u.arg))
      if u.op is Ops.CAST: return emit(u.src[0])      # sim is permissive on dtype
      # nl.where branches must be tiles, not scalars -> broadcast bare CONSTs
      if u.op is Ops.WHERE:
        def tile(s): return f"nl.full(t{ref}.shape, {emit(s)}, dtype=t{ref}.dtype)" if s.op is Ops.CONST else emit(s)
        return f"nl.where({emit(u.src[0])}, {tile(u.src[1])}, {tile(u.src[2])})"
      if u.op is Ops.MULACC: return f"nl.add(nl.multiply({emit(u.src[0])}, {emit(u.src[1])}), {emit(u.src[2])})"
      if u.op is Ops.EXP2: return f"nl.exp(nl.multiply({emit(u.src[0])}, {LN2!r}))"
      if u.op is Ops.LOG2: return f"nl.multiply(nl.log({emit(u.src[0])}), {1.0/LN2!r})"
      if u.op in NL_BINOP: return f"{NL_BINOP[u.op]}({emit(u.src[0])}, {emit(u.src[1])})"
      if u.op in NL_UNOP:  return f"{NL_UNOP[u.op]}({emit(u.src[0])})"
      raise NotImplementedError(f"NKI renderer: unhandled op {u.op}")

    expr = emit(store.src[1])
    ref_in = in_slots[0]
    lines.append(f"    out{out_slot} = nl.ndarray(in{ref_in}.shape, dtype=in{ref_in}.dtype, buffer=nl.shared_hbm)")
    lines.append(f"    nl.store(out{out_slot}, value={expr})")
    lines.append(f"    return out{out_slot}")

    meta = {"out_slot": out_slot, "in_slots": in_slots,
            "np_dtypes": {str(p.arg.slot): self._npname(p.dtype) for p in params}}
    sig = ", ".join(f"in{s}" for s in in_slots)
    src = (f"# TRAINIUM_META {json.dumps(meta)}\n"
           "import nki\nimport nki.language as nl\n\n"
           f"@nki.jit\ndef kernel({sig}):\n" + "\n".join(lines) + "\n")
    return src
