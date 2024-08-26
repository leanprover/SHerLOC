/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Constant
import SHerLOC.Parsing.Identifiers

namespace StableHLO

def parseAttributeName : PState String := do
  let st ← get
  if st.isAttributeName then
    let s := st.tok
    shift
    return s
  else
    throw <| st.error "Attribute name"

def parseAttribute : PState Attribute := do
  let id ← parseAttributeName
  parseItem "="
  let constant ← parseConstant
  return { id := id , constant := constant }

def parseAttributes : PState (List Attribute) := do
  parseList "{" "}" (some ",") parseAttribute

def parseOpName : PState OpName := do
  let st ← get
  match st.tok with
  | "stablehlo.abs" => shift ; return OpName.abs
  | "stablehlo.add" => shift ; return OpName.add
  | "stablehlo.after_all" => shift ; return OpName.after_all
  | "stablehlo.all_gather" => shift ; return OpName.all_gather
  | "stablehlo.all_reduce" => shift ; return OpName.all_reduce
  | "stablehlo.all_to_all" => shift ; return OpName.all_to_all
  | "stablehlo.and" => shift ; return OpName.and
  | "stablehlo.atan2" => shift ; return OpName.atan2
  | "stablehlo.batch_norm_grad" => shift ; return OpName.batch_norm_grad
  | "stablehlo.batch_norm_inference" => shift ; return OpName.batch_norm_inference
  | "stablehlo.batch_norm_training" => shift ; return OpName.batch_norm_training
  | "stablehlo.bitcast_convert" => shift ; return OpName.bitcast_convert
  | "stablehlo.broadcast_in_dim" => shift ; return OpName.broadcast_in_dim
  | "stablehlo.case" => shift ; return OpName.case
  | "stablehlo.cbrt" => shift ; return OpName.cbrt
  | "stablehlo.ceil" => shift ; return OpName.ceil
  | "stablehlo.cholesky" => shift ; return OpName.cholesky
  | "stablehlo.clamp" => shift ; return OpName.clamp
  | "stablehlo.collective_broadcast" => shift ; return OpName.collective_broadcast
  | "stablehlo.collective_permute" => shift ; return OpName.collective_permute
  | "stablehlo.compare" => shift ; return OpName.compare
  | "stablehlo.complex" => shift ; return OpName.complex
  | "stablehlo.composite" => shift ; return OpName.composite
  | "stablehlo.concatenate" => shift ; return OpName.concatenate
  | "stablehlo.constant" => shift ; return OpName.constant
  | "stablehlo.convert" => shift ; return OpName.convert
  | "stablehlo.convolution" => shift ; return OpName.convolution
  | "stablehlo.cosine" => shift ; return OpName.cosine
  | "stablehlo.count_leading_zeros" => shift ; return OpName.count_leading_zeros
  | "stablehlo.custom_call" => shift ; return OpName.custom_call
  | "stablehlo.divide" => shift ; return OpName.divide
  | "stablehlo.dot_general" => shift ; return OpName.dot_general
  | "stablehlo.dynamic_broadcast_in_dim" => shift ; return OpName.dynamic_broadcast_in_dim
  | "stablehlo.dynamic_conv" => shift ; return OpName.dynamic_conv
  | "stablehlo.dynamic_gather" => shift ; return OpName.dynamic_gather
  | "stablehlo.dynamic_iota" => shift ; return OpName.dynamic_iota
  | "stablehlo.dynamic_pad" => shift ; return OpName.dynamic_pad
  | "stablehlo.dynamic_reshape" => shift ; return OpName.dynamic_reshape
  | "stablehlo.dynamic_slice" => shift ; return OpName.dynamic_slice
  | "stablehlo.dynamic_update_slice" => shift ; return OpName.dynamic_update_slice
  | "stablehlo.exponential" => shift ; return OpName.exponential
  | "stablehlo.exponential_minus_one" => shift ; return OpName.exponential_minus_one
  | "stablehlo.fft" => shift ; return OpName.fft
  | "stablehlo.floor" => shift ; return OpName.floor
  | "stablehlo.gather" => shift ; return OpName.gather
  | "stablehlo.get_dimension_size" => shift ; return OpName.get_dimension_size
  | "stablehlo.get_tuple_element" => shift ; return OpName.get_tuple_element
  | "stablehlo.if" => shift ; return OpName.if
  | "stablehlo.imag" => shift ; return OpName.imag
  | "stablehlo.infeed" => shift ; return OpName.infeed
  | "stablehlo.iota" => shift ; return OpName.iota
  | "stablehlo.is_finite" => shift ; return OpName.is_finite
  | "stablehlo.log" => shift ; return OpName.log
  | "stablehlo.log_plus_one" => shift ; return OpName.log_plus_one
  | "stablehlo.logistic" => shift ; return OpName.logistic
  | "stablehlo.map" => shift ; return OpName.map
  | "stablehlo.maximum" => shift ; return OpName.maximum
  | "stablehlo.minimum" => shift ; return OpName.minimum
  | "stablehlo.multiply" => shift ; return OpName.multiply
  | "stablehlo.negate" => shift ; return OpName.negate
  | "stablehlo.not" => shift ; return OpName.not
  | "stablehlo.optimization_barrier" => shift ; return OpName.optimization_barrier
  | "stablehlo.or" => shift ; return OpName.or
  | "stablehlo.outfeed" => shift ; return OpName.outfeed
  | "stablehlo.pad" => shift ; return OpName.pad
  | "stablehlo.partition_id" => shift ; return OpName.partition_id
  | "stablehlo.popcnt" => shift ; return OpName.popcnt
  | "stablehlo.power" => shift ; return OpName.power
  | "stablehlo.real" => shift ; return OpName.real
  | "stablehlo.recv" => shift ; return OpName.recv
  | "stablehlo.reduce" => shift ; return OpName.reduce
  | "stablehlo.reduce_precision" => shift ; return OpName.reduce_precision
  | "stablehlo.reduce_scatter" => shift ; return OpName.reduce_scatter
  | "stablehlo.reduce_window" => shift ; return OpName.reduce_window
  | "stablehlo.remainder" => shift ; return OpName.remainder
  | "stablehlo.replica_id" => shift ; return OpName.replica_id
  | "stablehlo.reshape" => shift ; return OpName.reshape
  | "stablehlo.reverse" => shift ; return OpName.reverse
  | "stablehlo.rng" => shift ; return OpName.rng
  | "stablehlo.rng_bit_generator" => shift ; return OpName.rng_bit_generator
  | "stablehlo.round_nearest_afz" => shift ; return OpName.round_nearest_afz
  | "stablehlo.round_nearest_even" => shift ; return OpName.round_nearest_even
  | "stablehlo.rsqrt" => shift ; return OpName.rsqrt
  | "stablehlo.scatter" => shift ; return OpName.scatter
  | "stablehlo.select" => shift ; return OpName.select
  | "stablehlo.select_and_scatter" => shift ; return OpName.select_and_scatter
  | "stablehlo.send" => shift ; return OpName.send
  | "stablehlo.shift_left" => shift ; return OpName.shift_left
  | "stablehlo.shift_right_arithmetic" => shift ; return OpName.shift_right_arithmetic
  | "stablehlo.shift_right_logical" => shift ; return OpName.shift_right_logical
  | "stablehlo.sign" => shift ; return OpName.sign
  | "stablehlo.sine" => shift ; return OpName.sine
  | "stablehlo.slice" => shift ; return OpName.slice
  | "stablehlo.sort" => shift ; return OpName.sort
  | "stablehlo.sqrt" => shift ; return OpName.sqrt
  | "stablehlo.subtract" => shift ; return OpName.subtract
  | "stablehlo.tan" => shift ; return OpName.tan
  | "stablehlo.tanh" => shift ; return OpName.tanh
  | "stablehlo.transpose" => shift ; return OpName.transpose
  | "stablehlo.triangular_solve" => shift ; return OpName.triangular_solve
  | "stablehlo.tuple" => shift ; return OpName.tuple
  | "stablehlo.uniform_dequantize" => shift ; return OpName.uniform_dequantize
  | "stablehlo.uniform_quantize" => shift ; return OpName.uniform_quantize
  | "stablehlo.while" => shift ; return OpName.while
  | "stablehlo.xor" => shift ; return OpName.xor
  | _ => throw <| st.error "OpName"

def parseOpOutputs : PState (List ValueId) := do
  parseListAux "=" (some ",") parseValueId

def parseOpInputs : PState (List ValueId) := do
  parseList "(" ")" (some ",") parseValueId

def parseStableOp : PState Operation := do
  let st ← get
  let mut opOutputs := []
  if st.tok.get! ⟨ 0 ⟩ = '%' then
    opOutputs ← parseOpOutputs
    parseItem "="
  let opName ← parseOpName
  let arguments ← parseOpInputs
  parseItem ":"
  let functiontype ← parseFunctionType
  -- TODO inputFunctions and inputAttributes
  let operation := Operation.stable opName arguments [] [] opOutputs functiontype
  record st "Stable operation"
  return operation

def parseReturn : PState Operation := do
  let st ← get
  parseItem "stablehlo.return"
  let arguments ← parseOpInputs
  parseItem ":"
  let functiontype ← parseFunctionType
  let parseResult := Operation.return arguments functiontype
  record st "Return operation"
  return parseResult

-- TODO complete shortcut for now, ignoring return and call (and perhaps constant)
def parseOperation : PState Operation := do
  let st ← get
  if st.tok = "stablehlo.return" then parseReturn
  else if st.tok = "func.call" then throw <| st.error "Operation call"
  -- Missing call with results
  else parseStableOp

def parseOperations : PState (List Operation) :=
  parseList "{" "}" none parseOperation

end StableHLO
