/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Constants
import SHerLOC.Parsing.Identifiers

namespace StableHLO

def parseAttribute : PState Attribute := do
  push "parseAttribute"
  let id ← parseId
  parseItem "="
  let constant ← parseConstant
  pop "parseAttribute"
  return { id := id , constant := constant }

def parseAttributes : PState (List Attribute) := do
  push "parseAttributes"
  let r ← parseList "{" "}" (some ",") parseAttribute
  pop "parseAttributes"
  return r

def parseOpName : PState OpName := do
  push "parseOpName"
  let st ← get
  if st.is "stablehlo.abs" then parseItem "stablehlo.abs" ; pop "parseOpName" ; return OpName.abs
  if st.is "stablehlo.add" then parseItem "stablehlo.add" ; pop "parseOpName" ; return OpName.add
  if st.is "stablehlo.after_all" then parseItem "stablehlo.after_all" ; pop "parseOpName" ; return OpName.after_all
  if st.is "stablehlo.all_gather" then parseItem "stablehlo.all_gather" ; pop "parseOpName" ; return OpName.all_gather
  if st.is "stablehlo.all_reduce" then parseItem "stablehlo.all_reduce" ; pop "parseOpName" ; return OpName.all_reduce
  if st.is "stablehlo.all_to_all" then parseItem "stablehlo.all_to_all" ; pop "parseOpName" ; return OpName.all_to_all
  if st.is "stablehlo.and" then parseItem "stablehlo.and" ; pop "parseOpName" ; return OpName.and
  if st.is "stablehlo.atan2" then parseItem "stablehlo.atan2" ; pop "parseOpName" ; return OpName.atan2
  if st.is "stablehlo.batch_norm_grad" then parseItem "stablehlo.batch_norm_grad" ; pop "parseOpName" ; return OpName.batch_norm_grad
  if st.is "stablehlo.batch_norm_inference" then parseItem "stablehlo.batch_norm_inference" ; pop "parseOpName" ; return OpName.batch_norm_inference
  if st.is "stablehlo.batch_norm_training" then parseItem "stablehlo.batch_norm_training" ; pop "parseOpName" ; return OpName.batch_norm_training
  if st.is "stablehlo.bitcast_convert" then parseItem "stablehlo.bitcast_convert" ; pop "parseOpName" ; return OpName.bitcast_convert
  if st.is "stablehlo.broadcast_in_dim" then parseItem "stablehlo.broadcast_in_dim" ; pop "parseOpName" ; return OpName.broadcast_in_dim
  if st.is "stablehlo.case" then parseItem "stablehlo.case" ; pop "parseOpName" ; return OpName.case
  if st.is "stablehlo.cbrt" then parseItem "stablehlo.cbrt" ; pop "parseOpName" ; return OpName.cbrt
  if st.is "stablehlo.ceil" then parseItem "stablehlo.ceil" ; pop "parseOpName" ; return OpName.ceil
  if st.is "stablehlo.cholesky" then parseItem "stablehlo.cholesky" ; pop "parseOpName" ; return OpName.cholesky
  if st.is "stablehlo.clamp" then parseItem "stablehlo.clamp" ; pop "parseOpName" ; return OpName.clamp
  if st.is "stablehlo.collective_broadcast" then parseItem "stablehlo.collective_broadcast" ; pop "parseOpName" ; return OpName.collective_broadcast
  if st.is "stablehlo.collective_permute" then parseItem "stablehlo.collective_permute" ; pop "parseOpName" ; return OpName.collective_permute
  if st.is "stablehlo.compare" then parseItem "stablehlo.compare" ; pop "parseOpName" ; return OpName.compare
  if st.is "stablehlo.complex" then parseItem "stablehlo.complex" ; pop "parseOpName" ; return OpName.complex
  if st.is "stablehlo.composite" then parseItem "stablehlo.composite" ; pop "parseOpName" ; return OpName.composite
  if st.is "stablehlo.concatenate" then parseItem "stablehlo.concatenate" ; pop "parseOpName" ; return OpName.concatenate
  if st.is "stablehlo.constant" then parseItem "stablehlo.constant" ; pop "parseOpName" ; return OpName.constant
  if st.is "stablehlo.convert" then parseItem "stablehlo.convert" ; pop "parseOpName" ; return OpName.convert
  if st.is "stablehlo.convolution" then parseItem "stablehlo.convolution" ; pop "parseOpName" ; return OpName.convolution
  if st.is "stablehlo.cosine" then parseItem "stablehlo.cosine" ; pop "parseOpName" ; return OpName.cosine
  if st.is "stablehlo.count_leading_zeros" then parseItem "stablehlo.count_leading_zeros" ; pop "parseOpName" ; return OpName.count_leading_zeros
  if st.is "stablehlo.custom_call" then parseItem "stablehlo.custom_call" ; pop "parseOpName" ; return OpName.custom_call
  if st.is "stablehlo.divide" then parseItem "stablehlo.divide" ; pop "parseOpName" ; return OpName.divide
  if st.is "stablehlo.dot_general" then parseItem "stablehlo.dot_general" ; pop "parseOpName" ; return OpName.dot_general
  if st.is "stablehlo.dynamic_broadcast_in_dim" then parseItem "stablehlo.dynamic_broadcast_in_dim" ; pop "parseOpName" ; return OpName.dynamic_broadcast_in_dim
  if st.is "stablehlo.dynamic_conv" then parseItem "stablehlo.dynamic_conv" ; pop "parseOpName" ; return OpName.dynamic_conv
  if st.is "stablehlo.dynamic_gather" then parseItem "stablehlo.dynamic_gather" ; pop "parseOpName" ; return OpName.dynamic_gather
  if st.is "stablehlo.dynamic_iota" then parseItem "stablehlo.dynamic_iota" ; pop "parseOpName" ; return OpName.dynamic_iota
  if st.is "stablehlo.dynamic_pad" then parseItem "stablehlo.dynamic_pad" ; pop "parseOpName" ; return OpName.dynamic_pad
  if st.is "stablehlo.dynamic_reshape" then parseItem "stablehlo.dynamic_reshape" ; pop "parseOpName" ; return OpName.dynamic_reshape
  if st.is "stablehlo.dynamic_slice" then parseItem "stablehlo.dynamic_slice" ; pop "parseOpName" ; return OpName.dynamic_slice
  if st.is "stablehlo.dynamic_update_slice" then parseItem "stablehlo.dynamic_update_slice" ; pop "parseOpName" ; return OpName.dynamic_update_slice
  if st.is "stablehlo.exponential" then parseItem "stablehlo.exponential" ; pop "parseOpName" ; return OpName.exponential
  if st.is "stablehlo.exponential_minus_one" then parseItem "stablehlo.exponential_minus_one" ; pop "parseOpName" ; return OpName.exponential_minus_one
  if st.is "stablehlo.fft" then parseItem "stablehlo.fft" ; pop "parseOpName" ; return OpName.fft
  if st.is "stablehlo.floor" then parseItem "stablehlo.floor" ; pop "parseOpName" ; return OpName.floor
  if st.is "stablehlo.gather" then parseItem "stablehlo.gather" ; pop "parseOpName" ; return OpName.gather
  if st.is "stablehlo.get_dimension_size" then parseItem "stablehlo.get_dimension_size" ; pop "parseOpName" ; return OpName.get_dimension_size
  if st.is "stablehlo.get_tuple_element" then parseItem "stablehlo.get_tuple_element" ; pop "parseOpName" ; return OpName.get_tuple_element
  if st.is "stablehlo.if" then parseItem "stablehlo.if" ; pop "parseOpName" ; return OpName.if
  if st.is "stablehlo.imag" then parseItem "stablehlo.imag" ; pop "parseOpName" ; return OpName.imag
  if st.is "stablehlo.infeed" then parseItem "stablehlo.infeed" ; pop "parseOpName" ; return OpName.infeed
  if st.is "stablehlo.iota" then parseItem "stablehlo.iota" ; pop "parseOpName" ; return OpName.iota
  if st.is "stablehlo.is_finite" then parseItem "stablehlo.is_finite" ; pop "parseOpName" ; return OpName.is_finite
  if st.is "stablehlo.log" then parseItem "stablehlo.log" ; pop "parseOpName" ; return OpName.log
  if st.is "stablehlo.log_plus_one" then parseItem "stablehlo.log_plus_one" ; pop "parseOpName" ; return OpName.log_plus_one
  if st.is "stablehlo.logistic" then parseItem "stablehlo.logistic" ; pop "parseOpName" ; return OpName.logistic
  if st.is "stablehlo.map" then parseItem "stablehlo.map" ; pop "parseOpName" ; return OpName.map
  if st.is "stablehlo.maximum" then parseItem "stablehlo.maximum" ; pop "parseOpName" ; return OpName.maximum
  if st.is "stablehlo.minimum" then parseItem "stablehlo.minimum" ; pop "parseOpName" ; return OpName.minimum
  if st.is "stablehlo.multiply" then parseItem "stablehlo.multiply" ; pop "parseOpName" ; return OpName.multiply
  if st.is "stablehlo.negate" then parseItem "stablehlo.negate" ; pop "parseOpName" ; return OpName.negate
  if st.is "stablehlo.not" then parseItem "stablehlo.not" ; pop "parseOpName" ; return OpName.not
  if st.is "stablehlo.optimization_barrier" then parseItem "stablehlo.optimization_barrier" ; pop "parseOpName" ; return OpName.optimization_barrier
  if st.is "stablehlo.or" then parseItem "stablehlo.or" ; pop "parseOpName" ; return OpName.or
  if st.is "stablehlo.outfeed" then parseItem "stablehlo.outfeed" ; pop "parseOpName" ; return OpName.outfeed
  if st.is "stablehlo.pad" then parseItem "stablehlo.pad" ; pop "parseOpName" ; return OpName.pad
  if st.is "stablehlo.partition_id" then parseItem "stablehlo.partition_id" ; pop "parseOpName" ; return OpName.partition_id
  if st.is "stablehlo.popcnt" then parseItem "stablehlo.popcnt" ; pop "parseOpName" ; return OpName.popcnt
  if st.is "stablehlo.power" then parseItem "stablehlo.power" ; pop "parseOpName" ; return OpName.power
  if st.is "stablehlo.real" then parseItem "stablehlo.real" ; pop "parseOpName" ; return OpName.real
  if st.is "stablehlo.recv" then parseItem "stablehlo.recv" ; pop "parseOpName" ; return OpName.recv
  if st.is "stablehlo.reduce" then parseItem "stablehlo.reduce" ; pop "parseOpName" ; return OpName.reduce
  if st.is "stablehlo.reduce_precision" then parseItem "stablehlo.reduce_precision" ; pop "parseOpName" ; return OpName.reduce_precision
  if st.is "stablehlo.reduce_scatter" then parseItem "stablehlo.reduce_scatter" ; pop "parseOpName" ; return OpName.reduce_scatter
  if st.is "stablehlo.reduce_window" then parseItem "stablehlo.reduce_window" ; pop "parseOpName" ; return OpName.reduce_window
  if st.is "stablehlo.remainder" then parseItem "stablehlo.remainder" ; pop "parseOpName" ; return OpName.remainder
  if st.is "stablehlo.replica_id" then parseItem "stablehlo.replica_id" ; pop "parseOpName" ; return OpName.replica_id
  if st.is "stablehlo.reshape" then parseItem "stablehlo.reshape" ; pop "parseOpName" ; return OpName.reshape
  if st.is "stablehlo.reverse" then parseItem "stablehlo.reverse" ; pop "parseOpName" ; return OpName.reverse
  if st.is "stablehlo.rng" then parseItem "stablehlo.rng" ; pop "parseOpName" ; return OpName.rng
  if st.is "stablehlo.rng_bit_generator" then parseItem "stablehlo.rng_bit_generator" ; pop "parseOpName" ; return OpName.rng_bit_generator
  if st.is "stablehlo.round_nearest_afz" then parseItem "stablehlo.round_nearest_afz" ; pop "parseOpName" ; return OpName.round_nearest_afz
  if st.is "stablehlo.round_nearest_even" then parseItem "stablehlo.round_nearest_even" ; pop "parseOpName" ; return OpName.round_nearest_even
  if st.is "stablehlo.rsqrt" then parseItem "stablehlo.rsqrt" ; pop "parseOpName" ; return OpName.rsqrt
  if st.is "stablehlo.scatter" then parseItem "stablehlo.scatter" ; pop "parseOpName" ; return OpName.scatter
  if st.is "stablehlo.select" then parseItem "stablehlo.select" ; pop "parseOpName" ; return OpName.select
  if st.is "stablehlo.select_and_scatter" then parseItem "stablehlo.select_and_scatter" ; pop "parseOpName" ; return OpName.select_and_scatter
  if st.is "stablehlo.send" then parseItem "stablehlo.send" ; pop "parseOpName" ; return OpName.send
  if st.is "stablehlo.shift_left" then parseItem "stablehlo.shift_left" ; pop "parseOpName" ; return OpName.shift_left
  if st.is "stablehlo.shift_right_arithmetic" then parseItem "stablehlo.shift_right_arithmetic" ; pop "parseOpName" ; return OpName.shift_right_arithmetic
  if st.is "stablehlo.shift_right_logical" then parseItem "stablehlo.shift_right_logical" ; pop "parseOpName" ; return OpName.shift_right_logical
  if st.is "stablehlo.sign" then parseItem "stablehlo.sign" ; pop "parseOpName" ; return OpName.sign
  if st.is "stablehlo.sine" then parseItem "stablehlo.sine" ; pop "parseOpName" ; return OpName.sine
  if st.is "stablehlo.slice" then parseItem "stablehlo.slice" ; pop "parseOpName" ; return OpName.slice
  if st.is "stablehlo.sort" then parseItem "stablehlo.sort" ; pop "parseOpName" ; return OpName.sort
  if st.is "stablehlo.sqrt" then parseItem "stablehlo.sqrt" ; pop "parseOpName" ; return OpName.sqrt
  if st.is "stablehlo.subtract" then parseItem "stablehlo.subtract" ; pop "parseOpName" ; return OpName.subtract
  if st.is "stablehlo.tan" then parseItem "stablehlo.tan" ; pop "parseOpName" ; return OpName.tan
  if st.is "stablehlo.tanh" then parseItem "stablehlo.tanh" ; pop "parseOpName" ; return OpName.tanh
  if st.is "stablehlo.transpose" then parseItem "stablehlo.transpose" ; pop "parseOpName" ; return OpName.transpose
  if st.is "stablehlo.triangular_solve" then parseItem "stablehlo.triangular_solve" ; pop "parseOpName" ; return OpName.triangular_solve
  if st.is "stablehlo.tuple" then parseItem "stablehlo.tuple" ; pop "parseOpName" ; return OpName.tuple
  if st.is "stablehlo.uniform_dequantize" then parseItem "stablehlo.uniform_dequantize" ; pop "parseOpName" ; return OpName.uniform_dequantize
  if st.is "stablehlo.uniform_quantize" then parseItem "stablehlo.uniform_quantize" ; pop "parseOpName" ; return OpName.uniform_quantize
  if st.is "stablehlo.while" then parseItem "stablehlo.while" ; pop "parseOpName" ; return OpName.while
  if st.is "stablehlo.xor" then parseItem "stablehlo.xor" ; pop "parseOpName" ; return OpName.xor
  throw <| st.error "OpName"

def parseOpOutputs : PState (List ValueId) := do
  push "parseOpOutputs"
  let r ← parseListAux "=" (some ",") parseValueId
  pop "parseOpOutputs"
  return r

def parseOpInputValues : PState (List ValueId) := do
  push "parseOpInputValues"
  let r ← parseListAux ":" (some ",") parseValueId
  pop "parseOpInputValues"
  return r

def parseInputFuncInput : PState FuncInput := do
  push "parseInputFuncInput"
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  pop "parseInputFuncInput"
  return { id := id , typ := typ, attrs := [] }

def parseInputFuncInputs : PState (List FuncInput) := do
  push "parseInputFuncInputs"
  let r ← parseList "(" ")" (some ",") parseInputFuncInput
  pop "parseInputFuncInputs"
  return r

def parseOpInputAttrs : PState (List Attribute) := do
  push "parseOpInputAttrs"
  let r ← parseAttributes
  pop "parseOpInputAttrs"
  return r

def parseReturn : PState Operation := do
  push "parseReturn"
  parseItem "return"
  let arguments ← parseOpInputValues
  parseItem ":"
  let functiontype ← parseFunctionTypeShort
  let parseResult := Operation.return arguments functiontype
  pop "parseReturn"
  return parseResult

mutual

partial def parseInputFunc : PState InputFunc := do
  push "parseInputFunc"
  parseItem "{"
  let id ← parseUnusedId
  let funcInputs ← parseInputFuncInputs
  parseItem ":"
  let body ← parseInputFuncBody
  parseItem "}"
  pop "parseInputFunc"
  return InputFunc.mk id funcInputs body

partial def parseOpInputFuncs : PState (List InputFunc) := do
  push "parseOpInputFuncs"
  let r ← parseList "(" ")" (some ",") parseInputFunc
  pop "parseOpInputFuncs"
  return r

partial def parseStableOp : PState Operation := do
  push "parseStableOp"
  let st ← get
  let mut opOutputs := []
  if st.is "%" then
    opOutputs ← parseOpOutputs
    parseItem "="
  let st₀ ← get
  if st₀.is "stablehlo.constant" then
    let _ ← parseOpName
    let constant ← parseConstant
    let operation := Operation.constant opOutputs constant
    pop "parseStableOp"
    return operation
  else
    let opName ← parseOpName
    let opInputValues ← parseOpInputValues
    let mut opInputFuncs := []
    let st₁ ← get
    if st₁.is "(" then opInputFuncs ← parseOpInputFuncs
    let mut opInputAttrs := []
    let st₂ ← get
    if st₂.is "{" then opInputAttrs ← parseOpInputAttrs
    parseItem ":"
    -- Unfortunately, StableHLO seems to use both short and long notations for operation types
    -- However, it appears that parenthesis only appear for domains
    let st₃ ← get
    if st₃.is "(" then
      let functiontype ← parseFunctionTypeLong
      let operation := Operation.stable opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
      pop "parseStableOp"
      return operation
    else
      let functiontype ← parseFunctionTypeShort
      let operation := Operation.stable opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
      pop "parseStableOp"
      return operation

partial def parseOperation : PState Operation := do
  push "parseOperation"
  let st ← get
  if st.is "return" then
    let r ← parseReturn
    pop "parseOperation"
    return r
  else
    let r ← parseStableOp
    pop "parseOperation"
    return r

partial def parseInputFuncBody : PState (List Operation) := do
  push "parseInputFuncBody"
  let r ← parseListAux "}" none parseOperation
  pop "parseInputFuncBody"
  return r

end

def parseOperations : PState (List Operation) := do
  push "parseOperations"
  let r ← parseList "{" "}" none parseOperation
  pop "parseOperations"
  return r

end StableHLO
