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
  let id ← parseId
  parseItem "="
  let constant ← parseConstant
  return { id := id , constant := constant }

def parseAttributes : PState (List Attribute) := do
  parseList "{" "}" (some ",") parseAttribute

def parseOpName : PState OpName := do
  let st ← get
  if st.is "stablehlo.abs" then parseItem "stablehlo.abs" ; return OpName.abs
  if st.is "stablehlo.add" then parseItem "stablehlo.add" ; return OpName.add
  if st.is "stablehlo.after_all" then parseItem "stablehlo.after_all" ; return OpName.after_all
  if st.is "stablehlo.all_gather" then parseItem "stablehlo.all_gather" ; return OpName.all_gather
  if st.is "stablehlo.all_reduce" then parseItem "stablehlo.all_reduce" ; return OpName.all_reduce
  if st.is "stablehlo.all_to_all" then parseItem "stablehlo.all_to_all" ; return OpName.all_to_all
  if st.is "stablehlo.and" then parseItem "stablehlo.and" ; return OpName.and
  if st.is "stablehlo.atan2" then parseItem "stablehlo.atan2" ; return OpName.atan2
  if st.is "stablehlo.batch_norm_grad" then parseItem "stablehlo.batch_norm_grad" ; return OpName.batch_norm_grad
  if st.is "stablehlo.batch_norm_inference" then parseItem "stablehlo.batch_norm_inference" ; return OpName.batch_norm_inference
  if st.is "stablehlo.batch_norm_training" then parseItem "stablehlo.batch_norm_training" ; return OpName.batch_norm_training
  if st.is "stablehlo.bitcast_convert" then parseItem "stablehlo.bitcast_convert" ; return OpName.bitcast_convert
  if st.is "stablehlo.broadcast_in_dim" then parseItem "stablehlo.broadcast_in_dim" ; return OpName.broadcast_in_dim
  if st.is "stablehlo.case" then parseItem "stablehlo.case" ; return OpName.case
  if st.is "stablehlo.cbrt" then parseItem "stablehlo.cbrt" ; return OpName.cbrt
  if st.is "stablehlo.ceil" then parseItem "stablehlo.ceil" ; return OpName.ceil
  if st.is "stablehlo.cholesky" then parseItem "stablehlo.cholesky" ; return OpName.cholesky
  if st.is "stablehlo.clamp" then parseItem "stablehlo.clamp" ; return OpName.clamp
  if st.is "stablehlo.collective_broadcast" then parseItem "stablehlo.collective_broadcast" ; return OpName.collective_broadcast
  if st.is "stablehlo.collective_permute" then parseItem "stablehlo.collective_permute" ; return OpName.collective_permute
  if st.is "stablehlo.compare" then parseItem "stablehlo.compare" ; return OpName.compare
  if st.is "stablehlo.complex" then parseItem "stablehlo.complex" ; return OpName.complex
  if st.is "stablehlo.composite" then parseItem "stablehlo.composite" ; return OpName.composite
  if st.is "stablehlo.concatenate" then parseItem "stablehlo.concatenate" ; return OpName.concatenate
  if st.is "stablehlo.constant" then parseItem "stablehlo.constant" ; return OpName.constant
  if st.is "stablehlo.convert" then parseItem "stablehlo.convert" ; return OpName.convert
  if st.is "stablehlo.convolution" then parseItem "stablehlo.convolution" ; return OpName.convolution
  if st.is "stablehlo.cosine" then parseItem "stablehlo.cosine" ; return OpName.cosine
  if st.is "stablehlo.count_leading_zeros" then parseItem "stablehlo.count_leading_zeros" ; return OpName.count_leading_zeros
  if st.is "stablehlo.custom_call" then parseItem "stablehlo.custom_call" ; return OpName.custom_call
  if st.is "stablehlo.divide" then parseItem "stablehlo.divide" ; return OpName.divide
  if st.is "stablehlo.dot_general" then parseItem "stablehlo.dot_general" ; return OpName.dot_general
  if st.is "stablehlo.dynamic_broadcast_in_dim" then parseItem "stablehlo.dynamic_broadcast_in_dim" ; return OpName.dynamic_broadcast_in_dim
  if st.is "stablehlo.dynamic_conv" then parseItem "stablehlo.dynamic_conv" ; return OpName.dynamic_conv
  if st.is "stablehlo.dynamic_gather" then parseItem "stablehlo.dynamic_gather" ; return OpName.dynamic_gather
  if st.is "stablehlo.dynamic_iota" then parseItem "stablehlo.dynamic_iota" ; return OpName.dynamic_iota
  if st.is "stablehlo.dynamic_pad" then parseItem "stablehlo.dynamic_pad" ; return OpName.dynamic_pad
  if st.is "stablehlo.dynamic_reshape" then parseItem "stablehlo.dynamic_reshape" ; return OpName.dynamic_reshape
  if st.is "stablehlo.dynamic_slice" then parseItem "stablehlo.dynamic_slice" ; return OpName.dynamic_slice
  if st.is "stablehlo.dynamic_update_slice" then parseItem "stablehlo.dynamic_update_slice" ; return OpName.dynamic_update_slice
  if st.is "stablehlo.exponential" then parseItem "stablehlo.exponential" ; return OpName.exponential
  if st.is "stablehlo.exponential_minus_one" then parseItem "stablehlo.exponential_minus_one" ; return OpName.exponential_minus_one
  if st.is "stablehlo.fft" then parseItem "stablehlo.fft" ; return OpName.fft
  if st.is "stablehlo.floor" then parseItem "stablehlo.floor" ; return OpName.floor
  if st.is "stablehlo.gather" then parseItem "stablehlo.gather" ; return OpName.gather
  if st.is "stablehlo.get_dimension_size" then parseItem "stablehlo.get_dimension_size" ; return OpName.get_dimension_size
  if st.is "stablehlo.get_tuple_element" then parseItem "stablehlo.get_tuple_element" ; return OpName.get_tuple_element
  if st.is "stablehlo.if" then parseItem "stablehlo.if" ; return OpName.if
  if st.is "stablehlo.imag" then parseItem "stablehlo.imag" ; return OpName.imag
  if st.is "stablehlo.infeed" then parseItem "stablehlo.infeed" ; return OpName.infeed
  if st.is "stablehlo.iota" then parseItem "stablehlo.iota" ; return OpName.iota
  if st.is "stablehlo.is_finite" then parseItem "stablehlo.is_finite" ; return OpName.is_finite
  if st.is "stablehlo.log" then parseItem "stablehlo.log" ; return OpName.log
  if st.is "stablehlo.log_plus_one" then parseItem "stablehlo.log_plus_one" ; return OpName.log_plus_one
  if st.is "stablehlo.logistic" then parseItem "stablehlo.logistic" ; return OpName.logistic
  if st.is "stablehlo.map" then parseItem "stablehlo.map" ; return OpName.map
  if st.is "stablehlo.maximum" then parseItem "stablehlo.maximum" ; return OpName.maximum
  if st.is "stablehlo.minimum" then parseItem "stablehlo.minimum" ; return OpName.minimum
  if st.is "stablehlo.multiply" then parseItem "stablehlo.multiply" ; return OpName.multiply
  if st.is "stablehlo.negate" then parseItem "stablehlo.negate" ; return OpName.negate
  if st.is "stablehlo.not" then parseItem "stablehlo.not" ; return OpName.not
  if st.is "stablehlo.optimization_barrier" then parseItem "stablehlo.optimization_barrier" ; return OpName.optimization_barrier
  if st.is "stablehlo.or" then parseItem "stablehlo.or" ; return OpName.or
  if st.is "stablehlo.outfeed" then parseItem "stablehlo.outfeed" ; return OpName.outfeed
  if st.is "stablehlo.pad" then parseItem "stablehlo.pad" ; return OpName.pad
  if st.is "stablehlo.partition_id" then parseItem "stablehlo.partition_id" ; return OpName.partition_id
  if st.is "stablehlo.popcnt" then parseItem "stablehlo.popcnt" ; return OpName.popcnt
  if st.is "stablehlo.power" then parseItem "stablehlo.power" ; return OpName.power
  if st.is "stablehlo.real" then parseItem "stablehlo.real" ; return OpName.real
  if st.is "stablehlo.recv" then parseItem "stablehlo.recv" ; return OpName.recv
  if st.is "stablehlo.reduce" then parseItem "stablehlo.reduce" ; return OpName.reduce
  if st.is "stablehlo.reduce_precision" then parseItem "stablehlo.reduce_precision" ; return OpName.reduce_precision
  if st.is "stablehlo.reduce_scatter" then parseItem "stablehlo.reduce_scatter" ; return OpName.reduce_scatter
  if st.is "stablehlo.reduce_window" then parseItem "stablehlo.reduce_window" ; return OpName.reduce_window
  if st.is "stablehlo.remainder" then parseItem "stablehlo.remainder" ; return OpName.remainder
  if st.is "stablehlo.replica_id" then parseItem "stablehlo.replica_id" ; return OpName.replica_id
  if st.is "stablehlo.reshape" then parseItem "stablehlo.reshape" ; return OpName.reshape
  if st.is "stablehlo.reverse" then parseItem "stablehlo.reverse" ; return OpName.reverse
  if st.is "stablehlo.rng" then parseItem "stablehlo.rng" ; return OpName.rng
  if st.is "stablehlo.rng_bit_generator" then parseItem "stablehlo.rng_bit_generator" ; return OpName.rng_bit_generator
  if st.is "stablehlo.round_nearest_afz" then parseItem "stablehlo.round_nearest_afz" ; return OpName.round_nearest_afz
  if st.is "stablehlo.round_nearest_even" then parseItem "stablehlo.round_nearest_even" ; return OpName.round_nearest_even
  if st.is "stablehlo.rsqrt" then parseItem "stablehlo.rsqrt" ; return OpName.rsqrt
  if st.is "stablehlo.scatter" then parseItem "stablehlo.scatter" ; return OpName.scatter
  if st.is "stablehlo.select" then parseItem "stablehlo.select" ; return OpName.select
  if st.is "stablehlo.select_and_scatter" then parseItem "stablehlo.select_and_scatter" ; return OpName.select_and_scatter
  if st.is "stablehlo.send" then parseItem "stablehlo.send" ; return OpName.send
  if st.is "stablehlo.shift_left" then parseItem "stablehlo.shift_left" ; return OpName.shift_left
  if st.is "stablehlo.shift_right_arithmetic" then parseItem "stablehlo.shift_right_arithmetic" ; return OpName.shift_right_arithmetic
  if st.is "stablehlo.shift_right_logical" then parseItem "stablehlo.shift_right_logical" ; return OpName.shift_right_logical
  if st.is "stablehlo.sign" then parseItem "stablehlo.sign" ; return OpName.sign
  if st.is "stablehlo.sine" then parseItem "stablehlo.sine" ; return OpName.sine
  if st.is "stablehlo.slice" then parseItem "stablehlo.slice" ; return OpName.slice
  if st.is "stablehlo.sort" then parseItem "stablehlo.sort" ; return OpName.sort
  if st.is "stablehlo.sqrt" then parseItem "stablehlo.sqrt" ; return OpName.sqrt
  if st.is "stablehlo.subtract" then parseItem "stablehlo.subtract" ; return OpName.subtract
  if st.is "stablehlo.tan" then parseItem "stablehlo.tan" ; return OpName.tan
  if st.is "stablehlo.tanh" then parseItem "stablehlo.tanh" ; return OpName.tanh
  if st.is "stablehlo.transpose" then parseItem "stablehlo.transpose" ; return OpName.transpose
  if st.is "stablehlo.triangular_solve" then parseItem "stablehlo.triangular_solve" ; return OpName.triangular_solve
  if st.is "stablehlo.tuple" then parseItem "stablehlo.tuple" ; return OpName.tuple
  if st.is "stablehlo.uniform_dequantize" then parseItem "stablehlo.uniform_dequantize" ; return OpName.uniform_dequantize
  if st.is "stablehlo.uniform_quantize" then parseItem "stablehlo.uniform_quantize" ; return OpName.uniform_quantize
  if st.is "stablehlo.while" then parseItem "stablehlo.while" ; return OpName.while
  if st.is "stablehlo.xor" then parseItem "stablehlo.xor" ; return OpName.xor
  throw <| st.error "OpName"

def parseOpOutputs : PState (List ValueId) := do
  parseListAux "=" (some ",") parseValueId

def parseOpInputValues : PState (List ValueId) := do
  parseListAux ":" (some ",") parseValueId
  --parseList "(" ")" (some ",") parseValueId

def parseInputFuncInput : PState FuncInput := do
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  return { id := id , typ := typ }

def parseInputFuncInputs : PState (List FuncInput) := do
  parseList "(" ")" (some ",") parseInputFuncInput

def parseOpInputAttrs : PState (List Attribute) := do
  parseAttributes

def parseReturn : PState Operation := do
  parseItem "return"
  let arguments ← parseOpInputValues
  parseItem ":"
  let functiontype ← parseFunctionTypeShort
  let parseResult := Operation.return arguments functiontype
  return parseResult

mutual

partial def parseInputFunc : PState InputFunc := do
  parseItem "{"
  let id ← parseUnusedId
  let funcInputs ← parseInputFuncInputs
  parseItem ":"
  let body ← parseInputFuncBody
  parseItem "}"
  return InputFunc.mk id funcInputs body

partial def parseOpInputFuncs : PState (List InputFunc) := do
  parseList "(" ")" (some ",") parseInputFunc

partial def parseStableOp : PState Operation := do
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
      return operation
    else
      let functiontype ← parseFunctionTypeShort
      let operation := Operation.stable opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
      return operation

partial def parseOperation : PState Operation := do
  let st ← get
  if st.is "return" then parseReturn
  else parseStableOp

partial def parseInputFuncBody : PState (List Operation) :=
  parseListAux "}" none parseOperation

end

def parseOperations : PState (List Operation) :=
  parseList "{" "}" none parseOperation

end StableHLO
