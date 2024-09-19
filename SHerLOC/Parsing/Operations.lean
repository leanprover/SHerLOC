/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Constants
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Intermediate

namespace StableHLO.Parsing

def parseOpOutputs : PState (List ValueId) := do
  push "parseOpOutputs"
  let r ← parseListAux "=" "," parseValueIdRes
  pop "parseOpOutputs"
  return r

def parseInputFuncInput : PState FuncInput := do
  push "parseInputFuncInput"
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  pop "parseInputFuncInput"
  return { id := id , typ := typ }

def parseInputFuncInputs : PState (List FuncInput) := do
  push "parseInputFuncInputs"
  let r ← parseList "(" ")" "," parseInputFuncInput
  pop "parseInputFuncInputs"
  return r

def parseReturn : PState Operation := do
  push "parseReturn"
  let arguments ← parseValueUseList
  parseItem ":"
  let functiontype ← parseFunctionType
  let parseResult := Operation.return arguments functiontype
  pop "parseReturn"
  return parseResult

def parseCall (outputs : List ValueId) : PState Operation := do
  push "parseCall"
  parseItem "\"func.call\""
  let arguments ← parseValueUseList
  parseItem "<{"
  parseItem "callee"
  parseItem "="
  let callee ← parseFuncId
  parseItem "}>"
  parseItem ":"
  let typ ← parseFunctionType
  let r := Operation.call callee arguments outputs typ
  pop "parseCall"
  return r

def parseOpCode : PState OpCode := do
  parseItems ["\"", "stablehlo."]
  let opCodeString ← parseId
  let mut opCode : Option OpCode := none
  match opCodeString with
  | "abs" => opCode := some OpCode.abs
  | "add" => opCode := some OpCode.add
  | "after_all" => opCode := some OpCode.afterAll
  | "all_gather" => opCode := some OpCode.allGather
  | "all_reduce" => opCode := some OpCode.allReduce
  | "all_to_all" => opCode := some OpCode.allToAll
  | "and" => opCode := some OpCode.and
  | "atan2" => opCode := some OpCode.atan2
  | "batch_norm_grad" => opCode := some OpCode.batchNormGrad
  | "batch_norm_inference" => opCode := some OpCode.batchNormInference
  | "batch_norm_training" => opCode := some OpCode.batchNormTraining
  | "bitcast_convert" => opCode := some OpCode.bitcastConvert
  | "broadcast_in_dim" => opCode := some OpCode.broadcastInDim
  | "case" => opCode := some OpCode.case
  | "cbrt" => opCode := some OpCode.cbrt
  | "ceil" => opCode := some OpCode.ceil
  | "cholesky" => opCode := some OpCode.cholesky
  | "clamp" => opCode := some OpCode.clamp
  | "collective_broadcast" => opCode := some OpCode.collectiveBroadcast
  | "collective_permute" => opCode := some OpCode.collectivePermute
  | "compare" => opCode := some OpCode.compare
  | "complex" => opCode := some OpCode.complex
  | "composite" => opCode := some OpCode.composite
  | "concatenate" => opCode := some OpCode.concatenate
  | "constant" => opCode := some OpCode.constant
  | "convert" => opCode := some OpCode.convert
  | "convolution" => opCode := some OpCode.convolution
  | "cosine" => opCode := some OpCode.cosine
  | "count_leading_zeros" => opCode := some OpCode.countLeadingZeros
  | "custom_call" => opCode := some OpCode.customCall
  | "divide" => opCode := some OpCode.divide
  | "dot_general" => opCode := some OpCode.dotGeneral
  | "dynamic_broadcast_in_dim" => opCode := some OpCode.dynamicBroadcastInDim
  | "dynamic_conv" => opCode := some OpCode.dynamicConv
  | "dynamic_gather" => opCode := some OpCode.dynamicGather
  | "dynamic_iota" => opCode := some OpCode.dynamicIota
  | "dynamic_pad" => opCode := some OpCode.dynamicPad
  | "dynamic_reshape" => opCode := some OpCode.dynamicReshape
  | "dynamic_slice" => opCode := some OpCode.dynamicSlice
  | "dynamic_update_slice" => opCode := some OpCode.dynamicUpdateSlice
  | "exponential" => opCode := some OpCode.exponential
  | "exponential_minus_one" => opCode := some OpCode.exponentialMinusOne
  | "fft" => opCode := some OpCode.fft
  | "floor" => opCode := some OpCode.floor
  | "gather" => opCode := some OpCode.gather
  | "get_dimension_size" => opCode := some OpCode.getDimensionSize
  | "get_tuple_element" => opCode := some OpCode.getTupleElement
  | "if" => opCode := some OpCode.if
  | "imag" => opCode := some OpCode.imag
  | "infeed" => opCode := some OpCode.infeed
  | "iota" => opCode := some OpCode.iota
  | "is_finite" => opCode := some OpCode.isFinite
  | "log" => opCode := some OpCode.log
  | "log_plus_one" => opCode := some OpCode.logPlusOne
  | "logistic" => opCode := some OpCode.logistic
  | "map" => opCode := some OpCode.map
  | "maximum" => opCode := some OpCode.maximum
  | "minimum" => opCode := some OpCode.minimum
  | "multiply" => opCode := some OpCode.multiply
  | "negate" => opCode := some OpCode.negate
  | "not" => opCode := some OpCode.not
  | "optimization_barrier" => opCode := some OpCode.optimizationBarrier
  | "or" => opCode := some OpCode.or
  | "outfeed" => opCode := some OpCode.outfeed
  | "pad" => opCode := some OpCode.pad
  | "partition_id" => opCode := some OpCode.partitionId
  | "popcnt" => opCode := some OpCode.popcnt
  | "power" => opCode := some OpCode.power
  | "real" => opCode := some OpCode.real
  | "real_dynamic_slice" => opCode := some OpCode.realDynamicSlice
  | "recv" => opCode := some OpCode.recv
  | "reduce" => opCode := some OpCode.reduce
  | "reduce_precision" => opCode := some OpCode.reducePrecision
  | "reduce_scatter" => opCode := some OpCode.reduceScatter
  | "reduce_window" => opCode := some OpCode.reduceWindow
  | "remainder" => opCode := some OpCode.remainder
  | "replica_id" => opCode := some OpCode.replicaId
  | "reshape" => opCode := some OpCode.reshape
  | "reverse" => opCode := some OpCode.reverse
  | "rng" => opCode := some OpCode.rng
  | "rng_bit_generator" => opCode := some OpCode.rngBitGenerator
  | "round_nearest_afz" => opCode := some OpCode.roundNearestAfz
  | "round_nearest_even" => opCode := some OpCode.roundNearestEven
  | "rsqrt" => opCode := some OpCode.rsqrt
  | "scatter" => opCode := some OpCode.scatter
  | "select" => opCode := some OpCode.select
  | "select_and_scatter" => opCode := some OpCode.selectAndScatter
  | "send" => opCode := some OpCode.send
  | "shift_left" => opCode := some OpCode.shiftLeft
  | "shift_right_arithmetic" => opCode := some OpCode.shiftRightArithmetic
  | "shift_right_logical" => opCode := some OpCode.shiftRightLogical
  | "sign" => opCode := some OpCode.sign
  | "sine" => opCode := some OpCode.sine
  | "slice" => opCode := some OpCode.slice
  | "sort" => opCode := some OpCode.sort
  | "sqrt" => opCode := some OpCode.sqrt
  | "subtract" => opCode := some OpCode.subtract
  | "tan" => opCode := some OpCode.tan
  | "tanh" => opCode := some OpCode.tanh
  | "transpose" => opCode := some OpCode.transpose
  | "triangular_solve" => opCode := some OpCode.triangularSolve
  | "tuple" => opCode := some OpCode.tuple
  | "uniform_dequantize" => opCode := some OpCode.uniformDequantize
  | "uniform_quantize" => opCode := some OpCode.uniformQuantize
  | "while" => opCode := some OpCode.while
  | "xor" => opCode := some OpCode.xor
  | _ => opCode := none
  if let some op := opCode then
    parseItem "\""
    return op
  else throw <| (← error "op code")

mutual

  partial def parseInputFunc : PState InputFunc := do
    push "parseInputFunc"
    parseItem "{"
    let mut funcInputs : List FuncInput := []
    if ← is "^" then
      discard <| parseUnusedId
      funcInputs ← parseInputFuncInputs
      parseItem ":"
    let body ← parseInputFuncBody
    parseItem "}"
    pop "parseInputFunc"
    return InputFunc.mk funcInputs body

  partial def parseOpInputFuncs : PState (List InputFunc) := do
    push "parseOpInputFuncs"
    let r ← parseList "(" ")" "," parseInputFunc
    pop "parseOpInputFuncs"
    return r

  partial def parseOperationDictionaryAttributes : PState (List Attribute) := do
    push "parseOperationDictionaryAttributes"
    let r ← parseList "<{" "}>" "," parseAttribute
    pop "parseOperationDictionaryAttributes"
    return r

  partial def parseOperationBasic (op : OpCode) (opOutputs : List ValueId) : PState Operation := do
    push "parseOperationBasic"
    let opInputValues ← parseValueUseList
    let mut opInputAttrs := []
    if ← is "<{" then
      opInputAttrs ← parseOperationDictionaryAttributes
    let mut opInputFuncs := []
    if ← is "(" then
      opInputFuncs ← parseOpInputFuncs
    parseItem ":"
    let functiontype ← parseFunctionType
    let operation := Operation.stablehlo op opInputValues opInputFuncs opInputAttrs opOutputs functiontype
    pop "parseOperationBasic"
    return operation

  partial def parseOtherDialect (opOutputs : List ValueId) : PState Operation := do
    push "parseOtherDialect"
    let name ← parseString
    report s!"undocumented operation: {name}"
    let opInputValues ← parseValueUseList
    let mut opInputAttrs := []
    if ← is "<{" then
      opInputAttrs ← parseOperationDictionaryAttributes
    let mut opInputFuncs := []
    if ← is "(" then
      opInputFuncs ← parseOpInputFuncs
    parseItem ":"
    let functiontype ← parseFunctionType
    let operation := Operation.other name opInputValues opInputFuncs opInputAttrs opOutputs functiontype
    pop "parseOtherDialect"
    return operation

partial def parseStableHLO (opOutputs : List ValueId) : PState Operation := do
  let opCode ← parseOpCode
  match opCode with
  | OpCode.abs => parseOperationBasic OpCode.abs opOutputs
  | OpCode.add => parseOperationBasic OpCode.add opOutputs
  | OpCode.afterAll => parseOperationBasic OpCode.afterAll opOutputs
  | OpCode.allGather => parseOperationBasic OpCode.allGather opOutputs
  | OpCode.allReduce => parseOperationBasic OpCode.allReduce opOutputs
  | OpCode.allToAll => parseOperationBasic OpCode.allToAll opOutputs
  | OpCode.and => parseOperationBasic OpCode.and opOutputs
  | OpCode.atan2 => parseOperationBasic OpCode.atan2 opOutputs
  | OpCode.batchNormGrad => parseOperationBasic OpCode.batchNormGrad opOutputs
  | OpCode.batchNormInference => parseOperationBasic OpCode.batchNormInference opOutputs
  | OpCode.batchNormTraining => parseOperationBasic OpCode.batchNormTraining opOutputs
  | OpCode.bitcastConvert => parseOperationBasic OpCode.bitcastConvert opOutputs
  | OpCode.broadcastInDim => parseOperationBasic OpCode.broadcastInDim opOutputs
  | OpCode.case => parseOperationBasic OpCode.case opOutputs
  | OpCode.cbrt => parseOperationBasic OpCode.cbrt opOutputs
  | OpCode.ceil => parseOperationBasic OpCode.ceil opOutputs
  | OpCode.cholesky => parseOperationBasic OpCode.cholesky opOutputs
  | OpCode.clamp => parseOperationBasic OpCode.clamp opOutputs
  | OpCode.collectiveBroadcast => parseOperationBasic OpCode.collectiveBroadcast opOutputs
  | OpCode.collectivePermute => parseOperationBasic OpCode.collectivePermute opOutputs
  | OpCode.compare => parseOperationBasic OpCode.compare opOutputs
  | OpCode.complex => parseOperationBasic OpCode.complex opOutputs
  | OpCode.composite => parseOperationBasic OpCode.composite opOutputs
  | OpCode.concatenate => parseOperationBasic OpCode.concatenate opOutputs
  | OpCode.constant => parseOperationBasic OpCode.constant opOutputs
  | OpCode.convert => parseOperationBasic OpCode.convert opOutputs
  | OpCode.convolution => parseOperationBasic OpCode.convolution opOutputs
  | OpCode.cosine => parseOperationBasic OpCode.cosine opOutputs
  | OpCode.countLeadingZeros => parseOperationBasic OpCode.countLeadingZeros opOutputs
  | OpCode.customCall => parseOperationBasic OpCode.customCall opOutputs
  | OpCode.divide => parseOperationBasic OpCode.divide opOutputs
  | OpCode.dotGeneral => parseOperationBasic OpCode.dotGeneral opOutputs
  | OpCode.dynamicBroadcastInDim => parseOperationBasic OpCode.dynamicBroadcastInDim opOutputs
  | OpCode.dynamicConv => parseOperationBasic OpCode.dynamicConv opOutputs
  | OpCode.dynamicGather => parseOperationBasic OpCode.dynamicGather opOutputs
  | OpCode.dynamicIota => parseOperationBasic OpCode.dynamicIota opOutputs
  | OpCode.dynamicPad => parseOperationBasic OpCode.dynamicPad opOutputs
  | OpCode.dynamicReshape => parseOperationBasic OpCode.dynamicReshape opOutputs
  | OpCode.dynamicSlice => parseOperationBasic OpCode.dynamicSlice opOutputs
  | OpCode.dynamicUpdateSlice => parseOperationBasic OpCode.dynamicUpdateSlice opOutputs
  | OpCode.exponential => parseOperationBasic OpCode.exponential opOutputs
  | OpCode.exponentialMinusOne => parseOperationBasic OpCode.exponentialMinusOne opOutputs
  | OpCode.fft => parseOperationBasic OpCode.fft opOutputs
  | OpCode.floor => parseOperationBasic OpCode.floor opOutputs
  | OpCode.gather => parseOperationBasic OpCode.gather opOutputs
  | OpCode.getDimensionSize => parseOperationBasic OpCode.getDimensionSize opOutputs
  | OpCode.getTupleElement => parseOperationBasic OpCode.getTupleElement opOutputs
  | OpCode.if => parseOperationBasic OpCode.if opOutputs
  | OpCode.imag => parseOperationBasic OpCode.imag opOutputs
  | OpCode.infeed =>
    report "Semantics implementation defined infeed"
    parseOperationBasic OpCode.infeed opOutputs
  | OpCode.iota => parseOperationBasic OpCode.iota opOutputs
  | OpCode.isFinite => parseOperationBasic OpCode.isFinite opOutputs
  | OpCode.log => parseOperationBasic OpCode.log opOutputs
  | OpCode.logPlusOne => parseOperationBasic OpCode.logPlusOne opOutputs
  | OpCode.logistic => parseOperationBasic OpCode.logistic opOutputs
  | OpCode.map => parseOperationBasic OpCode.map opOutputs
  | OpCode.maximum => parseOperationBasic OpCode.maximum opOutputs
  | OpCode.minimum => parseOperationBasic OpCode.minimum opOutputs
  | OpCode.multiply => parseOperationBasic OpCode.multiply opOutputs
  | OpCode.negate => parseOperationBasic OpCode.negate opOutputs
  | OpCode.not => parseOperationBasic OpCode.not opOutputs
  | OpCode.optimizationBarrier => parseOperationBasic OpCode.optimizationBarrier opOutputs
  | OpCode.or => parseOperationBasic OpCode.or opOutputs
  | OpCode.outfeed => parseOperationBasic OpCode.outfeed opOutputs
  | OpCode.pad => parseOperationBasic OpCode.pad opOutputs
  | OpCode.partitionId => parseOperationBasic OpCode.partitionId opOutputs
  | OpCode.popcnt => parseOperationBasic OpCode.popcnt opOutputs
  | OpCode.power => parseOperationBasic OpCode.power opOutputs
  | OpCode.real => parseOperationBasic OpCode.real opOutputs
  | OpCode.realDynamicSlice => parseOperationBasic OpCode.real opOutputs -- Undocumented
  | OpCode.recv => parseOperationBasic OpCode.recv opOutputs
  | OpCode.reduce => parseOperationBasic OpCode.reduce opOutputs
  | OpCode.reducePrecision => parseOperationBasic OpCode.reducePrecision opOutputs
  | OpCode.reduceScatter => parseOperationBasic OpCode.reduceScatter opOutputs
  | OpCode.reduceWindow => parseOperationBasic OpCode.reduceWindow opOutputs
  | OpCode.remainder => parseOperationBasic OpCode.remainder opOutputs
  | OpCode.replicaId => parseOperationBasic OpCode.replicaId opOutputs
  | OpCode.reshape => parseOperationBasic OpCode.reshape opOutputs
  | OpCode.reverse => parseOperationBasic OpCode.reverse opOutputs
  | OpCode.rng =>
    report "explore for deprecation rng"
    parseOperationBasic OpCode.rng opOutputs
  | OpCode.rngBitGenerator => parseOperationBasic OpCode.rngBitGenerator opOutputs
  | OpCode.roundNearestAfz => parseOperationBasic OpCode.roundNearestAfz opOutputs
  | OpCode.roundNearestEven => parseOperationBasic OpCode.roundNearestEven opOutputs
  | OpCode.rsqrt => parseOperationBasic OpCode.rsqrt opOutputs
  | OpCode.scatter => parseOperationBasic OpCode.scatter opOutputs
  | OpCode.select => parseOperationBasic OpCode.select opOutputs
  | OpCode.selectAndScatter => parseOperationBasic OpCode.selectAndScatter opOutputs
  | OpCode.send => parseOperationBasic OpCode.send opOutputs
  | OpCode.shiftLeft => parseOperationBasic OpCode.shiftLeft opOutputs
  | OpCode.shiftRightArithmetic => parseOperationBasic OpCode.shiftRightArithmetic opOutputs
  | OpCode.shiftRightLogical => parseOperationBasic OpCode.shiftRightLogical opOutputs
  | OpCode.sign => parseOperationBasic OpCode.sign opOutputs
  | OpCode.sine => parseOperationBasic OpCode.sine opOutputs
  | OpCode.slice => parseOperationBasic OpCode.slice opOutputs
  | OpCode.sort => parseOperationBasic OpCode.sort opOutputs
  | OpCode.sqrt => parseOperationBasic OpCode.sqrt opOutputs
  | OpCode.subtract => parseOperationBasic OpCode.subtract opOutputs
  | OpCode.tan => parseOperationBasic OpCode.tan opOutputs
  | OpCode.tanh => {
    let opInputValues ← parseValueUseList
    -- could provide better error messages by ensuring not dictionnary
    if opInputValues.length ≠ 1 then throw <| (← error "tanh operation: wrong number of arguments")
    else {
      parseItem ":"
      let functionType ← parseFunctionType
      return Operation.tanh (opInputValues.get! 0) functionType
    }
  }
  | OpCode.transpose => parseOperationBasic OpCode.transpose opOutputs
  | OpCode.triangularSolve => parseOperationBasic OpCode.triangularSolve opOutputs
  | OpCode.tuple => parseOperationBasic OpCode.tuple opOutputs
  | OpCode.uniformDequantize => parseOperationBasic OpCode.uniformDequantize opOutputs
  | OpCode.uniformQuantize => parseOperationBasic OpCode.uniformQuantize opOutputs
  | OpCode.while =>
    report "semantics not fully decided "
    parseOperationBasic OpCode.while opOutputs
  | OpCode.xor => parseOperationBasic OpCode.xor opOutputs

  partial def parseOperation : PState Operation := do
    push "parseOperation"
    if ← isParse "\"func.return\"" then
      let r ← parseReturn
      pop "parseOperation"
      return r
    if ← isParse "\"stablehlo.return\"" then
      let r ← parseReturn
      pop "parseOperation"
      return r
    let mut opOutputs := []
    if ← is "%" then
      opOutputs ← parseOpOutputs
      parseItem "="
    if ← is "\"func.call\"" then
      let r ← parseCall opOutputs
      pop "parseOperation"
      return r

    if ← is "\"check." then
      let r ← parseOtherDialect opOutputs
      pop "parseOperation"
      return r

    if ← is "\"interpreter." then
      let r ← parseOtherDialect opOutputs
      pop "parseOperation"
      return r

    if ← is "\"chlo." then
      let r ← parseOtherDialect opOutputs
      pop "parseOperation"
      return r

    let operation ← parseStableHLO opOutputs

    pop "parseOperation"
    return operation

  partial def parseInputFuncBody : PState (List Operation) := do
    push "parseInputFuncBody"
    let r ← parseListAuxNoSep "}" parseOperation []
    pop "parseInputFuncBody"
    return r

end

end StableHLO.Parsing
