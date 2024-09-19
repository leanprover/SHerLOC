/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Intermediate

namespace StableHLO.Parsing

def parseOpOutputs : PState (List ValueId) := do
  let r ← parseListAux "=" "," parseValueIdRes
  return r

def parseInputFuncInput : PState FuncInput := do
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  return { id := id , typ := typ }

def parseInputFuncInputs : PState (List FuncInput) := do
  let r ← parseList "(" ")" "," parseInputFuncInput
  return r

-- def parseReturn : PState Operation := do
--   let arguments ← parseValueUseList
--   parseItem ":"
--   let functiontype ← parseFunctionType
--   let parseResult := Operation.return arguments functiontype
--   return parseResult

-- def parseCall (outputs : List ValueId) : PState Operation := do
--   parseItem "\"func.call\""
--   let arguments ← parseValueUseList
--   parseItem "<{"
--   parseItem "callee"
--   parseItem "="
--   let callee ← parseFuncId
--   parseItem "}>"
--   parseItem ":"
--   let typ ← parseFunctionType
--   let r := Operation.call callee arguments outputs typ
--   return r

def toOpCode (opCodeString : String) : PState OpCode := do
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
  | "return" => opCode := some OpCode.return
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
    return op
  else throw <| (← error "op code")

structure PreOperation where
  dialect : String
  operation : String
  inputValues : List ValueId
  inputFunctions : List InputFunc
  inputAttributes : List Attribute
  outputs : List ValueId
  signature : FunctionType
  deriving Repr, Inhabited, Nonempty

def parseTanh (p : PreOperation) : PState Operation := do
    -- could provide better error messages by ensuring not dictionnary
    if p.outputs.length ≠ 1 then throw <| (← error "tanh operation: wrong number of arguments")
    if p.inputValues.length ≠ 1 then throw <| (← error "tanh operation: wrong number of arguments")
    return Operation.tanh (p.outputs.get! 0) (p.inputValues.get! 0) p.signature

mutual

  partial def parseInputFunc : PState InputFunc := do
    parseItem "{"
    let mut funcInputs : List FuncInput := []
    if ← is "^" then
      discard <| parseUnusedId
      funcInputs ← parseInputFuncInputs
      parseItem ":"
    let body ← parseInputFuncBody
    parseItem "}"
    return InputFunc.mk funcInputs body

  partial def parseOpInputFuncs : PState (List InputFunc) := do
    let r ← parseList "(" ")" "," parseInputFunc
    return r

  partial def parseOperationDictionaryAttributes : PState (List Attribute) := do
    let r ← parseList "<{" "}>" "," parseAttribute
    return r

  partial def parseWithoutWFCheck (p : PreOperation) : PState Operation := do
    return Operation.other p.dialect p.operation p.inputValues p.inputFunctions p.inputAttributes p.outputs p.signature

  partial def toStableHLO (p : PreOperation) : PState Operation := do
    let opCode ← toOpCode p.operation
    match opCode with
    | OpCode.abs => parseWithoutWFCheck p
    | OpCode.add => parseWithoutWFCheck p
    | OpCode.afterAll => parseWithoutWFCheck p
    | OpCode.allGather => parseWithoutWFCheck p
    | OpCode.allReduce => parseWithoutWFCheck p
    | OpCode.allToAll => parseWithoutWFCheck p
    | OpCode.and => parseWithoutWFCheck p
    | OpCode.atan2 => parseWithoutWFCheck p
    | OpCode.batchNormGrad => parseWithoutWFCheck p
    | OpCode.batchNormInference => parseWithoutWFCheck p
    | OpCode.batchNormTraining => parseWithoutWFCheck p
    | OpCode.bitcastConvert => parseWithoutWFCheck p
    | OpCode.broadcastInDim => parseWithoutWFCheck p
    | OpCode.case => parseWithoutWFCheck p
    | OpCode.cbrt => parseWithoutWFCheck p
    | OpCode.ceil => parseWithoutWFCheck p
    | OpCode.cholesky => parseWithoutWFCheck p
    | OpCode.clamp => parseWithoutWFCheck p
    | OpCode.collectiveBroadcast => parseWithoutWFCheck p
    | OpCode.collectivePermute => parseWithoutWFCheck p
    | OpCode.compare => parseWithoutWFCheck p
    | OpCode.complex => parseWithoutWFCheck p
    | OpCode.composite => parseWithoutWFCheck p
    | OpCode.concatenate => parseWithoutWFCheck p
    | OpCode.constant => parseWithoutWFCheck p
    | OpCode.convert => parseWithoutWFCheck p
    | OpCode.convolution => parseWithoutWFCheck p
    | OpCode.cosine => parseWithoutWFCheck p
    | OpCode.countLeadingZeros => parseWithoutWFCheck p
    | OpCode.customCall => parseWithoutWFCheck p
    | OpCode.divide => parseWithoutWFCheck p
    | OpCode.dotGeneral => parseWithoutWFCheck p
    | OpCode.dynamicBroadcastInDim => parseWithoutWFCheck p
    | OpCode.dynamicConv => parseWithoutWFCheck p
    | OpCode.dynamicGather => parseWithoutWFCheck p
    | OpCode.dynamicIota => parseWithoutWFCheck p
    | OpCode.dynamicPad => parseWithoutWFCheck p
    | OpCode.dynamicReshape => parseWithoutWFCheck p
    | OpCode.dynamicSlice => parseWithoutWFCheck p
    | OpCode.dynamicUpdateSlice => parseWithoutWFCheck p
    | OpCode.exponential => parseWithoutWFCheck p
    | OpCode.exponentialMinusOne => parseWithoutWFCheck p
    | OpCode.fft => parseWithoutWFCheck p
    | OpCode.floor => parseWithoutWFCheck p
    | OpCode.gather => parseWithoutWFCheck p
    | OpCode.getDimensionSize => parseWithoutWFCheck p
    | OpCode.getTupleElement => parseWithoutWFCheck p
    | OpCode.if => parseWithoutWFCheck p
    | OpCode.imag => parseWithoutWFCheck p
    | OpCode.infeed =>
      report "Semantics implementation defined infeed"
      parseWithoutWFCheck p
    | OpCode.iota => parseWithoutWFCheck p
    | OpCode.isFinite => parseWithoutWFCheck p
    | OpCode.log => parseWithoutWFCheck p
    | OpCode.logPlusOne => parseWithoutWFCheck p
    | OpCode.logistic => parseWithoutWFCheck p
    | OpCode.map => parseWithoutWFCheck p
    | OpCode.maximum => parseWithoutWFCheck p
    | OpCode.minimum => parseWithoutWFCheck p
    | OpCode.multiply => parseWithoutWFCheck p
    | OpCode.negate => parseWithoutWFCheck p
    | OpCode.not => parseWithoutWFCheck p
    | OpCode.optimizationBarrier => parseWithoutWFCheck p
    | OpCode.or => parseWithoutWFCheck p
    | OpCode.outfeed => parseWithoutWFCheck p
    | OpCode.pad => parseWithoutWFCheck p
    | OpCode.partitionId => parseWithoutWFCheck p
    | OpCode.popcnt => parseWithoutWFCheck p
    | OpCode.power => parseWithoutWFCheck p
    | OpCode.real => parseWithoutWFCheck p
    | OpCode.realDynamicSlice => parseWithoutWFCheck p -- Undocumented
    | OpCode.recv => parseWithoutWFCheck p
    | OpCode.reduce => parseWithoutWFCheck p
    | OpCode.reducePrecision => parseWithoutWFCheck p
    | OpCode.reduceScatter => parseWithoutWFCheck p
    | OpCode.reduceWindow => parseWithoutWFCheck p
    | OpCode.remainder => parseWithoutWFCheck p
    | OpCode.replicaId => parseWithoutWFCheck p
    | OpCode.reshape => parseWithoutWFCheck p
    | OpCode.return => parseWithoutWFCheck p
    | OpCode.reverse => parseWithoutWFCheck p
    | OpCode.rng =>
      report "explore for deprecation rng"
      parseWithoutWFCheck p
    | OpCode.rngBitGenerator => parseWithoutWFCheck p
    | OpCode.roundNearestAfz => parseWithoutWFCheck p
    | OpCode.roundNearestEven => parseWithoutWFCheck p
    | OpCode.rsqrt => parseWithoutWFCheck p
    | OpCode.scatter => parseWithoutWFCheck p
    | OpCode.select => parseWithoutWFCheck p
    | OpCode.selectAndScatter => parseWithoutWFCheck p
    | OpCode.send => parseWithoutWFCheck p
    | OpCode.shiftLeft => parseWithoutWFCheck p
    | OpCode.shiftRightArithmetic => parseWithoutWFCheck p
    | OpCode.shiftRightLogical => parseWithoutWFCheck p
    | OpCode.sign => parseWithoutWFCheck p
    | OpCode.sine => parseWithoutWFCheck p
    | OpCode.slice => parseWithoutWFCheck p
    | OpCode.sort => parseWithoutWFCheck p
    | OpCode.sqrt => parseWithoutWFCheck p
    | OpCode.subtract => parseWithoutWFCheck p
    | OpCode.tan => parseWithoutWFCheck p
    | OpCode.tanh => parseTanh p
    | OpCode.transpose => parseWithoutWFCheck p
    | OpCode.triangularSolve => parseWithoutWFCheck p
    | OpCode.tuple => parseWithoutWFCheck p
    | OpCode.uniformDequantize => parseWithoutWFCheck p
    | OpCode.uniformQuantize => parseWithoutWFCheck p
    | OpCode.while =>
      report "semantics not fully decided "
      parseWithoutWFCheck p
    | OpCode.xor => parseWithoutWFCheck p

  partial def parseOperation : PState Operation := do
    let mut opOutputs := []
    if ← is "%" then
      opOutputs ← parseOpOutputs
      parseItem "="
    parseItem "\""
    let name ← parseId
    parseItem "\""
    let opInputValues ← parseValueUseList
    let mut opInputAttrs := []
    if ← is "<{" then
      opInputAttrs ← parseOperationDictionaryAttributes
    let mut opInputFuncs := []
    if ← is "(" then
      opInputFuncs ← parseOpInputFuncs
    parseItem ":"
    let functiontype ← parseFunctionType
    let nameSplit := name.splitOn "."
    if nameSplit.length ≠ 2 then throw <| (← error "invalid operation format")
    let dialect := nameSplit.get! 0
    let opName := nameSplit.get! 1

    let preOperation : PreOperation := { dialect := dialect,
                                         operation := opName,
                                         inputValues := opInputValues,
                                         inputFunctions := opInputFuncs,
                                         inputAttributes := opInputAttrs,
                                         outputs := opOutputs,
                                         signature := functiontype}

    match dialect with
    | "stablehlo" => toStableHLO preOperation
    | _ => report s!"undocumented operation {dialect} {opName}" ; parseWithoutWFCheck preOperation

  partial def parseInputFuncBody : PState (List Operation) := do
    parseListAuxNoSep "}" parseOperation []

end

end StableHLO.Parsing
