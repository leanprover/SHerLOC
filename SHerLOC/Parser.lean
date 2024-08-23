/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic

namespace StableHLO

structure TokWithPos where
  tok : String
  line : Nat
  column : Nat
  deriving Repr, Inhabited, Nonempty

instance : ToString TokWithPos where
  toString := fun t : TokWithPos => s!"({t.line},{t.column},{t.tok})"

structure NonTerminal where
  startLine : Nat
  startColumn : Nat
  endLine : Nat
  endColumn : Nat
  nonTerminal : String
  deriving Repr, Inhabited, Nonempty

instance : ToString NonTerminal where
  toString := fun t : NonTerminal => s!"({t.startLine},{t.startColumn}):({t.endLine},{t.endColumn}):{t.nonTerminal}"

structure ParsingState where
  source : List TokWithPos     -- Source data being parsed
  index : Nat                  -- Index into source data
  status : List NonTerminal    -- For debugging the parser
  deriving Repr, Inhabited, Nonempty

def ParsingState.tok (st : ParsingState) : String :=
  st.source[st.index]!.tok

def ParsingState.is (st : ParsingState) (keyword : String) : Bool :=
  st.tok = keyword

def ParsingState.line (st : ParsingState) : Nat :=
  st.source[st.index]!.line

def ParsingState.column (st : ParsingState) : Nat :=
  st.source[st.index]!.column

def ParsingState.lookahead (st : ParsingState) (shift : Nat) : String :=
  st.source[st.index + shift]!.tok

def ParsingState.isName (st : ParsingState) : Bool :=
  (st.tok.get ⟨0⟩).isAlphanum
  && st.tok.all fun c => c.isAlphanum || c = '_'

def ParsingState.isAttributeName (st : ParsingState) : Bool :=
  (st.tok.get ⟨0⟩).isAlphanum
  && st.tok.all fun c => c.isAlphanum || c = '_' || c = '.'

def ParsingState.isDecimal (st : ParsingState) : Option Nat :=
  if st.tok.all fun c => c.isDigit then some st.tok.toNat! else none

def ParsingState.isHexaDecimal (st : ParsingState) : Bool := Id.run do
  let b₁ := (st.tok.get ⟨0⟩) == '0'
  let b₂ := (st.tok.get ⟨1⟩) == 'x'
  let mut b₃ := true
  for i in [2:st.tok.length] do
    let c := st.tok.get ⟨i⟩
    if ! (c.isDigit || c.val ≥ 48 && c.val ≤ 57 || c.val ≥ 65 && c.val ≤ 70 || c.val ≥ 97 && c.val ≤ 102) then
      b₃ := false
  return b₁ && b₂ && b₃

def ParsingState.error (st : ParsingState) (msg : String) : String :=
  s!"{st.status.reverse} \n\n Parsing error line {st.line}, column {st.column}: expected {msg}, found {st.tok} "

abbrev PState (T : Type) := StateT ParsingState (Except String) T

def record (st : ParsingState) (nonTerminal : String) : PState Unit := do
  let st' ← get
  let info : NonTerminal := { startLine := st.line, startColumn := st.column, endLine := st'.line, endColumn := st'.column, nonTerminal := nonTerminal }
  set { st' with status := info :: st'.status }

def shift : PState Unit := do
  let st ← get
  set { st with index := st.index + 1 }

def parseItem (keyword : String) : PState Unit := do
  let st ← get
  if ! st.is keyword then
    throw <| st.error keyword
  shift

def parseId : PState String := do
  let st ← get
  if st.isName then
    shift
    return st.tok
  else
    throw <| st.error "Name"

def parseBooleanLiteral : PState Bool := do
  let st ← get
  if st.is "true" then
    shift
    return true
  else if st.is "false" then
    shift
    return false
  else throw <| st.error "Boolean literal"

def parseBooleanConstant : PState Constant := do
  let b ← parseBooleanLiteral
  return Constant.booleanConstant b

def parseDecimal : PState Nat := do
  let st ← get
  if let some i := st.isDecimal then shift ; return i
  else throw <| st.error "Decimal number"

def parseIntegerLiteral : PState Nat := do
  let st ← get
  let mut negative := false
  if st.is "+" then shift
  if st.is "-" then shift ; negative := true
  parseDecimal

def parseIntegerType : PState IntegerType := do
  let st ← get
  match st.tok with
  | "si2" => shift ; return { sign := Signedness.signed , size := IntegerSize.b2 }
  | "si4" => shift ; return { sign := Signedness.signed , size := IntegerSize.b4 }
  | "si8" => shift ; return { sign := Signedness.signed , size := IntegerSize.b8 }
  | "si16" => shift ; return { sign := Signedness.signed , size := IntegerSize.b16 }
  | "si32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | "si64" => shift ; return { sign := Signedness.signed , size := IntegerSize.b64 }
  | "ui2" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b2 }
  | "ui4" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b4 }
  | "ui8" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b8 }
  | "ui16" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b16 }
  | "ui32" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b32 }
  | "ui64" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b64 }
  -- Jax compatibility
  | "i32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | _ => throw <| st.error "Integer type"

def parseIntegerConstant : PState Constant := do
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  return Constant.integerConstant i t

def parseStringLiteral : PState String := do
  let st ← get
  parseItem "\""
  if (st.lookahead 1) = "\"" then return ""
  else
    let content ← parseId -- Verify String encoding
    parseItem "\""
    return content

def parseStringConstant : PState Constant := do
  let str ← parseStringLiteral
  return Constant.stringConstant str

def parseConstant : PState Constant := do
  let st ← get
  if st.tok.get ⟨ 0 ⟩ = '"' then return ← parseStringConstant
  match st.tok with
  | "true" | "false" => parseBooleanConstant
  | _ =>
    match (st.lookahead 2).get! ⟨ 0 ⟩ with
    | 's' | 'u' | 'i' /- Jax compatibility -/ => parseIntegerConstant
    | _ => throw <| st.error s!"Constant"

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

partial def parseListAux (closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  let st ← get
  if st.is closingMark then return []
  if let some sep := separator then if st.is sep then shift ; return ← parseListAux closingMark separator parse
  let attr ← parse
  let attrs ← parseListAux closingMark separator parse
  return attr :: attrs

def parseList (openingMark closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  parseItem openingMark
  let attrs ← parseListAux closingMark separator parse
  parseItem closingMark
  return attrs

def parseAttributes : PState (List Attribute) := do
  parseList "{" "}" (some ",") parseAttribute

partial def parseShape : PState (List Nat) := do
  let st ← get
  if (st.lookahead 1).get! ⟨ 0 ⟩ = 'x'
  then
    let i ← parseDecimal
    let shape ← parseShape
    return i :: shape
  else return []

def parseTensorElementType : PState TensorElementType := do
  let st ← get
  if st.is "i1" then return TensorElementType.booleanType
  let c := st.tok.get! ⟨ 0 ⟩
  if c = 's' || c = 'u' || c = 'i' then return TensorElementType.integerType <| ← parseIntegerType
  else throw <| st.error "TensorElementType"

def parseTensorType : PState TensorType := do
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorElementType ← parseTensorElementType
  parseItem ">"
  return { shape := shape , tensorElementType := tensorElementType}

-- temporary, shortcut for testing
def parseValueType : PState ValueType := do
  let temporary ← parseTensorType
  return ValueType.tensorType temporary

def parseValueId : PState String := do
  parseItem "%"
  parseId

def parseFuncInput : PState FuncInput := do
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  discard <| parseAttributes
  return { id := id , typ := typ }

def parseFuncInputs : PState (List FuncInput) := do
  parseList "(" ")" (some ",") parseFuncInput

def parseFuncOutput : PState ValueType := do
  let typ ← parseValueType
  discard <| parseAttributes
  return typ

def parseFuncOutputs : PState (List ValueType) := do
  parseList "(" ")" (some ",") parseFuncOutput

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

def parseValueTypes : PState (List ValueType) := do
  parseList "(" ")" (some ",") parseValueType

def parseFunctionType : PState FunctionType := do
  let inputTypes ← parseValueTypes
  parseItem "-"
  parseItem ">"
  let outputType ← parseValueTypes
  let functionType := { domain := inputTypes , range := outputType }
  return functionType

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
  return operation

-- TODO complete shortcut for now, ignoring return and call (and perhaps constant)
def parseOperation : PState Operation := do
  parseStableOp

def parseOperations : PState (List Operation) :=
  parseList "{" "}" none parseOperation

def parseFunction : PState Function := do
  let stStart ← get
  parseItem "func.func"
  parseItem "public"
  parseItem "@"
  let funcId ← parseId
  let funcInputs ← parseFuncInputs
  parseItem "-"
  parseItem ">"
  let funcOutputs ← parseFuncOutputs
  let body ← parseOperations
  let func := { funcId := funcId , funcInputs := funcInputs , funcOutputs := funcOutputs , funcBody := body }
  record stStart "Function"
  return func

def parseFunctions : PState (List Function) := do
  parseList "{" "}" none parseFunction

def parseModule : PState Program := do
  parseItem "module"
  parseItem "@"
  let id ← parseId
  parseItem "attributes"
  let attrs ← parseAttributes
  let funcs ← parseFunctions
  return funcs

def tokenize (src : List Char) : List TokWithPos := Id.run do
  let mut lineNumber : Nat := 0
  let mut lastColumnNumber : Nat := 0
  let mut columnNumber : Nat := 0
  let mut tokens : List TokWithPos := []
  let mut current : String := ""
  for c in src do
    if c.isAlphanum || c = '_' || c = '+' || c = '-' || c = '.' then
      current := current.push c
      columnNumber := columnNumber + 1
    else
      if current != "" then
        tokens := { tok := current, line := lineNumber, column := lastColumnNumber } :: tokens
        current := ""
        lastColumnNumber := columnNumber
      match c with
      | ' ' | ',' => columnNumber := columnNumber + 1
      | '\n' => columnNumber := 0 ; lineNumber := lineNumber + 1
      | '\t' => panic "tokenize: tab not implemented yet"
      | c =>
        tokens := { tok := c.toString , line := lineNumber, column := lastColumnNumber } :: tokens
        columnNumber := columnNumber + 1
      lastColumnNumber := columnNumber
  return tokens.reverse

def parse (src : List Char) : Program := Id.run do
  let tokens := tokenize src
  let r ← parseModule.run' <| ParsingState.mk tokens 0 []
  match r with
  | .ok p => p
  | .error e => panic e

end StableHLO
