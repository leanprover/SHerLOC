/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Types

namespace StableHLO

def tryParseBooleanLiteral (tok : String) : Option BooleanLiteral := do
  match tok with
  | "true" => some BooleanLiteral.true
  | "false" => some BooleanLiteral.false
  | _ => none

def parseBooleanLiteral : PState BooleanLiteral := do
  let st ← get
  if st.is "true" then
    shift
    return BooleanLiteral.true
  else if st.is "false" then
    shift
    return BooleanLiteral.false
  else throw <| st.error "Boolean literal"

def parseBooleanConstant : PState Constant := do
  let b ← parseBooleanLiteral
  return Constant.booleanConstant b

def parseIntegerLiteral : PState IntegerLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then shift
  if st.is "-" then shift ; sign := Sign.minus
  let nat ← parseDecimal
  let parseResult := { sign := sign , decimal := nat }
  return parseResult

def parseIntegerConstant : PState Constant := do
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  return Constant.integerConstant i t

def parseFloatLiteral : PState FloatLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then shift
  if st.is "-" then shift ; sign := Sign.minus
  let nat ← parseDecimal
  let integerPart := { sign := sign , decimal := nat }
  let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
  if st.is "." then
    shift
    fractionalPart := {fractionalPart with decimal := ← parseDecimal}
  let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
  if st.is "e" || st.is "E" then
    shift
    let mut scientificSign := Sign.plus
    if st.is "+" then shift
    if st.is "-" then shift ; scientificSign := Sign.minus
    let nat ← parseDecimal
    scientificPart := { sign := scientificSign, decimal := nat }
  let parseResult :=
    { integerPart := integerPart,
      fractionalPart := fractionalPart,
      scientificPart := scientificPart
    }
  return parseResult

def parseFloatConstant : PState Constant := do
  let floatLiteral ← parseFloatLiteral
  parseItem ":"
  let floatType ← parseFloatType
  return Constant.floatConstant floatLiteral floatType

def parseComplexLiteral : PState ComplexLiteral := do
  parseItem "("
  let realPart ← parseFloatLiteral
  parseItem ","
  let imaginaryPart ← parseFloatLiteral
  parseItem ")"
  let parseResult := { real := realPart, imaginary := imaginaryPart }
  return parseResult

def parseComplexConstant : PState Constant := do
  let complexLiteral ← parseComplexLiteral
  parseItem ":"
  let complexType ← parseComplexType
  return Constant.complexConstant complexLiteral complexType

def parseTensorConstant : PState Constant := do
  let st ← get
  throw <| st.error s!"Parser NIY: parseTensorConstant"

def parseQuantizedTensorConstant : PState Constant := do
  let st ← get
  throw <| st.error s!"Parser NIY: parseQuantizedTensorConstant"

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

def tryParseComparisonDirection (tok : String) : Option ComparisonDirection :=
  match tok with
  | "EQ" => some ComparisonDirection.eq
  | "NE" => some ComparisonDirection.ne
  | "GE" => some ComparisonDirection.ge
  | "GT" => some ComparisonDirection.gt
  | "LE" => some ComparisonDirection.le
  | "LT" => some ComparisonDirection.lt
  | _ => none

def tryParseCompareType (tok : String) : Option CompareType :=
  match tok with
  | "FLOAT" => some CompareType.float
  | "TOTALORDER" => some CompareType.totalOrder
  | "SIGNED" => some CompareType.signed
  | "UNSIGNED" => some CompareType.unsigned
  | _ => none

def tryParsePrecisionConfig (tok : String) : Option PrecisionConfig :=
  match tok with
  | "DEFAULT" => some PrecisionConfig.default
  | "HIGH" => some PrecisionConfig.high
  | "HIGHEST" => some PrecisionConfig.highest
  | _ => none

def tryParseFftType (tok : String) : Option FftType :=
  match tok with
  | "FFT" => some FftType.fft
  | "IFFT" => some FftType.ifft
  | "RFFT" => some FftType.rfft
  | "IRFFT" => some FftType.irfft
  | _ => none

def tryParseChannelType (tok : String) : Option ChannelType :=
  match tok with
  | "DEVICE_TO_DEVICE" => some ChannelType.deviceToDevice
  | "HOST_TO_DEVICE" => some ChannelType.hostToDevice
  | _ => none

def tryParseRngDistribution (tok : String) : Option RngDistribution :=
  match tok with
  | "UNIFORM" => some RngDistribution.uniform
  | "NORMAL" => some RngDistribution.normal
  | _ => none

def tryParseRngAlgorithm (tok : String) : Option RngAlgorithm :=
  match tok with
  | "DEFAULT" => some RngAlgorithm.default
  | "THREE_FRY" => some RngAlgorithm.threeFry
  | "PHILOX" => some RngAlgorithm.philox
  | _ => none

def tryParseTransposeA (tok : String) : Option TransposeA :=
  match tok with
  | "NO_TRANSPOSE" => some TransposeA.noTranspose
  | "TRANSPOSE" => some TransposeA.transpose
  | "ADJOINT" => some TransposeA.adjoint
  | _ => none

def tryParseEnumLiteral (tok : String) : Option EnumLiteral :=
  if let some r := tryParseComparisonDirection tok then EnumLiteral.comparisonDirection r
  else if let some r := tryParseCompareType tok then EnumLiteral.compareType r
  else if let some r := tryParsePrecisionConfig tok then EnumLiteral.precisionConfig r
  else if let some r := tryParseFftType tok then EnumLiteral.fftType r
  else if let some r := tryParseChannelType tok then EnumLiteral.channelType r
  else if let some r := tryParseRngDistribution tok then EnumLiteral.rngDistribution r
  else if let some r := tryParseRngAlgorithm tok then EnumLiteral.rngAlgorithm r
  else if let some r := tryParseTransposeA tok then EnumLiteral.transposeA r
  else none

-- def parseEnumConstant : PState Constant := do
--   let st ← get
--   throw <| st.error s!"Constant"

def parseConstant : PState Constant := do
  let st ← get
  if let some r := tryParseEnumLiteral st.tok then shift ; return Constant.enumConstant r
  if st.tok.get ⟨ 0 ⟩ = '"' then return ← parseStringConstant
  match st.tok with
  | "true" | "false" => parseBooleanConstant
  | _ =>
    match (st.lookahead 2).get! ⟨ 0 ⟩ with
    | 's' | 'u' | 'i' /- Jax compatibility -/ => parseIntegerConstant
    | _ => throw <| st.error s!"Constant"

end StableHLO
