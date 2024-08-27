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

def parseComplexLiteral : PState ComplexLiteral := do
  parseItem "("
  let realPart ← parseFloatLiteral
  parseItem ","
  let imaginaryPart ← parseFloatLiteral
  parseItem ")"
  let parseResult := { real := realPart, imaginary := imaginaryPart }
  return parseResult

def parseComplexConstant : PState ComplexConstant := do
  let complexLiteral ← parseComplexLiteral
  parseItem ":"
  let complexType ← parseComplexType
  return { literal := complexLiteral, type := complexType }

--def parseElementLiteral : PState ElementLiteral := do

def parseTensorLiteral : PState TensorLiteral := do sorry

def parseTensorConstant : PState TensorConstant := do
  let tensorLiteral ← parseTensorLiteral
  parseItem ":"
  let tensorType ← parseTensorType
  return { literal := tensorLiteral, type := tensorType }

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

-- Types do not appear in literals
-- Note that 'dense<3.4> : tensor<si32>' is described as a syntax error
-- So we need type information during parsing
-- Otherwise we could just parse a number and type check later

-- 8 categories of constants
-- of this 8 categories, 3 do not have an explicit type: String, Boolean, Enum
-- With 1 token, we can disambiguate these 3
-- The 5 others must have a type annotation following a ':'
-- The type annotation is unambiguous, but not with only 1 token
-- we can decide with 1 token if it is an integer
-- we can decide with 1 token if it is a float
-- we can decide with 1 token if it is a complex ('complex')
-- we can decide with 1 token if it is a tensor
-- but there are two types for tensors, and we need an arbitrary number of token to disambiguate them

def parseConstant : PState Constant := do
  let st ← get

  -- There are 8 categories of constants
  -- Of those 8, 3 do not have explicit types but can be chosen unambiguously with 1 token
  -- Enumerations
  if let some r := tryParseEnumLiteral st.tok then shift ; return Constant.enumConstant r
  -- Booleans:
  if st.tok = "true" || st.tok = "false" then return ← parseBooleanConstant
  -- Strings:
  -- Isn't the following a token on its own?
  if st.tok.get ⟨ 0 ⟩ = '"' then return ← parseStringConstant

  -- For the 5 other categories, we have an explicit type
  -- We need information about this type to disambiguate how to parse the literal
  -- That information follows the ":" token
  -- Of those 5 possible types, 3 can be chosen unambiguously with a single token (after ":")

  -- The last two types are for Tensors (quantized or not)

  match st.tok with
  | "true" | "false" => parseBooleanConstant
  | _ =>
    match (st.lookahead 2).get! ⟨ 0 ⟩ with
    | 's' | 'u' | 'i' /- Jax compatibility -/ => return Constant.integerConstant <| ← parseIntegerConstant
    | _ => throw <| st.error s!"Constant"

end StableHLO
