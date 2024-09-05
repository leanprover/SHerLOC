/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Types

namespace StableHLO

def tryParseBooleanLiteral : PState (Option BooleanLiteral) := do
  push "tryParseBooleanLiteral"
  let mut r : Option BooleanLiteral := none
  if ← isParse "true" then r := some BooleanLiteral.true
  if ← isParse "false" then r := some BooleanLiteral.false
  pop "tryParseBooleanLiteral"
  return r

def parseComplexLiteral : PState ComplexLiteral := do
  push "parseComplexLiteral"
  parseItem "("
  let realPart ← parseFloatLiteral
  parseItem ","
  let imaginaryPart ← parseFloatLiteral
  parseItem ")"
  let parseResult := { real := realPart, imaginary := imaginaryPart }
  pop "parseComplexLiteral"
  return parseResult

def parseComplexConstant : PState ComplexConstant := do
  push "parseComplexConstant"
  let complexLiteral ← parseComplexLiteral
  parseItem ":"
  let complexType ← parseComplexType
  pop "parseComplexConstant"
  return { literal := complexLiteral, type := complexType }

def parseElementLiteral : PState ElementLiteral := do
  push "parseElementLiteral"
  if ← is "(" then pop "parseElementLiteral" ; return ElementLiteral.complexLiteral <| ← parseComplexLiteral
  if let some r ← tryParseBooleanLiteral then pop "parseElementLiteral" ; return ElementLiteral.booleanLiteral  r
  pop "parseElementLiteral"
  return ElementLiteral.floatLiteral <| ← parseFloatLiteral

def parseDenseElements (closingMark : String) : PState (List ElementLiteral) := do
  push "parseDenseElements"
  let r ← parseListAux closingMark (some ",") parseElementLiteral
  pop "parseDenseElements"
  return r

partial def parseDenseLiteral : PState DenseLiteral := do
  push "parseDenseLiteral"
  if ← is "[" then
    let denseDimension ← parseList "[" "]" (some ",") parseDenseLiteral
    pop "parseDenseLiteral"
    return DenseLiteral.denseDimension denseDimension
  else
    let denseElements ← parseDenseElements "]"
    pop "parseDenseLiteral"
    return DenseLiteral.denseElements denseElements

def parseTensorLiteral : PState TensorLiteral := do
  push "parseTensorLiteral"
  parseItem "dense"
  parseItem "<"
  if ← is "[" then
    let denseLiteral ← parseDenseLiteral
    parseItem ">"
    pop "parseTensorLiteral"
    return denseLiteral
  else
    let denseElements ← parseDenseElements ">"
    let denseLiteral := DenseLiteral.denseElements denseElements
    parseItem ">"
    pop "parseTensorLiteral"
    return denseLiteral

def parseTensorConstant : PState TensorConstant := do
  push "parseTensorConstant"
  let tensorLiteral ← parseTensorLiteral
  parseItem ":"
  let tensorType ← parseTensorType
  pop "parseTensorConstant"
  return { literal := tensorLiteral, type := tensorType }

def parseArrayConstant : PState TensorConstant := do
  push "parseArrayConstant"
  parseItem "array"
  parseItem "<"
  let typ : TensorElementType ← parseTensorElementType
  let typ : TensorElementTypeGen := TensorElementTypeGen.classic typ
  let typ : TensorType := { shape := [], tensorElementTypeGen := typ }
  if ← isParse ">" then
    let literal := DenseLiteral.denseElements []
    pop "parseArrayConstant"
    return { literal := literal, type := typ }
  else if ← isParse ":" then
    let literal ← parseListAux ">" "," parseElementLiteral -- This will surely need to be generalized
    parseItem ">"
    let literal := DenseLiteral.denseElements literal
    pop "parseArrayConstant"
    return { literal := literal, type := typ }
  else throw <| ← error "Array constant"

def parseStringLiteral : PState String := do
  push "parseStringLiteral"
  let r ← parseString
  pop "parseStringLiteral"
  return r

def parseStringConstant : PState Constant := do
  push "parseStringConstant"
  let str ← parseStringLiteral
  pop "parseStringConstant"
  return Constant.stringConstant str

def parseComparisonDirection : PState ComparisonDirection := do
  push "tryParseComparisonDirection"
  let mut r := none
  if ← isParse "EQ" then r := ComparisonDirection.eq
  if ← isParse "NE" then r := ComparisonDirection.ne
  if ← isParse "GE" then r := ComparisonDirection.ge
  if ← isParse "GT" then r := ComparisonDirection.gt
  if ← isParse "LE" then r := ComparisonDirection.le
  if ← isParse "LT" then r := ComparisonDirection.lt
  if let some res := r then
    pop "tryParseComparisonDirection"
    return res
  else throw <| ← error "comparison direction"

def parseCompareType : PState CompareType := do
  push "tryParseCompareType"
  let mut r := none
  if ← isParse "FLOAT" then r := CompareType.float
  if ← isParse "TOTALORDER" then r := CompareType.totalOrder
  if ← isParse "SIGNED" then r := CompareType.signed
  if ← isParse "UNSIGNED" then r := CompareType.unsigned
  if let some res := r then
    pop "tryParseCompareType"
    return res
  else throw <| ← error "compaare type"

def parsePrecisionConfig : PState PrecisionConfig := do
  push "tryParsePrecisionConfig"
  let mut r := none
  if ← isParse "DEFAULT" then r := PrecisionConfig.default
  if ← isParse "HIGH" then r := PrecisionConfig.high
  if ← isParse "HIGHEST" then r := PrecisionConfig.highest
  if let some res := r then
    pop "tryParsePrecisionConfig"
    return res
  else throw <| ← error "precision config"

def parseFftType : PState FftType := do
  push "tryParseFftType"
  let mut r := none
  if ← isParse "FFT" then r := FftType.fft
  if ← isParse "IFFT" then r := FftType.ifft
  if ← isParse "RFFT" then r := FftType.rfft
  if ← isParse "IRFFT" then r := FftType.irfft
  if let some res := r then
    pop "tryParseFftType"
    return res
  else throw <| ← error "FFT type"

def parseChannelType : PState ChannelType := do
  push "tryParseChannelType"
  let mut r := none
  if ← isParse "DEVICE_TO_DEVICE" then r := ChannelType.deviceToDevice
  if ← isParse "HOST_TO_DEVICE" then r := ChannelType.hostToDevice
  if let some res := r then
    pop "tryParseChannelType"
    return res
  else throw <| ← error "channel type"

def parseRngDistribution : PState RngDistribution := do
  push "tryParseRngDistribution"
  let mut r := none
  if ← isParse "UNIFORM" then r := RngDistribution.uniform
  if ← isParse "NORMAL" then r := RngDistribution.normal
  if let some res := r then
    pop "tryParseRngDistribution"
    return res
  else throw <| ← error "rng distribution"

def parseRngAlgorithm : PState RngAlgorithm := do
  push "tryParseRngAlgorithm"
  let mut r := none
  if ← isParse "DEFAULT" then r := RngAlgorithm.default
  if ← isParse "THREE_FRY" then r := RngAlgorithm.threeFry
  if ← isParse "PHILOX" then r := RngAlgorithm.philox
  if let some res := r then
    pop "tryParseRngAlgorithm"
    return res
  else throw <| ← error "rng algorithm"

def parseTransposeA : PState TransposeA := do
  push "tryParseTransposeA"
  let mut r := none
  if ← isParse "NO_TRANSPOSE" then r := TransposeA.noTranspose
  if ← isParse "TRANSPOSE" then r := TransposeA.transpose
  if ← isParse "ADJOINT" then r := TransposeA.adjoint
  if let some res := r then
    pop "tryParseTransposeA"
    return res
  else throw <| ← error "tranpose annotation"

def parseEnumLiteral : PState EnumLiteral := do
  push "tryParseEnumLiteral"
  parseItem "#stablehlo"
  parseItem "<"
  let mut r := none
  if ← isParse "comparison_direction" then r := EnumLiteral.comparisonDirection <| ← parseComparisonDirection
  if ← isParse "comparison_type" then r := EnumLiteral.compareType <| ← parseCompareType
  if ← isParse "precision_config" then r := EnumLiteral.precisionConfig <| ← parsePrecisionConfig
  if ← isParse "fft_type" then r := EnumLiteral.fftType <| ← parseFftType
  if ← isParse "channel_type" then r := EnumLiteral.channelType <| ← parseChannelType
  if ← isParse "rng_distribution" then r := EnumLiteral.rngDistribution <| ← parseRngDistribution
  if ← isParse "rng_algorithm" then r := EnumLiteral.rngAlgorithm <| ← parseRngAlgorithm
  if ← isParse "transpose_a" then r := EnumLiteral.transposeA <| ← parseTransposeA
  if let some res := r then
    parseItem ">"
    pop "tryParseEnumLiteral"
    return res
  else throw <| ← error "enumeration"

def parseConstant : PState Constant := do
  push "parseConstant"
  if let some r ← tryParseBooleanLiteral then pop "parseConstant" ; return Constant.booleanConstant r
  if ← is "\"" then pop "parseConstant" ; return ← parseStringConstant
  if ← is "(" then pop "parseConstant" ; return Constant.complexConstant <| ← parseComplexConstant
  if ← is "dense" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseTensorConstant
  if ← is "array" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseArrayConstant
  if ← isParse "#stablehlo.conv" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special
  if ← isParse "#stablehlo.dot_algorithm" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special
  if ← isParse "#stablehlo.dot" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special
  if ← isParse "#stablehlo.channel_handle" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special
  if ← isParse "#stablehlo.scatter" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special
  if ← isParse "#stablehlo.gather" then flyOver "<" ">"; pop "parseConstant" ; return Constant.special


  if ← is "#stablehlo" then pop "parseConstant" ; return Constant.enumConstant <| ← parseEnumLiteral
  if ← is "[[" then flyOver "[[" "]]"; pop "parseConstant" ; return Constant.special
  if ← is "[" then flyOver "[" "]"; pop "parseConstant" ; return Constant.special
  let r ← parseNumberConstant
  pop "parseConstant"
  return Constant.numberConsant <| r

end StableHLO
