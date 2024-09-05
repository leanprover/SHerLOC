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
  if ← is "true" then parseItem "true" ; pop "tryParseBooleanLiteral" ; return some BooleanLiteral.true
  if ← is "false" then parseItem "false" ; pop "tryParseBooleanLiteral" ; return some BooleanLiteral.false
  pop "tryParseBooleanLiteral"
  return none

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
  if ← is ">" then
    parseItem ">"
    let literal := DenseLiteral.denseElements []
    pop "parseArrayConstant"
    return { literal := literal, type := typ }
  else if ← is ":" then
    parseItem ":"
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

def tryParseComparisonDirection : PState (Option ComparisonDirection) := do
  push "tryParseComparisonDirection"
  if ← is "EQ" then parseItem "EQ" ; pop "tryParseComparisonDirection"; return ComparisonDirection.eq
  if ← is "NE" then parseItem "NE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.ne
  if ← is "GE" then parseItem "GE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.ge
  if ← is "GT" then parseItem "GT" ; pop "tryParseComparisonDirection"; return ComparisonDirection.gt
  if ← is "LE" then parseItem "LE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.le
  if ← is "LT" then parseItem "LT" ; pop "tryParseComparisonDirection"; return ComparisonDirection.lt
  pop "tryParseComparisonDirection"
  return none

def tryParseCompareType : PState (Option CompareType) := do
  push "tryParseCompareType"
  if ← is "FLOAT" then parseItem "FLOAT" ; pop "tryParseCompareType"; return CompareType.float
  if ← is "TOTALORDER" then parseItem "TOTALORDER" ; pop "tryParseCompareType"; return CompareType.totalOrder
  if ← is "SIGNED" then parseItem "SIGNED" ; pop "tryParseCompareType"; return CompareType.signed
  if ← is "UNSIGNED" then parseItem "UNSIGNED" ; pop "tryParseCompareType"; return CompareType.unsigned
  pop "tryParseCompareType"
  return none

def tryParsePrecisionConfig : PState (Option PrecisionConfig) := do
  push "tryParsePrecisionConfig"
  if ← is "DEFAULT" then parseItem "DEFAULT" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.default
  if ← is "HIGH" then parseItem "HIGH" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.high
  if ← is "HIGHEST" then parseItem "HIGHEST" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.highest
  pop "tryParsePrecisionConfig"
  return none

def tryParseFftType : PState (Option FftType) := do
  push "tryParseFftType"
  if ← is "FFT" then parseItem "FFT" ; pop "tryParseFftType"; return FftType.fft
  if ← is "IFFT" then parseItem "IFFT" ; pop "tryParseFftType"; return FftType.ifft
  if ← is "RFFT" then parseItem "RFFT" ; pop "tryParseFftType"; return FftType.rfft
  if ← is "IRFFT" then parseItem "IRFFT" ; pop "tryParseFftType"; return FftType.irfft
  pop "tryParseFftType"
  return none

def tryParseChannelType : PState (Option ChannelType) := do
  push "tryParseChannelType"
  if ← is "DEVICE_TO_DEVICE" then parseItem "DEVICE_TO_DEVICE" ; pop "tryParseChannelType"; return ChannelType.deviceToDevice
  if ← is "HOST_TO_DEVICE" then parseItem "HOST_TO_DEVICE" ; pop "tryParseChannelType"; return ChannelType.hostToDevice
  pop "tryParseChannelType"
  return none

def tryParseRngDistribution : PState (Option RngDistribution) := do
  push "tryParseRngDistribution"
  if ← is "UNIFORM" then parseItem "UNIFORM" ; pop "tryParseRngDistribution"; return RngDistribution.uniform
  if ← is "NORMAL" then parseItem "NORMAL" ; pop "tryParseRngDistribution"; return RngDistribution.normal
  pop "tryParseRngDistribution"
  return none

def tryParseRngAlgorithm : PState (Option RngAlgorithm) := do
  push "tryParseRngAlgorithm"
  if ← is "DEFAULT" then parseItem "DEFAULT" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.default
  if ← is "THREE_FRY" then parseItem "THREE_FRY" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.threeFry
  if ← is "PHILOX" then parseItem "PHILOX" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.philox
  pop "tryParseRngAlgorithm"
  return none

def tryParseTransposeA : PState (Option TransposeA) := do
  push "tryParseTransposeA"
  if ← is "NO_TRANSPOSE" then parseItem "NO_TRANSPOSE" ; pop "tryParseTransposeA"; return TransposeA.noTranspose
  if ← is "TRANSPOSE" then parseItem "TRANSPOSE" ; pop "tryParseTransposeA"; return TransposeA.transpose
  if ← is "ADJOINT" then parseItem "ADJOINT" ; pop "tryParseTransposeA"; return TransposeA.adjoint
  pop "tryParseTransposeA"
  return none

def tryParseEnumLiteral : PState (Option EnumLiteral) := do
  push "tryParseEnumLiteral"
  if let some r ← tryParseComparisonDirection then pop "tryParseEnumLiteral"; return EnumLiteral.comparisonDirection r
  else if let some r ← tryParseCompareType then pop "tryParseEnumLiteral"; return EnumLiteral.compareType r
  else if let some r ← tryParsePrecisionConfig then pop "tryParseEnumLiteral"; return EnumLiteral.precisionConfig r
  else if let some r ← tryParseFftType then pop "tryParseEnumLiteral"; return EnumLiteral.fftType r
  else if let some r ← tryParseChannelType then pop "tryParseEnumLiteral"; return EnumLiteral.channelType r
  else if let some r ← tryParseRngDistribution then pop "tryParseEnumLiteral"; return EnumLiteral.rngDistribution r
  else if let some r ← tryParseRngAlgorithm then pop "tryParseEnumLiteral"; return EnumLiteral.rngAlgorithm r
  else if let some r ← tryParseTransposeA then pop "tryParseEnumLiteral"; return EnumLiteral.transposeA r
  else pop "tryParseEnumLiteral"; return none

def parseConstant : PState Constant := do
  push "parseConstant"
  if let some r ← tryParseEnumLiteral then pop "parseConstant" ; return Constant.enumConstant r
  if let some r ← tryParseBooleanLiteral then pop "parseConstant" ; return Constant.booleanConstant r
  if ← is "\"" then pop "parseConstant" ; return ← parseStringConstant
  if ← is "(" then pop "parseConstant" ; return Constant.complexConstant <| ← parseComplexConstant
  if ← is "dense" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseTensorConstant
  if ← is "array" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseArrayConstant
  pop "parseConstant"
  return Constant.numberConsant <| ← parseNumberConstant

end StableHLO
