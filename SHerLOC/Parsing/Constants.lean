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

def tryParseComparisonDirection : PState (Option ComparisonDirection) := do
  push "tryParseComparisonDirection"
  let mut r := none
  if ← isParse "EQ" then r := ComparisonDirection.eq
  if ← isParse "NE" then r := ComparisonDirection.ne
  if ← isParse "GE" then r := ComparisonDirection.ge
  if ← isParse "GT" then r := ComparisonDirection.gt
  if ← isParse "LE" then r := ComparisonDirection.le
  if ← isParse "LT" then r := ComparisonDirection.lt
  pop "tryParseComparisonDirection"
  return r

def tryParseCompareType : PState (Option CompareType) := do
  push "tryParseCompareType"
  let mut r := none
  if ← isParse "FLOAT" then r := CompareType.float
  if ← isParse "TOTALORDER" then r := CompareType.totalOrder
  if ← isParse "SIGNED" then r := CompareType.signed
  if ← isParse "UNSIGNED" then r := CompareType.unsigned
  pop "tryParseCompareType"
  return r

def tryParsePrecisionConfig : PState (Option PrecisionConfig) := do
  push "tryParsePrecisionConfig"
  let mut r := none
  if ← isParse "DEFAULT" then r := PrecisionConfig.default
  if ← isParse "HIGH" then r := PrecisionConfig.high
  if ← isParse "HIGHEST" then r := PrecisionConfig.highest
  pop "tryParsePrecisionConfig"
  return r

def tryParseFftType : PState (Option FftType) := do
  push "tryParseFftType"
  let mut r := none
  if ← isParse "FFT" then r := FftType.fft
  if ← isParse "IFFT" then r := FftType.ifft
  if ← isParse "RFFT" then r := FftType.rfft
  if ← isParse "IRFFT" then r := FftType.irfft
  pop "tryParseFftType"
  return r

def tryParseChannelType : PState (Option ChannelType) := do
  push "tryParseChannelType"
  let mut r := none
  if ← isParse "DEVICE_TO_DEVICE" then r := ChannelType.deviceToDevice
  if ← isParse "HOST_TO_DEVICE" then r := ChannelType.hostToDevice
  pop "tryParseChannelType"
  return r

def tryParseRngDistribution : PState (Option RngDistribution) := do
  push "tryParseRngDistribution"
  let mut r := none
  if ← isParse "UNIFORM" then r := RngDistribution.uniform
  if ← isParse "NORMAL" then r := RngDistribution.normal
  pop "tryParseRngDistribution"
  return r

def tryParseRngAlgorithm : PState (Option RngAlgorithm) := do
  push "tryParseRngAlgorithm"
  let mut r := none
  if ← isParse "DEFAULT" then r := RngAlgorithm.default
  if ← isParse "THREE_FRY" then r := RngAlgorithm.threeFry
  if ← isParse "PHILOX" then r := RngAlgorithm.philox
  pop "tryParseRngAlgorithm"
  return r

def tryParseTransposeA : PState (Option TransposeA) := do
  push "tryParseTransposeA"
  let mut r := none
  if ← isParse "NO_TRANSPOSE" then r := TransposeA.noTranspose
  if ← isParse "TRANSPOSE" then r := TransposeA.transpose
  if ← isParse "ADJOINT" then r := TransposeA.adjoint
  pop "tryParseTransposeA"
  return r

def tryParseEnumLiteral : PState (Option EnumLiteral) := do
  push "tryParseEnumLiteral"
  let mut res := none
  if let some r ← tryParseComparisonDirection then res := EnumLiteral.comparisonDirection r
  if let some r ← tryParseCompareType then res := EnumLiteral.compareType r
  if let some r ← tryParsePrecisionConfig then res := EnumLiteral.precisionConfig r
  if let some r ← tryParseFftType then res := EnumLiteral.fftType r
  if let some r ← tryParseChannelType then res := EnumLiteral.channelType r
  if let some r ← tryParseRngDistribution then res := EnumLiteral.rngDistribution r
  if let some r ← tryParseRngAlgorithm then res := EnumLiteral.rngAlgorithm r
  if let some r ← tryParseTransposeA then res := EnumLiteral.transposeA r
  pop "tryParseEnumLiteral";
  return res

def parseConstant : PState Constant := do
  push "parseConstant"
  if let some r ← tryParseBooleanLiteral then pop "parseConstant" ; return Constant.booleanConstant r
  if ← is "\"" then pop "parseConstant" ; return ← parseStringConstant
  if ← is "(" then pop "parseConstant" ; return Constant.complexConstant <| ← parseComplexConstant
  if ← is "dense" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseTensorConstant
  if ← is "array" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseArrayConstant
  if let some r ← tryParseEnumLiteral then pop "parseConstant" ; return Constant.enumConstant r
  let r ← parseNumberConstant
  pop "parseConstant"
  return Constant.numberConsant <| r

end StableHLO
