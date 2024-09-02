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
  let st ← get
  if st.is "true" then parseItem "true" ; return some BooleanLiteral.true
  if st.is "false" then parseItem "false" ; return some BooleanLiteral.false
  return none

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

def parseElementLiteral : PState ElementLiteral := do
  let st ← get
  if st.is "(" then return ElementLiteral.complexLiteral <| ← parseComplexLiteral
  if let some r ← tryParseBooleanLiteral then return ElementLiteral.booleanLiteral  r
  return ElementLiteral.floatLiteral <| ← parseFloatLiteral

def parseDenseElements (closingMark : String) : PState (List ElementLiteral) := do
  parseListAux closingMark (some ",") parseElementLiteral

partial def parseDenseLiteral : PState DenseLiteral := do
  let st ← get
  if st.is "[" then
    let denseDimension ← parseList "[" "]" (some ",") parseDenseLiteral
    return DenseLiteral.denseDimension denseDimension
  else
    let denseElements ← parseDenseElements "]"
    return DenseLiteral.denseElements denseElements

def parseTensorLiteral : PState TensorLiteral := do
  parseItem "dense"
  parseItem "<"
  let st ← get
  if st.is "[" then
    let denseLiteral ← parseDenseLiteral
    parseItem ">"
    return denseLiteral
  else
    let denseElements ← parseDenseElements ">"
    let denseLiteral := DenseLiteral.denseElements denseElements
    parseItem ">"
    return denseLiteral

def parseTensorConstant : PState TensorConstant := do
  let tensorLiteral ← parseTensorLiteral
  parseItem ":"
  let tensorType ← parseTensorType
  return { literal := tensorLiteral, type := tensorType }

def parseStringLiteral : PState String := do
  parseString

def parseStringConstant : PState Constant := do
  let str ← parseStringLiteral
  return Constant.stringConstant str

def tryParseComparisonDirection : PState (Option ComparisonDirection) := do
  let st ← get
  if st.is "EQ" then parseItem "EQ" ; return ComparisonDirection.eq
  if st.is "NE" then parseItem "NE" ; return ComparisonDirection.ne
  if st.is "GE" then parseItem "GE" ; return ComparisonDirection.ge
  if st.is "GT" then parseItem "GT" ; return ComparisonDirection.gt
  if st.is "LE" then parseItem "LE" ; return ComparisonDirection.le
  if st.is "LT" then parseItem "LT" ; return ComparisonDirection.lt
  return none

def tryParseCompareType : PState (Option CompareType) := do
  let st ← get
  if st.is "FLOAT" then parseItem "FLOAT" ; return CompareType.float
  if st.is "TOTALORDER" then parseItem "TOTALORDER" ; return CompareType.totalOrder
  if st.is "SIGNED" then parseItem "SIGNED" ; return CompareType.signed
  if st.is "UNSIGNED" then parseItem "UNSIGNED" ; return CompareType.unsigned
  return none

def tryParsePrecisionConfig : PState (Option PrecisionConfig) := do
  let st ← get
  if st.is "DEFAULT" then parseItem "DEFAULT" ; return PrecisionConfig.default
  if st.is "HIGH" then parseItem "HIGH" ; return PrecisionConfig.high
  if st.is "HIGHEST" then parseItem "HIGHEST" ; return PrecisionConfig.highest
  return none

def tryParseFftType : PState (Option FftType) := do
  let st ← get
  if st.is "FFT" then parseItem "FFT" ; return FftType.fft
  if st.is "IFFT" then parseItem "IFFT" ; return FftType.ifft
  if st.is "RFFT" then parseItem "RFFT" ; return FftType.rfft
  if st.is "IRFFT" then parseItem "IRFFT" ; return FftType.irfft
  return none

def tryParseChannelType : PState (Option ChannelType) := do
  let st ← get
  if st.is "DEVICE_TO_DEVICE" then parseItem "DEVICE_TO_DEVICE" ; return ChannelType.deviceToDevice
  if st.is "HOST_TO_DEVICE" then parseItem "HOST_TO_DEVICE" ; return ChannelType.hostToDevice
  return none

def tryParseRngDistribution : PState (Option RngDistribution) := do
  let st ← get
  if st.is "UNIFORM" then parseItem "UNIFORM" ; return RngDistribution.uniform
  if st.is "NORMAL" then parseItem "NORMAL" ; return RngDistribution.normal
  return none

def tryParseRngAlgorithm : PState (Option RngAlgorithm) := do
  let st ← get
  if st.is "DEFAULT" then parseItem "DEFAULT" ; return RngAlgorithm.default
  if st.is "THREE_FRY" then parseItem "THREE_FRY" ; return RngAlgorithm.threeFry
  if st.is "PHILOX" then parseItem "PHILOX" ; return RngAlgorithm.philox
  return none

def tryParseTransposeA : PState (Option TransposeA) := do
  let st ← get
  if st.is "NO_TRANSPOSE" then parseItem "NO_TRANSPOSE" ; return TransposeA.noTranspose
  if st.is "TRANSPOSE" then parseItem "TRANSPOSE" ; return TransposeA.transpose
  if st.is "ADJOINT" then parseItem "ADJOINT" ; return TransposeA.adjoint
  return none

def tryParseEnumLiteral : PState (Option EnumLiteral) := do
  if let some r ← tryParseComparisonDirection then return EnumLiteral.comparisonDirection r
  else if let some r ← tryParseCompareType then return EnumLiteral.compareType r
  else if let some r ← tryParsePrecisionConfig then return EnumLiteral.precisionConfig r
  else if let some r ← tryParseFftType then return EnumLiteral.fftType r
  else if let some r ← tryParseChannelType then return EnumLiteral.channelType r
  else if let some r ← tryParseRngDistribution then return EnumLiteral.rngDistribution r
  else if let some r ← tryParseRngAlgorithm then return EnumLiteral.rngAlgorithm r
  else if let some r ← tryParseTransposeA then return EnumLiteral.transposeA r
  else return none

def parseConstant : PState Constant := do
  let st ← get
  if let some r ← tryParseEnumLiteral then return Constant.enumConstant r
  if let some r ← tryParseBooleanLiteral then return Constant.booleanConstant r
  if st.is "\"" then return ← parseStringConstant
  if st.is "(" then return Constant.complexConstant <| ← parseComplexConstant
  if st.is "dense" then return Constant.tensorConstant <| ← parseTensorConstant
  return Constant.numberConsant <| ← parseNumberConstant

end StableHLO
