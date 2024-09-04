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
  let st ← get
  if st.is "true" then parseItem "true" ; pop "tryParseBooleanLiteral" ; return some BooleanLiteral.true
  if st.is "false" then parseItem "false" ; pop "tryParseBooleanLiteral" ; return some BooleanLiteral.false
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
  let st ← get
  if st.is "(" then pop "parseElementLiteral" ; return ElementLiteral.complexLiteral <| ← parseComplexLiteral
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
  let st ← get
  if st.is "[" then
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
  let st ← get
  if st.is "[" then
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
  let st ← get
  if st.is ">" then
    parseItem ">"
    let literal := DenseLiteral.denseElements []
    pop "parseArrayConstant"
    return { literal := literal, type := typ }
  else if st.is ":" then
    parseItem ":"
    let literal ← parseListAux ">" "," parseElementLiteral -- This will surely need to be generalized
    parseItem ">"
    let literal := DenseLiteral.denseElements literal
    pop "parseArrayConstant"
    return { literal := literal, type := typ }
  else throw <| st.error "Array constant"

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
  let st ← get
  if st.is "EQ" then parseItem "EQ" ; pop "tryParseComparisonDirection"; return ComparisonDirection.eq
  if st.is "NE" then parseItem "NE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.ne
  if st.is "GE" then parseItem "GE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.ge
  if st.is "GT" then parseItem "GT" ; pop "tryParseComparisonDirection"; return ComparisonDirection.gt
  if st.is "LE" then parseItem "LE" ; pop "tryParseComparisonDirection"; return ComparisonDirection.le
  if st.is "LT" then parseItem "LT" ; pop "tryParseComparisonDirection"; return ComparisonDirection.lt
  pop "tryParseComparisonDirection"
  return none

def tryParseCompareType : PState (Option CompareType) := do
  push "tryParseCompareType"
  let st ← get
  if st.is "FLOAT" then parseItem "FLOAT" ; pop "tryParseCompareType"; return CompareType.float
  if st.is "TOTALORDER" then parseItem "TOTALORDER" ; pop "tryParseCompareType"; return CompareType.totalOrder
  if st.is "SIGNED" then parseItem "SIGNED" ; pop "tryParseCompareType"; return CompareType.signed
  if st.is "UNSIGNED" then parseItem "UNSIGNED" ; pop "tryParseCompareType"; return CompareType.unsigned
  pop "tryParseCompareType"
  return none

def tryParsePrecisionConfig : PState (Option PrecisionConfig) := do
  push "tryParsePrecisionConfig"
  let st ← get
  if st.is "DEFAULT" then parseItem "DEFAULT" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.default
  if st.is "HIGH" then parseItem "HIGH" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.high
  if st.is "HIGHEST" then parseItem "HIGHEST" ; pop "tryParsePrecisionConfig"; return PrecisionConfig.highest
  pop "tryParsePrecisionConfig"
  return none

def tryParseFftType : PState (Option FftType) := do
  push "tryParseFftType"
  let st ← get
  if st.is "FFT" then parseItem "FFT" ; pop "tryParseFftType"; return FftType.fft
  if st.is "IFFT" then parseItem "IFFT" ; pop "tryParseFftType"; return FftType.ifft
  if st.is "RFFT" then parseItem "RFFT" ; pop "tryParseFftType"; return FftType.rfft
  if st.is "IRFFT" then parseItem "IRFFT" ; pop "tryParseFftType"; return FftType.irfft
  pop "tryParseFftType"
  return none

def tryParseChannelType : PState (Option ChannelType) := do
  push "tryParseChannelType"
  let st ← get
  if st.is "DEVICE_TO_DEVICE" then parseItem "DEVICE_TO_DEVICE" ; pop "tryParseChannelType"; return ChannelType.deviceToDevice
  if st.is "HOST_TO_DEVICE" then parseItem "HOST_TO_DEVICE" ; pop "tryParseChannelType"; return ChannelType.hostToDevice
  pop "tryParseChannelType"
  return none

def tryParseRngDistribution : PState (Option RngDistribution) := do
  push "tryParseRngDistribution"
  let st ← get
  if st.is "UNIFORM" then parseItem "UNIFORM" ; pop "tryParseRngDistribution"; return RngDistribution.uniform
  if st.is "NORMAL" then parseItem "NORMAL" ; pop "tryParseRngDistribution"; return RngDistribution.normal
  pop "tryParseRngDistribution"
  return none

def tryParseRngAlgorithm : PState (Option RngAlgorithm) := do
  push "tryParseRngAlgorithm"
  let st ← get
  if st.is "DEFAULT" then parseItem "DEFAULT" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.default
  if st.is "THREE_FRY" then parseItem "THREE_FRY" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.threeFry
  if st.is "PHILOX" then parseItem "PHILOX" ; pop "tryParseRngAlgorithm"; return RngAlgorithm.philox
  pop "tryParseRngAlgorithm"
  return none

def tryParseTransposeA : PState (Option TransposeA) := do
  push "tryParseTransposeA"
  let st ← get
  if st.is "NO_TRANSPOSE" then parseItem "NO_TRANSPOSE" ; pop "tryParseTransposeA"; return TransposeA.noTranspose
  if st.is "TRANSPOSE" then parseItem "TRANSPOSE" ; pop "tryParseTransposeA"; return TransposeA.transpose
  if st.is "ADJOINT" then parseItem "ADJOINT" ; pop "tryParseTransposeA"; return TransposeA.adjoint
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
  let st ← get
  if let some r ← tryParseEnumLiteral then pop "parseConstant" ; return Constant.enumConstant r
  if let some r ← tryParseBooleanLiteral then pop "parseConstant" ; return Constant.booleanConstant r
  if st.is "\"" then pop "parseConstant" ; return ← parseStringConstant
  if st.is "(" then pop "parseConstant" ; return Constant.complexConstant <| ← parseComplexConstant
  if st.is "dense" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseTensorConstant
  if st.is "array" then pop "parseConstant" ; return Constant.tensorConstant <| ← parseArrayConstant
  pop "parseConstant"
  return Constant.numberConsant <| ← parseNumberConstant

end StableHLO
