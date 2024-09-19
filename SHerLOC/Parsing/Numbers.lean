/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers

namespace StableHLO.Parsing

def parseBooleanLiteral : PState BooleanLiteral := do
  if ← isParse "true" then return BooleanLiteral.true
  if ← isParse "false" then return BooleanLiteral.false
  throw <| ← error "Boolean literal"

def parseIntegerLiteral : PState IntegerLiteral := do
  let mut sign := Sign.plus
  if ← isParse "+" then sign := Sign.plus
  else if ← isParse "-" then sign := Sign.minus
  let mut nat : Option Nat := none
  if ← is "0x" then
    nat ← parseHexaDecimal
  else
    nat ← parseDecimal
  if let some v := nat then
    let parseResult := { sign := sign , decimal := v }
    return parseResult
  else
    throw <| ← error "Integer literal"

def parseFloatLiteral : PState FloatLiteral := do
  let mut sign := Sign.plus
  if ← isParse "+" then sign := Sign.plus
  else if ← isParse "-" then sign := Sign.minus
  if ← is "0x" then
    let nat ← parseHexaDecimal
    return FloatLiteral.hexaDecimal nat
  else
    let nat ← parseDecimal
    let integerPart : IntegerLiteral := { sign := sign , decimal := nat }
    let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
    if ← isParse "." then
      fractionalPart := {fractionalPart with decimal := ← parseDecimal}
    let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
    if (← isParse "e") || (← isParse "E") then
      let mut scientificSign := Sign.plus
      if ← isParse "+" then scientificSign := Sign.plus
      else if ← isParse "-" then scientificSign := Sign.minus
      let nat ← parseDecimal
      scientificPart := { sign := scientificSign, decimal := nat }
    let parseResult := FloatLiteral.decimal
      { integerPart := integerPart,
        fractionalPart := fractionalPart,
        scientificPart := scientificPart
      }
    return parseResult

def parseComplexLiteral : PState ComplexLiteral := do
  parseItem "("
  let realPart ← parseFloatLiteral
  parseItem ","
  let imaginaryPart ← parseFloatLiteral
  parseItem ")"
  let parseResult := { real := realPart, imaginary := imaginaryPart }
  return parseResult

def parseElementLiteral : PState ElementLiteral := do
  skip
  if (← isDigit) || (← is "+") || (← is "-") then
    return ElementLiteral.floatLiteral <| ← parseFloatLiteral
  if (← is "t") || (← is "f") then
    return ElementLiteral.booleanLiteral <| ← parseBooleanLiteral
  if (← is "(") then
    return ElementLiteral.complexLiteral <| ← parseComplexLiteral
  if (← is "\"") then -- Not a good idea to try to parse these directly as numbers, they can be extremely large
    return ElementLiteral.stringLiteral <| ← parseString
  throw <| ← error "Element literal"

def parseDenseElements (closingMark : String) : PState (List ElementLiteral) := do
  parseListAux closingMark "," parseElementLiteral

partial def parseDenseLiteral : PState DenseLiteral := do
  if ← is "[" then
    let denseDimension ← parseList "[" "]" "," parseDenseLiteral
    return DenseLiteral.denseDimension denseDimension
  else
    let denseElements ← parseDenseElements "]"
    return DenseLiteral.denseElements denseElements

def parseTensorLiteral : PState TensorLiteral := do
  parseItem "dense"
  parseItem "<"
  if ← is "[" then
    let denseLiteral ← parseDenseLiteral
    parseItem ">"
    return denseLiteral
  else
    let denseElements ← parseDenseElements ">"
    let denseLiteral := DenseLiteral.denseElements denseElements
    parseItem ">"
    return denseLiteral

def parseStringLiteral : PState String := do
  parseString

def parseComparisonDirection : PState ComparisonDirection := do
  let mut r := none
  if ← isParse "EQ" then r := ComparisonDirection.eq
  if ← isParse "NE" then r := ComparisonDirection.ne
  if ← isParse "GE" then r := ComparisonDirection.ge
  if ← isParse "GT" then r := ComparisonDirection.gt
  if ← isParse "LE" then r := ComparisonDirection.le
  if ← isParse "LT" then r := ComparisonDirection.lt
  if let some res := r then
    return res
  else throw <| ← error "comparison direction"

def parseCompareType : PState CompareType := do
  let mut r := none
  if ← isParse "FLOAT" then r := CompareType.float
  if ← isParse "TOTALORDER" then r := CompareType.totalOrder
  if ← isParse "SIGNED" then r := CompareType.signed
  if ← isParse "UNSIGNED" then r := CompareType.unsigned
  if let some res := r then
    return res
  else throw <| ← error "compaare type"

def parsePrecisionConfig : PState PrecisionConfig := do
  let mut r := none
  if ← isParse "DEFAULT" then r := PrecisionConfig.default
  if ← isParse "HIGHEST" then r := PrecisionConfig.highest
  if ← isParse "HIGH" then r := PrecisionConfig.high
  if let some res := r then
    return res
  else throw <| ← error "precision config"

def parseFftType : PState FftType := do
  let mut r := none
  if ← isParse "FFT" then r := FftType.fft
  if ← isParse "IFFT" then r := FftType.ifft
  if ← isParse "RFFT" then r := FftType.rfft
  if ← isParse "IRFFT" then r := FftType.irfft
  if let some res := r then
    return res
  else throw <| ← error "FFT type"

def parseChannelType : PState ChannelType := do
  let mut r := none
  if ← isParse "DEVICE_TO_DEVICE" then r := ChannelType.deviceToDevice
  if ← isParse "HOST_TO_DEVICE" then r := ChannelType.hostToDevice
  if let some res := r then
    return res
  else throw <| ← error "channel type"

def parseRngDistribution : PState RngDistribution := do
  let mut r := none
  if ← isParse "UNIFORM" then r := RngDistribution.uniform
  if ← isParse "NORMAL" then r := RngDistribution.normal
  if let some res := r then
    return res
  else throw <| ← error "rng distribution"

def parseRngAlgorithm : PState RngAlgorithm := do
  let mut r := none
  if ← isParse "DEFAULT" then r := RngAlgorithm.default
  if ← isParse "THREE_FRY" then r := RngAlgorithm.threeFry
  if ← isParse "PHILOX" then r := RngAlgorithm.philox
  if let some res := r then
    return res
  else throw <| ← error "rng algorithm"

def parseTransposeA : PState TransposeA := do
  let mut r := none
  if ← isParse "NO_TRANSPOSE" then r := TransposeA.noTranspose
  if ← isParse "TRANSPOSE" then r := TransposeA.transpose
  if ← isParse "ADJOINT" then r := TransposeA.adjoint
  if let some res := r then
    return res
  else throw <| ← error "tranpose annotation"

def parseEnumLiteral : PState EnumLiteral := do
  parseItem "<"
  let mut r := none
  if ← isParse "comparison_direction" then r := EnumLiteral.comparisonDirection <| ← parseComparisonDirection
  if ← isParse "comparison_type" then r := EnumLiteral.compareType <| ← parseCompareType
  if ← isParse "precision" then r := EnumLiteral.precisionConfig <| ← parsePrecisionConfig
  if ← isParse "fft_type" then r := EnumLiteral.fftType <| ← parseFftType
  if ← isParse "channel_type" then r := EnumLiteral.channelType <| ← parseChannelType
  if ← isParse "rng_distribution" then r := EnumLiteral.rngDistribution <| ← parseRngDistribution
  if ← isParse "rng_algorithm" then r := EnumLiteral.rngAlgorithm <| ← parseRngAlgorithm
  if ← isParse "transpose" then r := EnumLiteral.transposeA <| ← parseTransposeA
  if let some res := r then
    parseItem ">"
    return res
  else throw <| ← error "enumeration"

def parseArrayLiteral : PState ArrayLiteral := do
  parseItems ["array", "<"]
  if ← isParse "i64" then
    let mut r := []
    if ← isParse ":" then
      r ← parseListAux ">" "," parseIntegerLiteral
    parseItem ">"
    return ArrayLiteral.array64 r
  if ← isParse "i1" then
    let mut r := []
    if ← isParse ":" then
      r ← parseListAux ">" "," parseBooleanLiteral
    parseItem ">"
    return ArrayLiteral.array1 r
  throw <| ← error "array literal"

def parseConvolutionMode : PState ConvolutionMode := do
  let mut r := none
  if (← isParse "o") then r := ConvolutionMode.o
  else if (← isParse "f") then r := ConvolutionMode.f
  else if (← isParse "i") then r := ConvolutionMode.i
  else if (← isParse "0") then r := ConvolutionMode.zero
  else if (← isParse "1") then r := ConvolutionMode.one
  else if (← isParse "b") then r := ConvolutionMode.b
  else if (← isParse "2") then r := ConvolutionMode.two
  if let some res := r then return res
  else throw <| ← error "convolution mode"

def parseConvolutionModes : PState (List ConvolutionMode) := do
  parseList "[" "]" "," parseConvolutionMode

def parseConvolution : PState Convolution := do
  parseItem "<"
  let lhs ← parseConvolutionModes
  parseItem "x"
  let rhs ← parseConvolutionModes
  parseItem "->"
  let result ← parseConvolutionModes
  parseItem ">"
  return { lhs, rhs, result }

end StableHLO.Parsing
