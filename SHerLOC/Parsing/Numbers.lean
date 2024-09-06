/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers

namespace StableHLO

def parseBooleanLiteral : PState BooleanLiteral := do
  if ← isParse "true" then return BooleanLiteral.true
  if ← isParse "false" then return BooleanLiteral.false
  throw <| ← error "Boolean literal"

def parseIntegerLiteral : PState IntegerLiteral := do
  push "parseIntegerLiteral"
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
    pop "parseIntegerLiteral"
    return parseResult
  else
    throw <| ← error "Integer literal"

def parseFloatLiteral : PState FloatLiteral := do
  push "parseFloatLiteral"
  let mut sign := Sign.plus
  if ← isParse "+" then sign := Sign.plus
  else if ← isParse "-" then sign := Sign.minus
  if ← is "0x" then
    let nat ← parseHexaDecimal
    pop "parseFloatLiteral"
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
    pop "parseFloatLiteral"
    return parseResult

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

def parseElementLiteral : PState ElementLiteral := do
  skip
  if (← isDigit) || (← is "+") || (← is "-") then
    return ElementLiteral.floatLiteral <| ← parseFloatLiteral
  if (← is "t") || (← is "f") then
    return ElementLiteral.booleanLiteral <| ← parseBooleanLiteral
  if (← is "(") then
    return ElementLiteral.complexLiteral <| ← parseComplexLiteral
  throw <| ← error "Element literal"

def parseDenseElements (closingMark : String) : PState (List ElementLiteral) := do
  push "parseDenseElements"
  let r ← parseListAux closingMark "," parseElementLiteral
  pop "parseDenseElements"
  return r

partial def parseDenseLiteral : PState DenseLiteral := do
  push "parseDenseLiteral"
  if ← is "[" then
    let denseDimension ← parseList "[" "]" "," parseDenseLiteral
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

def parseStringLiteral : PState String := do
  push "parseStringLiteral"
  let r ← parseString
  pop "parseStringLiteral"
  return r

def parseComparisonDirection : PState ComparisonDirection := do
  push "parseComparisonDirection"
  let mut r := none
  if ← isParse "EQ" then r := ComparisonDirection.eq
  if ← isParse "NE" then r := ComparisonDirection.ne
  if ← isParse "GE" then r := ComparisonDirection.ge
  if ← isParse "GT" then r := ComparisonDirection.gt
  if ← isParse "LE" then r := ComparisonDirection.le
  if ← isParse "LT" then r := ComparisonDirection.lt
  if let some res := r then
    pop "parseComparisonDirection"
    return res
  else throw <| ← error "comparison direction"

def parseCompareType : PState CompareType := do
  push "parseCompareType"
  let mut r := none
  if ← isParse "FLOAT" then r := CompareType.float
  if ← isParse "TOTALORDER" then r := CompareType.totalOrder
  if ← isParse "SIGNED" then r := CompareType.signed
  if ← isParse "UNSIGNED" then r := CompareType.unsigned
  if let some res := r then
    pop "parseCompareType"
    return res
  else throw <| ← error "compaare type"

def parsePrecisionConfig : PState PrecisionConfig := do
  push "parsePrecisionConfig"
  let mut r := none
  if ← isParse "DEFAULT" then r := PrecisionConfig.default
  if ← isParse "HIGH" then r := PrecisionConfig.high
  if ← isParse "HIGHEST" then r := PrecisionConfig.highest
  if let some res := r then
    pop "parsePrecisionConfig"
    return res
  else throw <| ← error "precision config"

def parseFftType : PState FftType := do
  push "parseFftType"
  let mut r := none
  if ← isParse "FFT" then r := FftType.fft
  if ← isParse "IFFT" then r := FftType.ifft
  if ← isParse "RFFT" then r := FftType.rfft
  if ← isParse "IRFFT" then r := FftType.irfft
  if let some res := r then
    pop "parseFftType"
    return res
  else throw <| ← error "FFT type"

def parseChannelType : PState ChannelType := do
  push "parseChannelType"
  let mut r := none
  if ← isParse "DEVICE_TO_DEVICE" then r := ChannelType.deviceToDevice
  if ← isParse "HOST_TO_DEVICE" then r := ChannelType.hostToDevice
  if let some res := r then
    pop "parseChannelType"
    return res
  else throw <| ← error "channel type"

def parseRngDistribution : PState RngDistribution := do
  push "parseRngDistribution"
  let mut r := none
  if ← isParse "UNIFORM" then r := RngDistribution.uniform
  if ← isParse "NORMAL" then r := RngDistribution.normal
  if let some res := r then
    pop "parseRngDistribution"
    return res
  else throw <| ← error "rng distribution"

def parseRngAlgorithm : PState RngAlgorithm := do
  push "parseRngAlgorithm"
  let mut r := none
  if ← isParse "DEFAULT" then r := RngAlgorithm.default
  if ← isParse "THREE_FRY" then r := RngAlgorithm.threeFry
  if ← isParse "PHILOX" then r := RngAlgorithm.philox
  if let some res := r then
    pop "parseRngAlgorithm"
    return res
  else throw <| ← error "rng algorithm"

def parseTransposeA : PState TransposeA := do
  push "parseTransposeA"
  let mut r := none
  if ← isParse "NO_TRANSPOSE" then r := TransposeA.noTranspose
  if ← isParse "TRANSPOSE" then r := TransposeA.transpose
  if ← isParse "ADJOINT" then r := TransposeA.adjoint
  if let some res := r then
    pop "parseTransposeA"
    return res
  else throw <| ← error "tranpose annotation"

def parseEnumLiteral : PState EnumLiteral := do
  push "parseEnumLiteral"
  parseItem "#stablehlo"
  parseItem "<"
  let mut r := none
  if ← isParse "comparison_direction" then r := EnumLiteral.comparisonDirection <| ← parseComparisonDirection
  if ← isParse "comparison_type" then r := EnumLiteral.compareType <| ← parseCompareType
  if ← isParse "precision" then r := EnumLiteral.precisionConfig <| ← parsePrecisionConfig
  if ← isParse "fft_type" then r := EnumLiteral.fftType <| ← parseFftType
  if ← isParse "channel_type" then r := EnumLiteral.channelType <| ← parseChannelType
  if ← isParse "rng_distribution" then r := EnumLiteral.rngDistribution <| ← parseRngDistribution
  if ← isParse "rng_algorithm" then r := EnumLiteral.rngAlgorithm <| ← parseRngAlgorithm
  if ← isParse "transpose_a" then r := EnumLiteral.transposeA <| ← parseTransposeA
  if let some res := r then
    parseItem ">"
    pop "parseEnumLiteral"
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

def parseExperiment1 : PState (List FuncId) := do
  parseList "[" "]" "," parseFuncId

def parseExperiment2 : PState (List (List FuncId)) := do
  parseList "[" "]" "," parseExperiment1

def parseExperiment3 : PState (List EnumLiteral) := do
  parseList "[" "]" "," parseEnumLiteral

def parseChannelHandle : PState (ChannelHandle) := do
  parseItems ["<"]
  parseItems ["handle", "="]
  let handle ← parseDecimal
  parseItem ","
  parseItems ["type", "="]
  let typ ← parseDecimal
  parseItem ">"
  return { handle := handle, typ := typ }

def parseLiteral : PState Literal := do
  skip
  if (← isDigit) || (← isChar '+') || (← isChar '-') then
    return Literal.element <| ElementLiteral.floatLiteral <| ← parseFloatLiteral
  if ← isChar 'd' then
    return Literal.tensor <| ← parseTensorLiteral
  if (← isChar 't') || (← isChar 'f') then
    return Literal.element <| ElementLiteral.booleanLiteral <| ← parseBooleanLiteral
  if (← isChar '(') then
    return Literal.element <| ElementLiteral.complexLiteral <| ← parseComplexLiteral
  if ← isChar '"' then
    return Literal.string <| ← parseStringLiteral
  if ← isChar 'a' then
    return Literal.array <| ← parseArrayLiteral

  if ← isChar '#' then {
    if ← isParse "#stablehlo.conv" then flyOver "<" ">"; return Literal.special
    if ← isParse "#stablehlo.dot_algorithm" then flyOver "<" ">"; return Literal.special
    if ← isParse "#stablehlo.dot" then flyOver "<" ">"; return Literal.special
    if ← isParse "#stablehlo.channel_handle" then
      return Literal.channelHandle <| ← parseChannelHandle
    if ← isParse "#stablehlo.scatter" then flyOver "<" ">"; return Literal.special
    if ← isParse "#stablehlo.gather" then flyOver "<" ">"; return Literal.special
    if ← is "#stablehlo" then return Literal.enum <| ← parseEnumLiteral
  }

  if ← isChar '[' then {
    if ← is "[[" then
      return Literal.experiment2 <| ← parseExperiment2
    if ← is "[#" then
      return Literal.experiment3 <| ← parseExperiment3
    if ← is "[" then
      return Literal.experiment1 <| ← parseExperiment1
  }

  if ← isChar '{' then
    flyOver "{" "}"; return Literal.special

  if ← isChar '@' then
    return Literal.func <| ← parseFuncId

  throw <| ← error "literal"

end StableHLO
