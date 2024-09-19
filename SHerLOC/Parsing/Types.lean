/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Numbers

namespace StableHLO.Parsing

def tryParseIntegerType : PState (Option IntegerType) := do
  let mut r : Option IntegerType := none
  if ← isChar 'i' then {
    if ← isParse "i32" then r := some { sign := Signedness.signed , size := IntegerSize.b32 }
    if ← isParse "i64" then r := some { sign := Signedness.signed , size := IntegerSize.b64 }
    if ← isParse "i2" then r := some { sign := Signedness.signed , size := IntegerSize.b2 }
    if ← isParse "i4" then r := some { sign := Signedness.signed , size := IntegerSize.b4 }
    if ← isParse "i8" then r := some { sign := Signedness.signed , size := IntegerSize.b8 }
    if ← isParse "i16" then r := some { sign := Signedness.signed , size := IntegerSize.b16 }
  }
  if ← isParse "ui32" then r := some { sign := Signedness.unsigned , size := IntegerSize.b32 }
  if ← isParse "ui64" then r := some { sign := Signedness.unsigned , size := IntegerSize.b64 }
  if ← isParse "ui2" then r := some { sign := Signedness.unsigned , size := IntegerSize.b2 }
  if ← isParse "ui4" then r := some { sign := Signedness.unsigned , size := IntegerSize.b4 }
  if ← isParse "ui8" then r := some { sign := Signedness.unsigned , size := IntegerSize.b8 }
  if ← isParse "ui16" then r := some { sign := Signedness.unsigned , size := IntegerSize.b16 }
  return r

def parseIntegerType : PState IntegerType := do
  if let some r ← tryParseIntegerType then return r
  else throw <| ← error "Integer type"

def tryParseFloatType : PState (Option FloatType) := do
  let mut r : Option FloatType := none
  if ← isChar 'f' then {
    if ← isParse "f16" then r := some FloatType.f16
    if ← isParse "f32" then r := some FloatType.f32
    if ← isParse "f64" then r := some FloatType.f64
    if ← isParse "f8E3M4" then r := some FloatType.f8E3M4
    if ← isParse "f8E4M3B11FNUZ" then r := some FloatType.f8E4M3B11FNUZ
    if ← isParse "f8E4M3FNUZ" then r := some FloatType.f8E4M3FNUZ
    if ← isParse "f8E4M3FN" then r := some FloatType.f8E4M3FN
    if ← isParse "f8E4M3" then r := some FloatType.f8E4M3
    if ← isParse "f8E5M2FNUZ" then r := some FloatType.f8E5M2FNUZ
    if ← isParse "f8E5M2" then r := some FloatType.f8E5M2
  }
  if ← isParse "bf16" then r := some FloatType.bf16
  if ← isParse "tf32" then r := some FloatType.tf32
  return r

def parseFloatType : PState FloatType := do
  if let some r ← tryParseFloatType then return r
  else throw <| ← error "Float type"

def parseNumberType : PState NumberType := do
  if let some r ← tryParseIntegerType then return NumberType.integerType r
  else if let some r ← tryParseFloatType then return NumberType.floatType r
  else throw <| ← error "Number type"

def parseComplexElementType : PState ComplexType := do
  if ← isParse "f32" then return ComplexType.f32
  else if ← isParse "f64" then return ComplexType.f64
  else throw <| ← error "Complex element type"

def parseComplexType : PState ComplexType := do
  parseItem "complex"
  parseItem "<"
  let t ← parseComplexElementType
  parseItem ">"
  return t

def tryParseDimensionSize : PState (Option DimensionSize) := do
  let mut r := none
  if (← isDigit) then
    r := some <| DimensionSize.known <| ← parseDecimal
  if (← isParse "?") then
    r := some <| DimensionSize.unknown
  return r

partial def parseShape : PState (List DimensionSize) := do
  if let some dim ← tryParseDimensionSize then
    parseItem "x"
    let dims ← parseShape
    return dim :: dims
  else
    return []

def parseTensorElementType : PState TensorElementType := do
  if let some r ← tryParseIntegerType then return TensorElementType.integerType r
  if ← isParse "i1" then return TensorElementType.booleanType
  if ← is "complex" then return TensorElementType.complexType <| ← parseComplexType
  if let some r ← tryParseFloatType then return TensorElementType.floatType r
  throw <| ← error "TensorElementType"

def parseQuantizationParameter : PState QuantizationParameter := do
  let quantizationScale ← parseFloatLiteral
  let mut quantizationZeroPoint := { sign := Sign.plus , decimal := 0 }
  if (← isParse ":") then
    quantizationZeroPoint ← parseIntegerLiteral
  let parseResult :=
    { quantizationScale := quantizationScale,
      quantizationZeroPoint := quantizationZeroPoint
    }
  return parseResult

def parseQuantizationParameters : PState (List QuantizationParameter) := do
  if ← is "{" then
    let quantizationParameters ← parseList "{" "}" "," parseQuantizationParameter
    return quantizationParameters
  else
    let quantizationParameter ← parseQuantizationParameter
    return [quantizationParameter]

def parseQuantizedTensorElementType : PState QuantizedTensorElementType := do
  parseItem "!quant.uniform"
  parseItem "<"
  let quantizationStorageType ← parseIntegerType
  let mut quantizationStorageMinMax := none
  if ← isParse "<" then
    let min ← parseIntegerLiteral
    parseItem ":"
    let max ← parseIntegerLiteral
    quantizationStorageMinMax := some (min,max)
    parseItem ">"
  parseItem ":"
  let quantizationExpressedType ← parseFloatType
  let mut quantizationDimension := none
  if ← isParse ":" then
    quantizationDimension := some (← parseIntegerLiteral)
  parseItem ","
  let quantizationParameters ← parseQuantizationParameters
  parseItem ">"
  let quantizationBasics : QuantizationBasics :=
    { quantizationStorageType := quantizationStorageType,
      quantizationStorageMinMax := quantizationStorageMinMax,
      quantizationExpressedType := quantizationExpressedType,
      quantizationDimension := quantizationDimension
    }
  let parseResult : QuantizedTensorElementType :=
    { quantizationBasics := quantizationBasics
      quantizationParameters := quantizationParameters
    }
  return parseResult

def parseTensorElementTypeGen : PState TensorElementTypeGen := do
  if ← is "!quant.uniform"
  then
    let quantizedTensorElementType ← parseQuantizedTensorElementType
    return TensorElementTypeGen.quantized quantizedTensorElementType
  else
    let tensorElementType ← parseTensorElementType
    return TensorElementTypeGen.classic tensorElementType

def parseTensorType : PState TensorType := do
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorElementTypeGen ← parseTensorElementTypeGen
  parseItem ">"
  return { shape := shape, tensorElementTypeGen := tensorElementTypeGen }

def parseTokenType : PState ValueType := do
  parseItem "!stablehlo.token"
  return ValueType.tokenType

mutual

partial def parseTupleType : PState ValueType := do
  parseItem "tuple"
  let TupleElementTypes ← parseList "<" ">" "," parseValueType
  return ValueType.tupleType TupleElementTypes

partial def parseValueType : PState ValueType := do
  if ← is "tensor" then return ValueType.tensorType <| ← parseTensorType
  else if ← is "tuple" then
    let r ← parseTupleType
    return r
  else if ← is "!stablehlo.token" then
    let r ← parseTokenType
    return r
  else throw <| ← error "Value Type"

end

def parseValueTypesOutput : PState (List ValueType) := do
  let mut valueTypes : List ValueType := []
  if ← is "(" then
    valueTypes ← parseList "(" ")" "," parseValueType
  else
    let r ← parseValueType
    valueTypes := [r]
  return valueTypes

def parseValueTypes : PState (List ValueType) := do
  let r ← parseList "(" ")" "," parseValueType
  return r

def parseFunctionType : PState FunctionType := do
  let inputTypes ← parseValueTypes
  parseItem "-"
  parseItem ">"
  let outputType ← parseValueTypesOutput
  return { domain := inputTypes, range := outputType }

def parseStringType : PState NonValueType := do
  parseItem "string"
  return NonValueType.stringType

def parseType : PState SType := do
  if (← is "tensor") || (← is "tuple") || (← is "!stablehlo.token") then
    return SType.valueType <|  ← parseValueType
  return SType.nonValueType <| NonValueType.tensorElementType <| ← parseTensorElementType

end StableHLO.Parsing
