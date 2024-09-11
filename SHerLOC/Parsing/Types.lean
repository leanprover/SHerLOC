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
  push "tryParseIntegerType"
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
  pop "tryParseIntegerType"
  return r

def parseIntegerType : PState IntegerType := do
  push "parseIntegerType"
  if let some r ← tryParseIntegerType then pop "parseIntegerType" ; return r
  else throw <| ← error "Integer type"

def tryParseFloatType : PState (Option FloatType) := do
  push "tryParseFloatType"
  let mut r : Option FloatType := none
  if ← isChar 'f' then {
    if ← isParse "f16" then r := some FloatType.f16
    if ← isParse "f32" then r := some FloatType.f32
    if ← isParse "f64" then r := some FloatType.f64
    if ← isParse "f8E4M3B11FNUZ" then r := some FloatType.f8E4M3B11FNUZ
    if ← isParse "f8E4M3FNUZ" then r := some FloatType.f8E4M3FNUZ
    if ← isParse "f8E4M3FN" then r := some FloatType.f8E4M3FN
    if ← isParse "f8E5M2FNUZ" then r := some FloatType.f8E5M2FNUZ
    if ← isParse "f8E5M2" then r := some FloatType.f8E5M2
  }
  if ← isParse "bf16" then r := some FloatType.bf16
  pop "tryParseFloatType"
  return r

def parseFloatType : PState FloatType := do
  push "parseFloatType"
  if let some r ← tryParseFloatType then pop "parseFloatType"; return r
  else throw <| ← error "Float type"

def parseNumberType : PState NumberType := do
  push "parseNumberType"
  if let some r ← tryParseIntegerType then pop "parseNumberType"; return NumberType.integerType r
  else if let some r ← tryParseFloatType then pop "parseNumberType"; return NumberType.floatType r
  else throw <| ← error "Number type"

def parseComplexElementType : PState ComplexType := do
  push "parseComplexElementType"
  if ← isParse "f32" then pop "parseComplexElementType"; return ComplexType.f32
  else if ← isParse "f64" then pop "parseComplexElementType"; return ComplexType.f64
  else throw <| ← error "Complex element type"

def parseComplexType : PState ComplexType := do
  push "parseComplexType"
  parseItem "complex"
  parseItem "<"
  let t ← parseComplexElementType
  parseItem ">"
  pop "parseComplexType"
  return t

partial def parseShape : PState (List Nat) := do
  push "parseShape"
  if ! (← isDigit) then pop "parseShape"; return []
  else
    let dim ← parseDecimal
    parseItem "x"
    let dims ← parseShape
    pop "parseShape"
    return dim :: dims

def parseTensorElementType : PState TensorElementType := do
  push "parseTensorElementType"
  if let some r ← tryParseIntegerType then pop "parseTensorElementType"; return TensorElementType.integerType r
  if ← isParse "i1" then pop "parseTensorElementType"; return TensorElementType.booleanType
  if ← is "complex" then pop "parseTensorElementType"; return TensorElementType.complexType <| ← parseComplexType
  if let some r ← tryParseFloatType then pop "parseTensorElementType"; return TensorElementType.floatType r
  throw <| ← error "TensorElementType"

def parseQuantizationParameter : PState QuantizationParameter := do
  push "parseQuantizationParameter"
  let quantizationScale ← parseFloatLiteral
  parseItem ":"
  let quantizationZeroPoint ← parseIntegerLiteral
  let parseResult :=
    { quantizationScale := quantizationScale,
      quantizationZeroPoint := quantizationZeroPoint
    }
  pop "parseQuantizationParameter"
  return parseResult

def parseQuantizationParameters : PState (List QuantizationParameter) := do
  push "parseQuantizationParameters"
  if ← is "{" then
    let quantizationParameters ← parseList "{" "}" "," parseQuantizationParameter
    pop "parseQuantizationParameters"
    return quantizationParameters
  else
    let quantizationParameter ← parseQuantizationParameter
    pop "parseQuantizationParameters"
    return [quantizationParameter]

def parseQuantizedTensorElementType : PState QuantizedTensorElementType := do
  push "parseQuantizedTensorElementType"
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
  pop "parseQuantizedTensorElementType"
  return parseResult

def parseTensorElementTypeGen : PState TensorElementTypeGen := do
  push "parseTensorElementTypeGen"
  if ← is "!quant.uniform"
  then
    let quantizedTensorElementType ← parseQuantizedTensorElementType
    pop "parseTensorElementTypeGen"
    return TensorElementTypeGen.quantized quantizedTensorElementType
  else
    let tensorElementType ← parseTensorElementType
    pop "parseTensorElementTypeGen"
    return TensorElementTypeGen.classic tensorElementType

def parseTensorType : PState TensorType := do
  push "parseTensorType"
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorElementTypeGen ← parseTensorElementTypeGen
  parseItem ">"
  pop "parseTensorType"
  return { shape := shape, tensorElementTypeGen := tensorElementTypeGen }

def parseTokenType : PState ValueType := do
  push "parseTokenType"
  parseItem "!stablehlo.token"
  pop "parseTokenType"
  return ValueType.tokenType

mutual

partial def parseTupleType : PState ValueType := do
  push "parseTupleType"
  parseItem "tuple"
  let TupleElementTypes ← parseList "<" ">" "," parseValueType
  pop "parseTupleType"
  return ValueType.tupleType TupleElementTypes

partial def parseValueType : PState ValueType := do
  push "parseValueType"
  if ← is "tensor" then pop "parseValueType"; return ValueType.tensorType <| ← parseTensorType
  else if ← is "tuple" then
    let r ← parseTupleType
    pop "parseValueType"
    return r
  else if ← is "!stablehlo.token" then
    let r ← parseTokenType
    pop "parseValueType"
    return r
  else throw <| ← error "Value Type"

end

-- Temporary? Mulitple results?
def parseValueTypesOutput : PState (List ValueType) := do
  push "parseValueTypesOutput"
  let mut valueTypes : List ValueType := []
  if ← is "(" then
    valueTypes ← parseList "(" ")" "," parseValueType
  else
    let r ← parseValueType
    valueTypes := [r]
  pop "parseValueTypesOutput"
  return valueTypes

def parseValueTypes : PState (List ValueType) := do
  push "parseValueTypes"
  let r ← parseList "(" ")" "," parseValueType
  pop "parseValueTypes"
  return r

def parseFunctionType : PState FunctionType := do
  push "parseFunctionTypeLong"
  let inputTypes ← parseValueTypes
  parseItem "-"
  parseItem ">"
  let outputType ← parseValueTypesOutput
  pop "parseFunctionTypeLong"
  return { domain := inputTypes, range := outputType }

def parseStringType : PState NonValueType := do
  push "parseStringType"
  parseItem "string"
  pop "parseStringType"
  return NonValueType.stringType

def parseType : PState SType := do
  if (← is "tensor") || (← is "tuple") || (← is "!stablehlo.token") then
    return SType.valueType <|  ← parseValueType
  return SType.nonValueType <| NonValueType.tensorElementType <| ← parseTensorElementType

end StableHLO.Parsing
