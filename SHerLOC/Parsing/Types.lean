/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Numbers

namespace StableHLO

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

def parseQuantizationStorageType : PState IntegerType := do
  push "parseQuantizationStorageType"
  let r ← parseIntegerType
  pop "parseQuantizationStorageType"
  return r

def parseQuantizationStorageMinMax : PState (IntegerConstant × IntegerConstant) := do
  push "parseQuantizationStorageMinMax"
  let min ← parseIntegerConstant
  let max ← parseIntegerConstant
  pop "parseQuantizationStorageMinMax"
  return (min,max)

def parseQuantizationExpressedType : PState FloatType := do
  push "parseQuantizationExpressedType"
  let r ← parseFloatType
  pop "parseQuantizationExpressedType"
  return r

def parseQuantizationDimension : PState IntegerConstant := do
  push "parseQuantizationDimension"
  let r ← parseIntegerConstant
  pop "parseQuantizationDimension"
  return r

def parseQuantizationScale : PState FloatConstant := do
  push "parseQuantizationScale"
  let r ← parseFloatConstant
  pop "parseQuantizationScale"
  return r

def parseQuantizationZeroPoint : PState IntegerConstant := do
  push "parseQuantizationZeroPoint"
  let r ← parseIntegerConstant
  pop "parseQuantizationZeroPoint"
  return r

def parseQuantizationParameter : PState QuantizationParameter := do
  push "parseQuantizationParameter"
  let quantizationScale ← parseQuantizationScale
  parseItem ":"
  let quantizationZeroPoint ← parseQuantizationZeroPoint
  let parseResult :=
    { quantizationScale := quantizationScale,
      quantizationZeroPoint := quantizationZeroPoint
    }
  pop "parseQuantizationParameter"
  return parseResult

def parseQuantizationParameters : PState (List QuantizationParameter) := do
  push "parseQuantizationParameters"
  if ← is "{" then
    let quantizationParameters ← parseList "{" "}" (some ",") parseQuantizationParameter
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
  let quantizationStorageType ← parseQuantizationStorageType
  let mut quantizationStorageMinMax := none
  if ← is "<" then
    quantizationStorageMinMax := some <| ← parseQuantizationStorageMinMax
  parseItem ":"
  let quantizationExpressedType ← parseQuantizationExpressedType
  let mut quantizationDimension := none
  if ← is ":" then
    quantizationDimension := some <| ← parseQuantizationDimension
  parseItem ","
  let quantizationParameters ← parseQuantizationParameters
  parseItem ">"
  let parseResult : QuantizedTensorElementType :=
    { quantizationStorageType := quantizationStorageType,
      quantizationStorageMinMax := quantizationStorageMinMax,
      quantizationExpressedType := quantizationExpressedType,
      quantizationDimension := quantizationDimension,
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
  parseItem "token"
  pop "parseTokenType"
  return ValueType.tokenType

mutual

partial def parseTupleType : PState ValueType := do
  push "parseTupleType"
  parseItem "tuple"
  let TupleElementTypes ← parseList "<" ">" (some ",") parseValueType
  pop "parseTupleType"
  return ValueType.tupleType TupleElementTypes

partial def parseValueType : PState ValueType := do
  push "parseValueType"
  if ← is "tensor" then pop "parseValueType"; return ValueType.tensorType <| ← parseTensorType
  else if ← is "tuple" then
    let r ← parseTupleType
    pop "parseValueType"
    return r
  else if ← is "token" then
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
  let r ← parseList "(" ")" (some ",") parseValueType
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

end StableHLO
