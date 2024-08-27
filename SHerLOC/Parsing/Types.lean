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
  let st ← get
  match st.tok with
  | "f32" => shift ; return ComplexType.f32
  | "f64" => shift ; return ComplexType.f64
  | _ => throw <| st.error "Complex element type"

def parseComplexType : PState ComplexType := do
  parseItem "complex"
  parseItem "<"
  let t ← parseComplexElementType
  parseItem ">"
  return t

partial def parseShape : PState (List Nat) := do
  let st ← get
  if (st.lookahead 1).get! ⟨ 0 ⟩ = 'x'
  then
    let i ← parseDecimal
    let shape ← parseShape
    return i :: shape
  else return []

def parseTensorElementType : PState TensorElementType := do
  let st ← get
  if st.is "i1" then shift ; return TensorElementType.booleanType
  let c := st.tok.get! ⟨ 0 ⟩
  if c = 's' || c = 'u' || c = 'i' then return TensorElementType.integerType <| ← parseIntegerType
  if c = 'f' || c = 'b' then return TensorElementType.floatType <| ← parseFloatType
  if st.is "complex" then return TensorElementType.complexType <| ← parseComplexType
  else throw <| st.error "TensorElementType"

def parseQuantizationStorageType : PState IntegerType := do
  parseIntegerType

def parseQuantizationStorageMinMax : PState (IntegerConstant × IntegerConstant) := do
  let min ← parseIntegerConstant
  let max ← parseIntegerConstant
  return (min,max)

def parseQuantizationExpressedType : PState FloatType := do
  parseFloatType

def parseQuantizationDimension : PState IntegerConstant := do
  parseIntegerConstant

def parseQuantizationScale : PState FloatConstant := do
  parseFloatConstant

def parseQuantizationZeroPoint : PState IntegerConstant := do
  parseIntegerConstant

def parseQuantizationParameter : PState QuantizationParameter := do
  let quantizationScale ← parseQuantizationScale
  parseItem ":"
  let quantizationZeroPoint ← parseQuantizationZeroPoint
  let parseResult :=
    { quantizationScale := quantizationScale,
      quantizationZeroPoint := quantizationZeroPoint
    }
  return parseResult

def parseQuantizationParameters : PState (List QuantizationParameter) := do
  let st ← get
  if st.is "{" then
    let quantizationParameters ← parseList "{" "}" (some ",") parseQuantizationParameter
    return quantizationParameters
  else
    let quantizationParameter ← parseQuantizationParameter
    return [quantizationParameter]

def parseQuantizedTensorElementType : PState QuantizedTensorElementType := do
  let st ← get
  parseItem "!quant.uniform"
  parseItem "<"
  let quantizationStorageType ← parseQuantizationStorageType
  let mut quantizationStorageMinMax := none
  let st₁ ← get
  if st₁.is "<" then
    quantizationStorageMinMax := some <| ← parseQuantizationStorageMinMax
  parseItem ":"
  let quantizationExpressedType ← parseQuantizationExpressedType
  let mut quantizationDimension := none
  let st₂ ← get
  if st₂.is ":" then
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
  return parseResult

def parseTensorElementTypeGen : PState TensorElementTypeGen := do
  let st ← get
  if st.tok = "!quant.uniform"
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
  parseItem "token"
  return ValueType.tokenType

mutual

partial def parseTupleType : PState ValueType := do
  parseItem "tuple"
  let TupleElementTypes ← parseList "<" ">" (some ",") parseValueType
  return ValueType.tupleType TupleElementTypes

partial def parseValueType : PState ValueType := do
  let st ← get
  match st.tok with
  | "tensor" => return ValueType.tensorType <| ← parseTensorType
  | "tuple" => parseTupleType
  | "token" => parseTokenType
  | _ => throw <| st.error "Value Type"

end

def parseValueTypes : PState (List ValueType) := do
  parseList "(" ")" (some ",") parseValueType

def parseStringType : PState NonValueType := do
  parseItem "string"
  return NonValueType.stringType

def parseFunctionType : PState FunctionType := do
  let inputTypes ← parseValueTypes
  parseItem "-"
  parseItem ">"
  let outputType ← parseValueTypes
  let functionType := { domain := inputTypes , range := outputType }
  return functionType

end StableHLO
