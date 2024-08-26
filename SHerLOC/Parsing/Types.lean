/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def parseIntegerType : PState IntegerType := do
  let st ← get
  match st.tok with
  | "si2" => shift ; return { sign := Signedness.signed , size := IntegerSize.b2 }
  | "si4" => shift ; return { sign := Signedness.signed , size := IntegerSize.b4 }
  | "si8" => shift ; return { sign := Signedness.signed , size := IntegerSize.b8 }
  | "si16" => shift ; return { sign := Signedness.signed , size := IntegerSize.b16 }
  | "si32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | "si64" => shift ; return { sign := Signedness.signed , size := IntegerSize.b64 }
  | "ui2" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b2 }
  | "ui4" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b4 }
  | "ui8" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b8 }
  | "ui16" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b16 }
  | "ui32" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b32 }
  | "ui64" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b64 }
  -- Jax compatibility
  | "i32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | _ => throw <| st.error "Integer type"

def parseFloatType : PState FloatType := do
  let st ← get
  match st.tok with
  | "f8E4M3FN" => shift ; return FloatType.f8E4M3FN
  | "f8E5M2" => shift ; return FloatType.f8E5M2
  | "f8E4M3FNUZ" => shift ; return FloatType.f8E4M3FNUZ
  | "f8E5M2FNUZ" => shift ; return FloatType.f8E5M2FNUZ
  | "f8E4M3B11FNUZ" => shift ; return FloatType.f8E4M3B11FNUZ
  | "bf16" => shift ; return FloatType.bf16
  | "f16" => shift ; return FloatType.f16
  | "f32" => shift ; return FloatType.f32
  | "f64" => shift ; return FloatType.f64
  | _ => throw <| st.error "Float type"

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

def parseTensorElementType : PState TensorElementType := do
  let st ← get
  if st.is "i1" then return TensorElementType.booleanType
  let c := st.tok.get! ⟨ 0 ⟩
  if c = 's' || c = 'u' || c = 'i' then return TensorElementType.integerType <| ← parseIntegerType
  if c = 'f' || c = 'b' then return TensorElementType.floatType <| ← parseFloatType
  if st.is "complex" then return TensorElementType.complexType <| ← parseComplexType
  else throw <| st.error "TensorElementType"

partial def parseShape : PState (List Nat) := do
  let st ← get
  if (st.lookahead 1).get! ⟨ 0 ⟩ = 'x'
  then
    let i ← parseDecimal
    let shape ← parseShape
    return i :: shape
  else return []

def parseTensorType : PState ValueType := do
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorElementType ← parseTensorElementType
  parseItem ">"
  return ValueType.tensorType { shape := shape , tensorElementType := tensorElementType}

def parseQuantizationStorageType : PState IntegerType := do
  parseIntegerType

-- makes parsing of types and constants mutually recursive
def parseQuantizationStorageMinMax : PState (Int × Int) := do sorry

def parseQuantizationExpressedType : PState FloatType := do
  parseFloatType

-- makes parsing of types and constants mutually recursive
def parseQuantizationDimension : PState Int := do sorry

-- makes parsing of types and constants mutually recursive
def parseQuantizationScale : PState Float := do sorry

-- makes parsing of types and constants mutually recursive
def parseQuantizationZeroPoint : PState Int := do sorry

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
  if st.is "<" then
    quantizationStorageMinMax := some <| ← parseQuantizationStorageMinMax
  parseItem ":"
  let quantizationExpressedType ← parseQuantizationExpressedType
  let mut quantizationDimension := none
  if st.is ":" then
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

def parseQuantizedTensorType : PState ValueType := do
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorQuantizedTensorElementType ← parseQuantizedTensorElementType
  parseItem ">"
  return ValueType.quantizedTensorType { shape := shape , quantizedTensorElementType := tensorQuantizedTensorElementType}

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
  | "tensor" =>
    if let some idx := st.search ">" then
      (match st.lookahead (idx + 1) with
      | "!quant.uniform" => parseQuantizedTensorType
      | _ => parseTensorType
      )
    else throw <| st.error "Value Type: coult not disambiguate between tensor and quantized tensore"
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
