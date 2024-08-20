/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/

/-!
# Types

-/

namespace StableHLO

inductive Signedness where
  | signed
  | unsigned

inductive IntegerSize where
  | b2
  | b4
  | b8
  | b16
  | b32
  | b64

structure IntegerType where
  sign : Signedness
  size : IntegerSize

inductive FloatType where
  | f8E4M3FN
  | f8E5M2
  | f8E4M3FNUZ
  | f8E5M2FNUZ
  | f8E4M3B11FNUZ
  | bf16
  | f16
  | f32
  | f64
  | tf32

inductive ComplexType where
  | f32
  | f64

inductive TensorElementType where
  | booleanType
  | integerType (t : IntegerType)
  | floatType (t: FloatType)
  | complexType (t: ComplexType)

inductive QuantizedTensorElementType where
  | quant : Signedness → IntegerSize → Int → Int → FloatSize → Int → List (Float × Int) → QuantizedTensorElementType

inductive ValueType where
  | tensorType (shape : List Int) (typ : TensorElementType)
  | quantizedTensorType (shape : List Int) (typ : QuantizedTensorElementType)
  | tokenType
  | tupleType (elements : List Valuetype)

inductive StringType where

structure FunctionType where
  domain : List ValueType
  range : List ValueType

inductive NonValueType where
  | tensorElementType (t : TensorElementType)
  | quantizedTensorElementType (t: QuantizedTensorElementType)
  | functionType (t : FunctionType)
  | stringType (t : StringType)

inductive SType where
  | valueType (t : ValueType)
  | nonValueType (t : NonValueType)

end StableHLO
