/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/

/-!
# Types

-/

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

inductive IntegerType where
  | integerType (sign : Signedness) (size : IntegerSize)

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
  | tensorType : List Int → TensorElementType → ValueType
  | quantizedTensorType : List Int → QuantizedTensorElementType → ValueType
  | tokenType
  | tupleType : List Valuetype → ValueType

inductive TensorFloar32 where

inductive StringType where

inductive FunctionType where
  | functionType : List ValueType → List ValueType → FunctionType
