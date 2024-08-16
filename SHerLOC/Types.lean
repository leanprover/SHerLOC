/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/

/-!
# Types

This file describes the type algegra of StableHLO.

-/

inductive Signedness where
  | Signed
  | Unsigned

inductive IntegerSize where
  | b2
  | b4
  | b8
  | b16
  | b32
  | b64

inductive FloatSize where
  | f8E4M3FN
  | f8E5M2
  | f8E4M3FNUZ
  | f8E5M2FNUZ
  | f8E4M3B11FNUZ
  | bf16
  | f16
  | f32
  | f64

inductive ComplexSize where
  | f32
  | f64

inductive TensorElementType where
  | BooleanType : TensorElementType
  | IntegerType : Signedness → IntegerSize → TensorElementType
  | FloatType : FloatSize → TensorElementType
  | ComplexType : ComplexSize → TensorElementType

inductive QuantizedTensorElementType where
  | Quant : Signedness → IntegerSize → Int → Int → FloatSize → Int → List (Float × Int) → QuantizedTensorElementType

inductive ValueType where
  | TensorType : List Int → TensorElementType → ValueType
  | QuantizedTensorType : List Int → QuantizedTensorElementType → ValueType
  | TokenType
  | TupleType : List Valuetype → ValueType

inductive TensorFloar32 where

inductive StringType where

inductive FunctionType where
  | FunctionType : List ValueType → List ValueType → FunctionType
