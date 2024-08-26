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
  deriving Repr, Inhabited, Nonempty

inductive IntegerSize where
  | b2
  | b4
  | b8
  | b16
  | b32
  | b64
  deriving Repr, Inhabited, Nonempty

structure IntegerType where
  sign : Signedness
  size : IntegerSize
  deriving Repr, Inhabited, Nonempty

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
  deriving Repr, Inhabited, Nonempty

inductive ComplexType where
  | f32
  | f64
  deriving Repr, Inhabited, Nonempty

inductive TensorElementType where
  | booleanType
  | integerType (t : IntegerType)
  | floatType (t: FloatType)
  | complexType (t: ComplexType)
  deriving Repr, Inhabited, Nonempty

structure QuantizationParameter where
  quantizationScale : Float
  quantizationZeroPoint: Int
  deriving Repr, Inhabited, Nonempty

structure QuantizedTensorElementType where
  quantizationStorageType : IntegerType
  quantizationStorageMinMax : Option (Int × Int)
  quantizationExpressedType : FloatType
  quantizationDimension : Option Int
  quantizationParameters : List QuantizationParameter
  deriving Repr, Inhabited, Nonempty

structure TensorType where
  shape : List Nat
  tensorElementType : TensorElementType
  deriving Repr, Inhabited, Nonempty

structure QuantizedTensorType where
  shape : List Nat
  quantizedTensorElementType : QuantizedTensorElementType
  deriving Repr, Inhabited, Nonempty

inductive ValueType where
  | tensorType (tensor : TensorType)
  | quantizedTensorType (typ : QuantizedTensorType)
  | tokenType
  | tupleType (elements : List ValueType)
  deriving Repr, Inhabited, Nonempty

structure FunctionType where
  domain : List ValueType
  range : List ValueType
  deriving Repr, Inhabited, Nonempty

inductive NonValueType where
  | tensorElementType (t : TensorElementType)
  | quantizedTensorElementType (t: QuantizedTensorElementType)
  | functionType (t : FunctionType)
  | stringType
  deriving Repr, Inhabited, Nonempty

inductive SType where
  | valueType (t : ValueType)
  | nonValueType (t : NonValueType)
  deriving Repr, Inhabited, Nonempty

end StableHLO
