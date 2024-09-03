/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Numbers

/-!
# Types

-/

namespace StableHLO

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
  quantizationScale : FloatConstant
  quantizationZeroPoint: IntegerConstant
  deriving Repr, Inhabited, Nonempty

structure QuantizedTensorElementType where
  quantizationStorageType : IntegerType
  quantizationStorageMinMax : Option (IntegerConstant × IntegerConstant)
  quantizationExpressedType : FloatType
  quantizationDimension : Option IntegerConstant
  quantizationParameters : List QuantizationParameter
  deriving Repr, Inhabited, Nonempty

-- Here I deviate from the spec because it allows me to keep
-- the grammer LL(1)
inductive TensorElementTypeGen where
  | classic (t : TensorElementType)
  | quantized (t : QuantizedTensorElementType)
  deriving Repr, Inhabited, Nonempty

structure TensorType where
  shape : List Nat
  tensorElementTypeGen : TensorElementTypeGen
  deriving Repr, Inhabited, Nonempty

inductive ValueType where
  | tensorType (tensor : TensorType)
  | tokenType
  | tupleType (elements : List ValueType)
  deriving Repr, Inhabited, Nonempty

inductive FunctionType where
  | short (range : List ValueType)
  | long (domain range : List ValueType)
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
