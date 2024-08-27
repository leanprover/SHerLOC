/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Numbers
import SHerLOC.AST.Types

/-!
# Constants

-/

namespace StableHLO

inductive BooleanLiteral where
  | true
  | false
  deriving Repr, Inhabited, Nonempty

structure ComplexLiteral where
  real : FloatLiteral
  imaginary : FloatLiteral
  deriving Repr, Inhabited, Nonempty

structure ComplexConstant where
  literal : ComplexLiteral
  type : ComplexType
  deriving Repr, Inhabited, Nonempty

-- Diverging from spec
-- I am going to defer until type checking to decide
-- whether the literal matches its type
-- This is because otherwise I have to lookahead arbitrarily far
-- to figure out the details of the tensor type
inductive ElementLiteral where
  | booleanLiteral (literal : BooleanLiteral)
  --| integerLiteral (literal : IntegerLiteral)
  | floatLiteral (literal : FloatLiteral)
  | complexLiteral (literal : ComplexLiteral)
  deriving Repr, Inhabited, Nonempty

abbrev TensorLiteral := List ElementLiteral

-- inductive TensorLiteral where
--   | element (elt : ElementLiteral)
--   | dimensions (dims : List TensorLiteral)
--   deriving Repr, Inhabited, Nonempty

structure TensorConstant where
  literal : TensorLiteral
  type : TensorType
  deriving Repr, Inhabited, Nonempty

-- inductive QuantizedTensorLiteral where
--   | element (elt : ElementLiteral)
--   | dimensions (dims : List QuantizedTensorLiteral)
--   deriving Repr, Inhabited, Nonempty

inductive ComparisonDirection where
  | eq
  | ne
  | ge
  | gt
  | le
  | lt
  deriving Repr, Inhabited, Nonempty

inductive CompareType where
  | float
  | totalOrder
  | signed
  | unsigned
  deriving Repr, Inhabited, Nonempty

inductive PrecisionConfig where
  | default
  | high
  | highest
  deriving Repr, Inhabited, Nonempty

inductive FftType where
  | fft
  | ifft
  | rfft
  | irfft
  deriving Repr, Inhabited, Nonempty

inductive ChannelType where
  | deviceToDevice
  | hostToDevice
  deriving Repr, Inhabited, Nonempty

inductive RngDistribution where
  | uniform
  | normal
  deriving Repr, Inhabited, Nonempty

inductive RngAlgorithm where
  | default
  | threeFry
  | philox
  deriving Repr, Inhabited, Nonempty

inductive TransposeA where
  | noTranspose
  | transpose
  | adjoint
  deriving Repr, Inhabited, Nonempty

inductive EnumLiteral where
  | comparisonDirection (enum : ComparisonDirection)
  | compareType (enum : CompareType)
  | precisionConfig (enum : PrecisionConfig)
  | fftType (enum : FftType)
  | channelType (enum : ChannelType)
  | rngDistribution (enum : RngDistribution)
  | rngAlgorithm (enum : RngAlgorithm)
  | transposeA (enum : TransposeA)
  deriving Repr, Inhabited, Nonempty

-- Deviation. There's not difference between classic and quantized tensor
-- literals, and I gave them a common type in the hierarchy
inductive Constant where
  | booleanConstant (literal : BooleanLiteral)
  | integerConstant (constant : IntegerConstant)
  | floatConstant (constant : FloatConstant)
  | numberConsant (constant : NumberConstant)
  | complexConstant (constant : ComplexConstant)
  | tensorConstant (constant : TensorConstant)
--  | quantizedTensorConstant (literal : QuantizedTensorLiteral) (type : QuantizedTensorType)
  | stringConstant (literal : String)
  | enumConstant (literal : EnumLiteral)
  deriving Repr, Inhabited, Nonempty

end StableHLO
