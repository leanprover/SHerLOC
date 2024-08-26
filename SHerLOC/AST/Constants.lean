/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Types

/-!
# Constants

-/

namespace StableHLO

inductive BooleanLiteral where
  | true
  | false
  deriving Repr, Inhabited, Nonempty

inductive Sign where
  | plus
  | minus
  deriving Repr, Inhabited, Nonempty

structure IntegerLiteral where
  sign : Sign
  decimal : Nat
  deriving Repr, Inhabited, Nonempty

structure FloatLiteral where
  integerPart : IntegerLiteral
  fractionalPart : IntegerLiteral
  scientificPart : IntegerLiteral
  deriving Repr, Inhabited, Nonempty

structure ComplexLiteral where
  real : FloatLiteral
  imaginary : FloatLiteral
  deriving Repr, Inhabited, Nonempty

inductive ElementLiteral where
  | booleanLiteral (literal : BooleanLiteral)
  | integerLiteral (literal : IntegerLiteral)
  | floatLiteral (literal : FloatLiteral)
  | complexLiteral (literal : ComplexLiteral)
  deriving Repr, Inhabited, Nonempty

inductive TensorLiteral where
  | element (elt : ElementLiteral)
  | dimensions (dims : List TensorLiteral)
  deriving Repr, Inhabited, Nonempty

inductive QuantizedTensorLiteral where
  | element (elt : ElementLiteral)
  | dimensions (dims : List QuantizedTensorLiteral)
  deriving Repr, Inhabited, Nonempty

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

inductive RngAlgorith where
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
  | rngAlgorith (enum : RngAlgorith)
  | transposeA (enum : TransposeA)
  deriving Repr, Inhabited, Nonempty

inductive Constant where
  | booleanConstant (literal : Bool)
  | integerConstant (literal : Int) (type : IntegerType)
  | floatConstant (literal : FloatLiteral) (type : FloatType)
  | complexConstant (literal : ComplexLiteral) (type : ComplexType)
  | tensorConstant (literal : TensorLiteral) (type : TensorType)
  | quantizedTensorConstant (literal : QuantizedTensorLiteral) (type : QuantizedTensorType)
  | stringConstant (literal : String)
  | enumConstant (literal : EnumLiteral)
  deriving Repr, Inhabited, Nonempty

end StableHLO
