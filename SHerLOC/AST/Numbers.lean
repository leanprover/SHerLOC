/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/

/-!
# Numbers

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

inductive Sign where
  | plus
  | minus
  deriving Repr, Inhabited, Nonempty

structure IntegerLiteral where
  sign : Sign
  decimal : Nat
  deriving Repr, Inhabited, Nonempty

structure FloatLiteralDecimal where
  integerPart : IntegerLiteral
  fractionalPart : IntegerLiteral
  scientificPart : IntegerLiteral
  deriving Repr, Inhabited, Nonempty

inductive FloatLiteral where
  | decimal (literal : FloatLiteralDecimal)
  | hexaDecimal (literal : Nat)
  deriving Repr, Inhabited, Nonempty

inductive BooleanLiteral where
  | true
  | false
  deriving Repr, Inhabited, Nonempty

structure ComplexLiteral where
  real : FloatLiteral
  imaginary : FloatLiteral
  deriving Repr, Inhabited, Nonempty

inductive ElementLiteral where
  | booleanLiteral (literal : BooleanLiteral)
  | floatLiteral (literal : FloatLiteral)
  | complexLiteral (literal : ComplexLiteral)
  deriving Repr, Inhabited, Nonempty

inductive DenseLiteral where
  | denseDimension (literal : List DenseLiteral)
  | denseElements (literal : List ElementLiteral)
  deriving Repr, Inhabited, Nonempty

abbrev TensorLiteral := DenseLiteral

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

inductive Literal where
  | enum (literal : EnumLiteral)
  | element (literal : ElementLiteral)
  | tensor (literal : TensorLiteral)
  | string (literal : String)
  | use_global_device_ids
  | array (literal : List IntegerLiteral)
  | special
  deriving Repr, Inhabited, Nonempty

end StableHLO
