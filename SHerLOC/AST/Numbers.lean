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

inductive Sign where
  | plus
  | minus
  deriving Repr, Inhabited, Nonempty

structure IntegerLiteral where
  sign : Sign
  decimal : Nat
  deriving Repr, Inhabited, Nonempty

structure IntegerConstant where
  literal : IntegerLiteral
  type : IntegerType
  deriving Repr, Inhabited, Nonempty

structure FloatLiteral where
  integerPart : IntegerLiteral
  fractionalPart : IntegerLiteral
  scientificPart : IntegerLiteral
  deriving Repr, Inhabited, Nonempty

structure FloatConstant where
  literal : FloatLiteral
  type : FloatType
  deriving Repr, Inhabited, Nonempty

end StableHLO
