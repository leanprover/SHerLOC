/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.Types

/-!
# Constants

-/

inductive Constant where
  | BooleanConstant (literal : Bool)
  | IntegerConstant (literal : Int) (type : IntegerType)
  | FloatConstant (literal : Float) (type : FloatType)
  | ComplexConstant (real imaginary : ComplexType)
  | TensorConstant
  | QuantizedTensorConstant
  | StringConstant (literal : String)
  | EnumConstant
