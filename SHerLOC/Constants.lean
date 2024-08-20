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
  | booleanConstant (literal : Bool)
  | integerConstant (literal : Int) (type : IntegerType)
  | floatConstant (literal : Float) (type : FloatType)
  | complexConstant (real imaginary : ComplexType)
  | tensorConstant
  | quantizedTensorConstant
  | stringConstant (literal : String)
  | enumConstant
