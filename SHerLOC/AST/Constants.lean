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

structure Constant where
  literal : Literal
  typ : Option SType
  deriving Repr, Inhabited, Nonempty

-- structure IntegerConstant where
--   literal : IntegerLiteral
--   type : IntegerType
--   deriving Repr, Inhabited, Nonempty

-- structure FloatConstant where
--   literal : FloatLiteral
--   type : FloatType
--   deriving Repr, Inhabited, Nonempty

-- structure NumberConstant where
--   literal : FloatLiteral
--   type : NumberType
--   deriving Repr, Inhabited, Nonempty

-- structure ComplexConstant where
--   literal : ComplexLiteral
--   type : ComplexType
--   deriving Repr, Inhabited, Nonempty

-- structure TensorConstant where
--   literal : TensorLiteral
--   type : TensorType
--   deriving Repr, Inhabited, Nonempty

-- inductive Constant where
--   | booleanConstant (literal : BooleanLiteral)
--   | integerConstant (constant : IntegerConstant)
--   | floatConstant (constant : FloatConstant)
--   | numberConsant (constant : NumberConstant)
--   | complexConstant (constant : ComplexConstant)
--   | tensorConstant (constant : TensorConstant)
--   | stringConstant (literal : String)
--   | enumConstant (literal : EnumLiteral)
--   | special -- For complicated dictionnary properties we do not parse yet
--   deriving Repr, Inhabited, Nonempty

end StableHLO
