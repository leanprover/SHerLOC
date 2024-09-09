/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Constants
import SHerLOC.AST.Identifiers
import SHerLOC.AST.Types

/-!
# Operations

-/

namespace StableHLO

structure Attribute where
  id : AttrId
  constant : Constant
  deriving Repr, Inhabited, Nonempty

structure FuncInput where
  id : FuncId
  typ : ValueType
  deriving Repr, Inhabited, Nonempty

mutual

  inductive InputFunc where
    | mk
      (funcInputs : List FuncInput)
      (body : List Operation)
    deriving Repr, Inhabited, Nonempty

  inductive Operation where
    | stablehlo
      (name : String)
      (inputValues : List ValueId)
      (inputFunctions : List InputFunc)
      (inputAttributes : List Attribute)
      (outputs : List ValueId)
      (signature : FunctionType)
    | return
      (operands : List ValueId)
      (signature : FunctionType)
    | call
      (callee : FuncId)
      (inputValues : List ValueId)
      (outputs : List ValueId)
      (signature : FunctionType)
    deriving Repr, Inhabited, Nonempty

end

end StableHLO
