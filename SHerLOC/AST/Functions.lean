/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Types
import SHerLOC.AST.Identifiers
import SHerLOC.AST.Operations

/-!
# Functions

-/

namespace StableHLO

structure FuncOutput where
  typ : ValueType
  attrs : List Attribute
  deriving Repr, Inhabited, Nonempty

structure Function where
  funcId : FuncId
  funcInputs : List FuncInput
  funcOutputs : List FuncOutput
  funcBody : List Operation
  deriving Repr, Inhabited, Nonempty

end StableHLO
