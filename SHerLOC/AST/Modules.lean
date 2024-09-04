/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Functions

/-!
# Modules

-/

namespace StableHLO

structure Module where
  modId : Option FuncId
  modAttrs : List Attribute
  modFuncs : List Function
  deriving Repr, Inhabited, Nonempty

end StableHLO
