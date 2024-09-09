/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Modules

/-!
# Programs

-/

namespace StableHLO

def Program := List Module
  deriving Repr, Inhabited, Nonempty

end StableHLO
