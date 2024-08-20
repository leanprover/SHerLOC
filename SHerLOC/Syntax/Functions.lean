/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.Syntax.Types
import SHerLOC.Syntax.Identifiers
import SHerLOC.Syntax.Operations

/-!
# Functions

-/

namespace StableHLO

structure Function where
  funcId : FuncId
  funcInputs : List (ValueId × ValueType)
  funcOutputs : List ValueType
  funcBody : List Operation

end StableHLO
