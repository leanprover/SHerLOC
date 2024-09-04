/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Operations
import SHerLOC.Parsing.Functions
import SHerLOC.Parsing.Intermediate

namespace StableHLO

def parseModule : PState Module := do
  push "parseModule"
  parseItem "\"builtin.module\""
  let valueUseList ← parseValueUseList
  let dictionaryProperties ← parseDictionaryProperties
  let region ← parseRegion parseFunction
  let attributes ← parseAttributes
  parseItem ":"
  let functiontype ← parseFunctionType
  let r : Module := { modId := "", modAttrs := attributes, modFuncs := region }
  pop "parseModule"
  return r

end StableHLO
