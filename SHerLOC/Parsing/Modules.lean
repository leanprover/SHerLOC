/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Operations
import SHerLOC.Parsing.Functions

namespace StableHLO

def parseModule : PState Module := do
  push "parseModule"
  parseItem "module"
  parseItem "@"
  let modId ← parseId
  parseItem "attributes"
  let modAttrs ← parseAttributes
  let modFuncs ← parseFunctions
  let r : Module := { modId := modId, modAttrs, modFuncs }
  pop "parseModule"
  return r

end StableHLO
