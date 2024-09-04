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
  parseItems ["\"builtin.module\"", "(", ")"]
  parseItem "<{"
  parseItem "sym_name"
  parseItem "="
  let name ← parseString
  parseItem "}>"
  parseItem "("
  let region ← parseFunctions
  parseItem ")"
  let attributes ← parseAttributes
  parseItems [":","(",")","->","(",")"]
  let r : Module := { modId := name, modAttrs := attributes, modFuncs := region }
  pop "parseModule"
  return r

end StableHLO
