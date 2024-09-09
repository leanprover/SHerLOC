/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Types

namespace StableHLO

def parseConstant : PState Constant := do
  push "parseConstant"
  let literal ← parseLiteral
  let mut typ : Option SType := none
  if ← isParse ":" then
    typ ← parseType
  let r : Constant := { literal := literal, typ := typ }
  pop "parseConstant"
  return r

end StableHLO
