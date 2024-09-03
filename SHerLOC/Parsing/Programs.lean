/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Modules

namespace StableHLO

def parse (src : List Char) : Except (String × List Trace × List Derivation) Program := do
  parseModule.run' <| ParsingState.mk src 0 src.length 1 0 [] []

end StableHLO
