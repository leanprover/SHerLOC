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

def parseModule : PState Program := do
  parseItem "module"
  parseItem "@"
  discard <| parseId
  parseItem "attributes"
  discard <| parseAttributes
  let funcs ← parseFunctions
  return funcs

def parse (src : List Char) : Program := Id.run do
  let tokens := tokenize src
  let r ← parseModule.run' <| ParsingState.mk tokens 0 []
  match r with
  | .ok p => p
  | .error e => panic e

end StableHLO
