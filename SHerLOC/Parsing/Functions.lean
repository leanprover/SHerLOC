/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Operations

namespace StableHLO

def parseFuncInput : PState FuncInput := do
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  discard <| parseAttributes
  return { id := id , typ := typ }

def parseFuncInputs : PState (List FuncInput) := do
  parseList "(" ")" (some ",") parseFuncInput

def parseFuncOutput : PState ValueType := do
  let typ ← parseValueType
  discard <| parseAttributes
  return typ

def parseFuncOutputs : PState (List ValueType) := do
  parseList "(" ")" (some ",") parseFuncOutput

def parseFunction : PState Function := do
  let stStart ← get
  parseItem "func.func"
  parseItem "public"
  parseItem "@"
  let funcId ← parseId
  let funcInputs ← parseFuncInputs
  parseItem "-"
  parseItem ">"
  let funcOutputs ← parseFuncOutputs
  let body ← parseOperations
  let func := { funcId := funcId , funcInputs := funcInputs , funcOutputs := funcOutputs , funcBody := body }
  record stStart "Function"
  return func

def parseFunctions : PState (List Function) := do
  parseList "{" "}" none parseFunction

end StableHLO
