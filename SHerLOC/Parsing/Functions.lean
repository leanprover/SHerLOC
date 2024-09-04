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
  push "parseFuncInput"
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  let attrs ←  parseAttributes
  pop "parseFuncInput"
  return { id := id , typ := typ , attrs := attrs }

def parseFuncInputs : PState (List FuncInput) := do
  push "parseFuncInputs"
  let r ← parseList "(" ")" (some ",") parseFuncInput
  pop "parseFuncInputs"
  return r

def parseFuncOutput : PState FuncOutput := do
  push "parseFuncOutput"
  let typ ← parseValueType
  let attrs ← parseAttributes
  pop "parseFuncOutput"
  return { typ := typ , attrs := attrs }

def parseFuncOutputs : PState (List FuncOutput) := do
  push "parseFuncOutputs"
  let r ← parseList "(" ")" (some ",") parseFuncOutput
  pop "parseFuncOutputs"
  return r

def parseFunction : PState Function := do
  push "parseFunction"
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
  pop "parseFunction"
  return func

def parseFunctions : PState (List Function) := do
  push "parseFunctions"
  let r ← parseList "{" "}" none parseFunction
  pop "parseFunctions"
  return r

end StableHLO
