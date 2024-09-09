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
  let mut name : Option FuncId := none
  if ← is "<{" then
    parseItem "<{"
    parseItem "sym_name"
    parseItem "="
    name ← parseString
    parseItem "}>"
  parseItem "("
  let region ← parseFunctions
  parseItem ")"
  let mut attributes : List Attribute := []
  if ← is "{" then
    attributes ← parseAttributes
  parseItems [":","(",")","->","(",")"]
  let r : Module := { modId := name, modAttrs := attributes, modFuncs := region }
  pop "parseModule"
  return r

def parseModules : PState (List Module) := do
  parseItems ["\"builtin.module\"", "(", ")"]
  let mut r : List Module := []
  if ← is "<{" then
    reset
    r := [← parseModule]
  else
    parseItems ["(","{"]
    if ← is "\"builtin.module\"" then
      r ← parseListAuxNoSep "})" parseModule []
    else
      reset
      r := [← parseModule]
  return r

end StableHLO
