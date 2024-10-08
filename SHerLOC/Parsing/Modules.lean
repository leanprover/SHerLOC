/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Operations
import SHerLOC.Parsing.Functions
import SHerLOC.Parsing.Intermediate

namespace StableHLO.Parsing

def parseModule : PState Module := do
  parseItems ["\"builtin.module\"", "(", ")"]
  let mut name : Option FuncId := none
  if ← is "<{" then
    parseItem "<{"
    parseItem "sym_name"
    parseItem "="
    name ← parseString
    parseItem "}>"
  parseItem "("
  if (← isParse "{") then
    if (← isParse "^bb0:") then -- Empty module
      let r : Module := { modId := name, modAttrs := [], modFuncs := [] }
      parseItems ["}",")"]
      parseItems [":","(",")","->","(",")"]
      return r
    let region ← parseFunctions
    parseItems ["}",")"]
    let mut attributes : List Attribute := []
    if ← is "{" then
      attributes ← parseAttributes
    parseItems [":","(",")","->","(",")"]
    let r : Module := { modId := name, modAttrs := attributes, modFuncs := region }
    return r
  else
    let r : Module := { modId := name, modAttrs := [], modFuncs := [] }
    return r

partial def parseModules : PState (List Module) := do
  let done ← done?
  if done then
    return []
  else
    let mod ← parseModule
    let mods ← parseModules
    return mod :: mods



end StableHLO.Parsing
