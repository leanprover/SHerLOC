/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Types
import SHerLOC.Parsing.Constants

namespace StableHLO

def parseAttribute : PState Attribute := do
  push "parseAttribute"
  if ← isParse "use_global_device_ids" then -- Review
    pop "parseAttribute"
    return { id := "use_global_device_ids" , constant := Constant.booleanConstant BooleanLiteral.true }
  else
    let id ← parseId
    parseItem "="
    let constant ← parseConstant
    pop "parseAttribute"
    return { id := id , constant := constant }

def parseAttributes : PState (List Attribute) := do
  push "parseAttributes"
  let r ← parseList "{" "}" "," parseAttribute
  pop "parseAttributes"
  return r

def parseValueUseList : PState (List ValueId) := do
  push "parseValueUseList"
  let r ← parseList "(" ")" "," parseValueIdOpArg
  pop "parseValueUseList"
  return r

def tryParseDictionaryEntry (name : String) (parser : PState T) : PState (Option T) := do
  if ← is name then
    parseItem name
    parseItem "="
    let t ← parser
    return some t
  else return none

def parseDictionaryProperties : PState (List Attribute) := do
  push "parseDictionaryProperties"
  let r ← parseList "<{" "}>" "," parseAttribute
  pop "parseDictionaryProperties"
  return r

end StableHLO
