/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Types
import SHerLOC.Parsing.Constants

namespace StableHLO.Parsing

mutual

  partial def parseLiteral : PState Literal := do
    skip
    if (← isDigit) || (← isChar '+') || (← isChar '-') then
      return Literal.element <| ElementLiteral.floatLiteral <| ← parseFloatLiteral
    if ← isChar 'd' then
      return Literal.tensor <| ← parseTensorLiteral
    if (← isChar 't') || (← isChar 'f') then
      return Literal.element <| ElementLiteral.booleanLiteral <| ← parseBooleanLiteral
    if (← isChar '(') then
      return Literal.element <| ElementLiteral.complexLiteral <| ← parseComplexLiteral
    if ← isChar '"' then
      return Literal.string <| ← parseStringLiteral
    if ← isChar 'a' then
      return Literal.array <| ← parseArrayLiteral

    if ← isParse "#stablehlo" then {
      if (← isParse ".") then {
        if ← isParse "conv" then return Literal.convolution <| ← parseConvolution
        if ← isParse "dot_algorithm" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "dot" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "channel_handle" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "scatter" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "gather" then return Literal.stableHLORecord <| ← parseRecord
      } else return Literal.enum <| ← parseEnumLiteral
    }

    if ← isChar '[' then
      return Literal.list <| ← parseList "[" "]" "," parseLiteral

    if ← isChar '{' then
      return Literal.dictionary <| ← parseAttributes

    if ← isChar '@' then
      return Literal.func <| ← parseFuncId

    throw <| ← error "literal"

  partial def parseConstant : PState Constant := do
    push "parseConstant"
    let literal ← parseLiteral
    let mut typ : Option SType := none
    if ← isParse ":" then
      typ ← parseType
    let r : Constant := Constant.mk literal typ
    pop "parseConstant"
    return r

  partial def parseAttribute : PState Attribute := do
    push "parseAttribute"
    if ← isParse "use_global_device_ids" then -- Review
      pop "parseAttribute"
      throw <| ← error "use_global_device_ids"
      -- return -- Attribute.mk "use_global_device_ids" { literal := Literal.use_global_device_ids, typ := none }}
    else
      let id ← parseId
      parseItem "="
      let constant ← parseConstant
      pop "parseAttribute"
      return Attribute.mk id constant

partial def parseAttributes : PState (List Attribute) := do
  push "parseAttributes"
  let r ← parseList "{" "}" "," parseAttribute
  pop "parseAttributes"
  return r

end

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

end StableHLO.Parsing
