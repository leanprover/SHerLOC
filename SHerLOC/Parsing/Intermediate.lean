/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Types

namespace StableHLO.Parsing

def parseStableHLORecordFieldValue : PState (StableHLORecordFieldValue) := do
  if (← is "[") then
    let value ← parseDecimals
    return StableHLORecordFieldValue.many value
  else if (← isDigit) then
    let value ← parseDecimal
    return StableHLORecordFieldValue.one value
  else if (← isParse "true") then
    return StableHLORecordFieldValue.bool true
  else if (← isParse "false") then
    return StableHLORecordFieldValue.bool false
  else
    let type ← parseFloatType
    return StableHLORecordFieldValue.type type

def parseStableHLORecordField : PState (StableHLORecordField) := do
  let name ← parseId
  parseItem "="
  let value ← parseStableHLORecordFieldValue
  return StableHLORecordField.mk name value

def parseRecord : PState (List StableHLORecordField) := do
  let r ← parseList "<" ">" "," parseStableHLORecordField
  return r

mutual

  partial def parseLiteral : PState Literal := do
    skip
    if (← isDigit) || (← isChar '+') || (← isChar '-') then
      return Literal.element <| ElementLiteral.floatLiteral <| ← parseFloatLiteral
    if ← isChar 'd' then
      return Literal.tensor <| ← parseTensorLiteral
    if (← is "tr") || (← is "fa") then
      return Literal.element <| ElementLiteral.booleanLiteral <| ← parseBooleanLiteral
    if (← isChar '(') then
      return Literal.element <| ElementLiteral.complexLiteral <| ← parseComplexLiteral
    if ← isChar '"' then
      return Literal.string <| ← parseStringLiteral
    if ← isChar 'a' then
      report "literal array"
      return Literal.array <| ← parseArrayLiteral

    if ← isParse "#stablehlo" then {
      if (← isParse ".") then {
        report "literal record"
        if ← isParse "conv" then return Literal.convolution <| ← parseConvolution
        if ← isParse "dot_algorithm" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "dot" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "channel_handle" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "scatter" then return Literal.stableHLORecord <| ← parseRecord
        if ← isParse "gather" then return Literal.stableHLORecord <| ← parseRecord
      } else return Literal.enum <| ← parseEnumLiteral
    }

    if ← isChar '[' then
      report "literal list"
      return Literal.list <| ← parseList "[" "]" "," parseLiteral

    if ← isChar '{' then
      report "literal attribute"
      return Literal.dictionary <| ← parseAttributes

    if ← isChar '@' then
      report "literal function"
      return Literal.func <| ← parseFuncId

    throw <| (← error "literal")

  partial def parseConstant : PState Constant := do
    let literal ← parseLiteral
    let mut typ : Option SType := none
    if ← isParse ":" then
      typ ← parseType
    let r : Constant := Constant.mk literal typ
    return r

  partial def parseAttribute : PState Attribute := do
    if ← isParse "use_global_device_ids" then
      report "literal use_global_device_ids"
      return Attribute.mk "use_global_device_ids" <| Constant.mk (Literal.element (ElementLiteral.booleanLiteral BooleanLiteral.true)) none
    else
      let id ← parseId
      parseItem "="
      let constant ← parseConstant
      return Attribute.mk id constant

partial def parseAttributes : PState (List Attribute) := do
  let r ← parseList "{" "}" "," parseAttribute
  return r

end

def parseValueUseList : PState (List ValueId) := do
  let r ← parseList "(" ")" "," parseValueIdOpArg
  return r

def tryParseDictionaryEntry (name : String) (parser : PState T) : PState (Option T) := do
  if ← is name then
    parseItem name
    parseItem "="
    let t ← parser
    return some t
  else return none

def parseDictionaryProperties : PState (List Attribute) := do
  let r ← parseList "<{" "}>" "," parseAttribute
  return r

end StableHLO.Parsing
