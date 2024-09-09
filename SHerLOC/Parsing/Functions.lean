/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Operations
import SHerLOC.Parsing.Intermediate

namespace StableHLO

def tryParseDEntryFunctionType : PState (Option FunctionType) := do
  tryParseDictionaryEntry "function_type" parseFunctionType

def parseDictionaryAttributesInner : PState (List Attribute) := do
  parseList "{" "}" "," parseAttribute

def parseDictionaryAttributesOutter : PState (List (List Attribute)) := do
  parseList "[" "]" "," parseDictionaryAttributesInner

def tryParseDEntryResultAttributes : PState (Option (List (List Attribute))) := do
  tryParseDictionaryEntry "res_attrs" parseDictionaryAttributesOutter

def tryParseDEntryArgAttributes : PState (Option (List (List Attribute))) := do
  tryParseDictionaryEntry "arg_attrs" parseDictionaryAttributesOutter

def tryParseDEntrySymName : PState (Option String) := do
  tryParseDictionaryEntry "sym_name" parseString

def tryParseDEntrySymVisibility : PState (Option String) := do
  tryParseDictionaryEntry "sym_visibility" parseString

def parseFunctionDictionaryAttributes : PState (String × FunctionType × (List (List Attribute)) × (List (List Attribute))) := do
  let mut functionName : Option String := none
  let mut functionType : Option FunctionType := none
  let mut functionVisibility : Option String := none
  let mut functionResultAttributes : List (List Attribute) := []
  let mut functionArgAttributes : List (List Attribute) := []
  let mut functionNameTodo : Bool := true
  let mut functionTypeTodo : Bool := true
  let mut functionVisibilityTodo : Bool := true
  let mut functionResultAttributesTodo : Bool := true
  let mut functionArgAttributesTodo : Bool := true
  let mut count := 0

  for _ in [:5] do
    if functionNameTodo then
      if let some name ← tryParseDEntrySymName then
        functionName := name
        functionNameTodo := false
        count := count + 1
        if ← is "," then parseItem "," else break

    if functionTypeTodo then
      if let some t ← tryParseDEntryFunctionType then
        functionType := t
        functionTypeTodo := false
        count := count + 1
        if ← is "," then parseItem "," else break

    if functionResultAttributesTodo then
      if let some res ← tryParseDEntryResultAttributes then
        functionResultAttributes := res
        functionResultAttributesTodo := false
        count := count + 1
        if ← is "," then parseItem "," else break

    if functionArgAttributesTodo then
      if let some res ← tryParseDEntryArgAttributes then
        functionArgAttributes := res
        functionArgAttributesTodo := false
        count := count + 1
        if ← is "," then parseItem "," else break

    if functionVisibilityTodo then
      if let some visibility ← tryParseDEntrySymVisibility then
        functionVisibility := visibility
        functionVisibilityTodo := false
        count := count + 1
        if ← is "," then parseItem "," else break

    if count = 5 then break

  if let some name := functionName then
    if let some typ := functionType then
      return (name, typ, functionArgAttributes, functionResultAttributes)
    else
      throw <| ← error "A5"
  else
    throw <| ← error "A6"

def parseFunction : PState Function := do
  push "parseFunction"
  parseItems ["\"func.func\"", "(", ")"]
  parseItem "<{"
  let (name,typ,argAttrs,resAttrs) ← parseFunctionDictionaryAttributes
  parseItem "}>"
  let mut funcInputs : List FuncInput := []
  parseItem "({"
  if ← is "^" then
    discard <| parseUnusedId
    funcInputs ← parseInputFuncInputs
    parseItem ":"
  let operations ← parseListAuxNoSep "}" parseOperation []
  let body : InputFunc := InputFunc.mk funcInputs operations
  parseItem "})"
  parseItems [":","(",")","->","(",")"]
  let r : Function := { funcId := name , funcArgAttrs := argAttrs , funcResAttrs := resAttrs , funcType := typ, funcBody := body }
  pop "parseFunction"
  return r

def parseFunctions : PState (List Function) := do
  push "parseFunctions"
  let r ← parseListNoSep "{" "}" parseFunction
  pop "parseFunctions"
  return r

end StableHLO
