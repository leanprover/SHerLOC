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

def tryParseDEntryFunctionType : PState (Option FunctionType) := do
  tryParseDictionaryEntry "function_type" parseFunctionType

def parseDictionaryAttributes : PState (List Attribute) := do
  parseList "[{" "}]" "," parseAttribute

def tryParseDEntryResultAttributes : PState (Option (List Attribute)) := do
  tryParseDictionaryEntry "res_attrs" parseDictionaryAttributes

def tryParseDEntrySymName : PState (Option String) := do
  tryParseDictionaryEntry "sym_name" parseString

def tryParseDEntrySymVisibility : PState (Option String) := do
  tryParseDictionaryEntry "sym_visibility" parseString

def parseFunctionDictionaryAttributes : PState (String × FunctionType) := do
  let mut functionName : Option String := none
  let mut functionType : Option FunctionType := none
  let mut functionVisibility : Option String := none
  let mut functionResultAttributes : Option (List Attribute) := none
  for i in [:4] do
    dbg_trace s!"Iteration {i}"
    if let some name ← tryParseDEntrySymName then
      dbg_trace "name"
      functionName := name
      let st ← get
      if st.is "," then parseItem "," else break
    if let some t ← tryParseDEntryFunctionType then
      dbg_trace "type"
      functionType := t
      let st ← get
      if st.is "," then parseItem "," else break
    if let some res ← tryParseDEntryResultAttributes then
      dbg_trace "attributes"
      functionResultAttributes := res
      let st ← get
      if st.is "," then parseItem "," else break
    if let some visibility ← tryParseDEntrySymVisibility then
      dbg_trace "visibility"
      functionVisibility := visibility
      let st ← get
      if st.is "," then parseItem "," else break
  dbg_trace s!"{repr functionName} {repr functionType} {repr functionVisibility} {repr functionResultAttributes}"
  let st ← get
  if let some name := functionName then
    if let some typ := functionType then
      return (name, typ)
    else
      throw <| st.error "A5"
  else
    throw <| st.error "A6"

def parseFunction : PState Function := do
  push "parseFunction"
  parseItem "\"func.func\""
  let valueUseList ← parseValueUseList
  parseItem "<{"
  let (name,typ) ← parseFunctionDictionaryAttributes
  parseItem "}>"
  let region ← parseRegion parseOperation
  parseItem ":"
  let functiontype ← parseFunctionType
  let r : Function := { funcId := name , funcInputs := [] , funcOutputs := [] , funcBody := region }
  pop "parseFunction"
  return r

def parseFunctions : PState (List Function) := do
  push "parseFunctions"
  let r ← parseList "{" "}" none parseFunction
  pop "parseFunctions"
  return r

end StableHLO
