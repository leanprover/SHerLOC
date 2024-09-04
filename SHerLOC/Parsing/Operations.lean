/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Constants
import SHerLOC.Parsing.Identifiers
import SHerLOC.Parsing.Intermediate

namespace StableHLO

def parseOpOutputs : PState (List ValueId) := do
  push "parseOpOutputs"
  let r ← parseListAux "=" (some ",") parseValueId
  pop "parseOpOutputs"
  return r

def parseOpInputValues : PState (List ValueId) := do
  push "parseOpInputValues"
  let r ← parseListAux ":" (some ",") parseValueId
  pop "parseOpInputValues"
  return r

def parseInputFuncInput : PState FuncInput := do
  push "parseInputFuncInput"
  let id ← parseValueId
  parseItem ":"
  let typ ← parseValueType
  pop "parseInputFuncInput"
  return { id := id , typ := typ }

def parseInputFuncInputs : PState (List FuncInput) := do
  push "parseInputFuncInputs"
  let r ← parseList "(" ")" (some ",") parseInputFuncInput
  pop "parseInputFuncInputs"
  return r

def parseOpInputAttrs : PState (List Attribute) := do
  push "parseOpInputAttrs"
  let r ← parseAttributes
  pop "parseOpInputAttrs"
  return r

def parseReturn : PState Operation := do
  push "parseReturn"
  parseItem "\"func.return\""
  let arguments ← parseValueUseList
  parseItem ":"
  let functiontype ← parseFunctionType
  let parseResult := Operation.return arguments functiontype
  pop "parseReturn"
  return parseResult

def parseCall (outputs : List ValueId) : PState Operation := do
  push "parseCall"
  parseItem "\"func.call\""
  let arguments ← parseValueUseList
  parseItem "<{"
  parseItem "callee"
  parseItem "="
  let callee ← parseFuncId
  parseItem "}>"
  parseItem ":"
  let typ ← parseFunctionType
  let r := Operation.call callee arguments outputs typ
  pop "parseCall"
  return r

mutual

partial def parseInputFunc : PState InputFunc := do
  push "parseInputFunc"
  parseItem "{"
  discard <| parseUnusedId
  let funcInputs ← parseInputFuncInputs
  parseItem ":"
  let body ← parseInputFuncBody
  parseItem "}"
  pop "parseInputFunc"
  return InputFunc.mk funcInputs body

partial def parseOpInputFuncs : PState (List InputFunc) := do
  push "parseOpInputFuncs"
  let r ← parseList "(" ")" (some ",") parseInputFunc
  pop "parseOpInputFuncs"
  return r

partial def parseOperationDictionaryAttributes : PState (List Attribute) := do
  push "parseOperationDictionaryAttributes"
  let r ← parseList "<{" "}>" "," parseAttribute
  pop "parseOperationDictionaryAttributes"
  return r

partial def parseOperation : PState Operation := do
  push "parseOperation"
  let st ← get
  if st.is "\"func.return\"" then
    let r ← parseReturn
    pop "parseOperation"
    return r
  let mut opOutputs := []
  if st.is "%" then
    opOutputs ← parseOpOutputs
    parseItem "="
  let st₀ ← get
  if st₀.is "\"func.call\"" then
    let r ← parseCall opOutputs
    pop "parseOperation"
    return r
  let opName ← parseString
  --let opInputValues ← parseOpInputValues
  let opInputValues ← parseValueUseList
  let mut opInputAttrs := []
  let st₂ ← get
  if st₂.is "<{" then
    opInputAttrs ← parseOperationDictionaryAttributes
  let mut opInputFuncs := []
  let st₃ ← get
  if st₃.is "(" then
    opInputFuncs ← parseOpInputFuncs
  parseItem ":"
  let functiontype ← parseFunctionType
  let operation := Operation.stablehlo opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
  pop "parseOperation"
  return operation

partial def parseInputFuncBody : PState (List Operation) := do
  push "parseInputFuncBody"
  let r ← parseListAux "}" none parseOperation
  pop "parseInputFuncBody"
  return r

end

def parseOperations : PState (List Operation) := do
  push "parseOperations"
  let r ← parseList "{" "}" none parseOperation
  pop "parseOperations"
  return r

end StableHLO
