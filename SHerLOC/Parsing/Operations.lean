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

def parseAttributes : PState (List Attribute) := do
  push "parseAttributes"
  let r ← parseList "{" "}" (some ",") parseAttribute
  pop "parseAttributes"
  return r

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
  return { id := id , typ := typ, attrs := [] }

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
  let arguments ← parseOpInputValues
  parseItem ":"
  let functiontype ← parseFunctionType
  let parseResult := Operation.return arguments functiontype
  pop "parseReturn"
  return parseResult

mutual

partial def parseInputFunc : PState InputFunc := do
  push "parseInputFunc"
  parseItem "{"
  let id ← parseUnusedId
  let funcInputs ← parseInputFuncInputs
  parseItem ":"
  let body ← parseInputFuncBody
  parseItem "}"
  pop "parseInputFunc"
  return InputFunc.mk id funcInputs body

partial def parseOpInputFuncs : PState (List InputFunc) := do
  push "parseOpInputFuncs"
  let r ← parseList "(" ")" (some ",") parseInputFunc
  pop "parseOpInputFuncs"
  return r

-- partial def parseStableOp : PState Operation := do
--   push "parseStableOp"
--   let st ← get
--   let mut opOutputs := []
--   if st.is "%" then
--     opOutputs ← parseOpOutputs
--     parseItem "="
--   let st₀ ← get
--   if st₀.is "stablehlo.constant" then
--     let _ ← parseOpName
--     let constant ← parseConstant
--     let operation := Operation.constant opOutputs constant
--     pop "parseStableOp"
--     return operation
--   else
--     let opName ← parseOpName
--     let opInputValues ← parseOpInputValues
--     let mut opInputFuncs := []
--     let st₁ ← get
--     if st₁.is "(" then opInputFuncs ← parseOpInputFuncs
--     let mut opInputAttrs := []
--     let st₂ ← get
--     if st₂.is "{" then opInputAttrs ← parseOpInputAttrs
--     parseItem ":"
--     -- Unfortunately, StableHLO seems to use both short and long notations for operation types
--     -- However, it appears that parenthesis only appear for domains
--     let st₃ ← get
--     if st₃.is "(" then
--       let functiontype ← parseFunctionType
--       let operation := Operation.stable opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
--       pop "parseStableOp"
--       return operation
--     else
--       let functiontype ← parseFunctionType
--       let operation := Operation.stable opName opInputValues opInputFuncs opInputAttrs opOutputs functiontype
--       pop "parseStableOp"
--       return operation

partial def parseOperationDictionaryAttributes : PState (List Attribute) := do
  push "parseOperationDictionaryAttributes"
  let r ← parseList "<{" "}>" "," parseAttribute
  pop "parseOperationDictionaryAttributes"
  return r

partial def parseOperation : PState Operation := do
  push "parseOperation"
  let st ← get
  let mut opOutputs := []
  if st.is "%" then
    opOutputs ← parseOpOutputs
    parseItem "="
  let opName ← parseString
  --let opInputValues ← parseOpInputValues
  let opInputValues ← parseValueUseList
  let mut opInputAttrs := []
  let st₂ ← get
  if st₂.is "<{" then
    opInputAttrs ← parseOperationDictionaryAttributes
  parseItem ":"
  let functiontype ← parseFunctionType
  let operation := Operation.stable opName opInputValues [] opInputAttrs opOutputs functiontype
  pop "parseStableOp"
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
