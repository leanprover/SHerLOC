/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser
import SHerLOC.Parsing.Types

namespace StableHLO

def parseBooleanLiteral : PState Bool := do
  let st ← get
  if st.is "true" then
    shift
    return true
  else if st.is "false" then
    shift
    return false
  else throw <| st.error "Boolean literal"

def parseBooleanConstant : PState Constant := do
  let b ← parseBooleanLiteral
  return Constant.booleanConstant b

def parseIntegerLiteral : PState Nat := do
  let st ← get
  let mut negative := false
  if st.is "+" then shift
  if st.is "-" then shift ; negative := true
  parseDecimal

def parseIntegerConstant : PState Constant := do
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  return Constant.integerConstant i t

def parseStringLiteral : PState String := do
  let st ← get
  parseItem "\""
  if (st.lookahead 1) = "\"" then return ""
  else
    let content ← parseId -- Verify String encoding
    parseItem "\""
    return content

def parseStringConstant : PState Constant := do
  let str ← parseStringLiteral
  return Constant.stringConstant str

def parseConstant : PState Constant := do
  let st ← get
  if st.tok.get ⟨ 0 ⟩ = '"' then return ← parseStringConstant
  match st.tok with
  | "true" | "false" => parseBooleanConstant
  | _ =>
    match (st.lookahead 2).get! ⟨ 0 ⟩ with
    | 's' | 'u' | 'i' /- Jax compatibility -/ => parseIntegerConstant
    | _ => throw <| st.error s!"Constant"

end StableHLO
