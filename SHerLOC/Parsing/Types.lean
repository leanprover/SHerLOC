/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def parseIntegerType : PState IntegerType := do
  let st ← get
  match st.tok with
  | "si2" => shift ; return { sign := Signedness.signed , size := IntegerSize.b2 }
  | "si4" => shift ; return { sign := Signedness.signed , size := IntegerSize.b4 }
  | "si8" => shift ; return { sign := Signedness.signed , size := IntegerSize.b8 }
  | "si16" => shift ; return { sign := Signedness.signed , size := IntegerSize.b16 }
  | "si32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | "si64" => shift ; return { sign := Signedness.signed , size := IntegerSize.b64 }
  | "ui2" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b2 }
  | "ui4" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b4 }
  | "ui8" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b8 }
  | "ui16" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b16 }
  | "ui32" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b32 }
  | "ui64" => shift ; return { sign := Signedness.unsigned , size := IntegerSize.b64 }
  -- Jax compatibility
  | "i32" => shift ; return { sign := Signedness.signed , size := IntegerSize.b32 }
  | _ => throw <| st.error "Integer type"

partial def parseShape : PState (List Nat) := do
  let st ← get
  if (st.lookahead 1).get! ⟨ 0 ⟩ = 'x'
  then
    let i ← parseDecimal
    let shape ← parseShape
    return i :: shape
  else return []

def parseTensorElementType : PState TensorElementType := do
  let st ← get
  if st.is "i1" then return TensorElementType.booleanType
  let c := st.tok.get! ⟨ 0 ⟩
  if c = 's' || c = 'u' || c = 'i' then return TensorElementType.integerType <| ← parseIntegerType
  else throw <| st.error "TensorElementType"

def parseTensorType : PState TensorType := do
  parseItem "tensor"
  parseItem "<"
  let shape ← parseShape
  let tensorElementType ← parseTensorElementType
  parseItem ">"
  return { shape := shape , tensorElementType := tensorElementType}

-- temporary, shortcut for testing
def parseValueType : PState ValueType := do
  let temporary ← parseTensorType
  return ValueType.tensorType temporary

def parseValueTypes : PState (List ValueType) := do
  parseList "(" ")" (some ",") parseValueType

def parseFunctionType : PState FunctionType := do
  let inputTypes ← parseValueTypes
  parseItem "-"
  parseItem ">"
  let outputType ← parseValueTypes
  let functionType := { domain := inputTypes , range := outputType }
  return functionType

end StableHLO
