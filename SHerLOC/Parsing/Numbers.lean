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

def parseIntegerLiteral : PState IntegerLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then shift
  if st.is "-" then shift ; sign := Sign.minus
  let nat ← parseDecimal
  let parseResult := { sign := sign , decimal := nat }
  return parseResult

def parseIntegerConstant : PState IntegerConstant := do
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  return { literal := i, type := t }

def parseFloatType : PState FloatType := do
  let st ← get
  match st.tok with
  | "f8E4M3FN" => shift ; return FloatType.f8E4M3FN
  | "f8E5M2" => shift ; return FloatType.f8E5M2
  | "f8E4M3FNUZ" => shift ; return FloatType.f8E4M3FNUZ
  | "f8E5M2FNUZ" => shift ; return FloatType.f8E5M2FNUZ
  | "f8E4M3B11FNUZ" => shift ; return FloatType.f8E4M3B11FNUZ
  | "bf16" => shift ; return FloatType.bf16
  | "f16" => shift ; return FloatType.f16
  | "f32" => shift ; return FloatType.f32
  | "f64" => shift ; return FloatType.f64
  | _ => throw <| st.error "Float type"

def parseFloatLiteral : PState FloatLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then shift
  if st.is "-" then shift ; sign := Sign.minus
  let nat ← parseDecimal
  let integerPart := { sign := sign , decimal := nat }
  let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
  if st.is "." then
    shift
    fractionalPart := {fractionalPart with decimal := ← parseDecimal}
  let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
  if st.is "e" || st.is "E" then
    shift
    let mut scientificSign := Sign.plus
    if st.is "+" then shift
    if st.is "-" then shift ; scientificSign := Sign.minus
    let nat ← parseDecimal
    scientificPart := { sign := scientificSign, decimal := nat }
  let parseResult :=
    { integerPart := integerPart,
      fractionalPart := fractionalPart,
      scientificPart := scientificPart
    }
  return parseResult

def parseFloatConstant : PState FloatConstant := do
  let floatLiteral ← parseFloatLiteral
  parseItem ":"
  let floatType ← parseFloatType
  return { literal := floatLiteral, type := floatType }

end StableHLO
