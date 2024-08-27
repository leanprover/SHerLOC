/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def tryParseIntegerType (tok : String) : Option IntegerType :=
  match tok with
  | "si2" => some { sign := Signedness.signed , size := IntegerSize.b2 }
  | "si4" => some { sign := Signedness.signed , size := IntegerSize.b4 }
  | "si8" => some { sign := Signedness.signed , size := IntegerSize.b8 }
  | "si16" => some { sign := Signedness.signed , size := IntegerSize.b16 }
  | "si32" => some { sign := Signedness.signed , size := IntegerSize.b32 }
  | "si64" => some { sign := Signedness.signed , size := IntegerSize.b64 }
  | "ui2" => some { sign := Signedness.unsigned , size := IntegerSize.b2 }
  | "ui4" => some { sign := Signedness.unsigned , size := IntegerSize.b4 }
  | "ui8" => some { sign := Signedness.unsigned , size := IntegerSize.b8 }
  | "ui16" => some { sign := Signedness.unsigned , size := IntegerSize.b16 }
  | "ui32" => some { sign := Signedness.unsigned , size := IntegerSize.b32 }
  | "ui64" => some { sign := Signedness.unsigned , size := IntegerSize.b64 }
  | _ => none

def parseIntegerType : PState IntegerType := do
  let st ← get
  if let some r := tryParseIntegerType st.tok then shift ; return r
  else throw <| st.error "Integer type"

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

def tryParseFloatType (tok : String) : Option FloatType := do
  match tok with
  | "f8E4M3FN" => some FloatType.f8E4M3FN
  | "f8E5M2" => some FloatType.f8E5M2
  | "f8E4M3FNUZ" => some FloatType.f8E4M3FNUZ
  | "f8E5M2FNUZ" => some FloatType.f8E5M2FNUZ
  | "f8E4M3B11FNUZ" => some FloatType.f8E4M3B11FNUZ
  | "bf16" => some FloatType.bf16
  | "f16" => some FloatType.f16
  | "f32" => some FloatType.f32
  | "f64" => some FloatType.f64
  | _ => none

def parseFloatType : PState FloatType := do
  let st ← get
  if let some r := tryParseFloatType st.tok then shift ; return r
  else throw <| st.error "Float type"

def parseFloatLiteral : PState FloatLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then shift
  if st.is "-" then shift ; sign := Sign.minus
  let nat ← parseDecimal
  let integerPart := { sign := sign , decimal := nat }
  let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
  let st₁ ← get
  if st₁.is "." then
    shift
    fractionalPart := {fractionalPart with decimal := ← parseDecimal}
  let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
  let st₂ ← get
  if st₂.is "e" || st₂.is "E" then
    shift
    let mut scientificSign := Sign.plus
    let st₃ ← get
    if st₃.is "+" then shift
    if st₃.is "-" then shift ; scientificSign := Sign.minus
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

def parseNumberType : PState NumberType := do
  let st ← get
  if let some r := tryParseIntegerType st.tok then shift ; return NumberType.integerType r
  else if let some r := tryParseFloatType st.tok then shift ; return NumberType.floatType r
  else throw <| st.error "Number type"

def parserNumberConstant : PState NumberConstant := do
  let literal ← parseFloatLiteral
  parseItem ":"
  let numberType ← parseNumberType
  return { literal := literal, type :=numberType }

end StableHLO
