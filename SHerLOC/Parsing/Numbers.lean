/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def tryParseIntegerType : PState (Option IntegerType) := do
  let st ← get
  if st.is "i2" then parseItem "i2" ; return some { sign := Signedness.signed , size := IntegerSize.b2 }
  if st.is "i4" then parseItem "i4" ; return some { sign := Signedness.signed , size := IntegerSize.b4 }
  if st.is "i8" then parseItem "i8" ; return some { sign := Signedness.signed , size := IntegerSize.b8 }
  if st.is "i16" then parseItem "i16" ; return some { sign := Signedness.signed , size := IntegerSize.b16 }
  if st.is "i32" then parseItem "i32" ; return some { sign := Signedness.signed , size := IntegerSize.b32 }
  if st.is "i64" then parseItem "i64" ; return some { sign := Signedness.signed , size := IntegerSize.b64 }
  if st.is "ui2" then parseItem "ui2" ; return some { sign := Signedness.unsigned , size := IntegerSize.b2 }
  if st.is "ui4" then parseItem "ui4" ; return some { sign := Signedness.unsigned , size := IntegerSize.b4 }
  if st.is "ui8" then parseItem "ui8" ; return some { sign := Signedness.unsigned , size := IntegerSize.b8 }
  if st.is "ui16" then parseItem "ui16" ; return some { sign := Signedness.unsigned , size := IntegerSize.b16 }
  if st.is "ui32" then parseItem "ui32" ; return some { sign := Signedness.unsigned , size := IntegerSize.b32 }
  if st.is "ui64" then parseItem "ui64" ; return some { sign := Signedness.unsigned , size := IntegerSize.b64 }
  return none

def parseIntegerType : PState IntegerType := do
  let st ← get
  if let some r ← tryParseIntegerType then return r
  else throw <| st.error "Integer type"

def parseIntegerLiteral : PState IntegerLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then parseItem "+"
  else if st.is "-" then parseItem "-" ; sign := Sign.minus
  let nat ← parseDecimal
  let parseResult := { sign := sign , decimal := nat }
  return parseResult

def parseIntegerConstant : PState IntegerConstant := do
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  return { literal := i, type := t }

def tryParseFloatType : PState (Option FloatType) := do
  let st ← get
  if st.is "f8E4M3FN" then parseItem "f8E4M3FN" ; return some FloatType.f8E4M3FN
  if st.is "f8E5M2" then parseItem "f8E5M2" ; return some FloatType.f8E5M2
  if st.is "f8E4M3FNUZ" then parseItem "f8E4M3FNUZ" ; return some FloatType.f8E4M3FNUZ
  if st.is "f8E5M2FNUZ" then parseItem "f8E5M2FNUZ" ; return some FloatType.f8E5M2FNUZ
  if st.is "f8E4M3B11FNUZ" then parseItem "f8E4M3B11FNUZ" ; return some FloatType.f8E4M3B11FNUZ
  if st.is "bf16" then parseItem "bf16" ; return some FloatType.bf16
  if st.is "f16" then parseItem "f16" ; return some FloatType.f16
  if st.is "f32" then parseItem "f32" ; return some FloatType.f32
  if st.is "f64" then parseItem "f64" ; return some FloatType.f64
  return none

def parseFloatType : PState FloatType := do
  let st ← get
  if let some r ← tryParseFloatType then return r
  else throw <| st.error "Float type"

def parseFloatLiteral : PState FloatLiteral := do
  let st ← get
  let mut sign := Sign.plus
  if st.is "+" then parseItem "+"
  else if st.is "-" then parseItem "-" ; sign := Sign.minus
  let nat ← parseDecimal
  let integerPart := { sign := sign , decimal := nat }
  let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
  let st₁ ← get
  if st₁.is "." then
    parseItem "."
    fractionalPart := {fractionalPart with decimal := ← parseDecimal}
  let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
  let st₂ ← get
  if st₂.is "e" || st₂.is "E" then
    if st₂.is "e" then parseItem "e"
    if st₂.is "E" then parseItem "E"
    let mut scientificSign := Sign.plus
    let st₃ ← get
    if st₃.is "+" then parseItem "+"
    else if st₃.is "-" then parseItem "-" ; scientificSign := Sign.minus
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
  if let some r ← tryParseIntegerType then return NumberType.integerType r
  else if let some r ← tryParseFloatType then return NumberType.floatType r
  else throw <| st.error "Number type"

def parseNumberConstant : PState NumberConstant := do
  let literal ← parseFloatLiteral
  parseItem ":"
  let numberType ← parseNumberType
  return { literal := literal, type :=numberType }

end StableHLO
