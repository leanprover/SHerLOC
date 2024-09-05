/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def tryParseIntegerType : PState (Option IntegerType) := do
  push "tryParseIntegerType"
  let mut r : Option IntegerType := none
  if ← isParse "i2" then r := some { sign := Signedness.signed , size := IntegerSize.b2 }
  if ← isParse "i4" then r := some { sign := Signedness.signed , size := IntegerSize.b4 }
  if ← isParse "i8" then r := some { sign := Signedness.signed , size := IntegerSize.b8 }
  if ← isParse "i16" then r := some { sign := Signedness.signed , size := IntegerSize.b16 }
  if ← isParse "i32" then r := some { sign := Signedness.signed , size := IntegerSize.b32 }
  if ← isParse "i64" then r := some { sign := Signedness.signed , size := IntegerSize.b64 }
  if ← isParse "ui2" then r := some { sign := Signedness.unsigned , size := IntegerSize.b2 }
  if ← isParse "ui4" then r := some { sign := Signedness.unsigned , size := IntegerSize.b4 }
  if ← isParse "ui8" then r := some { sign := Signedness.unsigned , size := IntegerSize.b8 }
  if ← isParse "ui16" then r := some { sign := Signedness.unsigned , size := IntegerSize.b16 }
  if ← isParse "ui32" then r := some { sign := Signedness.unsigned , size := IntegerSize.b32 }
  if ← isParse "ui64" then r := some { sign := Signedness.unsigned , size := IntegerSize.b64 }
  pop "tryParseIntegerType"
  return r

def parseIntegerType : PState IntegerType := do
  push "parseIntegerType"
  if let some r ← tryParseIntegerType then pop "parseIntegerType" ; return r
  else throw <| ← error "Integer type"

def parseIntegerLiteral : PState IntegerLiteral := do
  push "parseIntegerLiteral"
  let mut sign := Sign.plus
  if ← isParse "+" then sign := Sign.plus
  else if ← isParse "-" then sign := Sign.minus
  let mut nat : Option Nat := none
  if ← is "0x" then
    nat ← parseHexaDecimal
  else
    nat ← parseDecimal
  if let some v := nat then
    let parseResult := { sign := sign , decimal := v }
    pop "parseIntegerLiteral"
    return parseResult
  else
    throw <| ← error "Integer literal"

def parseIntegerConstant : PState IntegerConstant := do
  push "parseIntegerConstant"
  let i ← parseIntegerLiteral
  parseItem ":"
  let t ← parseIntegerType
  pop "parseIntegerConstant"
  return { literal := i, type := t }

def tryParseFloatType : PState (Option FloatType) := do
  push "tryParseFloatType"
  let mut r : Option FloatType := none
  if ← isParse "f8E4M3B11FNUZ" then r := some FloatType.f8E4M3B11FNUZ
  if ← isParse "f8E4M3FNUZ" then r := some FloatType.f8E4M3FNUZ
  if ← isParse "f8E4M3FN" then r := some FloatType.f8E4M3FN
  if ← isParse "f8E5M2FNUZ" then r := some FloatType.f8E5M2FNUZ
  if ← isParse "f8E5M2" then r := some FloatType.f8E5M2
  if ← isParse "bf16" then r := some FloatType.bf16
  if ← isParse "f16" then r := some FloatType.f16
  if ← isParse "f32" then r := some FloatType.f32
  if ← isParse "f64" then r := some FloatType.f64
  pop "tryParseFloatType"
  return r

def parseFloatType : PState FloatType := do
  push "parseFloatType"
  if let some r ← tryParseFloatType then pop "parseFloatType"; return r
  else throw <| ← error "Float type"

def parseFloatLiteral : PState FloatLiteral := do
  push "parseFloatLiteral"
  let mut sign := Sign.plus
  if ← isParse "+" then sign := Sign.plus
  else if ← isParse "-" then sign := Sign.minus
  if ← is "0x" then
    let nat ← parseHexaDecimal
    pop "parseFloatLiteral"
    return FloatLiteral.hexaDecimal nat
  else
    let nat ← parseDecimal
    let integerPart : IntegerLiteral := { sign := sign , decimal := nat }
    let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
    if ← isParse "." then
      fractionalPart := {fractionalPart with decimal := ← parseDecimal}
    let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }
    if (← isParse "e") || (← isParse "E") then
      let mut scientificSign := Sign.plus
      if ← isParse "+" then scientificSign := Sign.plus
      else if ← isParse "-" then scientificSign := Sign.minus
      let nat ← parseDecimal
      scientificPart := { sign := scientificSign, decimal := nat }
    let parseResult := FloatLiteral.decimal
      { integerPart := integerPart,
        fractionalPart := fractionalPart,
        scientificPart := scientificPart
      }
    pop "parseFloatLiteral"
    return parseResult

def parseFloatConstant : PState FloatConstant := do
  push "parseFloatConstant"
  let floatLiteral ← parseFloatLiteral
  parseItem ":"
  let floatType ← parseFloatType
  pop "parseFloatConstant"
  return { literal := floatLiteral, type := floatType }

def parseNumberType : PState NumberType := do
  push "parseNumberType"
  if let some r ← tryParseIntegerType then pop "parseNumberType"; return NumberType.integerType r
  else if let some r ← tryParseFloatType then pop "parseNumberType"; return NumberType.floatType r
  else throw <| ← error "Number type"

def parseNumberConstant : PState NumberConstant := do
  push "parseNumberConstant"
  let literal ← parseFloatLiteral
  parseItem ":"
  let numberType ← parseNumberType
  pop "parseNumberConstant"
  return { literal := literal, type :=numberType }

end StableHLO
