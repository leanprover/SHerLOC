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
  if ← is "i2" then parseItem "i2" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b2 }
  if ← is "i4" then parseItem "i4" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b4 }
  if ← is "i8" then parseItem "i8" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b8 }
  if ← is "i16" then parseItem "i16" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b16 }
  if ← is "i32" then parseItem "i32" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b32 }
  if ← is "i64" then parseItem "i64" ; pop "tryParseIntegerType"; return some { sign := Signedness.signed , size := IntegerSize.b64 }
  if ← is "ui2" then parseItem "ui2" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b2 }
  if ← is "ui4" then parseItem "ui4" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b4 }
  if ← is "ui8" then parseItem "ui8" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b8 }
  if ← is "ui16" then parseItem "ui16" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b16 }
  if ← is "ui32" then parseItem "ui32" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b32 }
  if ← is "ui64" then parseItem "ui64" ; pop "tryParseIntegerType"; return some { sign := Signedness.unsigned , size := IntegerSize.b64 }
  pop "tryParseIntegerType"
  return none

def parseIntegerType : PState IntegerType := do
  push "parseIntegerType"
  if let some r ← tryParseIntegerType then pop "parseIntegerType" ; return r
  else throw <| ← error "Integer type"

def parseIntegerLiteral : PState IntegerLiteral := do
  push "parseIntegerLiteral"
  let mut sign := Sign.plus
  if ← is "+" then parseItem "+"
  else if ← is "-" then parseItem "-" ; sign := Sign.minus
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
  if ← is "f8E4M3FN" then parseItem "f8E4M3FN" ; pop "tryParseFloatType"; return some FloatType.f8E4M3FN
  if ← is "f8E5M2" then parseItem "f8E5M2" ; pop "tryParseFloatType"; return some FloatType.f8E5M2
  if ← is "f8E4M3FNUZ" then parseItem "f8E4M3FNUZ" ; pop "tryParseFloatType"; return some FloatType.f8E4M3FNUZ
  if ← is "f8E5M2FNUZ" then parseItem "f8E5M2FNUZ" ; pop "tryParseFloatType"; return some FloatType.f8E5M2FNUZ
  if ← is "f8E4M3B11FNUZ" then parseItem "f8E4M3B11FNUZ" ; pop "tryParseFloatType"; return some FloatType.f8E4M3B11FNUZ
  if ← is "bf16" then parseItem "bf16" ; pop "tryParseFloatType"; return some FloatType.bf16
  if ← is "f16" then parseItem "f16" ; pop "tryParseFloatType"; return some FloatType.f16
  if ← is "f32" then parseItem "f32" ; pop "tryParseFloatType"; return some FloatType.f32
  if ← is "f64" then parseItem "f64" ; pop "tryParseFloatType"; return some FloatType.f64
  pop "tryParseFloatType"
  return none

def parseFloatType : PState FloatType := do
  push "parseFloatType"
  if let some r ← tryParseFloatType then pop "parseFloatType"; return r
  else throw <| ← error "Float type"

def parseFloatLiteral : PState FloatLiteral := do
  push "parseFloatLiteral"
  let mut sign := Sign.plus
  if ← is "+" then parseItem "+"
  else if ← is "-" then parseItem "-" ; sign := Sign.minus
  if ← is "0x" then
    let nat ← parseHexaDecimal
    pop "parseFloatLiteral"
    return FloatLiteral.hexaDecimal nat
  else
    let nat ← parseDecimal
    let integerPart : IntegerLiteral := { sign := sign , decimal := nat }
    let mut fractionalPart : IntegerLiteral := { sign := Sign.plus, decimal := 0 }
    if ← is "." then
      parseItem "."
      fractionalPart := {fractionalPart with decimal := ← parseDecimal}
    let mut scientificPart : IntegerLiteral:= { sign := Sign.plus, decimal := 0 }

    if (← is "e") || (← is "E") then
      if ← is "e" then parseItem "e"
      if ← is "E" then parseItem "E"
      let mut scientificSign := Sign.plus
      if ← is "+" then parseItem "+"
      else if ← is "-" then parseItem "-" ; scientificSign := Sign.minus
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
