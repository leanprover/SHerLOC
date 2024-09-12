/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser

namespace StableHLO.Parsing

def parseValueId : PState String := do
  push "parseValueId"
  parseItem "%"
  let r ← parseId
  pop "parseValueId"
  return r

def parseValueIdRes : PState String := do
  push "parseValueIdRes"
  let r ← parseValueId
  let mut r' := ""
  if ← isParse ":" then
    r' ← parseId
    r' := ":" ++ r'
  let r := r ++ r'
  pop "parseValueIdRes"
  return r

def parseValueIdOpArg : PState String := do
  push "parseValueOpArg"
  let r ← parseValueId
  let mut r' := ""
  if ← isParse "#" then
    r' ← parseId
    r' := "#" ++ r'
  let r := r ++ r'
  pop "parseValueOpArg"
  return r

def parseFuncId : PState String := do
  push "parseFuncId"
  parseItem "@"
  let r ← parseFId
  pop "parseFuncId"
  return r

def parseUnusedId : PState String := do
  push "parseUnusedId"
  parseItem "^"
  let r ← parseId
  pop "parseUnusedId"
  return r

def parseAttrId : PState String := do
  push "parseAttrId"
  let r ← parseId
  pop "parseAttrId"
  return r

end StableHLO.Parsing
