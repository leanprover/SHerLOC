/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1
import SHerLOC.Parsing.Parser

namespace StableHLO.Parsing

def parseValueId : PState String := do
  parseItem "%"
  parseId

def parseValueIdRes : PState String := do
  let r ← parseValueId
  let mut r' := ""
  if ← isParse ":" then
    r' ← parseId
    r' := ":" ++ r'
  let r := r ++ r'
  return r

def parseValueIdOpArg : PState String := do
  let r ← parseValueId
  let mut r' := ""
  if ← isParse "#" then
    r' ← parseId
    r' := "#" ++ r'
  let r := r ++ r'
  return r

def parseFuncId : PState String := do
  parseItem "@"
  parseFId

def parseUnusedId : PState String := do
  parseItem "^"
  parseId

def parseAttrId : PState String := do
  parseId

end StableHLO.Parsing
