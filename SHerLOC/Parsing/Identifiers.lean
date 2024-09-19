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
  let r ← parseId
  return r

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
  let r ← parseFId
  return r

def parseUnusedId : PState String := do
  parseItem "^"
  let r ← parseId
  return r

def parseAttrId : PState String := do
  let r ← parseId
  return r

end StableHLO.Parsing
