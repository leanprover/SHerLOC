/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Parser

namespace StableHLO

def parseValueId : PState String := do
  parseItem "%"
  parseId

def parseFuncId : PState String := do
  parseItem "@"
  parseId

def parseUnusedId : PState String := do
  parseItem "^"
  parseId

def parseAttrId : PState String := do
  parseId

end StableHLO
