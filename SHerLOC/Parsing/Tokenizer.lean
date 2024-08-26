/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic

namespace StableHLO

structure TokWithPos where
  tok : String
  line : Nat
  column : Nat
  deriving Repr, Inhabited, Nonempty

instance : ToString TokWithPos where
  toString := fun t : TokWithPos => s!"({t.line},{t.column},{t.tok})"

def tokenize (src : List Char) : List TokWithPos := Id.run do
  let mut lineNumber : Nat := 1
  let mut lastColumnNumber : Nat := 1
  let mut columnNumber : Nat := 1
  let mut tokens : List TokWithPos := []
  let mut current : String := ""
  for c in src do
    if c.isAlphanum || c = '_' || c = '+' || c = '-' || c = '.' then
      current := current.push c
      columnNumber := columnNumber + 1
    else
      if current != "" then
        tokens := { tok := current, line := lineNumber, column := lastColumnNumber } :: tokens
        current := ""
        lastColumnNumber := columnNumber
      match c with
      | ' ' | ',' => columnNumber := columnNumber + 1
      | '\n' => columnNumber := 1 ; lineNumber := lineNumber + 1
      | '\t' => panic "tokenize: tab not implemented yet"
      | c =>
        tokens := { tok := c.toString , line := lineNumber, column := lastColumnNumber } :: tokens
        columnNumber := columnNumber + 1
      lastColumnNumber := columnNumber
  return tokens.reverse

end StableHLO
