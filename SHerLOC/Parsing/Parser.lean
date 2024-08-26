/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic
import SHerLOC.Parsing.Tokenizer

namespace StableHLO

structure NonTerminal where
  startLine : Nat
  startColumn : Nat
  endLine : Nat
  endColumn : Nat
  nonTerminal : String
  deriving Repr, Inhabited, Nonempty

instance : ToString NonTerminal where
  toString := fun t : NonTerminal => s!"({t.startLine},{t.startColumn}):({t.endLine},{t.endColumn}):{t.nonTerminal}"

structure ParsingState where
  source : List TokWithPos     -- Source data being parsed
  index : Nat                  -- Index into source data
  status : List NonTerminal    -- For debugging the parser
  deriving Repr, Inhabited, Nonempty

def ParsingState.tok (st : ParsingState) : String :=
  st.source[st.index]!.tok

def ParsingState.is (st : ParsingState) (keyword : String) : Bool :=
  st.tok = keyword

def ParsingState.line (st : ParsingState) : Nat :=
  st.source[st.index]!.line

def ParsingState.column (st : ParsingState) : Nat :=
  st.source[st.index]!.column

def ParsingState.lookahead (st : ParsingState) (shift : Nat) : String :=
  st.source[st.index + shift]!.tok

partial def ParsingState.search (st : ParsingState) (item : String) : Option Nat :=
  if st.is item then some 0
  else if st.index >= st.source.length then none
  else match { st with index := st.index + 1 }.search item with
    | some v => some (1 + v)
    | none => none

def ParsingState.isName (st : ParsingState) : Bool :=
  (st.tok.get ⟨0⟩).isAlphanum
  && st.tok.all fun c => c.isAlphanum || c = '_'

def ParsingState.isAttributeName (st : ParsingState) : Bool :=
  (st.tok.get ⟨0⟩).isAlphanum
  && st.tok.all fun c => c.isAlphanum || c = '_' || c = '.'

def ParsingState.isDecimal (st : ParsingState) : Option Nat :=
  if st.tok.all fun c => c.isDigit then some st.tok.toNat! else none

def ParsingState.isHexaDecimal (st : ParsingState) : Bool := Id.run do
  let b₁ := (st.tok.get ⟨0⟩) == '0'
  let b₂ := (st.tok.get ⟨1⟩) == 'x'
  let mut b₃ := true
  for i in [2:st.tok.length] do
    let c := st.tok.get ⟨i⟩
    if ! (c.isDigit || c.val ≥ 48 && c.val ≤ 57 || c.val ≥ 65 && c.val ≤ 70 || c.val ≥ 97 && c.val ≤ 102) then
      b₃ := false
  return b₁ && b₂ && b₃

def ParsingState.error (st : ParsingState) (msg : String) : String :=
  s!"{st.status.reverse} \n\n Parsing error line {st.line}, column {st.column}: expected {msg}, found {st.tok} "

abbrev PState (T : Type) := StateT ParsingState (Except String) T

def record (st : ParsingState) (nonTerminal : String) : PState Unit := do
  let st' ← get
  let info : NonTerminal := { startLine := st.line, startColumn := st.column, endLine := st'.line, endColumn := st'.column, nonTerminal := nonTerminal }
  set { st' with status := info :: st'.status }

def shift : PState Unit := do
  let st ← get
  set { st with index := st.index + 1 }

def parseItem (keyword : String) : PState Unit := do
  let st ← get
  if ! st.is keyword then
    throw <| st.error keyword
  shift

def parseId : PState String := do
  let st ← get
  if st.isName then
    shift
    return st.tok
  else
    throw <| st.error "Name"

def parseDecimal : PState Nat := do
  let st ← get
  if let some i := st.isDecimal then shift ; return i
  else throw <| st.error "Decimal number"

partial def parseListAux (closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  let st ← get
  if st.is closingMark then return []
  if let some sep := separator then if st.is sep then shift ; return ← parseListAux closingMark separator parse
  let attr ← parse
  let attrs ← parseListAux closingMark separator parse
  return attr :: attrs

def parseList (openingMark closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  parseItem openingMark
  let attrs ← parseListAux closingMark separator parse
  parseItem closingMark
  return attrs

end StableHLO
