/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST1

namespace StableHLO.Parsing

structure Trace where
  startLine : Nat
  startColumn : Nat
  parser : String
  deriving Repr, Inhabited, Nonempty

instance : ToString Trace where
  toString := fun t : Trace => s!"({t.startLine},{t.startColumn}):{t.parser}"

instance : ToString (List Trace) where
  toString := fun t : List Trace => t.foldl (fun s : String => fun t : Trace => s ++ s!"{t}\n") "\n"

structure Derivation where
  startLine : Nat
  startColumn : Nat
  endLine : Nat
  endColumn : Nat
  parser : String
  deriving Repr, Inhabited, Nonempty

instance : ToString Derivation where
  toString := fun t : Derivation => s!"{t.parser}        ({t.startLine},{t.startColumn}):({t.endLine},{t.endColumn})"

instance : ToString (List Derivation) where
  toString := fun t : List Derivation => t.foldl (fun s : String => fun t : Derivation => s ++ s!"{t}\n") "\n"

structure ParsingState where
  source : String    -- Source data being parsed
  index : Nat                  -- Index into source data
  stop : Nat
  lineNumber : Nat
  columnNumber : Nat
  trace : List Trace            -- For debugging the parser
  derivations : List Derivation -- For debugging the parser
  report : List String
  deriving Repr, Inhabited, Nonempty

abbrev PState (T : Type) := StateT ParsingState (Except (String × List Trace × List Derivation)) T

def error (msg : String) : PState (String × (List Trace) × (List Derivation)) := do
  let st ← get
  let mut token := ""
  let mut started := false
  for i in [st.index:st.stop] do
    let c := if let some c := st.source.get? ⟨ i ⟩ then c else panic s!"Indexing error in ParsingState.error"
    if ! started then
      if c = ' ' || c = '\t' || c = '\n' then continue
      else
        started := true
        token := token.push c
    else if c = ' ' || c = '\t' || c = '\n' then break
    else token := token.push c
  let errorMsg := s!"Parsing error line {st.lineNumber}, column {st.columnNumber} : expected {msg} but found {token}"
  return (errorMsg, st.trace, st.derivations)

def errorSimple (msg : String) : PState (String × (List Trace) × (List Derivation)) := do
  let st ← get
  let errorMsg := s!"Parsing error line {st.lineNumber}, column {st.columnNumber} : {msg}"
  return (errorMsg, st.trace, st.derivations)

def report (msg : String) : PState Unit := do
  let st ← get
  let msg := s!"line {st.lineNumber}, column {st.columnNumber}: {msg}\n"
  set { st with report := msg :: st.report}

def skipComment (index : Nat) (st : ParsingState) : Nat := Id.run do
  let mut count := 0
  for i in [index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    count := count + 1
    if c = '\n' then
      break
  return count

def skip : PState Unit := do
  let st ← get
  let mut count := 0
  let mut lines := 0
  let mut column := st.columnNumber
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if c = '\n' then
      count := count + 1
      lines := lines + 1
      column := 0
    else if c = ' ' then
      count := count + 1
      column := column + 1
    else if c = '\t' then
      count := count + 1
      column := column + 8
    else if c = '/' && st.source.get! ⟨ i + 1 ⟩ = '/' then
      count := count + skipComment (st.index + count) (← get)
      lines := lines + 1
      column := 0
    else break
  set { st with
    index := st.index + count,
    lineNumber := st.lineNumber + lines,
    columnNumber := column
    }

def done? : PState Bool := do
  skip
  let st ← get
  if st.index >= st.stop then return true else return false

def parseItem (keyword : String) : PState Unit := do
  skip
  let st ← get
  let sub : Substring := { str := st.source, startPos := ⟨ st.index ⟩ , stopPos := ⟨ st.index + keyword.length ⟩ }
  if sub.beq keyword.toSubstring then
    set { st with
      index := st.index + keyword.length,
      columnNumber := st.columnNumber + keyword.length
      }
  else
    throw <| ← error keyword

def is (keyword : String) : PState Bool := do
  skip
  let st ← get
  let sub : Substring := { str := st.source, startPos := ⟨ st.index ⟩ , stopPos := ⟨ st.index + keyword.length ⟩ }
  return sub.beq keyword.toSubstring

def isParse (keyword : String) : PState Bool := do
  skip
  let st ← get
  let sub : Substring := { str := st.source, startPos := ⟨ st.index ⟩ , stopPos := ⟨ st.index + keyword.length ⟩ }
  if sub.beq keyword.toSubstring then
      set { st with
        index := st.index + keyword.length,
        columnNumber := st.columnNumber + keyword.length
        }
      return true
    else
      return false

def isDigit : PState Bool := do
  skip
  let st ← get
  let c := st.source.get! ⟨ st.index ⟩
  return c.isDigit

def isChar (c : Char) : PState Bool := do
  skip
  let st ← get
  let c' := st.source.get! ⟨ st.index ⟩
  return c = c'

def parseItems (keywords : List String) : PState Unit := do
  for i in [:keywords.length] do
    parseItem <| keywords[i]!

def parseFId : PState String := do
  skip
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if c.isAlphanum || c = '_' || c = '.' || c = '"' || c = '<' || c = '>' then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return token
  else
    throw <| ← error s!"Id"

def parseId : PState String := do
  skip
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if c.isAlphanum || c = '_' || c = '.' then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return token
  else
    throw <| ← error s!"Id"

def parseDecimal : PState Nat := do
  skip
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if c.isDigit then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return token.toNat!
  else
    throw <| ← error s!"Decimal"

def isHexDigit (c : Char) : Bool :=
  c.val ≥ 48 && c.val ≤ 57 || c.val ≥ 65 && c.val ≤ 70 || c.val ≥ 97 && c.val ≤ 102

def toNatHex (s : String) : Nat :=
  let r := s.foldl (fun n c =>  n*16 + (
    if c.isDigit then c.toNat - '0'.toNat
    else
      if c.val <= 70 then 10 + (c.toNat - 'A'.toNat)
      else 10 + (c.toNat - 'a'.toNat))) 0
  r

def parseHexaDecimal : PState Nat := do
  skip
  parseItem "0x"
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if isHexDigit c then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return toNatHex token
  else
    throw <| ← error s!"HexaDecimal"

def parseString : PState String := do
  skip
  parseItem "\""
  let st ← get
  let mut token := ""
  let mut escaped := false
  for i in [st.index:st.stop] do
    let c := st.source.get! ⟨ i ⟩
    if c = '"' then
      if escaped then
        token := token.push c
        escaped := false
      else
        break
    else if c = '\\' then
      if escaped then escaped := false
      else escaped := true
      token := token.push c
    else
      token := token.push c
  set { st with
    index := st.index + token.length,
    columnNumber := st.columnNumber + token.length
   }
  parseItem "\""
  return token

partial def parseListOneorMoreAux (separator : String) (parse : PState T) (acc : List T) : PState (List T) := do
  if ← isParse separator then
    parseListOneorMoreAux separator parse ((← parse) :: acc)
  else return acc.reverse

partial def parseListOneorMore (separator : String) (parse : PState T) : PState (List T) := do
  let head ← parse
  let tail ← parseListOneorMoreAux separator parse []
  return head :: tail

partial def parseListAux' (closingMark : String) (separator : String) (parse : PState T) (acc : List T) : PState (List T) := do
  if ← is closingMark then return acc.reverse
  if ← isParse separator then
    parseListAux' closingMark separator parse ((← parse) :: acc)
  else
    parseListAux' closingMark separator parse ((← parse) :: acc)

partial def parseListAux (closingMark : String) (separator : String) (parse : PState T) : PState (List T) := do
  parseListAux' closingMark separator parse []

def parseList (openingMark closingMark : String) (separator : String) (parse : PState T) : PState (List T) := do
  parseItem openingMark
  let attrs ← parseListAux closingMark separator parse
  parseItem closingMark
  return attrs

partial def parseListAuxNoSep (closingMark : String) (parse : PState T) (acc : List T) : PState (List T) := do
  if ← is closingMark then return acc.reverse
  parseListAuxNoSep closingMark parse ((← parse) :: acc)

def parseListNoSep (openingMark closingMark : String) (parse : PState T) : PState (List T) := do
  parseItem openingMark
  let attrs ← parseListAuxNoSep closingMark parse []
  parseItem closingMark
  return attrs

def parseDecimals : PState (List Nat) := do
  parseList "[" "]" "," parseDecimal

end StableHLO.Parsing
