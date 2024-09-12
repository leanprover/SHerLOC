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
  deriving Repr, Inhabited, Nonempty

abbrev PState (T : Type) := StateT ParsingState (Except (String × List Trace × List Derivation)) T

-- For backtracking at the very beginning of parsing when deciding if there are 1 or more modules
def reset : PState Unit := do
  let st ← get
  let src := st.source
  set (ParsingState.mk src 0 src.length 1 0 [] [])

def error (msg : String) : PState (String × (List Trace) × (List Derivation) ):= do
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

def skipComment (st : ParsingState) : Nat := Id.run do
  let mut count := 0
  for i in [st.index:st.stop] do
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
    else if c = '/' then
      if st.source.get! ⟨ i + 1 ⟩ = '/' then -- comment
        count := count + skipComment (← get) + 2
        lines := lines + 1
        column := 0
    else break
  set { st with
    index := st.index + count,
    lineNumber := st.lineNumber + lines,
    columnNumber := column
    }

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
    parseItem <| keywords.get! i

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

def flyOver (start stop : String) : PState Unit := do
  skip
  parseItem start
  let st ← get
  for _ in [st.index:st.stop] do
    let st ← get
    if ← isParse "->" then continue
    if ← is stop then break
    else set { st with index := st.index + 1, columnNumber := st.columnNumber + 1} -- Incorrect because of \n and \t but this code is temporary
  parseItem stop

def push (parser : String) : PState Unit := do
  let st ← get
  let traceItem : Trace := { startLine := st.lineNumber, startColumn := st.columnNumber, parser }
  set { st with trace := traceItem :: st.trace   }

def indent (n : Nat) : String := Id.run do
  let mut token := ""
  for _ in [:n] do
    token := token.push ' '
  return token

def pop (parser : String) : PState Unit := do
  let st ← get
  if let some tail := st.trace.tail? then
    let head := st.trace.head!
    if head.parser = parser then
      let derivation : Derivation := {
        startLine := head.startLine,
        startColumn := head.startColumn,
        endLine := st.lineNumber,
        endColumn := st.columnNumber,
        parser := (indent tail.length) ++ parser }
      set {st with trace := tail, derivations := derivation :: st.derivations }
    else panic! s!"Trace mismatch: expected {parser} but found {head}"
  else panic! "More pops than pushes, some parser is missing its push"

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

end StableHLO.Parsing
