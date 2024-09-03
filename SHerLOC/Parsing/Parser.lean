/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC.AST.Basic

namespace StableHLO

-- structure NonTerminal where
--   startLine : Nat
--   startColumn : Nat
--   endLine : Nat
--   endColumn : Nat
--   nonTerminal : String
--   deriving Repr, Inhabited, Nonempty

-- instance : ToString NonTerminal where
--   toString := fun t : NonTerminal => s!"({t.startLine},{t.startColumn}):({t.endLine},{t.endColumn}):{t.nonTerminal}"

structure ParsingState where
  source : List Char     -- Source data being parsed
  index : Nat                  -- Index into source data
  stop : Nat
  lineNumber : Nat
  columnNumber : Nat
  --status : List NonTerminal    -- For debugging the parser
  deriving Repr, Inhabited, Nonempty

abbrev PState (T : Type) := StateT ParsingState (Except String) T

def ParsingState.is (st : ParsingState) (keyword : String) : Bool := Id.run do
  let mut index := st.index
  for i in [st.index:st.stop] do
    if let some c := st.source[i]? then
      if c = ' ' || c = '\t' || c = '\n' then index := index + 1
      else break
    else return false
  let mut valid := true
  for i in [:keyword.length] do
    if let some c := st.source[index + i]? then
      if c != keyword.get! ⟨ i ⟩ then valid := false
    else return false
  return valid

def ParsingState.isDigit (st : ParsingState) : Bool :=
  if let some c := st.source[st.index]? then
    c.isDigit
  else false

def ParsingState.error (st : ParsingState) (msg : String) : String := Id.run do
  let mut token := ""
  let mut started := false
  for i in [st.index:st.stop] do
    let c := if let some c := st.source[i]? then c else panic s!"Indexing error in ParsingState.error"
    if ! started then
      if c = ' ' || c = '\t' || c = '\n' then ()
      else started := true
    else if c = ' ' || c = '\t' || c = '\n' then break
    else token := token.push c
  s!"Parsing error line {st.lineNumber}, column {st.columnNumber} : expected {msg} but found {token}"

-- def record (st : ParsingState) (nonTerminal : String) : PState Unit := do
--   let st' ← get
--   let info : NonTerminal := { startLine := st.line, startColumn := st.column, endLine := st'.line, endColumn := st'.column, nonTerminal := nonTerminal }
--   set { st' with status := info :: st'.status }

def skip : PState Unit := do
  let st ← get
  let mut count := 0
  let mut lines := 0
  let mut column := st.columnNumber
  for i in [st.index:st.stop] do
    let c := if let some c := st.source[i]? then c else panic s!"Indexing error in skip"
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
    else break
  set { st with
    index := st.index + count,
    lineNumber := st.lineNumber + lines,
    columnNumber := column
    }

def parseItem (keyword : String) : PState Unit := do
  skip
  let st ← get
  let mut success := true
  for i in [:keyword.length] do
    let c := if let some c := st.source[st.index + i]? then c else panic s!"Indexing error in parseItem"
    if ! c = keyword.get! ⟨ i ⟩ then
      success := false
      break
  if success then
    set { st with
      index := st.index + keyword.length,
      columnNumber := st.columnNumber + keyword.length
      }
  else
    throw <| st.error keyword

def parseId : PState String := do
  skip
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := if let some c := st.source[i]? then c else panic s!"Indexing error in parseId"
    if c.isAlphanum || c = '_' || c = '.' then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return token
  else
    throw <| st.error s!"Id"

def parseDecimal : PState Nat := do
  skip
  let st ← get
  let mut token := ""
  for i in [st.index:st.stop] do
    let c := if let some c := st.source[i]? then c else panic s!"Indexing error in parseDecimal"
    if c.isDigit then token := token.push c
    else break
  if token.length != 0 then
    set { st with
      index := st.index + token.length,
      columnNumber := st.columnNumber + token.length
    }
    return token.toNat!
  else
    throw <| st.error s!"Decimal"

def parseString : PState String := do
  skip
  parseItem "\""
  let st ← get
  let mut token := ""
  let mut escaped := false
  for i in [st.index:st.stop] do
    let c := if let some c := st.source[i]? then c else panic s!"Indexing error in parseString"
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

partial def parseListAux (closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  let st ← get
  if st.is closingMark then return []
  if let some sep := separator then if st.is sep then parseItem sep ; return ← parseListAux closingMark separator parse
  let attr ← parse
  let attrs ← parseListAux closingMark separator parse
  return attr :: attrs

def parseList (openingMark closingMark : String) (separator : Option String) (parse : PState T) : PState (List T) := do
  parseItem openingMark
  let attrs ← parseListAux closingMark separator parse
  parseItem closingMark
  return attrs

end StableHLO
