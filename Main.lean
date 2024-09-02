/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC

open System IO FilePath Process FS Std

def main : IO Unit := do
  let o ← output { cmd := "ls", args := #["Tests"] }
  let files := o.stdout.splitOn "\n"
  let files := files.filter fun s => s.takeRight 5 = ".mlir"
  let mut success : Bool := true
  let mut count : Nat := 0
  for file in files do
    IO.println s!"Reading {file}"
    let fp : FilePath := System.mkFilePath ["Tests", file]
    let content ← readFile fp
    let content := StableHLO.parse content.data
    match content with
    | .ok _ =>
      IO.println s!"Parsing {file}: success"
    | .error e =>
      IO.println s!"Parsing {file}: failure {e}"
      count := count + 1
      success := false
  if ! success then panic s!"{count} out of {files.length} tests failed"
