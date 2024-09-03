/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC

open System IO FilePath Process FS Std

def main (args : List String) : IO Unit := do
  if args.length = 0 then
    let o ← output { cmd := "ls", args := #["Tests"] }
    let files := o.stdout.splitOn "\n"
    let files := files.filter fun s => s.takeRight 5 = ".mlir"
    let mut passed := []
    let mut failed := []
    for file in files do
      let fp : FilePath := System.mkFilePath ["Tests", file]
      let content ← readFile fp
      let content := StableHLO.parse content.data
      match content with
      | .ok _ =>
        passed := file :: passed
      | .error _ =>
        failed := file :: failed
    IO.println "\nPassed:\n"
    for file in passed do
      IO.println file
    IO.println "\nFailed:\n"
    for file in failed do
      IO.println file
    if failed.length > 0 then panic! s!"Some tests failed"
  else if args.length = 1 then
    if let some _ := args[0]!.toNat? then
      let file : String := "test" ++ args[0]! ++ ".mlir"
      let fp : FilePath := System.mkFilePath ["Tests", file]
      let content ← readFile fp
      let content := StableHLO.parse content.data
      match content with
      | .ok p => IO.println s!"{repr p}"
      | .error e =>
        IO.println s!"{e.2}"
        IO.println s!"{e.1}"
    else panic! s!"Unexpected argument {args[0]!}, expected natural number"
  else panic! s!"Unexpected number of arguments: {args.length}"
