/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC

open System IO FilePath Process FS Std

def main (args : List String) : IO UInt32 := do
  if args.length = 0 then
    let o ← output { cmd := "ls", args := #["Tests"] }
    let files := o.stdout.splitOn "\n"
    let files := files.filter fun s => s.takeRight 5 = ".mlir"
    let mut passed := []
    let mut failed := []
    for file in files do
      let fp : FilePath := System.mkFilePath ["Tests", file]
      let content ← readFile fp
      let content := StableHLO.parse content
      IO.print s!"Parsing {file}... "
      match content with
      | .ok _ =>
        passed := file :: passed
        IO.println "success"
      | .error _ =>
        failed := file :: failed
        IO.println "failure"
    IO.println "\nFailed tests:\n"
    for file in failed do
      IO.println file
    IO.println ""
    IO.println s!"Passed: {passed.length}, Failed {failed.length}"
    if failed.length > 0 then
      return 1
    else
      return 0
  else if args.length = 1 then
    let file := args[0]!
    let fp : FilePath := System.mkFilePath ["Tests", file]
    let content ← readFile fp
    let content := StableHLO.parse content
    match content with
    | .ok p =>
      IO.println s!"{repr p}"
      return 0
    | .error e =>
      IO.println s!"{e.2.2}"
      IO.println s!"{e.2.1}"
      IO.println s!"{e.1}"
      return 1

  else panic! s!"Unexpected number of arguments: {args.length}"
