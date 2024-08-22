/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC

open List System IO FS FilePath

def main (args : List String) : IO Unit := do
  if args.length != 1 then
    IO.println "Expected 1 argument"
    IO.Process.exit 1
  let file : FilePath := args[0]!
  let content ← readFile file
  let r := StableHLO.tokenize content.data
  IO.println s!"{r}"
  let content ← StableHLO.parse content.data
  IO.println s!"{content}"
