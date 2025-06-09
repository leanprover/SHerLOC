/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC

open System IO FilePath Process FS Std

def helpMessage : String :=
  "Usage: sherloc [command] [options]\n" ++
  "Commands:\n" ++
  "  help, --help, -h: Show this help message\n" ++
  "  parse <file>: Parse the given file and output the AST to .ast and .report files [default]\n" ++
  "  ops <file>: List operations used in the given hlo program"

def main (args : List String) : IO UInt32 := do
  match args with
    | ["help"] | ["--help"] | ["-h"] =>
      IO.println helpMessage
      return 0
    | [file] | ["parse", file] =>
      let fp : FilePath := System.mkFilePath [file]
      let content ← readFile fp
      let content := StableHLO.Parsing.parse content
      match content with
      | .ok p =>
        let fpAST : FilePath := System.mkFilePath [file ++ ".ast"]
        let fpReport : FilePath := System.mkFilePath [file ++ ".report"]
        writeFile fpAST s!"{repr p.1}\n"
        writeFile fpReport s!"{p.2.report}\n"
        return 0
      | .error e =>
        IO.println s!"{e.2.2}"
        IO.println s!"{e.2.1}"
        IO.println s!"{e.1}"
        return 1
    | ["ops", file] =>
      let fp : FilePath := System.mkFilePath [file]
      let content ← readFile fp
      let content := StableHLO.Parsing.parse content
      match content with
      | .ok (p, _) =>
        let ops := StableHLO.Analysis.uniqueOps p
        ops.forM IO.println
        return 0
      | .error e =>
        IO.println "Error parsing file:"
        IO.println s!"{e}"
        return 1
    | ["graph", file] =>
      let fp : FilePath := System.mkFilePath [file]
      let content ← readFile fp
      let content := StableHLO.Parsing.parse content
      match content with
      | .ok (p, _) =>
        let graph := StableHLO.Analysis.graph p
        IO.println graph
        return 0
      | .error e =>
        IO.println "Error parsing file:"
        IO.println s!"{e}"
        return 1
    | _ => panic! s!"Invalid command. Use 'sherloc help' for usage information."

