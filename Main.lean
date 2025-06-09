/-
Copyright (c) 2024 Amazon.com, Inc. or its affiliates. All Rights Reserved.
Released under Apache 2.0 license as described in the file LICENSE.
Authors: Jean-Baptiste Tristan
-/
import SHerLOC
import Cli

open System IO FilePath FS Std
open Cli

def helpMessage : String :=
  "Usage: sherloc [command] [options]\n" ++
  "Commands:\n" ++
  "  help, --help, -h: Show this help message\n" ++
  "  parse <file>: Parse the given file and output the AST to .ast and .report files [default]\n" ++
  "  ops <file>: List operations used in the given hlo program"

def parse (P : Parsed) : IO UInt32 := do
  let file := P.positionalArg! "hloFileName" |>.as! String
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
def ops (P : Parsed) : IO UInt32 := do
  let file := P.positionalArg! "hloFileName" |>.as! String
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
def graph (P : Parsed) : IO UInt32 := do
  let file := P.positionalArg! "hloFileName" |>.as! String
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

def parseCmd := `[Cli|
  "parse" VIA parse;
  "Parse a StableHLO file and output .ast and .report files."

  ARGS:
    hloFileName : String;      "The StableHLO file to parse (e.g., 'example.mlir')"
]

def opsCmd := `[Cli|
  "ops" VIA ops;
  "List all unique operators used in a StableHLO program."

  ARGS:
    hloFileName : String;      "The StableHLO file to parse (e.g., 'example.mlir')"
]
def graphCmd := `[Cli|
  "graph" VIA graph;
  "Generate a graph representation of the StableHLO program in graphviz format."

  ARGS:
    hloFileName : String;      "The StableHLO file to parse (e.g., 'example.mlir')"
]

def sherlocCmd : Cmd := `[Cli|
  sherloc NOOP; ["1.1.0"]
  "SHerLOC is a utility for analyzing and transforming StableHLO programs."

  SUBCOMMANDS:
    parseCmd;
    opsCmd;
    graphCmd
]

def main (args : List String) : IO UInt32 := do
  if args.isEmpty then do
    IO.println sherlocCmd.help
    return 0

  try sherlocCmd.validate args
  catch e =>
    match e with
    | .userError s => IO.eprintln s
    | e => IO.eprintln s!"{e}"
    pure 1
