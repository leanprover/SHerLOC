# SHerLOC

SHerLOC is a program analyzer for [StableHLO programs](https://openxla.org/stablehlo). It is written in [Lean](https://leanprover-community.github.io/index.html). 

SHerLOC aims to transform a StableHLO program written in concrete generic syntax into a well-formed, typed, abstract syntax tree. It also reports information such as use of undocumented/unspecified/underspecified/deprecated constructions and provides tools for analyzing and visualizing HLO programs.

## Installation

To use SHerLOC, you must [install Lean](https://leanprover-community.github.io/get_started.html). If you want to use SHerLOC on StableHLO programs written in pretty syntax, you also need to [install StableHLO](https://github.com/openxla/stablehlo?tab=readme-ov-file#build-instructions) (note that you do not need to build the Python bindings).

You should then clone this repository.

## Usage

To run SHerLOC, go to the SHerLOC directory and run 

```
lake exe sherloc myprogram.mlir
```

This will produce two files, `myprogram.mlir.ast` and `myprogram.mlir.report` that contain respectively a dump of the abstract syntax tree and the reported information about the program.

If the StableHLO program is in pretty syntax, you can convert it to generic syntax using `stablehlo-opt`

```
stablehlo-opt -mlir-print-op-generic myprogrampretty.mlir > myprogramgeneric.mlir
```

Other subcommands are also available:

`ops` - prints the operators used in the program:
```bash
lake exe sherloc ops myprogram.mlir
```

`graph` - prints the program in graphviz format for visualization
```bash
lake exe sherloc graph myprogram.mlir
```

## Exporting Models to StableHLO

For examples of how to produce a StableHLO program from a JAX model, see `export_hlo.py`. Example usage:

```bash
python export_hlo.py attention > attention.mlir
```

