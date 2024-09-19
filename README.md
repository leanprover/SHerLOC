# SHerLOC

SHerLOC is a program analyzer for [StableHLO programs](https://openxla.org/stablehlo). It is written in [Lean](https://leanprover-community.github.io/index.html). 

SHerLOC aims to transform a StableHLO program written in concrete generic syntax into a well-formed, typed, abstract syntax tree. It also reports information such as use of undocumented/unspecified/underspecified/deprecated constructions.

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

To produce a StableHLO program in generic syntax from Jax, you can use the following Python example:

```python
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir

# Returns prettyprint of StableHLO module as generic print
def get_stablehlo_asm(module_str):
  with jax_mlir.make_ir_context():
    stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
    return stablehlo_module.operation.get_asm(print_generic_op_form=True, enable_debug_info=False)

## ----- 

import jax
from jax import export
import jax.numpy as jnp
import numpy as np

def plus(x,y):
  return jnp.add(x,y)

# Create abstract input shapes:
inputs = (np.int32(1), np.int32(1),)
input_shapes = [jax.ShapeDtypeStruct(input.shape, input.dtype) for input in inputs]
stablehlo_add = export.export(jax.jit(plus))(*input_shapes).mlir_module()

print(get_stablehlo_asm(stablehlo_add))
```
