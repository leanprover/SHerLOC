"""
This script exports a JAX model to StableHLO format and prints the assembly code.

Tested with Python 3.12.10
"""

import argparse
import jax
from jax import export
import jax.numpy as jnp
from flax import nnx

"""
These are technically private implementation details and could change at any time, but
this is the recommended way to get pretty-printed StableHLO output from JAX.

See: https://openxla.org/stablehlo/tutorials/jax-export#define_get_stablehlo_asm_to_help_with_mlir_printing
"""
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir


# Returns prettyprint of StableHLO module as generic print
def get_stablehlo_asm(module_str):
    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
        return stablehlo_module.operation.get_asm(print_generic_op_form=True, enable_debug_info=False)

def do_export(model, sample_inputs):
    model_jit = jax.jit(model)
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in sample_inputs]
    exported = export.export(model_jit)(*input_shapes)
    hlo = get_stablehlo_asm(exported.mlir_module())
    print(hlo)

def main():
    parser = argparse.ArgumentParser(description='Export JAX model to StableHLO format and print assembly code')
    parser.add_argument('model', help='Name of the JAX model to export (e.g., add, matmul, mlp, attention, resnet, llama)')
    
    args = parser.parse_args()

    model = args.model
        
    if model == "add":
        def add(x,y):
            return jnp.add(x,y)
        sample_inputs = [jnp.float32((20,10)), jnp.float32((20,10))]
        do_export(add, sample_inputs)

    elif model == "matmul":
        def matmul(x,y):
            return jnp.matmul(x,y)
        sample_inputs = [jnp.float32((20,10)), jnp.float32((10,30))]
        do_export(matmul, sample_inputs)

    elif model == "mlp":
        layer = nnx.Linear(4,2, rngs=nnx.Rngs(0))
        model = layer
        sample_inputs = [jax.random.uniform(jax.random.key(0),(1,4))]
        do_export(model, sample_inputs)

    elif model == "attention":
        layer = nnx.MultiHeadAttention(num_heads=8, in_features=5, qkv_features=16,
                                       decode=False, rngs=nnx.Rngs(0))
        keys = jax.random.split(jax.random.key(0), 3)
        shape = (4, 3, 2, 5)
        sample_inputs = [jax.random.uniform(keys[i], shape) for i in range(3)]
        model = layer
        do_export(model, sample_inputs)

    elif model == "resnet":
        """
        These models are pulled from HuggingFace, and therefore may change over time
        making the exported StableHLO non-reproducible.
        """
        from transformers import FlaxResNetModel
        model = FlaxResNetModel.from_pretrained("microsoft/resnet-18", return_dict=False)
        sample_inputs = [jax.random.uniform(jax.random.key(0), (1, 3, 224, 224))]
        do_export(model, sample_inputs)

    elif model == "llama":
        from transformers import FlaxLlamaForCausalLM
        model = FlaxLlamaForCausalLM.from_pretrained("afmck/testing-llama-tiny")
        sample_inputs = [jnp.asarray([[    1, 16644, 31844,   629,  3924,   322, 14138]])]
        do_export(model, sample_inputs)
    else:
        raise Exception("Unrecognized model: {}".format(model))
    
if __name__ == "__main__":
    main()
