"""
This script exports various JAX models to StableHLO format and prints the assembly code.

Tested with Python 3.12.10
"""

import argparse
import jax
from jax import export
import jax.numpy as jnp
from flax import nnx
from flax import linen as nn

"""
These are technically private implementation details and could change at any time, but
this is the recommended way to get pretty-printed StableHLO output from JAX.

See: https://openxla.org/stablehlo/tutorials/jax-export#define_get_stablehlo_asm_to_help_with_mlir_printing
"""
from jax._src.interpreters import mlir as jax_mlir
from jax._src.lib.mlir import ir


def get_stablehlo_asm(module_str):
    """Returns prettyprint of StableHLO module as generic print."""
    with jax_mlir.make_ir_context():
        stablehlo_module = ir.Module.parse(module_str, context=jax_mlir.make_ir_context())
        return stablehlo_module.operation.get_asm(print_generic_op_form=True, enable_debug_info=False)
def do_export(model_jit, outfile, *args, **kwargs):
    """Exports model `model_jit` to `outfile` by running it with `args` and `kwargs`."""
    exported = export.export(model_jit)(*args, **kwargs)
    hlo = get_stablehlo_asm(exported.mlir_module())
    with open(outfile, 'w') as f:
        f.write(hlo)

def do_export_jax(model, sample_inputs, outfile):
    """Exports a plain JAX model"""
    model_jit = jax.jit(model)
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in sample_inputs]
    do_export(model_jit, outfile, *input_shapes)
def do_export_linen(model, sample_inputs, outfile):
    """Exports a model that uses the Flax Linen API.
    
    These models are more complicated as they use a purely functional API, so
    the user must pass the model's parameters explicitly. That means before
    exporting, we must first initialize some parameters and then pass
    them to the export function.
    """
    params = model.init(jax.random.key(42), *sample_inputs)
    model_jit = jax.jit(model.apply)
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in sample_inputs]
    do_export(model_jit, outfile, params, *input_shapes)
def do_export_transformers_linen(model, sample_inputs, outfile):
    """Exports a model from the Transformers library that uses the Flax Linen API.
    
    Exporting these models is similar to exporting a regular Linen model, except the
    parameters are initialized using the model's `init_weights` method and we call
    the model normally instead of using `apply`.
    """
    shapes = [x.shape for x in sample_inputs]
    params = model.init_weights(jax.random.key(42), *shapes)
    model_jit = jax.jit(model)
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in sample_inputs]
    do_export(model_jit, outfile, *input_shapes, params=params)

@jax.jit
def forward(graphdef: nnx.GraphDef, state: nnx.State, x) -> tuple[jax.Array, nnx.State]:
    m = nnx.merge(graphdef, state)
    out = m(*x)
    _, state = nnx.split(m)
    return out
def do_export_nnx(model, sample_inputs, outfile):
    """Exports a Flax model that uses the newer nnx API.
    
    These models include their state in the model itself, but if we naively
    export them the weights will be included as constants in HLO instead of
    arguments. To avoid this, we use the `nnx.split` function to separate
    the model's graph definition from its state, and then we pass the state
    explicitly when we export the model.
    """

    graphdef, state = nnx.split(model)
    input_shapes = [jax.ShapeDtypeStruct(x.shape, x.dtype) for x in sample_inputs]
    do_export(forward, outfile, graphdef, state, input_shapes)

def main():
    parser = argparse.ArgumentParser(description='Export JAX model to StableHLO format and print assembly code')
    parser.add_argument('model',
                        choices=['add', 'matmul', 'mlp', 'attention', 'attention-linen',
                                 'resnet', 'resnet18', 'llama3-tiny', 'llama3'],
                        help='Name of the JAX model to export')
    parser.add_argument('outfile', help='Output file to save the StableHLO assembly code')
    
    args = parser.parse_args()

    modelName = args.model
    outfile = args.outfile
        
    if modelName == "add":
        def add(x,y):
            return jnp.add(x,y)
        sample_inputs = [jnp.float32((20,10)), jnp.float32((20,10))]
        do_export_jax(add, sample_inputs, outfile)
    elif modelName == "matmul":
        def matmul(x,y):
            return jnp.matmul(x,y)
        sample_inputs = [jnp.float32((20,10)), jnp.float32((10,30))]
        do_export_jax(matmul, sample_inputs, outfile)
    elif modelName == "mlp":
        layer = nnx.Linear(4,2, rngs=nnx.Rngs(0))
        model = layer
        sample_inputs = [jax.random.uniform(jax.random.key(0),(1,4))]
        do_export_nnx(model, sample_inputs, outfile)
    elif modelName == "attention":
        # This is a MultiHeadAttention layer that provides an example of how to export a
        # model that uses the nnx API.
        layer = nnx.MultiHeadAttention(num_heads=8, in_features=5, qkv_features=16,
                                       decode=False, rngs=nnx.Rngs(0))
        keys = jax.random.split(jax.random.key(0), 3)
        shape = (4, 3, 2, 5)
        sample_inputs = [jax.random.uniform(keys[i], shape) for i in range(3)]
        model = layer
        do_export_nnx(model, sample_inputs, outfile)
    elif modelName == "attention-linen":
        # An attention layer using the Linen API.
        # This attention layer should be mostly identical to the nnx version, but
        # including it here serves as an example for exporting a Flax Linen model.
        layer = nn.MultiHeadDotProductAttention(num_heads=8,
                                                qkv_features=16,
                                       decode=False,
                                       )
        keys = jax.random.split(jax.random.key(0), 3)
        shape = (4, 3, 2, 5)
        sample_inputs = [jax.random.uniform(keys[i], shape) for i in range(3)]
        model = layer
        do_export_linen(model, sample_inputs, outfile)
    elif modelName == "resnet" or modelName == "resnet18":
        from transformers import FlaxResNetModel, ResNetConfig
        if modelName == "resnet18":
            # A ResNet-18 configuration for testing purposes.
            config = ResNetConfig(
                architectures= [ "ResNetForImageClassification" ],
                depths=[ 2, 2, 2, 2 ],
                downsample_in_first_stage=False,
                embedding_size=64,
                hidden_act="relu",
                hidden_sizes=[ 64, 128, 256, 512 ],
                layer_type="basic",
                num_channels=3,
                torch_dtype="float32",
            )
            model = FlaxResNetModel(config)
        else:
            # A ResNet configuration that matches the default ResNet-50.
            model = FlaxResNetModel(ResNetConfig())

        sample_input = jax.random.uniform(jax.random.key(0), (1, 3, 224, 224))

        # The external interface expects (batch, channels, height, width), but
        # here we're using the internal interface which expects (batch, height, width, channels).
        transposed_shape = sample_input.transpose((0, 2, 3, 1)).shape
        params = model.init_weights(jax.random.key(42), transposed_shape)

        model_jit = jax.jit(model)
        shape = jax.ShapeDtypeStruct(sample_input.shape, sample_input.dtype)
        do_export(model_jit, outfile, shape, params=params)
    elif modelName == "llama3-tiny":
        # A smaller llama3 model for testing purposes.
        from transformers import FlaxLlamaForCausalLM, LlamaConfig
        configuration = LlamaConfig(
            vocab_size = 32000,
            hidden_size = 4,
            intermediate_size = 14,
            num_hidden_layers = 2,
            num_attention_heads = 2,
            max_position_embeddings = 2048)
        model = FlaxLlamaForCausalLM(configuration)
        sample_inputs = [jnp.asarray([[    1, 16644, 31844,   629,  3924,   322, 14138]])]
        do_export_transformers_linen(model, sample_inputs, outfile)
    elif modelName == "llama3":
        # The default llama3-7B model.
        from transformers import FlaxLlamaForCausalLM, LlamaConfig
        configuration = LlamaConfig()
        model = FlaxLlamaForCausalLM(configuration)
        sample_inputs = [jnp.asarray([[    1, 16644, 31844,   629,  3924,   322, 14138]])]
        do_export_transformers_linen(model, sample_inputs, outfile)
    else:
        raise Exception("Unrecognized model: {}".format(model))
    
if __name__ == "__main__":
    main()
