"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<3x1xi32>}> : () -> tensor<3x1xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-0.317652494, -0.498273045, -1.63233531, -0.124743178, 2.18847871, 1.92351472, 1.37014866, -3.42049432, -2.30765843, 2.53218222]> : tensor<10xf32>}> : () -> tensor<10xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<3x2xf32>}> : () -> tensor<3x2xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<10xf32>) -> tensor<10x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 2>}> : (tensor<10x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x1xi32>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<3x2x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<3x2x!quant.uniform<i8:f32, 0.0038690987755270567:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<3x2x!quant.uniform<i8:f32, 0.0038690987755270567:-128>>) -> tensor<3x2xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<3x2xf32>, tensor<3x2xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

