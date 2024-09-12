"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0], [2]]> : tensor<2x1xi32>}> : () -> tensor<2x1xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-4.45134878, -1.72604203, 5.85744715, -1.34194362, 0.103943698]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %2 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.984438836]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0], start_index_map = [0], index_vector_dim = 1>, slice_sizes = array<i64: 1>}> : (tensor<5x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>, tensor<2x1xi32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<2x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.0038605444571551155:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<2x!quant.uniform<i8:f32, 0.0038605444571551155:-128>>) -> tensor<2xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

