"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1]]> : tensor<1x2xi32>}> : () -> tensor<1x2xi32>
    %1 = "stablehlo.constant"() <{value = dense<[[-2.72349977, -0.208018199]]> : tensor<1x2xf32>}> : () -> tensor<1x2xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<1x2xf32>) -> tensor<1x2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<collapsed_slice_dims = [0, 1], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 1>}> : (tensor<1x2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>, tensor<1x2xi32>) -> tensor<1x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<1x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.0038181253508025523:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<1x!quant.uniform<i8:f32, 0.0038181253508025523:-128>>) -> tensor<1xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

