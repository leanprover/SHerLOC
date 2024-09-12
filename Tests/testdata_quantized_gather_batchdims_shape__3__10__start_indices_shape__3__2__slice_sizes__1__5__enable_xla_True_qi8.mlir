"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0], [1, 8], [2, 0]]> : tensor<3x2xi32>}> : () -> tensor<3x2xi32>
    %1 = "stablehlo.constant"() <{value = dense<[[-0.786750376, -0.429459691, -2.42140698, 0.0205181241, -0.394114822, -2.58621716, -1.07088399, 3.29197717, -3.44814229, -0.25225088], [1.27824605, -2.20641971, 1.13592541, 2.04215646, -1.61209357, 3.22753859, -1.28165495, 3.17407966, 2.02299929, 2.47564316], [0.905838906, 3.71254492, 1.97064459, 3.77753663, 1.49392521, 4.79311323, 3.70975041, -1.04468286, 3.31870532, 1.45112896]]> : tensor<3x10xf32>}> : () -> tensor<3x10xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.000000e+00, 0.000000e+00, 0.000000e+00, 0.0195749179, 0.000000e+00], [0.998320758, 0.000000e+00, 0.998320758, 0.998320758, 0.998320758], [0.904361188, 0.998320758, 0.998320758, 0.998320758, 0.998320758]]> : tensor<3x5xf32>}> : () -> tensor<3x5xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<3x10xf32>) -> tensor<3x10x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 5>}> : (tensor<3x10x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>, tensor<3x2xi32>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<3x5x!quant.uniform<i8:f32, 0.0039189298947652183:-128>>) -> tensor<3x5x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<3x5x!quant.uniform<i8:f32, 0.0039149835997936769:-128>>) -> tensor<3x5xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<3x5xf32>, tensor<3x5xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

