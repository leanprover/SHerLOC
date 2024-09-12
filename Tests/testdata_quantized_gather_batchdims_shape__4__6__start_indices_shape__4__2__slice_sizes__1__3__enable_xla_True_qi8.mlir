"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1], [1, 2], [2, 3], [3, 2]]> : tensor<4x2xi32>}> : () -> tensor<4x2xi32>
    %1 = "stablehlo.constant"() <{value = dense<[[-2.82557797, 2.39072633, 1.59782159, 5.14471102, -0.118122488, 1.23312056], [-1.81219053, -2.04905701, 2.10215306, -1.29667866, -0.0825303718, 1.88295043], [2.51706767, 0.0771943628, 2.18911791, -0.366536409, -2.39656186, 0.698230087], [2.96748114, 0.137859881, 1.44472873, -1.30095637, 1.24915195, -2.93037224]]> : tensor<4x6xf32>}> : () -> tensor<4x6xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.997595727, 0.997595727, 0.997595727], [0.997595727, 0.000000e+00, 0.000000e+00], [0.000000e+00, 0.000000e+00, 0.696360946], [0.997595727, 0.000000e+00, 0.997595727]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<4x6xf32>) -> tensor<4x6x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %4 = "stablehlo.gather"(%3, %0) <{dimension_numbers = #stablehlo.gather<offset_dims = [1], collapsed_slice_dims = [0], start_index_map = [0, 1], index_vector_dim = 1>, slice_sizes = array<i64: 1, 3>}> : (tensor<4x6x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<4x2xi32>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %5 = "stablehlo.uniform_quantize"(%4) : (tensor<4x3x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>) -> tensor<4x3x!quant.uniform<i8:f32, 0.0039121401076223335:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<4x3x!quant.uniform<i8:f32, 0.0039121401076223335:-128>>) -> tensor<4x3xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<4x3xf32>, tensor<4x3xf32>) -> tensor<i1>
    "func.return"(%7) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

