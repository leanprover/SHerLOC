"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[5.000000e-01, 1.200000e+00, 1.500000e+00, 1.700000e+00, 2.500000e+00], [-5.000000e-01, -1.200000e+00, -1.500000e+00, -1.700000e+00, -2.500000e+00]]> : tensor<2x5xf32>}> : () -> tensor<2x5xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[0.996078491, 0.996078491, 0.996078491, 0.996078491, 0.996078491], [-0.996078491, -0.996078491, -0.996078491, -0.996078491, -0.996078491]]> : tensor<2x5xf32>}> : () -> tensor<2x5xf32>
    %2 = "stablehlo.uniform_quantize"(%0) : (tensor<2x5xf32>) -> tensor<2x5x!quant.uniform<i8:f32, 0.0078306291617599184>>
    %3 = "stablehlo.round_nearest_afz"(%2) : (tensor<2x5x!quant.uniform<i8:f32, 0.0078306291617599184>>) -> tensor<2x5x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>
    %4 = "stablehlo.uniform_dequantize"(%3) : (tensor<2x5x!quant.uniform<i8:f32, 0.0078431372549019607:-1>>) -> tensor<2x5xf32>
    %5 = "stablehlo.custom_call"(%1, %4) <{call_target_name = "check.eq"}> : (tensor<2x5xf32>, tensor<2x5xf32>) -> tensor<i1>
    "func.return"(%5) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

