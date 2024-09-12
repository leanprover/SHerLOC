"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1.000000e+00], [2.000000e+00], [5.000000e+00], [6.000000e+00]], [[3.000000e+00], [4.000000e+00], [7.000000e+00], [8.000000e+00]], [[1.000000e+01], [1.100000e+01], [1.400000e+01], [1.500000e+01]], [[1.200000e+01], [1.300000e+01], [1.600000e+01], [1.700000e+01]]]]> : tensor<1x4x4x1xf32>}> : () -> tensor<1x4x4x1xf32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<3x3x1x1xf32>}> : () -> tensor<3x3x1x1xf32>
    %2 = "stablehlo.constant"() <{value = dense<2.47494364> : tensor<1x2x2x1xf32>}> : () -> tensor<1x2x2x1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<3x3x1x1xf32>) -> tensor<3x3x1x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x4x4x1xf32>) -> tensor<1x4x4x1x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %5 = "stablehlo.convolution"(%4, %3) <{batch_group_count = 1 : i64, dimension_numbers = #stablehlo.conv<[b, 0, 1, f]x[0, 1, i, o]->[b, 0, 1, f]>, feature_group_count = 1 : i64, lhs_dilation = array<i64: 2, 2>, window_strides = array<i64: 4, 4>}> : (tensor<1x4x4x1x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<3x3x1x1x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>) -> tensor<1x2x2x1x!quant.uniform<i32:f32, 1.5350925350904778E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x2x2x1x!quant.uniform<i32:f32, 1.5350925350904778E-5>>) -> tensor<1x2x2x1x!quant.uniform<i8:f32, 0.0097056613248937273:-128>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x2x2x1x!quant.uniform<i8:f32, 0.0097056613248937273:-128>>) -> tensor<1x2x2x1xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1x2x2x1xf32>, tensor<1x2x2x1xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

