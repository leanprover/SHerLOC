"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[2.59234142, -2.35737705, 3.07461166, -5.77705336], [-3.64460349, 0.689637601, -0.0876942202, 1.62593222], [-4.65739489, 0.247004092, -1.08101177, -0.238710642]]]> : tensor<1x3x4xf32>}> : () -> tensor<1x3x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.64053726, 1.4618907, -0.867068588], [2.86438012, -2.70172548, -0.4580172], [0.140140817, -0.666462898, -7.101100e-01], [1.02142382, 0.236523077, 0.760420739]]]> : tensor<1x4x3xf32>}> : () -> tensor<1x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<-0.558793128> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %5 = "stablehlo.dot_general"(%4, %3) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGHEST>, #stablehlo<precision HIGHEST>]}> : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

