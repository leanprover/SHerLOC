"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[3.21390557, 2.41580057, -0.537137687, -1.02739561], [1.35577726, -2.48765302, 3.99296689, 3.09424305], [-2.41700459, -1.63692343, -6.27982473, 2.19841671]]]> : tensor<1x3x4xf32>}> : () -> tensor<1x3x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[-0.227901727, -0.527938426, 2.04744601], [0.46251452, -2.02832699, -5.24830675], [-1.4312098, -5.60030842, 6.4256258], [-0.754353106, -7.84103918, -1.26745594]]]> : tensor<1x4x3xf32>}> : () -> tensor<1x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<-0.372528762> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %5 = "stablehlo.dot_general"(%4, %3) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>}> : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

