"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[3.29589868, 1.23571205, 1.77729869, -1.18627894], [-0.334112585, -0.701673209, 1.37458372, 1.33988047], [-1.23167276, -1.47567272, -5.089730e+00, 4.67841911]]]> : tensor<1x3x4xf32>}> : () -> tensor<1x3x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[-1.98014688, 6.42148352, -4.60602903], [0.598645091, -1.56104231, 1.48972511], [2.90995979, -0.593875766, -4.25129318], [-1.8781029, 1.97643542, 2.91169143]]]> : tensor<1x4x3xf32>}> : () -> tensor<1x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<0.760579526> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %5 = "stablehlo.dot_general"(%4, %3) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>, precision_config = [#stablehlo<precision HIGH>, #stablehlo<precision HIGH>]}> : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

