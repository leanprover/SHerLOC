"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.02716792, -0.84180355, -1.21495497, 1.22473526], [1.533764, 3.72483683, -3.63489795, 1.59403718], [-2.40816736, -1.31785822, -1.78571689, 0.202816486]]]> : tensor<1x3x4xf32>}> : () -> tensor<1x3x4xf32>
    %1 = "stablehlo.constant"() <{value = dense<[[[2.14761162, 2.50920868, -4.64897203], [1.72184336, -1.84444284, -1.14315307], [-1.28549385, 2.20468378, 1.76219583], [-3.49526167, -2.27656484, 2.42410755]]]> : tensor<1x4x3xf32>}> : () -> tensor<1x4x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<-0.558793128> : tensor<1xf32>}> : () -> tensor<1xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<1x4x3xf32>) -> tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<1x3x4xf32>) -> tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>
    %5 = "stablehlo.dot_general"(%4, %3) <{dot_dimension_numbers = #stablehlo.dot<lhs_batching_dimensions = [0], rhs_batching_dimensions = [0], lhs_contracting_dimensions = [2, 1], rhs_contracting_dimensions = [1, 2]>}> : (tensor<1x3x4x!quant.uniform<i8:f32, 0.0039166809297075458>>, tensor<1x4x3x!quant.uniform<i8:f32, 0.0039153145808799592>>) -> tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>
    %6 = "stablehlo.uniform_quantize"(%5) : (tensor<1x!quant.uniform<i32:f32, 1.533503795273843E-5>>) -> tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>
    %7 = "stablehlo.uniform_dequantize"(%6) : (tensor<1x!quant.uniform<i8:f32, 0.0051740104076909085:-20>>) -> tensor<1xf32>
    %8 = "stablehlo.custom_call"(%2, %7) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%8) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

