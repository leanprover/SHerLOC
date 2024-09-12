"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<1xf32>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %1 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "stablehlo.uniform_quantize"(%0) : (tensor<1xf32>) -> tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>
    %3 = "stablehlo.divide"(%2, %2) : (tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>, tensor<1x!quant.uniform<i8:f32, 0.0039041399955749511:-128>>) -> tensor<1x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>
    %4 = "stablehlo.uniform_dequantize"(%3) : (tensor<1x!quant.uniform<i8:f32, 0.0039215686274509803:-128>>) -> tensor<1xf32>
    %5 = "stablehlo.custom_call"(%1, %4) <{call_target_name = "check.eq"}> : (tensor<1xf32>, tensor<1xf32>) -> tensor<i1>
    "func.return"(%4) : (tensor<1xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

