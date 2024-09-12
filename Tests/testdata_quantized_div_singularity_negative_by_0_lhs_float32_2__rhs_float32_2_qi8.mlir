"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<2xf32>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<-1.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<2xf32>}> : () -> tensor<2xf32>
    %2 = "stablehlo.constant"() <{value = dense<91.4060898> : tensor<2xf32>}> : () -> tensor<2xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>
    %5 = "stablehlo.divide"(%4, %3) : (tensor<2x!quant.uniform<i8:f32, 0.0039068778355916345:-128>>, tensor<2x!quant.uniform<i8:f32, 0.0039059886745378084:-128>>) -> tensor<2x!quant.uniform<i8:f32, 0.71411007151884187:-128>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<2x!quant.uniform<i8:f32, 0.71411007151884187:-128>>) -> tensor<2xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<2xf32>, tensor<2xf32>) -> tensor<i1>
    "func.return"(%6) : (tensor<2xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

