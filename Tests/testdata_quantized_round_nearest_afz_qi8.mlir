"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<i1>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2.500000e+00, 4.000000e-01, 5.000000e-01, 6.000000e-01, 2.500000e+00]> : tensor<5xf32>}> : () -> tensor<5xf32>
    %1 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<5xf32>}> : () -> tensor<5xf32>
    %2 = "stablehlo.uniform_quantize"(%0) : (tensor<5xf32>) -> tensor<5x!quant.uniform<i8:f32, 0.0038905945478701124:-1>>
    %3 = "stablehlo.round_nearest_afz"(%2) : (tensor<5x!quant.uniform<i8:f32, 0.0038905945478701124:-1>>) -> tensor<5x!quant.uniform<i8:f32, 7.843137254901961E-9>>
    %4 = "stablehlo.uniform_dequantize"(%3) : (tensor<5x!quant.uniform<i8:f32, 7.843137254901961E-9>>) -> tensor<5xf32>
    %5 = "stablehlo.custom_call"(%1, %4) <{call_target_name = "check.eq"}> : (tensor<5xf32>, tensor<5xf32>) -> tensor<i1>
    "func.return"(%5) : (tensor<i1>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

