"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "uniform_dequantize"}> ({
    %0 = "stablehlo.constant"() <{value = dense<10> : tensor<2xi8>}> : () -> tensor<2xi8>
    %1 = "stablehlo.bitcast_convert"(%0) : (tensor<2xi8>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>
    %2 = "stablehlo.uniform_dequantize"(%1) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>) -> tensor<2xf32>
    "check.expect_almost_eq_const"(%2) <{value = dense<[4.000000e+00, 1.500000e+01]> : tensor<2xf32>}> : (tensor<2xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

