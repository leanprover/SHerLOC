"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "uniform_quantize"}> ({
    %0 = "stablehlo.constant"() <{value = dense<10> : tensor<2xi8>}> : () -> tensor<2xi8>
    %1 = "stablehlo.bitcast_convert"(%0) : (tensor<2xi8>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>
    %2 = "stablehlo.uniform_quantize"(%1) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-20,2.000000e-01:-30}>>
    %3 = "stablehlo.bitcast_convert"(%2) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-20,2.000000e-01:-30}>>) -> tensor<2xi8>
    "check.expect_eq_const"(%3) <{value = dense<[20, 45]> : tensor<2xi8>}> : (tensor<2xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

