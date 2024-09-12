"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "uniform_quantize"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[4.000000e+00, 1.500000e+01]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "stablehlo.uniform_quantize"(%0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>
    %2 = "stablehlo.bitcast_convert"(%1) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>) -> tensor<2xi8>
    "check.expect_eq_const"(%2) <{value = dense<10> : tensor<2xi8>}> : (tensor<2xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
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

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "uniform_dequantize"}> ({
    %0 = "stablehlo.constant"() <{value = dense<10> : tensor<2xi8>}> : () -> tensor<2xi8>
    %1 = "stablehlo.bitcast_convert"(%0) : (tensor<2xi8>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>
    %2 = "stablehlo.uniform_dequantize"(%1) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>) -> tensor<2xf32>
    "check.expect_almost_eq_const"(%2) <{value = dense<[4.000000e+00, 1.500000e+01]> : tensor<2xf32>}> : (tensor<2xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "uniform_qdq"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[4.000000e+00, 1.500000e+01]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "stablehlo.uniform_quantize"(%0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>
    %2 = "stablehlo.uniform_dequantize"(%1) : (tensor<2x!quant.uniform<i8:f32:0, {1.000000e-01:-30,5.000000e-01:-20}>>) -> tensor<2xf32>
    "check.expect_almost_eq_const"(%2) <{value = dense<[4.000000e+00, 1.500000e+01]> : tensor<2xf32>}> : (tensor<2xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "quantized_add"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.000000e+00, 2.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %1 = "stablehlo.constant"() <{value = dense<[3.000000e+00, 4.000000e+00]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %2 = "stablehlo.uniform_quantize"(%0) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 1.000000e-01:-30>>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<2xf32>) -> tensor<2x!quant.uniform<i8:f32, 5.000000e-01:-20>>
    %4 = "stablehlo.add"(%2, %3) : (tensor<2x!quant.uniform<i8:f32, 1.000000e-01:-30>>, tensor<2x!quant.uniform<i8:f32, 5.000000e-01:-20>>) -> tensor<2x!quant.uniform<i8:f32, 5.000000e-01:-20>>
    %5 = "stablehlo.bitcast_convert"(%4) : (tensor<2x!quant.uniform<i8:f32, 5.000000e-01:-20>>) -> tensor<2xi8>
    "check.expect_eq_const"(%5) <{value = dense<[-12, -8]> : tensor<2xi8>}> : (tensor<2xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

