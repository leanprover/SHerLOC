"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_bf16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFF80, -2.500000e+00, -9.183550e-41, -0.000000e+00, 0.000000e+00, 9.183550e-41, 2.500000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>}> : () -> tensor<9xbf16>
    %1 = "stablehlo.floor"(%0) : (tensor<9xbf16>) -> tensor<9xbf16>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0xFF80, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>}> : (tensor<9xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFC00, -2.500000e+00, -5.960460e-08, -0.000000e+00, 0.000000e+00, 5.960460e-08, 2.500000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    %1 = "stablehlo.floor"(%0) : (tensor<9xf16>) -> tensor<9xf16>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0xFC00, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>}> : (tensor<9xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<9xf32>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFF800000, -2.500000e+00, -1.401300e-45, -0.000000e+00, 0.000000e+00, 1.401300e-45, 2.500000e+00, 0x7F800000, 0x7FC00000]> : tensor<9xf32>}> : () -> tensor<9xf32>
    %1 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<9xf32>}> : () -> tensor<9xf32>
    %2 = "stablehlo.uniform_quantize"(%0) : (tensor<9xf32>) -> tensor<9x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>
    %3 = "stablehlo.floor"(%2) : (tensor<9x!quant.uniform<i8:f32, 0.0039132908278820561:-128>>) -> tensor<9x!quant.uniform<i8:f32, 7.843137254901961E-9>>
    %4 = "stablehlo.uniform_dequantize"(%3) : (tensor<9x!quant.uniform<i8:f32, 7.843137254901961E-9>>) -> tensor<9xf32>
    %5 = "stablehlo.custom_call"(%1, %4) <{call_target_name = "check.eq"}> : (tensor<9xf32>, tensor<9xf32>) -> tensor<i1>
    "func.return"(%4) : (tensor<9xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000000, -2.500000e+00, -4.940660e-324, -0.000000e+00, 0.000000e+00, 4.940660e-324, 2.500000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : () -> tensor<9xf64>
    %1 = "stablehlo.floor"(%0) : (tensor<9xf64>) -> tensor<9xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0xFFF0000000000000, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : (tensor<9xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

