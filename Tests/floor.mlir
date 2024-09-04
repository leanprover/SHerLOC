"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_bf16"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0xFF80, -2.500000e+00, -9.183550e-41, -0.000000e+00, 0.000000e+00, 9.183550e-41, 2.500000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>}> : () -> tensor<9xbf16>
    %7 = "stablehlo.floor"(%6) : (tensor<9xbf16>) -> tensor<9xbf16>
    "check.expect_almost_eq_const"(%7) <{value = dense<[0xFF80, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7F80, 0x7FC0]> : tensor<9xbf16>}> : (tensor<9xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_f16"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[0xFC00, -2.500000e+00, -5.960460e-08, -0.000000e+00, 0.000000e+00, 5.960460e-08, 2.500000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>}> : () -> tensor<9xf16>
    %5 = "stablehlo.floor"(%4) : (tensor<9xf16>) -> tensor<9xf16>
    "check.expect_almost_eq_const"(%5) <{value = dense<[0xFC00, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7C00, 0x7E00]> : tensor<9xf16>}> : (tensor<9xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_f32"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[0xFF800000, -2.500000e+00, -1.401300e-45, -0.000000e+00, 0.000000e+00, 1.401300e-45, 2.500000e+00, 0x7F800000, 0x7FC00000]> : tensor<9xf32>}> : () -> tensor<9xf32>
    %3 = "stablehlo.floor"(%2) : (tensor<9xf32>) -> tensor<9xf32>
    "check.expect_almost_eq_const"(%3) <{value = dense<[0xFF800000, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7F800000, 0x7FC00000]> : tensor<9xf32>}> : (tensor<9xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "floor_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000000, -2.500000e+00, -4.940660e-324, -0.000000e+00, 0.000000e+00, 4.940660e-324, 2.500000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : () -> tensor<9xf64>
    %1 = "stablehlo.floor"(%0) : (tensor<9xf64>) -> tensor<9xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0xFFF0000000000000, -3.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000]> : tensor<9xf64>}> : (tensor<9xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

