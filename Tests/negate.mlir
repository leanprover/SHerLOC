"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_si4"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %31 = "stablehlo.negate"(%30) : (tensor<5xi4>) -> tensor<5xi4>
    "check.expect_eq_const"(%31) <{value = dense<[-8, 1, 0, -1, -7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_ui4"}> ({
    %28 = "stablehlo.constant"() <{value = dense<[0, 8, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    %29 = "stablehlo.negate"(%28) : (tensor<3xui4>) -> tensor<3xui4>
    "check.expect_eq_const"(%29) <{value = dense<[0, 8, 1]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_si8"}> ({
    %26 = "stablehlo.constant"() <{value = dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %27 = "stablehlo.negate"(%26) : (tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%27) <{value = dense<[-128, 9, 0, -8, -127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_ui8"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[0, 16, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %25 = "stablehlo.negate"(%24) : (tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%25) <{value = dense<[0, 240, 1]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_si16"}> ({
    %22 = "stablehlo.constant"() <{value = dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %23 = "stablehlo.negate"(%22) : (tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%23) <{value = dense<[-32768, 129, 0, -128, -32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_ui16"}> ({
    %20 = "stablehlo.constant"() <{value = dense<[0, 256, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    %21 = "stablehlo.negate"(%20) : (tensor<3xui16>) -> tensor<3xui16>
    "check.expect_eq_const"(%21) <{value = dense<[0, 65280, 1]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_si32"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %19 = "stablehlo.negate"(%18) : (tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%19) <{value = dense<[-2147483648, 65537, 0, -65536, -2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_ui32"}> ({
    %16 = "stablehlo.constant"() <{value = dense<[0, 65536, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    %17 = "stablehlo.negate"(%16) : (tensor<3xui32>) -> tensor<3xui32>
    "check.expect_eq_const"(%17) <{value = dense<[0, 4294901760, 1]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_si64"}> ({
    %14 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %15 = "stablehlo.negate"(%14) : (tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%15) <{value = dense<[-9223372036854775808, 2147483649, 0, -2147483648, -9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_ui64"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    %13 = "stablehlo.negate"(%12) : (tensor<3xui64>) -> tensor<3xui64>
    "check.expect_eq_const"(%13) <{value = dense<[0, 18446744069414584320, 1]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_bf16"}> ({
    %10 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %11 = "stablehlo.negate"(%10) : (tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%11) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000980e-01, -3.140630e+00, 0xFF80, 0x7F80, 0xFFFF, -9.183550e-41, 9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_f16"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %9 = "stablehlo.negate"(%8) : (tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%9) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -9.997550e-02, -3.140630e+00, 0xFC00, 0x7C00, 0xFFFF, -5.960460e-08, 5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_f32"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %7 = "stablehlo.negate"(%6) : (tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%7) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.14159274, 0xFF800000, 0x7F800000, 0xFFFFFFFF, -1.401300e-45, 1.401300e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_f64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %5 = "stablehlo.negate"(%4) : (tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%5) <{value = dense<[-0.000000e+00, 0.000000e+00, -1.000000e+00, -1.250000e-01, -1.000000e-01, -3.1415926535897931, 0xFFF0000000000000, 0x7FF0000000000000, 0xFFFFFFFFFFFFFFFF, -4.940660e-324, 4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_c64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %3 = "stablehlo.negate"(%2) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%3) <{value = dense<[(-1.500000e+00,-2.500000e+00), (-3.500000e+00,-4.500000e+00)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "negate_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.negate"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(-1.500000e+00,-2.500000e+00), (-3.500000e+00,-4.500000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

