"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 1]> : tensor<4xi2>}> : () -> tensor<4xi2>
    "check.expect_eq_const"(%0) <{value = dense<[-2, -1, 0, 1]> : tensor<4xi2>}> : (tensor<4xi2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2, 3]> : tensor<4xui2>}> : () -> tensor<4xui2>
    "check.expect_eq_const"(%0) <{value = dense<[0, 1, 2, 3]> : tensor<4xui2>}> : (tensor<4xui2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    "check.expect_eq_const"(%0) <{value = dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 8, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    "check.expect_eq_const"(%0) <{value = dense<[0, 8, 15]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    "check.expect_eq_const"(%0) <{value = dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 16, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    "check.expect_eq_const"(%0) <{value = dense<[0, 16, 255]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    "check.expect_eq_const"(%0) <{value = dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 256, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    "check.expect_eq_const"(%0) <{value = dense<[0, 256, 65535]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    "check.expect_eq_const"(%0) <{value = dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 65536, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    "check.expect_eq_const"(%0) <{value = dense<[0, 65536, 4294967295]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    "check.expect_eq_const"(%0) <{value = dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    "check.expect_eq_const"(%0) <{value = dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e3m4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.125000e+00, 0x7F, 0xFF, 1.562500e-02, -1.562500e-02]> : tensor<10xf8E3M4>}> : () -> tensor<10xf8E3M4>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.125000e+00, 0x7F, 0xFF, 1.562500e-02, -1.562500e-02]> : tensor<10xf8E3M4>}> : (tensor<10xf8E3M4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3b11_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 3.000000e+01, -3.000000e+01, 1.220700e-04, -1.220700e-04]> : tensor<10xf8E4M3B11FNUZ>}> : () -> tensor<10xf8E4M3B11FNUZ>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 3.000000e+01, -3.000000e+01, 1.220700e-04, -1.220700e-04]> : tensor<10xf8E4M3B11FNUZ>}> : (tensor<10xf8E4M3B11FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3>}> : () -> tensor<10xf8E4M3>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3>}> : (tensor<10xf8E4M3>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3_fn"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3FN>}> : () -> tensor<10xf8E4M3FN>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3FN>}> : (tensor<10xf8E4M3FN>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 2.400000e+02, -2.400000e+02, 9.765620e-04, -9.765620e-04]> : tensor<10xf8E4M3FNUZ>}> : () -> tensor<10xf8E4M3FNUZ>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 2.400000e+02, -2.400000e+02, 9.765620e-04, -9.765620e-04]> : tensor<10xf8E4M3FNUZ>}> : (tensor<10xf8E4M3FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e5m2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 0x7F, 0xFF, 1.525880e-05, -1.525880e-05]> : tensor<10xf8E5M2>}> : () -> tensor<10xf8E5M2>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 0x7F, 0xFF, 1.525880e-05, -1.525880e-05]> : tensor<10xf8E5M2>}> : (tensor<10xf8E5M2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e5m2_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : () -> tensor<10xf8E5M2FNUZ>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : (tensor<10xf8E5M2FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e5m2_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : () -> tensor<10xf8E5M2FNUZ>
    "check.expect_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : (tensor<10xf8E5M2FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_bf16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%0) <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

