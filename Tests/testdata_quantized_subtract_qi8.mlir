"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2, -3, 0]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %1 = "stablehlo.constant"() <{value = dense<[-8, -1, 2, -3, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<5xi4>, tensor<5xi4>) -> tensor<5xi4>
    "check.expect_eq_const"(%2) <{value = dense<[-8, 2, 0, 0, -7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 2]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %1 = "stablehlo.constant"() <{value = dense<[15, 3]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xui4>, tensor<2xui4>) -> tensor<2xui4>
    "check.expect_eq_const"(%2) <{value = dense<[1, 15]> : tensor<2xui4>}> : (tensor<2xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 8, -9, 0]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %1 = "stablehlo.constant"() <{value = dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%2) <{value = dense<[-128, 2, 0, 0, -127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %1 = "stablehlo.constant"() <{value = dense<[255, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
    "check.expect_eq_const"(%2) <{value = dense<[1, 0]> : tensor<2xui8>}> : (tensor<2xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 128, -129, 0]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %1 = "stablehlo.constant"() <{value = dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%2) <{value = dense<[-32768, 2, 0, 0, -32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %1 = "stablehlo.constant"() <{value = dense<[65535, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
    "check.expect_eq_const"(%2) <{value = dense<[1, 0]> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%2) <{value = dense<[-2147483648, 2, 0, 0, -2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %1 = "stablehlo.constant"() <{value = dense<[4294967295, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "check.expect_eq_const"(%2) <{value = dense<[1, 0]> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%2) <{value = dense<[-9223372036854775808, 2, 0, 0, -9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<[18446744073709551615, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "check.expect_eq_const"(%2) <{value = dense<[1, 0]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_bf16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0x7F80, 0xFF80, 0x7F80, 9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 0x7F80, 0xFF80, 0xFF80, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<11xbf16>, tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -2.011720e-01, 0.000000e+00, 0x7F80, 0x7FC0, 0x7FC0, 0x7F80, 1.836710e-40]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<11xf16>, tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -2.000730e-01, 0.000000e+00, 0x7C00, 0x7E00, 0x7E00, 0x7C00, 1.192090e-07]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> tensor<11xf32>, sym_name = "main"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %2 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 0.000000e+00, -0.624969602, -0.203303367, 0.000000e+00, 0.9562788, 0.000000e+00, 0.000000e+00, 0.9562788, 0.000000e+00]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %3 = "stablehlo.uniform_quantize"(%1) : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>
    %4 = "stablehlo.uniform_quantize"(%0) : (tensor<11xf32>) -> tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>
    %5 = "stablehlo.subtract"(%4, %3) : (tensor<11x!quant.uniform<i8:f32, 0.0039172410964965817:-128>>, tensor<11x!quant.uniform<i8:f32, 0.0039188104517319626:-128>>) -> tensor<11x!quant.uniform<i8:f32, 0.007529754264681947>>
    %6 = "stablehlo.uniform_dequantize"(%5) : (tensor<11x!quant.uniform<i8:f32, 0.007529754264681947>>) -> tensor<11xf32>
    %7 = "stablehlo.custom_call"(%2, %6) <{call_target_name = "check.eq"}> : (tensor<11xf32>, tensor<11xf32>) -> tensor<i1>
    "func.return"(%6) : (tensor<11xf32>) -> ()
  }) : () -> ()
}) {jax.uses_shape_polymorphism = true} : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<11xf64>, tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -0.19999999999999998, 0.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000, 0x7FF8000000000000, 0x7FF0000000000000, 9.881310e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %1 = "stablehlo.subtract"(%0, %0) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.subtract"(%0, %0) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

