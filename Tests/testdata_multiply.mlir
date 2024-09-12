"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 8, -9, 0]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %1 = "stablehlo.constant"() <{value = dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%2) <{value = dense<[0, -1, 64, 81, 0]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 16, 16]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %1 = "stablehlo.constant"() <{value = dense<[255, 16, 17]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%2) <{value = dense<[0, 0, 16]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 128, -129, 0]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %1 = "stablehlo.constant"() <{value = dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%2) <{value = dense<[0, -1, 16384, 16641, 0]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %1 = "stablehlo.constant"() <{value = dense<[65535, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
    "check.expect_eq_const"(%2) <{value = dense<0> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %1 = "stablehlo.constant"() <{value = dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%2) <{value = dense<[0, -1, 1073741824, 1073807361, 0]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %1 = "stablehlo.constant"() <{value = dense<[4294967295, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "check.expect_eq_const"(%2) <{value = dense<0> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%2) <{value = dense<[0, -1, 4611686018427387904, 4611686022722355201, 0]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<[18446744073709551615, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "check.expect_eq_const"(%2) <{value = dense<0> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_bf16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 2.658460e+36, 2.658460e+36, -2.658460e+36, 2.658460e+36, 9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 2.658460e+36, -2.658460e+36, -2.658460e+36, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<11xbf16>, tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.015140e-02, 9.875000e+00, 0.000000e+00, 0x7F80, 0x7F80, 0xFF80, -0.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<11xf16>, tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 2.999880e-02, 9.867180e+00, 0x7E00, 0x7C00, 0x7C00, 0xFC00, -0.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<11xf32>, tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 0.0300000012, 9.86960506, 0x7FC00000, 0x7F800000, 0x7F800000, 0xFF800000, -0.000000e+00]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<11xf64>, tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%2) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.000000e-02, 9.869604401089358, 0x7FF8000000000000, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, -0.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %1 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

