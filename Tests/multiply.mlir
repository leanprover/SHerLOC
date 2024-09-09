"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si8"}> ({
    %42 = "stablehlo.constant"() <{value = dense<[0, 1, 8, -9, 0]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %43 = "stablehlo.constant"() <{value = dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %44 = "stablehlo.multiply"(%42, %43) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%44) <{value = dense<[0, -1, 64, 81, 0]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui8"}> ({
    %39 = "stablehlo.constant"() <{value = dense<[0, 16, 16]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %40 = "stablehlo.constant"() <{value = dense<[255, 16, 17]> : tensor<3xui8>}> : () -> tensor<3xui8>
    %41 = "stablehlo.multiply"(%39, %40) : (tensor<3xui8>, tensor<3xui8>) -> tensor<3xui8>
    "check.expect_eq_const"(%41) <{value = dense<[0, 0, 16]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si16"}> ({
    %36 = "stablehlo.constant"() <{value = dense<[0, 1, 128, -129, 0]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %37 = "stablehlo.constant"() <{value = dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %38 = "stablehlo.multiply"(%36, %37) : (tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%38) <{value = dense<[0, -1, 16384, 16641, 0]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui16"}> ({
    %33 = "stablehlo.constant"() <{value = dense<[0, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %34 = "stablehlo.constant"() <{value = dense<[65535, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %35 = "stablehlo.multiply"(%33, %34) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
    "check.expect_eq_const"(%35) <{value = dense<0> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si32"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %31 = "stablehlo.constant"() <{value = dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %32 = "stablehlo.multiply"(%30, %31) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%32) <{value = dense<[0, -1, 1073741824, 1073807361, 0]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui32"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[0, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %28 = "stablehlo.constant"() <{value = dense<[4294967295, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %29 = "stablehlo.multiply"(%27, %28) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "check.expect_eq_const"(%29) <{value = dense<0> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_si64"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %25 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %26 = "stablehlo.multiply"(%24, %25) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%26) <{value = dense<[0, -1, 4611686018427387904, 4611686022722355201, 0]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_ui64"}> ({
    %21 = "stablehlo.constant"() <{value = dense<[0, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %22 = "stablehlo.constant"() <{value = dense<[18446744073709551615, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %23 = "stablehlo.multiply"(%21, %22) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "check.expect_eq_const"(%23) <{value = dense<0> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_i1"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %19 = "stablehlo.constant"() <{value = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %20 = "stablehlo.multiply"(%18, %19) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%20) <{value = dense<[false, false, false, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_bf16"}> ({
    %15 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 2.658460e+36, 2.658460e+36, -2.658460e+36, 2.658460e+36, 9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %16 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 2.658460e+36, -2.658460e+36, -2.658460e+36, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %17 = "stablehlo.multiply"(%15, %16) : (tensor<11xbf16>, tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%17) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.015140e-02, 9.875000e+00, 0.000000e+00, 0x7F80, 0x7F80, 0xFF80, -0.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f16"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %13 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %14 = "stablehlo.multiply"(%12, %13) : (tensor<11xf16>, tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%14) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 2.999880e-02, 9.867180e+00, 0x7E00, 0x7C00, 0x7C00, 0xFC00, -0.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f32"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %10 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %11 = "stablehlo.multiply"(%9, %10) : (tensor<11xf32>, tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%11) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 0.0300000012, 9.86960506, 0x7FC00000, 0x7F800000, 0x7F800000, 0xFF800000, -0.000000e+00]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_f64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %7 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %8 = "stablehlo.multiply"(%6, %7) : (tensor<11xf64>, tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%8) <{value = dense<[0.000000e+00, 0.000000e+00, 7.000000e+00, 9.375000e-02, 3.000000e-02, 9.869604401089358, 0x7FF8000000000000, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, -0.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_c64"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %5 = "stablehlo.multiply"(%3, %4) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%5) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "mul_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.multiply"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<[(-4.000000e+00,7.500000e+00), (2.600000e+01,8.250000e+01)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

