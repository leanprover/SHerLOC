"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si4"}> ({
    %45 = "stablehlo.constant"() <{value = dense<[0, 1, 2, -3, 0]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %46 = "stablehlo.constant"() <{value = dense<[-8, -1, 2, -3, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %47 = "stablehlo.subtract"(%45, %46) : (tensor<5xi4>, tensor<5xi4>) -> tensor<5xi4>
    "check.expect_eq_const"(%47) <{value = dense<[-8, 2, 0, 0, -7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui4"}> ({
    %42 = "stablehlo.constant"() <{value = dense<[0, 2]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %43 = "stablehlo.constant"() <{value = dense<[15, 3]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %44 = "stablehlo.subtract"(%42, %43) : (tensor<2xui4>, tensor<2xui4>) -> tensor<2xui4>
    "check.expect_eq_const"(%44) <{value = dense<[1, 15]> : tensor<2xui4>}> : (tensor<2xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si8"}> ({
    %39 = "stablehlo.constant"() <{value = dense<[0, 1, 8, -9, 0]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %40 = "stablehlo.constant"() <{value = dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %41 = "stablehlo.subtract"(%39, %40) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%41) <{value = dense<[-128, 2, 0, 0, -127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui8"}> ({
    %36 = "stablehlo.constant"() <{value = dense<[0, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %37 = "stablehlo.constant"() <{value = dense<[255, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %38 = "stablehlo.subtract"(%36, %37) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
    "check.expect_eq_const"(%38) <{value = dense<[1, 0]> : tensor<2xui8>}> : (tensor<2xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si16"}> ({
    %33 = "stablehlo.constant"() <{value = dense<[0, 1, 128, -129, 0]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %34 = "stablehlo.constant"() <{value = dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %35 = "stablehlo.subtract"(%33, %34) : (tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%35) <{value = dense<[-32768, 2, 0, 0, -32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui16"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[0, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %31 = "stablehlo.constant"() <{value = dense<[65535, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %32 = "stablehlo.subtract"(%30, %31) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
    "check.expect_eq_const"(%32) <{value = dense<[1, 0]> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si32"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %28 = "stablehlo.constant"() <{value = dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %29 = "stablehlo.subtract"(%27, %28) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%29) <{value = dense<[-2147483648, 2, 0, 0, -2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui32"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[0, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %25 = "stablehlo.constant"() <{value = dense<[4294967295, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %26 = "stablehlo.subtract"(%24, %25) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "check.expect_eq_const"(%26) <{value = dense<[1, 0]> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_si64"}> ({
    %21 = "stablehlo.constant"() <{value = dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %22 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %23 = "stablehlo.subtract"(%21, %22) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%23) <{value = dense<[-9223372036854775808, 2, 0, 0, -9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_ui64"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[0, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %19 = "stablehlo.constant"() <{value = dense<[18446744073709551615, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %20 = "stablehlo.subtract"(%18, %19) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "check.expect_eq_const"(%20) <{value = dense<[1, 0]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_bf16"}> ({
    %15 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0x7F80, 0xFF80, 0x7F80, 9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %16 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 0x7F80, 0xFF80, 0xFF80, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %17 = "stablehlo.subtract"(%15, %16) : (tensor<11xbf16>, tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%17) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -2.011720e-01, 0.000000e+00, 0x7F80, 0x7FC0, 0x7FC0, 0x7F80, 1.836710e-40]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_f16"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %13 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %14 = "stablehlo.subtract"(%12, %13) : (tensor<11xf16>, tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%14) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -2.000730e-01, 0.000000e+00, 0x7C00, 0x7E00, 0x7E00, 0x7C00, 1.192090e-07]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_f32"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %10 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %11 = "stablehlo.subtract"(%9, %10) : (tensor<11xf32>, tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%11) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -0.200000018, 0.000000e+00, 0x7F800000, 0x7FC00000, 0x7FC00000, 0x7F800000, 2.802600e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_f64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %7 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %8 = "stablehlo.subtract"(%6, %7) : (tensor<11xf64>, tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%8) <{value = dense<[0.000000e+00, 0.000000e+00, -6.000000e+00, -6.250000e-01, -0.19999999999999998, 0.000000e+00, 0x7FF0000000000000, 0x7FF8000000000000, 0x7FF8000000000000, 0x7FF0000000000000, 9.881310e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_c64"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %4 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %5 = "stablehlo.subtract"(%3, %4) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%5) <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "subtract_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %2 = "stablehlo.subtract"(%0, %1) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%2) <{value = dense<(0.000000e+00,0.000000e+00)> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

