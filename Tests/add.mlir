"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si2"}> ({
    %60 = "stablehlo.constant"() <{value = dense<[1, 0, -1, -2]> : tensor<4xi2>}> : () -> tensor<4xi2>
    %61 = "stablehlo.constant"() <{value = dense<[-1, 1, 1, 1]> : tensor<4xi2>}> : () -> tensor<4xi2>
    %62 = "stablehlo.add"(%60, %61) : (tensor<4xi2>, tensor<4xi2>) -> tensor<4xi2>
    "check.expect_eq_const"(%62) <{value = dense<[0, 1, 0, -1]> : tensor<4xi2>}> : (tensor<4xi2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui2"}> ({
    %57 = "stablehlo.constant"() <{value = dense<[0, 2]> : tensor<2xui2>}> : () -> tensor<2xui2>
    %58 = "stablehlo.constant"() <{value = dense<1> : tensor<2xui2>}> : () -> tensor<2xui2>
    %59 = "stablehlo.add"(%57, %58) : (tensor<2xui2>, tensor<2xui2>) -> tensor<2xui2>
    "check.expect_eq_const"(%59) <{value = dense<[1, 3]> : tensor<2xui2>}> : (tensor<2xui2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si4"}> ({
    %54 = "stablehlo.constant"() <{value = dense<[0, 1, 2, -3, 0]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %55 = "stablehlo.constant"() <{value = dense<[-8, -1, 2, -3, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    %56 = "stablehlo.add"(%54, %55) : (tensor<5xi4>, tensor<5xi4>) -> tensor<5xi4>
    "check.expect_eq_const"(%56) <{value = dense<[-8, 0, 4, -6, 7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui4"}> ({
    %51 = "stablehlo.constant"() <{value = dense<[0, 2]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %52 = "stablehlo.constant"() <{value = dense<[15, 3]> : tensor<2xui4>}> : () -> tensor<2xui4>
    %53 = "stablehlo.add"(%51, %52) : (tensor<2xui4>, tensor<2xui4>) -> tensor<2xui4>
    "check.expect_eq_const"(%53) <{value = dense<[15, 5]> : tensor<2xui4>}> : (tensor<2xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si8"}> ({
    %48 = "stablehlo.constant"() <{value = dense<[0, 1, 8, -9, 0]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %49 = "stablehlo.constant"() <{value = dense<[-128, -1, 8, -9, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    %50 = "stablehlo.add"(%48, %49) : (tensor<5xi8>, tensor<5xi8>) -> tensor<5xi8>
    "check.expect_eq_const"(%50) <{value = dense<[-128, 0, 16, -18, 127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui8"}> ({
    %45 = "stablehlo.constant"() <{value = dense<[0, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %46 = "stablehlo.constant"() <{value = dense<[255, 16]> : tensor<2xui8>}> : () -> tensor<2xui8>
    %47 = "stablehlo.add"(%45, %46) : (tensor<2xui8>, tensor<2xui8>) -> tensor<2xui8>
    "check.expect_eq_const"(%47) <{value = dense<[255, 32]> : tensor<2xui8>}> : (tensor<2xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si16"}> ({
    %42 = "stablehlo.constant"() <{value = dense<[0, 1, 128, -129, 0]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %43 = "stablehlo.constant"() <{value = dense<[-32768, -1, 128, -129, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    %44 = "stablehlo.add"(%42, %43) : (tensor<5xi16>, tensor<5xi16>) -> tensor<5xi16>
    "check.expect_eq_const"(%44) <{value = dense<[-32768, 0, 256, -258, 32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui16"}> ({
    %39 = "stablehlo.constant"() <{value = dense<[0, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %40 = "stablehlo.constant"() <{value = dense<[65535, 256]> : tensor<2xui16>}> : () -> tensor<2xui16>
    %41 = "stablehlo.add"(%39, %40) : (tensor<2xui16>, tensor<2xui16>) -> tensor<2xui16>
    "check.expect_eq_const"(%41) <{value = dense<[65535, 512]> : tensor<2xui16>}> : (tensor<2xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si32"}> ({
    %36 = "stablehlo.constant"() <{value = dense<[0, 1, 32768, -32769, 0]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %37 = "stablehlo.constant"() <{value = dense<[-2147483648, -1, 32768, -32769, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    %38 = "stablehlo.add"(%36, %37) : (tensor<5xi32>, tensor<5xi32>) -> tensor<5xi32>
    "check.expect_eq_const"(%38) <{value = dense<[-2147483648, 0, 65536, -65538, 2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui32"}> ({
    %33 = "stablehlo.constant"() <{value = dense<[0, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %34 = "stablehlo.constant"() <{value = dense<[4294967295, 65536]> : tensor<2xui32>}> : () -> tensor<2xui32>
    %35 = "stablehlo.add"(%33, %34) : (tensor<2xui32>, tensor<2xui32>) -> tensor<2xui32>
    "check.expect_eq_const"(%35) <{value = dense<[4294967295, 131072]> : tensor<2xui32>}> : (tensor<2xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_si64"}> ({
    %30 = "stablehlo.constant"() <{value = dense<[0, 1, 2147483648, -2147483649, 0]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %31 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -1, 2147483648, -2147483649, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %32 = "stablehlo.add"(%30, %31) : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi64>
    "check.expect_eq_const"(%32) <{value = dense<[-9223372036854775808, 0, 4294967296, -4294967298, 9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_ui64"}> ({
    %27 = "stablehlo.constant"() <{value = dense<[0, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %28 = "stablehlo.constant"() <{value = dense<[18446744073709551615, 4294967296]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %29 = "stablehlo.add"(%27, %28) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "check.expect_eq_const"(%29) <{value = dense<[18446744073709551615, 8589934592]> : tensor<2xui64>}> : (tensor<2xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_i1"}> ({
    %24 = "stablehlo.constant"() <{value = dense<[false, false, true, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %25 = "stablehlo.constant"() <{value = dense<[false, true, false, true]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %26 = "stablehlo.add"(%24, %25) : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%26) <{value = dense<[false, true, true, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_bf16"}> ({
    %21 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0x7F80, 0xFF80, 0x7F80, 9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %22 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.007810e-01, 3.140630e+00, 0.000000e+00, 0x7F80, 0xFF80, 0xFF80, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %23 = "stablehlo.add"(%21, %22) : (tensor<11xbf16>, tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%23) <{value = dense<[0.000000e+00, -0.000000e+00, 8.000000e+00, 8.750000e-01, 4.003910e-01, 6.281250e+00, 0x7F80, 0x7F80, 0xFF80, 0x7FC0, 0.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_f16"}> ({
    %18 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0x7C00, 0xFC00, 0x7C00, 5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %19 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000490e-01, 3.140630e+00, 0.000000e+00, 0x7C00, 0xFC00, 0xFC00, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %20 = "stablehlo.add"(%18, %19) : (tensor<11xf16>, tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%20) <{value = dense<[0.000000e+00, -0.000000e+00, 8.000000e+00, 8.750000e-01, 3.999020e-01, 6.281250e+00, 0x7C00, 0x7C00, 0xFC00, 0x7E00, 0.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_f32"}> ({
    %15 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0x7F800000, 0xFF800000, 0x7F800000, 1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %16 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.14159274, 0.000000e+00, 0x7F800000, 0xFF800000, 0xFF800000, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %17 = "stablehlo.add"(%15, %16) : (tensor<11xf32>, tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%17) <{value = dense<[0.000000e+00, -0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.28318548, 0x7F800000, 0x7F800000, 0xFF800000, 0x7FC00000, 0.000000e+00]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_f64"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF0000000000000, 4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %13 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 7.000000e+00, 7.500000e-01, 3.000000e-01, 3.1415926535897931, 0.000000e+00, 0x7FF0000000000000, 0xFFF0000000000000, 0xFFF0000000000000, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %14 = "stablehlo.add"(%12, %13) : (tensor<11xf64>, tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%14) <{value = dense<[0.000000e+00, -0.000000e+00, 8.000000e+00, 8.750000e-01, 4.000000e-01, 6.2831853071795862, 0x7FF0000000000000, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FF8000000000000, 0.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_c64"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %10 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %11 = "stablehlo.add"(%9, %10) : (tensor<2xcomplex<f32>>, tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%11) <{value = dense<[(3.000000e+00,5.000000e+00), (1.500000e+01,1.100000e+01)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_test_c128"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %7 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (7.500000e+00,5.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %8 = "stablehlo.add"(%6, %7) : (tensor<2xcomplex<f64>>, tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%8) <{value = dense<[(3.000000e+00,5.000000e+00), (1.500000e+01,1.100000e+01)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_scalar"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<i4>}> : () -> tensor<i4>
    %4 = "stablehlo.constant"() <{value = dense<3> : tensor<i4>}> : () -> tensor<i4>
    %5 = "stablehlo.add"(%3, %4) : (tensor<i4>, tensor<i4>) -> tensor<i4>
    "check.expect_eq_const"(%5) <{value = dense<5> : tensor<i4>}> : (tensor<i4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "add_op_tensor_shape_with_zero_dim_size"}> ({
    %0 = "stablehlo.constant"() <{value = dense<2> : tensor<2x0x3xi4>}> : () -> tensor<2x0x3xi4>
    %1 = "stablehlo.constant"() <{value = dense<3> : tensor<2x0x3xi4>}> : () -> tensor<2x0x3xi4>
    %2 = "stablehlo.add"(%0, %1) : (tensor<2x0x3xi4>, tensor<2x0x3xi4>) -> tensor<2x0x3xi4>
    "check.expect_eq_const"(%2) <{value = dense<> : tensor<2x0x3xi4>}> : (tensor<2x0x3xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

