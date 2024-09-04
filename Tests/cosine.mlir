"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_bf16"}> ({
    %10 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %11 = "stablehlo.cosine"(%10) : (tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%11) <{value = dense<[1.000000e+00, 1.000000e+00, 5.390630e-01, 9.921870e-01, 9.960930e-01, -1.000000e+00, 0xFFC0, 0xFFC0, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_f16"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %9 = "stablehlo.cosine"(%8) : (tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%9) <{value = dense<[1.000000e+00, 1.000000e+00, 5.405270e-01, 9.921870e-01, 9.951170e-01, -1.000000e+00, 0xFE00, 0xFE00, 0x7FFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_f32"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %7 = "stablehlo.cosine"(%6) : (tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%7) <{value = dense<[1.000000e+00, 1.000000e+00, 0.540302277, 0.992197692, 0.995004177, -1.000000e+00, 0xFFC00000, 0xFFC00000, 0x7FFFFFFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_f64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %5 = "stablehlo.cosine"(%4) : (tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%5) <{value = dense<[1.000000e+00, 1.000000e+00, 0.54030230586813977, 0.992197667229329, 0.99500416527802582, -1.000000e+00, 0xFFF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 1.000000e+00, 1.000000e+00]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_c64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %3 = "stablehlo.cosine"(%2) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%3) <{value = dense<[(4.337810e-01,-6.03504848), (-42.1537743,15.7863016)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "cosine_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.cosine"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(0.43378099760770306,-6.0350486377665726), (-42.153773835602316,15.786301507647636)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

