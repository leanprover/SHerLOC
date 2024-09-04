"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_bf16"}> ({
    %10 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %11 = "stablehlo.tanh"(%10) : (tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%11) <{value = dense<[0.000000e+00, -0.000000e+00, 7.617180e-01, 1.245120e-01, 9.960930e-02, 9.960930e-01, 1.000000e+00, -1.000000e+00, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_f16"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %9 = "stablehlo.tanh"(%8) : (tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%9) <{value = dense<[0.000000e+00, -0.000000e+00, 7.617180e-01, 1.243290e-01, 9.967040e-02, 9.960930e-01, 1.000000e+00, -1.000000e+00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_f32"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %7 = "stablehlo.tanh"(%6) : (tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%7) <{value = dense<[0.000000e+00, -0.000000e+00, 0.761594176, 1.243530e-01, 0.0996679961, 0.996272087, 1.000000e+00, -1.000000e+00, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_f64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %5 = "stablehlo.tanh"(%4) : (tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%5) <{value = dense<[0.000000e+00, -0.000000e+00, 0.76159415595576485, 0.12435300177159619, 0.099667994624955819, 0.99627207622074998, 1.000000e+00, -1.000000e+00, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_c32"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %3 = "stablehlo.tanh"(%2) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%3) <{value = dense<[(0.967786788,-0.0926378369), (1.00166273,7.52857188E-4)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "tanh_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.tanh"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(0.96778680215277412,-0.092637836268419898), (1.0016627850956348,7.5285721538218659E-4)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

