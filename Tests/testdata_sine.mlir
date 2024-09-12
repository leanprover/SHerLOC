"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_bf16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000980e-01, 3.140630e+00, 0x7F80, 0xFF80, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : () -> tensor<11xbf16>
    %1 = "stablehlo.sine"(%0) : (tensor<11xbf16>) -> tensor<11xbf16>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, -0.000000e+00, 8.398430e-01, 1.245120e-01, 1.000980e-01, 9.689330e-04, 0xFFC0, 0xFFC0, 0x7FFF, 9.183550e-41, -9.183550e-41]> : tensor<11xbf16>}> : (tensor<11xbf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_f16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.997550e-02, 3.140630e+00, 0x7C00, 0xFC00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : () -> tensor<11xf16>
    %1 = "stablehlo.sine"(%0) : (tensor<11xf16>) -> tensor<11xf16>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, -0.000000e+00, 8.413080e-01, 1.246950e-01, 9.979240e-02, 9.675020e-04, 0xFE00, 0xFE00, 0x7FFF, 5.960460e-08, -5.960460e-08]> : tensor<11xf16>}> : (tensor<11xf16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_f32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    %1 = "stablehlo.sine"(%0) : (tensor<11xf32>) -> tensor<11xf32>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, -0.000000e+00, 0.841470957, 0.12467473, 0.0998334214, -8.74227765E-8, 0xFFC00000, 0xFFC00000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.1415926535897931, 0x7FF0000000000000, 0xFFF0000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : () -> tensor<11xf64>
    %1 = "stablehlo.sine"(%0) : (tensor<11xf64>) -> tensor<11xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[0.000000e+00, -0.000000e+00, 0.8414709848078965, 0.12467473338522769, 0.099833416646828154, 1.2246467991473532E-16, 0xFFF8000000000000, 0xFFF8000000000000, 0x7FFFFFFFFFFFFFFF, 4.940660e-324, -4.940660e-324]> : tensor<11xf64>}> : (tensor<11xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f32>>}> : () -> tensor<2xcomplex<f32>>
    %1 = "stablehlo.sine"(%0) : (tensor<2xcomplex<f32>>) -> tensor<2xcomplex<f32>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(6.1169281,0.427974522), (-15.7901983,-42.1433716)]> : tensor<2xcomplex<f32>>}> : (tensor<2xcomplex<f32>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sine_op_test_c128"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[(1.500000e+00,2.500000e+00), (3.500000e+00,4.500000e+00)]> : tensor<2xcomplex<f64>>}> : () -> tensor<2xcomplex<f64>>
    %1 = "stablehlo.sine"(%0) : (tensor<2xcomplex<f64>>) -> tensor<2xcomplex<f64>>
    "check.expect_almost_eq_const"(%1) <{value = dense<[(6.1169280123693124,0.42797453450615125), (-15.790198357309713,-42.143370741504995)]> : tensor<2xcomplex<f64>>}> : (tensor<2xcomplex<f64>>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

