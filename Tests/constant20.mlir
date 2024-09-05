"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : () -> tensor<11xf32>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.000000e-01, 3.14159274, 0x7F800000, 0xFF800000, 0x7FFFFFFF, 1.401300e-45, -1.401300e-45]> : tensor<11xf32>}> : (tensor<11xf32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

