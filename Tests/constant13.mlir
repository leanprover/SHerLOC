"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3_fn"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3FN>}> : () -> tensor<10xf8E4M3FN>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 0x7F, 0xFF, 1.953130e-03, -1.953130e-03]> : tensor<10xf8E4M3FN>}> : (tensor<10xf8E4M3FN>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

