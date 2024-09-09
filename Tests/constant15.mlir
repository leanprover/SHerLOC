"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e5m2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 0x7F, 0xFF, 1.525880e-05, -1.525880e-05]> : tensor<10xf8E5M2>}> : () -> tensor<10xf8E5M2>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, -0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 0x7F, 0xFF, 1.525880e-05, -1.525880e-05]> : tensor<10xf8E5M2>}> : (tensor<10xf8E5M2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

