"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e5m2_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : () -> tensor<10xf8E5M2FNUZ>
    "check.expect_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 9.375000e-02, 3.000000e+00, 5.734400e+04, -5.734400e+04, 7.629390e-06, -7.629390e-06]> : tensor<10xf8E5M2FNUZ>}> : (tensor<10xf8E5M2FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

