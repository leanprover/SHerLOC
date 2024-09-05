"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_f8_e4m3b11_fnuz"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 3.000000e+01, -3.000000e+01, 1.220700e-04, -1.220700e-04]> : tensor<10xf8E4M3B11FNUZ>}> : () -> tensor<10xf8E4M3B11FNUZ>
    "check.expect_almost_eq_const"(%0) <{value = dense<[0.000000e+00, 0.000000e+00, 1.000000e+00, 1.250000e-01, 1.015630e-01, 3.250000e+00, 3.000000e+01, -3.000000e+01, 1.220700e-04, -1.220700e-04]> : tensor<10xf8E4M3B11FNUZ>}> : (tensor<10xf8E4M3B11FNUZ>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

