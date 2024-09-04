"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "round_nearest_afz_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2.500000e+00, 4.000000e-01, 5.000000e-01, 6.000000e-01, 2.500000e+00]> : tensor<5xf64>}> : () -> tensor<5xf64>
    %1 = "stablehlo.round_nearest_afz"(%0) : (tensor<5xf64>) -> tensor<5xf64>
    "check.expect_almost_eq_const"(%1) <{value = dense<[-3.000000e+00, 0.000000e+00, 1.000000e+00, 1.000000e+00, 3.000000e+00]> : tensor<5xf64>}> : (tensor<5xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

