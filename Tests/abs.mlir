"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "abs_op_test_si64"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %5 = "stablehlo.abs"(%4) : (tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%5) <{value = dense<[2, 0, 2]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "abs_op_test_f64"}> ({
    %2 = "stablehlo.constant"() <{value = dense<[2.310000e+01, -2.310000e+01, -0.000000e+00]> : tensor<3xf64>}> : () -> tensor<3xf64>
    %3 = "stablehlo.abs"(%2) : (tensor<3xf64>) -> tensor<3xf64>
    "check.expect_almost_eq_const"(%3) <{value = dense<[2.310000e+01, 2.310000e+01, 0.000000e+00]> : tensor<3xf64>}> : (tensor<3xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "abs_op_test_c64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<(3.000000e+00,4.000000e+00)> : tensor<complex<f64>>}> : () -> tensor<complex<f64>>
    %1 = "stablehlo.abs"(%0) : (tensor<complex<f64>>) -> tensor<f64>
    "check.expect_almost_eq_const"(%1) <{value = dense<5.000000e+00> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

