"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_eq"}> ({
    %9 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %10 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "check.expect_eq"(%9, %10) : (tensor<3xi64>, tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_eq_const"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "check.expect_eq_const"(%8) <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq"}> ({
    %6 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %7 = "stablehlo.constant"() <{value = dense<5.000100e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq"(%6, %7) : (tensor<f64>, tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_const"}> ({
    %5 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq_const"(%5) <{value = dense<5.000100e+00> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_tolerance"}> ({
    %3 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %4 = "stablehlo.constant"() <{value = dense<5.000100e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq"(%3, %4) <{tolerance = 1.000000e-01 : f64}> : (tensor<f64>, tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_const_tolerance"}> ({
    %2 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq_const"(%2) <{tolerance = 1.000000e-01 : f64, value = dense<5.100000e+00> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "check_close"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %1 = "stablehlo.constant"() <{value = dense<5.011720e+00> : tensor<f16>}> : () -> tensor<f16>
    "check.expect_close"(%0, %1) <{max_ulp_difference = 3 : ui64}> : (tensor<f16>, tensor<f16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

