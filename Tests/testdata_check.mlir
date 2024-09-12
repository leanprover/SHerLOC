"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_eq"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "check.expect_eq"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_eq_const"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "check.expect_eq_const"(%0) <{value = dense<[-2, 0, 2]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.constant"() <{value = dense<5.000100e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq"(%0, %1) : (tensor<f64>, tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_const"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq_const"(%0) <{value = dense<5.000100e+00> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_tolerance"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.constant"() <{value = dense<5.000100e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq"(%0, %1) <{tolerance = 1.000000e-01 : f64}> : (tensor<f64>, tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_almost_eq_const_tolerance"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f64>}> : () -> tensor<f64>
    "check.expect_almost_eq_const"(%0) <{tolerance = 1.000000e-01 : f64, value = dense<5.100000e+00> : tensor<f64>}> : (tensor<f64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "check_close"}> ({
    %0 = "stablehlo.constant"() <{value = dense<5.000000e+00> : tensor<f16>}> : () -> tensor<f16>
    %1 = "stablehlo.constant"() <{value = dense<5.011720e+00> : tensor<f16>}> : () -> tensor<f16>
    "check.expect_close"(%0, %1) <{max_ulp_difference = 3 : ui64}> : (tensor<f16>, tensor<f16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

