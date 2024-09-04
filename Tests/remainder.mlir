"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "remainder_op_test_si64"}> ({
    %6 = "stablehlo.constant"() <{value = dense<[17, -17, 17, -17]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %7 = "stablehlo.constant"() <{value = dense<[3, 3, -3, -3]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %8 = "stablehlo.remainder"(%6, %7) : (tensor<4xi64>, tensor<4xi64>) -> tensor<4xi64>
    "check.expect_eq_const"(%8) <{value = dense<[2, -2, 2, -2]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "remainder_op_test_ui64"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[17, 18, 19, 20]> : tensor<4xui64>}> : () -> tensor<4xui64>
    %4 = "stablehlo.constant"() <{value = dense<[3, 4, 5, 7]> : tensor<4xui64>}> : () -> tensor<4xui64>
    %5 = "stablehlo.remainder"(%3, %4) : (tensor<4xui64>, tensor<4xui64>) -> tensor<4xui64>
    "check.expect_eq_const"(%5) <{value = dense<[2, 2, 4, 6]> : tensor<4xui64>}> : (tensor<4xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "remainder_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1.710000e+01, -1.710000e+01, 1.710000e+01, -1.710000e+01]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %1 = "stablehlo.constant"() <{value = dense<[3.000000e+00, 3.000000e+00, -3.000000e+00, -3.000000e+00]> : tensor<4xf64>}> : () -> tensor<4xf64>
    %2 = "stablehlo.remainder"(%0, %1) : (tensor<4xf64>, tensor<4xf64>) -> tensor<4xf64>
    "check.expect_eq_const"(%2) <{value = dense<[2.1000000000000014, -2.1000000000000014, 2.1000000000000014, -2.1000000000000014]> : tensor<4xf64>}> : (tensor<4xf64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

