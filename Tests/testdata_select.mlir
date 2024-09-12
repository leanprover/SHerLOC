"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "select_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, false, true]> : tensor<3xi1>}> : () -> tensor<3xi1>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[3, 7, -3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %3 = "stablehlo.select"(%0, %1, %2) : (tensor<3xi1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[2, 7, -1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "select_op_test_si64_scalar"}> ({
    %0 = "stablehlo.constant"() <{value = dense<false> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[3, 7, -3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %3 = "stablehlo.select"(%0, %1, %2) : (tensor<i1>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[3, 7, -3]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

