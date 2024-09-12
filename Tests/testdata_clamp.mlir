"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 5, -5]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[3, 7, -3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %3 = "stablehlo.clamp"(%0, %1, %2) : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[2, 5, -3]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_min_scalar"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 0, -2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.clamp"(%0, %1, %2) : (tensor<3xi64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[1, 1, -1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_max_scalar"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<[1, 1, 4]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %3 = "stablehlo.clamp"(%0, %1, %2) : (tensor<i64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[1, 1, 0]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_min_max_both_scalar"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.clamp"(%0, %1, %2) : (tensor<i64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[1, 1, 0]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

