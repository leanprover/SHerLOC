"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64"}> ({
    %12 = "stablehlo.constant"() <{value = dense<[1, 5, -5]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %13 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %14 = "stablehlo.constant"() <{value = dense<[3, 7, -3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %15 = "stablehlo.clamp"(%12, %13, %14) : (tensor<3xi64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%15) <{value = dense<[2, 5, -3]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_min_scalar"}> ({
    %8 = "stablehlo.constant"() <{value = dense<[0, 0, -2]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %9 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %10 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %11 = "stablehlo.clamp"(%8, %9, %10) : (tensor<3xi64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
    "check.expect_eq_const"(%11) <{value = dense<[1, 1, -1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_max_scalar"}> ({
    %4 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %5 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %6 = "stablehlo.constant"() <{value = dense<[1, 1, 4]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %7 = "stablehlo.clamp"(%4, %5, %6) : (tensor<i64>, tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%7) <{value = dense<[1, 1, 0]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "clamp_op_test_si64_min_max_both_scalar"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.constant"() <{value = dense<[2, 3, -1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.clamp"(%0, %1, %2) : (tensor<i64>, tensor<3xi64>, tensor<i64>) -> tensor<3xi64>
    "check.expect_eq_const"(%3) <{value = dense<[1, 1, 0]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

