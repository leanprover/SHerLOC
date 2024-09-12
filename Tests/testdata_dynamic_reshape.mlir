"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "dynamic_reshape_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3, 4, 5, 6]]> : tensor<1x6xi64>}> : () -> tensor<1x6xi64>
    %1 = "stablehlo.constant"() <{value = dense<6> : tensor<1xi64>}> : () -> tensor<1xi64>
    %2 = "stablehlo.dynamic_reshape"(%0, %1) : (tensor<1x6xi64>, tensor<1xi64>) -> tensor<6xi64>
    "check.expect_eq_const"(%2) <{value = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi64>}> : (tensor<6xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

