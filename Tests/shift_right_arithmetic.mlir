"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "shift_right_arithmetic_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1, 0, 8]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[1, 2, 3]> : tensor<3xi64>}> : () -> tensor<3xi64>
    %2 = "stablehlo.shift_right_arithmetic"(%0, %1) : (tensor<3xi64>, tensor<3xi64>) -> tensor<3xi64>
    "check.expect_eq_const"(%2) <{value = dense<[-1, 0, 1]> : tensor<3xi64>}> : (tensor<3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

