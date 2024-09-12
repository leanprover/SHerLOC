"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "popcnt_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2, 127]> : tensor<4xi64>}> : () -> tensor<4xi64>
    %1 = "stablehlo.popcnt"(%0) : (tensor<4xi64>) -> tensor<4xi64>
    "check.expect_eq_const"(%1) <{value = dense<[0, 1, 1, 7]> : tensor<4xi64>}> : (tensor<4xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

