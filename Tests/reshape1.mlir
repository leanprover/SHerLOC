"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reshape_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[1, 2, 3, 4, 5, 6]> : tensor<6xi32>}> : () -> tensor<6xi32>
    %1 = "stablehlo.reshape"(%0) : (tensor<6xi32>) -> tensor<2x3xi32>
    "check.expect_eq_const"(%1) <{value = dense<[[1, 2, 3], [4, 5, 6]]> : tensor<2x3xi32>}> : (tensor<2x3xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

