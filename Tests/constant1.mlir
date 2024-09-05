"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1, 2, 3]> : tensor<4xui2>}> : () -> tensor<4xui2>
    "check.expect_eq_const"(%0) <{value = dense<[0, 1, 2, 3]> : tensor<4xui2>}> : (tensor<4xui2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

