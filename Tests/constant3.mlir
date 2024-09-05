"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 8, 15]> : tensor<3xui4>}> : () -> tensor<3xui4>
    "check.expect_eq_const"(%0) <{value = dense<[0, 8, 15]> : tensor<3xui4>}> : (tensor<3xui4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

