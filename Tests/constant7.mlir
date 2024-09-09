"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 256, 65535]> : tensor<3xui16>}> : () -> tensor<3xui16>
    "check.expect_eq_const"(%0) <{value = dense<[0, 256, 65535]> : tensor<3xui16>}> : (tensor<3xui16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

