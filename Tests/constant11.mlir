"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>}> : () -> tensor<3xui64>
    "check.expect_eq_const"(%0) <{value = dense<[0, 4294967296, 18446744073709551615]> : tensor<3xui64>}> : (tensor<3xui64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

