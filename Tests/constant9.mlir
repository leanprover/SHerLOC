"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 65536, 4294967295]> : tensor<3xui32>}> : () -> tensor<3xui32>
    "check.expect_eq_const"(%0) <{value = dense<[0, 65536, 4294967295]> : tensor<3xui32>}> : (tensor<3xui32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

