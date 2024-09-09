"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_ui8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 16, 255]> : tensor<3xui8>}> : () -> tensor<3xui8>
    "check.expect_eq_const"(%0) <{value = dense<[0, 16, 255]> : tensor<3xui8>}> : (tensor<3xui8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

