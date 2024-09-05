"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si2"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 1]> : tensor<4xi2>}> : () -> tensor<4xi2>
    "check.expect_eq_const"(%0) <{value = dense<[-2, -1, 0, 1]> : tensor<4xi2>}> : (tensor<4xi2>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

