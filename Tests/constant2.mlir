"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si4"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>}> : () -> tensor<5xi4>
    "check.expect_eq_const"(%0) <{value = dense<[-8, -1, 0, 1, 7]> : tensor<5xi4>}> : (tensor<5xi4>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

