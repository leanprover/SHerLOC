"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si16"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>}> : () -> tensor<5xi16>
    "check.expect_eq_const"(%0) <{value = dense<[-32768, -129, 0, 128, 32767]> : tensor<5xi16>}> : (tensor<5xi16>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

