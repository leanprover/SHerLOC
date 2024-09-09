"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si8"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>}> : () -> tensor<5xi8>
    "check.expect_eq_const"(%0) <{value = dense<[-128, -9, 0, 8, 127]> : tensor<5xi8>}> : (tensor<5xi8>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

