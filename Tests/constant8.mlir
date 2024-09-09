"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si32"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>}> : () -> tensor<5xi32>
    "check.expect_eq_const"(%0) <{value = dense<[-2147483648, -65537, 0, 65536, 2147483647]> : tensor<5xi32>}> : (tensor<5xi32>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

