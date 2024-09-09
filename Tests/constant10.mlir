"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "constant_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>}> : () -> tensor<5xi64>
    "check.expect_eq_const"(%0) <{value = dense<[-9223372036854775808, -2147483649, 0, 2147483648, 9223372036854775807]> : tensor<5xi64>}> : (tensor<5xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

