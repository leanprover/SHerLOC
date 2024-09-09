"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-2, -1, 0, 2, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %1 = "stablehlo.constant"() <{value = dense<[-2, -2, 0, 1, 2]> : tensor<5xi64>}> : () -> tensor<5xi64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<5xi64>, tensor<5xi64>) -> tensor<5xi1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<5xi1>}> : (tensor<5xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

