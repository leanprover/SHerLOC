"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false]> : tensor<2xi1>}> : (tensor<2xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

