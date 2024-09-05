"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<i1>}> : () -> tensor<i1>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<i1>, tensor<i1>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

