"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_ui64_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<ui64>}> : () -> tensor<ui64>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<ui64>, tensor<ui64>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<true> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

