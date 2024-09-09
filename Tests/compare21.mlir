"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64_default"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0xFFF0000000000001> : tensor<f64>}> : () -> tensor<f64>
    %1 = "stablehlo.constant"() <{value = dense<0xFFF0000000000001> : tensor<f64>}> : () -> tensor<f64>
    %2 = "stablehlo.compare"(%0, %1) <{comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<f64>, tensor<f64>) -> tensor<i1>
    "check.expect_eq_const"(%2) <{value = dense<false> : tensor<i1>}> : (tensor<i1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

