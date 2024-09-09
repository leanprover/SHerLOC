"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_i1"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[true, true, false, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %1 = "stablehlo.constant"() <{value = dense<[true, false, true, false]> : tensor<4xi1>}> : () -> tensor<4xi1>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type UNSIGNED>, comparison_direction = #stablehlo<comparison_direction LE>}> : (tensor<4xi1>, tensor<4xi1>) -> tensor<4xi1>
    "check.expect_eq_const"(%2) <{value = dense<[true, false, true, true]> : tensor<4xi1>}> : (tensor<4xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

