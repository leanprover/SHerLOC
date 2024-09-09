"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "compare_op_test_f64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0xFFF0000000000001, 0xFFF0000000000000, 0xFFF0000000000000, -2.000000e+00, -2.000000e+00, -0.000000e+00, -0.000000e+00, 0.000000e+00, 1.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FF0000000000001]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %1 = "stablehlo.constant"() <{value = dense<[0xFFF0000000000001, 0x7FF0000000000001, 0xFFF0000000000000, 0x7FF0000000000000, -2.000000e+00, -1.000000e+00, -0.000000e+00, 0.000000e+00, 0.000000e+00, 2.000000e+00, 2.000000e+00, 0x7FF0000000000000, 0x7FF0000000000001, 0x7FFFFFFFFFFFFFFF]> : tensor<14xf64>}> : () -> tensor<14xf64>
    %2 = "stablehlo.compare"(%0, %1) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<14xf64>, tensor<14xf64>) -> tensor<14xi1>
    "check.expect_eq_const"(%2) <{value = dense<[false, false, false, true, false, true, false, false, false, true, false, false, false, false]> : tensor<14xi1>}> : (tensor<14xi1>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

