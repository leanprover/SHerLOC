"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "count_leading_zeros_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1], [128, -1]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.count_leading_zeros"(%0) : (tensor<2x2xi64>) -> tensor<2x2xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[64, 63], [56, 0]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

