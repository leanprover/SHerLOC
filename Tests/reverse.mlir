"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reverse"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>}> : () -> tensor<3x2xi64>
    %1 = "stablehlo.reverse"(%0) <{dimensions = array<i64: 1, 0>}> : (tensor<3x2xi64>) -> tensor<3x2xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[6, 5], [4, 3], [2, 1]]> : tensor<3x2xi64>}> : (tensor<3x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

