"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "concatenate"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>}> : () -> tensor<3x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[7, 8]]> : tensor<1x2xi64>}> : () -> tensor<1x2xi64>
    %2 = "stablehlo.concatenate"(%0, %1) <{dimension = 0 : i64}> : (tensor<3x2xi64>, tensor<1x2xi64>) -> tensor<4x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[1, 2], [3, 4], [5, 6], [7, 8]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

