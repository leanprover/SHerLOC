"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reduce"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1, 2, 3, 4, 5]]> : tensor<1x6xi64>}> : () -> tensor<1x6xi64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.reduce"(%0, %1) <{dimensions = array<i64: 1>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%3) : (tensor<i64>) -> ()
    }) : (tensor<1x6xi64>, tensor<i64>) -> tensor<1xi64>
    "check.expect_eq_const"(%2) <{value = dense<15> : tensor<1xi64>}> : (tensor<1xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

