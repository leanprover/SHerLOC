"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "map_op_test_si64"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[4, 5], [6, 7]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2 = "stablehlo.map"(%0, %1) <{dimensions = array<i64: 0, 1>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%3) : (tensor<i64>) -> ()
    }) : (tensor<2x2xi64>, tensor<2x2xi64>) -> tensor<2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[0, 5], [12, 21]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

