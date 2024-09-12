"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reduce_window"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>}> : () -> tensor<3x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.reduce_window"(%0, %1) <{base_dilations = array<i64: 2, 1>, padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 3, 1>, window_dimensions = array<i64: 2, 1>, window_strides = array<i64: 4, 1>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%3) : (tensor<i64>) -> ()
    }) : (tensor<3x2xi64>, tensor<i64>) -> tensor<2x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[0, 0], [3, 4]]> : tensor<2x2xi64>}> : (tensor<2x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "reduce_window_issue_1662"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2], [3, 4], [5, 6]]> : tensor<3x2xi64>}> : () -> tensor<3x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %2 = "stablehlo.reduce_window"(%0, %1) <{base_dilations = array<i64: 2, 1>, padding = dense<[[2, 1], [0, 0]]> : tensor<2x2xi64>, window_dilations = array<i64: 3, 1>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 4, 1>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %3 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%3) : (tensor<i64>) -> ()
    }) : (tensor<3x2xi64>, tensor<i64>) -> tensor<1x2xi64>
    "check.expect_eq_const"(%2) <{value = dense<[[5, 6]]> : tensor<1x2xi64>}> : (tensor<1x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

