"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "select_and_scatter_op_test"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 5], [2, 5], [3, 6], [4, 4]]> : tensor<4x2xi64>}> : () -> tensor<4x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[5, 6], [7, 8]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %3 = "stablehlo.select_and_scatter"(%0, %1, %2) <{padding = dense<[[0, 1], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 3, 1>, window_strides = array<i64: 2, 1>}> ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %5 = "stablehlo.compare"(%arg2, %arg3) <{comparison_direction = #stablehlo<comparison_direction GE>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%5) : (tensor<i1>) -> ()
    }, {
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %4 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<4x2xi64>, tensor<2x2xi64>, tensor<i64>) -> tensor<4x2xi64>
    "check.expect_eq_const"(%3) <{value = dense<[[0, 0], [0, 0], [5, 14], [7, 0]]> : tensor<4x2xi64>}> : (tensor<4x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

