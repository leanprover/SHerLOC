"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sort_stable"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 2, 3], [3, 2, 1]]> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[3, 2, 1], [1, 2, 3]]> : tensor<2x3xi64>}> : () -> tensor<2x3xi64>
    %2:2 = "stablehlo.sort"(%0, %1) <{dimension = 0 : i64, is_stable = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>, %arg2: tensor<i64>, %arg3: tensor<i64>):
      %3 = "stablehlo.compare"(%arg0, %arg1) <{comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%3) : (tensor<i1>) -> ()
    }) : (tensor<2x3xi64>, tensor<2x3xi64>) -> (tensor<2x3xi64>, tensor<2x3xi64>)
    "check.expect_eq_const"(%2#0) <{value = dense<[[3, 2, 3], [1, 2, 1]]> : tensor<2x3xi64>}> : (tensor<2x3xi64>) -> ()
    "check.expect_eq_const"(%2#1) <{value = dense<[[1, 2, 1], [3, 2, 3]]> : tensor<2x3xi64>}> : (tensor<2x3xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

// -----
"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "sort_issue_2440", sym_visibility = "public"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 1, 1, -3, -3, -2, 0], [4, -1, 3, 0, 4, 0, -3], [1, 0, 0, 0, 0, 2, 3], [-3, -4, -4, -2, 0, 3, 3], [0, 5, 2, -2, 0, -2, 0]]> : tensor<5x7xi64>}> : () -> tensor<5x7xi64>
    %1 = "stablehlo.sort"(%0) <{dimension = 0 : i64}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %2 = "stablehlo.compare"(%arg0, %arg1) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
      "stablehlo.return"(%2) : (tensor<i1>) -> ()
    }) : (tensor<5x7xi64>) -> tensor<5x7xi64>
    "check.expect_eq_const"(%1) <{value = dense<[[-3, -4, -4, -3, -3, -2, -3], [0, -1, 0, -2, 0, -2, 0], [0, 0, 1, -2, 0, 0, 0], [1, 1, 2, 0, 0, 2, 3], [4, 5, 3, 0, 4, 3, 3]]> : tensor<5x7xi64>}> : (tensor<5x7xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

