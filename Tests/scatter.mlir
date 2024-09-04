"builtin.module"() ({
  "func.func"() <{function_type = () -> (), sym_name = "scatter_op_test"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>}> : () -> tensor<3x4x2xi64>
    %6 = "stablehlo.constant"() <{value = dense<[[[0, 2], [1, 0], [2, 1]], [[0, 1], [1, 0], [0, 9]]]> : tensor<2x3x2xi64>}> : () -> tensor<2x3x2xi64>
    %7 = "stablehlo.constant"() <{value = dense<1> : tensor<2x3x2x2xi64>}> : () -> tensor<2x3x2x2xi64>
    %8 = "stablehlo.scatter"(%5, %6, %7) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [2, 3], inserted_window_dims = [0], scatter_dims_to_operand_dims = [1, 0], index_vector_dim = 2>, unique_indices = false}> ({
    ^bb0(%arg2: tensor<i64>, %arg3: tensor<i64>):
      %9 = "stablehlo.add"(%arg2, %arg3) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%9) : (tensor<i64>) -> ()
    }) : (tensor<3x4x2xi64>, tensor<2x3x2xi64>, tensor<2x3x2x2xi64>) -> tensor<3x4x2xi64>
    "check.expect_eq_const"(%8) <{value = dense<[[[1, 2], [5, 6], [7, 8], [7, 8]], [[10, 11], [12, 13], [14, 15], [16, 17]], [[18, 19], [20, 21], [21, 22], [23, 24]]]> : tensor<3x4x2xi64>}> : (tensor<3x4x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (), sym_name = "scatter_op_with_batching_dim_test"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[[1, 2], [3, 4], [5, 6], [7, 8]], [[9, 10], [11, 12], [13, 14], [15, 16]], [[17, 18], [19, 20], [21, 22], [23, 24]]], [[[25, 26], [27, 28], [29, 30], [31, 32]], [[33, 34], [35, 36], [37, 38], [39, 40]], [[41, 42], [43, 44], [45, 46], [47, 48]]]]> : tensor<2x3x4x2xi64>}> : () -> tensor<2x3x4x2xi64>
    %1 = "stablehlo.constant"() <{value = dense<[[[[0, 0], [1, 0], [2, 1]], [[0, 1], [1, 1], [0, 9]]], [[[0, 0], [2, 1], [2, 2]], [[1, 2], [0, 1], [1, 0]]]]> : tensor<2x2x3x2xi64>}> : () -> tensor<2x2x3x2xi64>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<2x2x3x2x2xi64>}> : () -> tensor<2x2x3x2x2xi64>
    %3 = "stablehlo.scatter"(%0, %1, %2) <{indices_are_sorted = false, scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [3, 4], inserted_window_dims = [1], input_batching_dims = [0], scatter_indices_batching_dims = [1], scatter_dims_to_operand_dims = [2, 1], index_vector_dim = 3>, unique_indices = false}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %4 = "stablehlo.add"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%4) : (tensor<i64>) -> ()
    }) : (tensor<2x3x4x2xi64>, tensor<2x2x3x2xi64>, tensor<2x2x3x2x2xi64>) -> tensor<2x3x4x2xi64>
    "check.expect_eq_const"(%3) <{value = dense<[[[[3, 4], [6, 7], [6, 7], [7, 8]], [[9, 10], [11, 12], [15, 16], [17, 18]], [[17, 18], [19, 20], [22, 23], [24, 25]]], [[[25, 26], [28, 29], [30, 31], [31, 32]], [[35, 36], [38, 39], [38, 39], [39, 40]], [[41, 42], [44, 45], [46, 47], [47, 48]]]]> : tensor<2x3x4x2xi64>}> : (tensor<2x3x4x2xi64>) -> ()
    "func.return"() : () -> ()
  }) : () -> ()
}) : () -> ()

