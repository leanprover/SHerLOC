"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi32>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi32>
    %8 = "func.call"(%6) <{callee = @cumsum}> : (tensor<8x9xi32>) -> tensor<8x9xi32>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi32>, tensor<8x9xi32>) -> ()
    "func.return"(%8) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[2, -1, -4, 0, -1, 1, -3, -5, 0], [1, -2, 0, -2, -3, 2, 4, 2, 0], [0, -6, -3, 1, 2, 0, -2, -3, 0], [0, -1, -2, 0, 3, 3, 4, -3, -3], [2, 3, 0, 3, -1, 0, -4, 1, -2], [2, 1, -7, -1, 1, -2, 2, -2, -1], [3, -2, -2, -2, 0, -3, 3, 1, -3], [5, 1, -4, -1, -2, 0, 0, -2, -5]]> : tensor<8x9xi32>}> : () -> tensor<8x9xi32>
    "func.return"(%5) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[2, -1, -4, 0, -1, 1, -3, -5, 0], [3, -3, -4, -2, -4, 3, 1, -3, 0], [3, -9, -7, -1, -2, 3, -1, -6, 0], [3, -10, -9, -1, 1, 6, 3, -9, -3], [5, -7, -9, 2, 0, 6, -1, -8, -5], [7, -6, -16, 1, 1, 4, 1, -10, -6], [10, -8, -18, -1, 1, 1, 4, -9, -9], [15, -7, -22, -2, -1, 1, 4, -11, -14]]> : tensor<8x9xi32>}> : () -> tensor<8x9xi32>
    "func.return"(%4) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi32>) -> tensor<8x9xi32>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<i32>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%3) : (tensor<i32>) -> ()
    }) : (tensor<8x9xi32>, tensor<i32>) -> tensor<8x9xi32>
    "func.return"(%2) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

