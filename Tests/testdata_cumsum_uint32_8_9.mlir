"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui32>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui32>
    %8 = "func.call"(%6) <{callee = @cumsum}> : (tensor<8x9xui32>) -> tensor<8x9xui32>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui32>, tensor<8x9xui32>) -> ()
    "func.return"(%8) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[3, 1, 0, 2, 0, 1, 0, 2, 3], [0, 0, 0, 7, 0, 4, 2, 3, 5], [0, 4, 0, 0, 3, 0, 0, 0, 4], [0, 2, 5, 2, 0, 1, 0, 4, 0], [1, 4, 4, 1, 3, 5, 2, 3, 1], [5, 0, 5, 4, 0, 2, 2, 1, 2], [1, 2, 4, 0, 1, 3, 1, 6, 2], [1, 3, 0, 5, 3, 3, 5, 6, 0]]> : tensor<8x9xui32>}> : () -> tensor<8x9xui32>
    "func.return"(%5) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[3, 1, 0, 2, 0, 1, 0, 2, 3], [3, 1, 0, 9, 0, 5, 2, 5, 8], [3, 5, 0, 9, 3, 5, 2, 5, 12], [3, 7, 5, 11, 3, 6, 2, 9, 12], [4, 11, 9, 12, 6, 11, 4, 12, 13], [9, 11, 14, 16, 6, 13, 6, 13, 15], [10, 13, 18, 16, 7, 16, 7, 19, 17], [11, 16, 18, 21, 10, 19, 12, 25, 17]]> : tensor<8x9xui32>}> : () -> tensor<8x9xui32>
    "func.return"(%4) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui32>) -> tensor<8x9xui32>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui32>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<ui32>}> : () -> tensor<ui32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<ui32>) -> tensor<ui32>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%3) : (tensor<ui32>) -> ()
    }) : (tensor<8x9xui32>, tensor<ui32>) -> tensor<8x9xui32>
    "func.return"(%2) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

