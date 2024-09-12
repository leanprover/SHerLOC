"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi32>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi32>
    %8 = "func.call"(%6) <{callee = @cummax}> : (tensor<8x9xi32>) -> tensor<8x9xi32>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi32>, tensor<8x9xi32>) -> ()
    "func.return"(%8) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[-2, 1, 6, 0, 1, 0, 2, 0, -1], [-3, 4, 1, 2, 6, 3, 0, 0, -4], [0, 0, 0, 1, -3, -2, 1, 1, 0], [-4, 1, -4, -1, 4, 2, 0, -2, 2], [2, -3, -6, -1, 1, 3, 2, 1, 1], [4, 1, -4, 4, 3, -1, 1, -2, -6], [0, 2, 2, 1, -2, 0, -2, 0, 3], [0, 4, -7, 6, 2, 4, 1, -1, -3]]> : tensor<8x9xi32>}> : () -> tensor<8x9xi32>
    "func.return"(%5) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[-2, 1, 6, 0, 1, 0, 2, 0, -1], [-2, 4, 6, 2, 6, 3, 2, 0, -1], [0, 4, 6, 2, 6, 3, 2, 1, 0], [0, 4, 6, 2, 6, 3, 2, 1, 2], [2, 4, 6, 2, 6, 3, 2, 1, 2], [4, 4, 6, 4, 6, 3, 2, 1, 2], [4, 4, 6, 4, 6, 3, 2, 1, 3], [4, 4, 6, 6, 6, 4, 2, 1, 3]]> : tensor<8x9xi32>}> : () -> tensor<8x9xi32>
    "func.return"(%4) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi32>) -> tensor<8x9xi32>, sym_name = "cummax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi32>):
    %0 = "stablehlo.constant"() <{value = dense<-2147483648> : tensor<i32>}> : () -> tensor<i32>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i32>) -> tensor<i32>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i32>, %arg2: tensor<i32>):
      %3 = "stablehlo.maximum"(%arg1, %arg2) : (tensor<i32>, tensor<i32>) -> tensor<i32>
      "stablehlo.return"(%3) : (tensor<i32>) -> ()
    }) : (tensor<8x9xi32>, tensor<i32>) -> tensor<8x9xi32>
    "func.return"(%2) : (tensor<8x9xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

