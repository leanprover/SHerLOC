"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi64>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi64>
    %8 = "func.call"(%6) <{callee = @cumsum}> : (tensor<8x9xi64>) -> tensor<8x9xi64>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi64>, tensor<8x9xi64>) -> ()
    "func.return"(%8) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[0, 1, -1, 1, -1, 2, 1, 4, 1], [-4, 0, 2, 3, 0, 2, 3, 0, 1], [0, -2, 0, -3, 3, -4, -1, 2, -2], [-1, -1, -2, 0, 0, 0, -3, -1, -3], [-3, 0, 2, -1, 0, 1, 3, 2, -2], [0, 2, 0, 0, 1, 0, 1, 1, -4], [-2, -3, 1, -1, 1, 2, 5, 2, 5], [-2, 1, -2, 0, -4, -1, 0, 3, 2]]> : tensor<8x9xi64>}> : () -> tensor<8x9xi64>
    "func.return"(%5) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[0, 1, -1, 1, -1, 2, 1, 4, 1], [-4, 1, 1, 4, -1, 4, 4, 4, 2], [-4, -1, 1, 1, 2, 0, 3, 6, 0], [-5, -2, -1, 1, 2, 0, 0, 5, -3], [-8, -2, 1, 0, 2, 1, 3, 7, -5], [-8, 0, 1, 0, 3, 1, 4, 8, -9], [-10, -3, 2, -1, 4, 3, 9, 10, -4], [-12, -2, 0, -1, 0, 2, 9, 13, -2]]> : tensor<8x9xi64>}> : () -> tensor<8x9xi64>
    "func.return"(%4) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi64>) -> tensor<8x9xi64>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi64>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i64>) -> tensor<i64>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%3) : (tensor<i64>) -> ()
    }) : (tensor<8x9xi64>, tensor<i64>) -> tensor<8x9xi64>
    "func.return"(%2) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

