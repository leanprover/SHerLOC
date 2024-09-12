"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui16>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui16>
    %8 = "func.call"(%6) <{callee = @cumsum}> : (tensor<8x9xui16>) -> tensor<8x9xui16>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui16>, tensor<8x9xui16>) -> ()
    "func.return"(%8) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[1, 5, 3, 1, 7, 3, 5, 4, 7], [4, 1, 3, 6, 0, 2, 2, 0, 6], [2, 0, 0, 2, 0, 1, 0, 3, 0], [3, 0, 4, 6, 2, 4, 2, 0, 0], [1, 5, 0, 4, 4, 3, 0, 3, 1], [2, 1, 1, 0, 2, 0, 0, 0, 4], [0, 3, 3, 5, 1, 4, 3, 3, 2], [4, 2, 1, 1, 0, 5, 2, 1, 4]]> : tensor<8x9xui16>}> : () -> tensor<8x9xui16>
    "func.return"(%5) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[1, 5, 3, 1, 7, 3, 5, 4, 7], [5, 6, 6, 7, 7, 5, 7, 4, 13], [7, 6, 6, 9, 7, 6, 7, 7, 13], [10, 6, 10, 15, 9, 10, 9, 7, 13], [11, 11, 10, 19, 13, 13, 9, 10, 14], [13, 12, 11, 19, 15, 13, 9, 10, 18], [13, 15, 14, 24, 16, 17, 12, 13, 20], [17, 17, 15, 25, 16, 22, 14, 14, 24]]> : tensor<8x9xui16>}> : () -> tensor<8x9xui16>
    "func.return"(%4) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui16>) -> tensor<8x9xui16>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui16>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<ui16>}> : () -> tensor<ui16>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<ui16>) -> tensor<ui16>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<ui16>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%3) : (tensor<ui16>) -> ()
    }) : (tensor<8x9xui16>, tensor<ui16>) -> tensor<8x9xui16>
    "func.return"(%2) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

