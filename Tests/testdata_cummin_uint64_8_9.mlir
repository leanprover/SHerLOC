"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui64>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui64>
    %8 = "func.call"(%6) <{callee = @cummin}> : (tensor<8x9xui64>) -> tensor<8x9xui64>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui64>, tensor<8x9xui64>) -> ()
    "func.return"(%8) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 4, 0, 2, 0, 0, 1], [4, 3, 1, 3, 3, 1, 0, 0, 1], [1, 5, 2, 1, 7, 1, 3, 0, 4], [2, 5, 1, 4, 2, 1, 1, 2, 4], [5, 1, 1, 4, 3, 0, 3, 2, 4], [2, 1, 5, 2, 2, 6, 3, 0, 5], [1, 1, 4, 3, 0, 1, 0, 1, 1], [3, 3, 3, 6, 4, 4, 1, 2, 1]]> : tensor<8x9xui64>}> : () -> tensor<8x9xui64>
    "func.return"(%5) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[0, 0, 0, 4, 0, 2, 0, 0, 1], [0, 0, 0, 3, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 1, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1], [0, 0, 0, 1, 0, 0, 0, 0, 1]]> : tensor<8x9xui64>}> : () -> tensor<8x9xui64>
    "func.return"(%4) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui64>) -> tensor<8x9xui64>, sym_name = "cummin", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui64>):
    %0 = "stablehlo.constant"() <{value = dense<18446744073709551615> : tensor<ui64>}> : () -> tensor<ui64>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<ui64>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui64>, %arg2: tensor<ui64>):
      %3 = "stablehlo.minimum"(%arg1, %arg2) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%3) : (tensor<ui64>) -> ()
    }) : (tensor<8x9xui64>, tensor<ui64>) -> tensor<8x9xui64>
    "func.return"(%2) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

