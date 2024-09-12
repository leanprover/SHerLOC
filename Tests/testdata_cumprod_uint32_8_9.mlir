"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui32>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui32>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xui32>) -> tensor<8x9xui32>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui32>, tensor<8x9xui32>) -> ()
    "func.return"(%7) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[6, 0, 0, 2, 0, 3, 2, 0, 2], [2, 0, 1, 1, 2, 0, 0, 2, 3], [1, 1, 4, 4, 4, 3, 1, 1, 1], [2, 0, 2, 5, 4, 2, 2, 7, 0], [5, 4, 1, 6, 1, 3, 1, 1, 4], [0, 4, 3, 2, 0, 3, 1, 5, 0], [0, 5, 6, 3, 3, 2, 1, 0, 2], [2, 3, 2, 3, 4, 4, 0, 3, 0]]> : tensor<8x9xui32>}> : () -> tensor<8x9xui32>
    "func.return"(%4) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[6, 0, 0, 2, 0, 3, 2, 0, 2], [12, 0, 0, 2, 0, 0, 0, 0, 6], [12, 0, 0, 8, 0, 0, 0, 0, 6], [24, 0, 0, 40, 0, 0, 0, 0, 0], [120, 0, 0, 240, 0, 0, 0, 0, 0], [0, 0, 0, 480, 0, 0, 0, 0, 0], [0, 0, 0, 1440, 0, 0, 0, 0, 0], [0, 0, 0, 4320, 0, 0, 0, 0, 0]]> : tensor<8x9xui32>}> : () -> tensor<8x9xui32>
    "func.return"(%3) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui32>) -> tensor<8x9xui32>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui32>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<ui32>}> : () -> tensor<ui32>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui32>, %arg2: tensor<ui32>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%2) : (tensor<ui32>) -> ()
    }) : (tensor<8x9xui32>, tensor<ui32>) -> tensor<8x9xui32>
    "func.return"(%1) : (tensor<8x9xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

