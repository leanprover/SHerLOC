"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui16>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui16>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xui16>) -> tensor<8x9xui16>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui16>, tensor<8x9xui16>) -> ()
    "func.return"(%7) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[3, 0, 1, 5, 0, 3, 2, 1, 1], [0, 1, 0, 3, 5, 0, 3, 3, 1], [1, 5, 1, 0, 3, 2, 1, 4, 5], [1, 1, 2, 4, 4, 0, 4, 2, 1], [0, 3, 2, 1, 0, 5, 2, 2, 4], [3, 1, 2, 2, 1, 0, 0, 0, 2], [4, 1, 0, 3, 1, 3, 3, 4, 2], [0, 0, 2, 5, 0, 2, 0, 1, 0]]> : tensor<8x9xui16>}> : () -> tensor<8x9xui16>
    "func.return"(%4) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[3, 0, 1, 5, 0, 3, 2, 1, 1], [0, 0, 0, 15, 0, 0, 6, 3, 1], [0, 0, 0, 0, 0, 0, 6, 12, 5], [0, 0, 0, 0, 0, 0, 24, 24, 5], [0, 0, 0, 0, 0, 0, 48, 48, 20], [0, 0, 0, 0, 0, 0, 0, 0, 40], [0, 0, 0, 0, 0, 0, 0, 0, 80], [0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xui16>}> : () -> tensor<8x9xui16>
    "func.return"(%3) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui16>) -> tensor<8x9xui16>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui16>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<ui16>}> : () -> tensor<ui16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui16>, %arg2: tensor<ui16>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%2) : (tensor<ui16>) -> ()
    }) : (tensor<8x9xui16>, tensor<ui16>) -> tensor<8x9xui16>
    "func.return"(%1) : (tensor<8x9xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

