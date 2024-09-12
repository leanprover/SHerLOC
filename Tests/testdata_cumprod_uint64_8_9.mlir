"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui64>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui64>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xui64>) -> tensor<8x9xui64>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui64>, tensor<8x9xui64>) -> ()
    "func.return"(%7) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[7, 0, 2, 1, 1, 0, 1, 1, 0], [3, 0, 2, 0, 1, 3, 1, 2, 0], [2, 5, 1, 1, 2, 0, 9, 5, 2], [3, 1, 2, 3, 0, 0, 0, 0, 3], [4, 3, 3, 7, 2, 6, 2, 3, 1], [4, 4, 3, 4, 3, 1, 0, 0, 2], [0, 5, 1, 0, 3, 3, 3, 2, 4], [3, 1, 4, 0, 3, 3, 2, 1, 4]]> : tensor<8x9xui64>}> : () -> tensor<8x9xui64>
    "func.return"(%4) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[7, 0, 2, 1, 1, 0, 1, 1, 0], [21, 0, 4, 0, 1, 0, 1, 2, 0], [42, 0, 4, 0, 2, 0, 9, 10, 0], [126, 0, 8, 0, 0, 0, 0, 0, 0], [504, 0, 24, 0, 0, 0, 0, 0, 0], [2016, 0, 72, 0, 0, 0, 0, 0, 0], [0, 0, 72, 0, 0, 0, 0, 0, 0], [0, 0, 288, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xui64>}> : () -> tensor<8x9xui64>
    "func.return"(%3) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui64>) -> tensor<8x9xui64>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui64>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<ui64>}> : () -> tensor<ui64>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui64>, %arg2: tensor<ui64>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%2) : (tensor<ui64>) -> ()
    }) : (tensor<8x9xui64>, tensor<ui64>) -> tensor<8x9xui64>
    "func.return"(%1) : (tensor<8x9xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

