"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi64>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi64>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xi64>) -> tensor<8x9xi64>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi64>, tensor<8x9xi64>) -> ()
    "func.return"(%7) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[1, -3, -2, -9, 1, 2, -1, 1, 3], [-7, 1, 1, 3, -3, 5, 0, 3, 1], [0, -1, 1, 0, 0, 7, 2, -3, 2], [-5, -3, 2, 6, 1, 0, -1, -2, 1], [-1, 2, 0, -2, 0, -2, -2, 0, 2], [6, 3, 2, -1, -2, -2, 0, 5, 2], [-1, 0, -2, -1, -4, -3, -4, 2, 0], [2, 2, 2, 3, -1, 3, 0, 0, 4]]> : tensor<8x9xi64>}> : () -> tensor<8x9xi64>
    "func.return"(%4) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[1, -3, -2, -9, 1, 2, -1, 1, 3], [-7, -3, -2, -27, -3, 10, 0, 3, 3], [0, 3, -2, 0, 0, 70, 0, -9, 6], [0, -9, -4, 0, 0, 0, 0, 18, 6], [0, -18, 0, 0, 0, 0, 0, 0, 12], [0, -54, 0, 0, 0, 0, 0, 0, 24], [0, 0, 0, 0, 0, 0, 0, 0, 0], [0, 0, 0, 0, 0, 0, 0, 0, 0]]> : tensor<8x9xi64>}> : () -> tensor<8x9xi64>
    "func.return"(%3) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi64>) -> tensor<8x9xi64>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi64>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i64>}> : () -> tensor<i64>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i64>, %arg2: tensor<i64>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%2) : (tensor<i64>) -> ()
    }) : (tensor<8x9xi64>, tensor<i64>) -> tensor<8x9xi64>
    "func.return"(%1) : (tensor<8x9xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

