"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi8>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi8>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xi8>) -> tensor<8x9xi8>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi8>, tensor<8x9xi8>) -> ()
    "func.return"(%7) : (tensor<8x9xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[4, -6, -1, 1, 1, 3, 2, -1, 3], [-3, -1, 1, 6, 1, -1, -2, 0, 0], [-2, 2, 2, 1, -5, 2, -2, -6, 2], [-6, 1, -2, -1, 0, 5, 5, 1, -3], [5, 0, 2, -3, 3, 1, 0, -1, -2], [1, 0, -3, 3, -1, -1, 0, -4, 7], [2, 0, -1, -4, 0, 0, 0, 0, -3], [-2, 0, -1, 3, 2, -2, 0, 0, 0]]> : tensor<8x9xi8>}> : () -> tensor<8x9xi8>
    "func.return"(%4) : (tensor<8x9xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[4, -6, -1, 1, 1, 3, 2, -1, 3], [-12, 6, -1, 6, 1, -3, -4, 0, 0], [24, 12, -2, 6, -5, -6, 8, 0, 0], [112, 12, 4, -6, 0, -30, 40, 0, 0], [48, 0, 8, 18, 0, -30, 0, 0, 0], [48, 0, -24, 54, 0, 30, 0, 0, 0], [96, 0, 24, 40, 0, 0, 0, 0, 0], [64, 0, -24, 120, 0, 0, 0, 0, 0]]> : tensor<8x9xi8>}> : () -> tensor<8x9xi8>
    "func.return"(%3) : (tensor<8x9xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi8>) -> tensor<8x9xi8>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi8>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i8>}> : () -> tensor<i8>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i8>, %arg2: tensor<i8>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%2) : (tensor<i8>) -> ()
    }) : (tensor<8x9xi8>, tensor<i8>) -> tensor<8x9xi8>
    "func.return"(%1) : (tensor<8x9xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

