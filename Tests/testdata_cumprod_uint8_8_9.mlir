"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xui8>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xui8>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xui8>) -> tensor<8x9xui8>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xui8>, tensor<8x9xui8>) -> ()
    "func.return"(%7) : (tensor<8x9xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[2, 0, 3, 3, 3, 4, 0, 2, 1], [4, 1, 2, 5, 2, 3, 4, 1, 7], [2, 1, 2, 3, 1, 2, 2, 3, 4], [0, 0, 0, 4, 0, 6, 1, 1, 0], [1, 1, 1, 3, 6, 1, 3, 1, 0], [3, 1, 3, 2, 0, 4, 1, 0, 0], [5, 2, 0, 0, 0, 2, 1, 3, 0], [0, 3, 4, 1, 1, 1, 0, 1, 1]]> : tensor<8x9xui8>}> : () -> tensor<8x9xui8>
    "func.return"(%4) : (tensor<8x9xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[2, 0, 3, 3, 3, 4, 0, 2, 1], [8, 0, 6, 15, 6, 12, 0, 2, 7], [16, 0, 12, 45, 6, 24, 0, 6, 28], [0, 0, 0, 180, 0, 144, 0, 6, 0], [0, 0, 0, 28, 0, 144, 0, 6, 0], [0, 0, 0, 56, 0, 64, 0, 0, 0], [0, 0, 0, 0, 0, 128, 0, 0, 0], [0, 0, 0, 0, 0, 128, 0, 0, 0]]> : tensor<8x9xui8>}> : () -> tensor<8x9xui8>
    "func.return"(%3) : (tensor<8x9xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xui8>) -> tensor<8x9xui8>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xui8>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<ui8>}> : () -> tensor<ui8>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<ui8>, %arg2: tensor<ui8>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%2) : (tensor<ui8>) -> ()
    }) : (tensor<8x9xui8>, tensor<ui8>) -> tensor<8x9xui8>
    "func.return"(%1) : (tensor<8x9xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

