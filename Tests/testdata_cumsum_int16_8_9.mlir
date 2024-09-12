"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi16>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi16>
    %8 = "func.call"(%6) <{callee = @cumsum}> : (tensor<8x9xi16>) -> tensor<8x9xi16>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi16>, tensor<8x9xi16>) -> ()
    "func.return"(%8) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[0, 1, 2, 0, 0, 0, 0, 2, -3], [0, 2, -2, 0, 0, 2, -6, -2, 0], [-4, -2, -1, 6, 0, 1, 2, 3, 0], [-4, -3, 0, 1, 2, 2, 3, 4, 4], [-2, -2, 1, -3, -1, -1, 0, -1, -1], [0, 2, 0, 2, 5, 0, 2, 0, 2], [1, -1, 0, -3, 2, -4, 0, 4, 1], [0, 0, -2, 0, 1, 0, -2, 0, -4]]> : tensor<8x9xi16>}> : () -> tensor<8x9xi16>
    "func.return"(%5) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[0, 1, 2, 0, 0, 0, 0, 2, -3], [0, 3, 0, 0, 0, 2, -6, 0, -3], [-4, 1, -1, 6, 0, 3, -4, 3, -3], [-8, -2, -1, 7, 2, 5, -1, 7, 1], [-10, -4, 0, 4, 1, 4, -1, 6, 0], [-10, -2, 0, 6, 6, 4, 1, 6, 2], [-9, -3, 0, 3, 8, 0, 1, 10, 3], [-9, -3, -2, 3, 9, 0, -1, 10, -1]]> : tensor<8x9xi16>}> : () -> tensor<8x9xi16>
    "func.return"(%4) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi16>) -> tensor<8x9xi16>, sym_name = "cumsum", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi16>):
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<i16>}> : () -> tensor<i16>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<i16>) -> tensor<i16>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>):
      %3 = "stablehlo.add"(%arg1, %arg2) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%3) : (tensor<i16>) -> ()
    }) : (tensor<8x9xi16>, tensor<i16>) -> tensor<8x9xi16>
    "func.return"(%2) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

