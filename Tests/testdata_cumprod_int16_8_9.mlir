"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %5 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xi16>
    %6 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xi16>
    %7 = "func.call"(%5) <{callee = @cumprod}> : (tensor<8x9xi16>) -> tensor<8x9xi16>
    "stablehlo.custom_call"(%7, %6) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<8x9xi16>, tensor<8x9xi16>) -> ()
    "func.return"(%7) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[-5, -4, -1, -1, -5, 0, -4, 0, 8], [-3, 4, 1, 0, -3, -1, 3, 1, 8], [-1, 0, 0, 0, 1, 0, 3, 2, 1], [-2, -3, 1, -1, 4, 0, 0, -1, 1], [5, 0, 1, 3, -6, 0, -4, 2, -2], [1, -6, 0, 0, 2, 0, 7, -4, -2], [-3, 0, 0, 4, 1, -1, 1, 2, -1], [-1, -1, 0, 0, -1, 3, 6, 0, 2]]> : tensor<8x9xi16>}> : () -> tensor<8x9xi16>
    "func.return"(%4) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[-5, -4, -1, -1, -5, 0, -4, 0, 8], [15, -16, -1, 0, 15, 0, -12, 0, 64], [-15, 0, 0, 0, 15, 0, -36, 0, 64], [30, 0, 0, 0, 60, 0, 0, 0, 64], [150, 0, 0, 0, -360, 0, 0, 0, -128], [150, 0, 0, 0, -720, 0, 0, 0, 256], [-450, 0, 0, 0, -720, 0, 0, 0, -256], [450, 0, 0, 0, 720, 0, 0, 0, -512]]> : tensor<8x9xi16>}> : () -> tensor<8x9xi16>
    "func.return"(%3) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xi16>) -> tensor<8x9xi16>, sym_name = "cumprod", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xi16>):
    %0 = "stablehlo.constant"() <{value = dense<1> : tensor<i16>}> : () -> tensor<i16>
    %1 = "stablehlo.reduce_window"(%arg0, %0) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<i16>, %arg2: tensor<i16>):
      %2 = "stablehlo.multiply"(%arg1, %arg2) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%2) : (tensor<i16>) -> ()
    }) : (tensor<8x9xi16>, tensor<i16>) -> tensor<8x9xi16>
    "func.return"(%1) : (tensor<8x9xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

