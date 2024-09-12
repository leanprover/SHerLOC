"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xbf16>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xbf16>
    %8 = "func.call"(%6) <{callee = @cummin}> : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> ()
    "func.return"(%8) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[1.757810e+00, -3.359380e+00, 4.968750e+00, 5.195310e-01, -1.298830e-01, -3.242190e-01, 1.220700e-01, 4.000000e+00, 3.031250e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 8.125000e-01, 2.937500e+00, 1.078130e+00], [3.031250e+00, 8.593750e-01, 2.171880e+00, -4.031250e+00, 2.453130e+00, 2.093750e+00, 2.140630e+00, -1.304690e+00, 3.125000e+00], [-3.484380e+00, -3.652340e-01, 3.593750e+00, 6.787110e-02, -2.328130e+00, -8.375000e+00, -2.046880e+00, 1.195310e+00, 7.375000e+00], [-2.187500e+00, -5.812500e+00, 2.171880e+00, 4.156250e+00, 1.875000e+00, 2.406250e+00, 6.812500e+00, -6.445310e-01, -1.734380e+00], [-4.250000e+00, -3.171880e+00, -3.906250e-02, 7.617180e-02, -5.039060e-01, 2.640630e+00, 7.343750e-01, 3.375000e+00, 3.437500e+00], [5.125000e+00, 3.234380e+00, -2.078130e+00, 3.476560e-01, -3.125000e+00, 1.820310e+00, -1.699220e-01, -6.031250e+00, -3.105470e-01], [1.164060e+00, -3.171880e+00, 4.550780e-01, 1.390630e+00, 3.109380e+00, 3.457030e-01, -7.773430e-01, -1.984380e+00, 1.515630e+00]]> : tensor<8x9xbf16>}> : () -> tensor<8x9xbf16>
    "func.return"(%5) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[1.757810e+00, -3.359380e+00, 4.968750e+00, 5.195310e-01, -1.298830e-01, -3.242190e-01, 1.220700e-01, 4.000000e+00, 3.031250e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 1.220700e-01, 2.937500e+00, 1.078130e+00], [-1.984380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -4.941410e-01, 1.220700e-01, -1.304690e+00, 1.078130e+00], [-3.484380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, 1.078130e+00], [-3.484380e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -1.812500e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -1.304690e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -2.078130e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -6.031250e+00, -1.734380e+00], [-4.250000e+00, -7.312500e+00, -2.078130e+00, -5.375000e+00, -3.234380e+00, -8.375000e+00, -2.046880e+00, -6.031250e+00, -1.734380e+00]]> : tensor<8x9xbf16>}> : () -> tensor<8x9xbf16>
    "func.return"(%4) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xbf16>) -> tensor<8x9xbf16>, sym_name = "cummin", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<0x7F80> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<bf16>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %3 = "stablehlo.minimum"(%arg1, %arg2) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%3) : (tensor<bf16>) -> ()
    }) : (tensor<8x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    "func.return"(%2) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

