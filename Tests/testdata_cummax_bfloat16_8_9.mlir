"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %6 = "func.call"() <{callee = @inputs}> : () -> tensor<8x9xbf16>
    %7 = "func.call"() <{callee = @expected}> : () -> tensor<8x9xbf16>
    %8 = "func.call"(%6) <{callee = @cummax}> : (tensor<8x9xbf16>) -> tensor<8x9xbf16>
    "stablehlo.custom_call"(%8, %7) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<8x9xbf16>, tensor<8x9xbf16>) -> ()
    "func.return"(%8) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %5 = "stablehlo.constant"() <{value = dense<[[-1.218750e+00, 5.687500e+00, -5.187500e+00, 3.652340e-01, -1.148440e+00, 1.601560e+00, 3.968750e+00, -1.273440e+00, 3.031250e+00], [-5.125000e+00, 2.406250e+00, 6.445310e-02, 8.242180e-01, 4.593750e+00, -2.406250e+00, 4.000000e+00, 1.820310e+00, -3.140630e+00], [-5.031250e+00, 1.187500e+00, 2.421880e+00, -2.031250e+00, 3.671880e+00, 2.187500e+00, 4.375000e-01, 2.296880e+00, 6.250000e-01], [-2.578130e+00, -8.046880e-01, 8.562500e+00, -2.109380e+00, -4.156250e+00, 2.875000e+00, 2.031250e+00, -1.390630e+00, 5.585940e-01], [-3.562500e+00, 4.718750e+00, 3.140630e+00, 5.187500e+00, 3.320310e-01, -2.578130e+00, -7.031250e+00, 8.906250e-01, 4.550780e-01], [4.472660e-01, -2.218750e+00, 9.921870e-01, -1.562500e+00, 4.394530e-01, -9.140620e-01, -5.625000e-01, 5.093750e+00, -5.593750e+00], [3.078130e+00, 3.085940e-01, -1.562500e+00, 1.289060e+00, 2.875000e+00, -3.000000e+00, -2.451170e-01, 2.328130e+00, 1.070310e+00], [1.203130e+00, -4.042970e-01, -6.093750e-01, 3.093750e+00, -4.238280e-01, 1.304690e+00, 2.373050e-01, -6.679690e-01, -3.296880e+00]]> : tensor<8x9xbf16>}> : () -> tensor<8x9xbf16>
    "func.return"(%5) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8x9xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %4 = "stablehlo.constant"() <{value = dense<[[-1.218750e+00, 5.687500e+00, -5.187500e+00, 3.652340e-01, -1.148440e+00, 1.601560e+00, 3.968750e+00, -1.273440e+00, 3.031250e+00], [-1.218750e+00, 5.687500e+00, 6.445310e-02, 8.242180e-01, 4.593750e+00, 1.601560e+00, 4.000000e+00, 1.820310e+00, 3.031250e+00], [-1.218750e+00, 5.687500e+00, 2.421880e+00, 8.242180e-01, 4.593750e+00, 2.187500e+00, 4.000000e+00, 2.296880e+00, 3.031250e+00], [-1.218750e+00, 5.687500e+00, 8.562500e+00, 8.242180e-01, 4.593750e+00, 2.875000e+00, 4.000000e+00, 2.296880e+00, 3.031250e+00], [-1.218750e+00, 5.687500e+00, 8.562500e+00, 5.187500e+00, 4.593750e+00, 2.875000e+00, 4.000000e+00, 2.296880e+00, 3.031250e+00], [4.472660e-01, 5.687500e+00, 8.562500e+00, 5.187500e+00, 4.593750e+00, 2.875000e+00, 4.000000e+00, 5.093750e+00, 3.031250e+00], [3.078130e+00, 5.687500e+00, 8.562500e+00, 5.187500e+00, 4.593750e+00, 2.875000e+00, 4.000000e+00, 5.093750e+00, 3.031250e+00], [3.078130e+00, 5.687500e+00, 8.562500e+00, 5.187500e+00, 4.593750e+00, 2.875000e+00, 4.000000e+00, 5.093750e+00, 3.031250e+00]]> : tensor<8x9xbf16>}> : () -> tensor<8x9xbf16>
    "func.return"(%4) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<8x9xbf16>) -> tensor<8x9xbf16>, sym_name = "cummax", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<8x9xbf16>):
    %0 = "stablehlo.constant"() <{value = dense<0xFF80> : tensor<bf16>}> : () -> tensor<bf16>
    %1 = "stablehlo.broadcast_in_dim"(%0) <{broadcast_dimensions = array<i64>}> : (tensor<bf16>) -> tensor<bf16>
    %2 = "stablehlo.reduce_window"(%arg0, %1) <{padding = dense<[[7, 0], [0, 0]]> : tensor<2x2xi64>, window_dimensions = array<i64: 8, 1>}> ({
    ^bb0(%arg1: tensor<bf16>, %arg2: tensor<bf16>):
      %3 = "stablehlo.maximum"(%arg1, %arg2) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%3) : (tensor<bf16>) -> ()
    }) : (tensor<8x9xbf16>, tensor<bf16>) -> tensor<8x9xbf16>
    "func.return"(%2) : (tensor<8x9xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

