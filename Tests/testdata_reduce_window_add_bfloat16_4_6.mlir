"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xbf16>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%6) : (tensor<bf16>) -> ()
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    "func.return"(%5) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.023440e+00, 4.750000e+00, -4.218750e+00, -3.925780e-01, 4.593750e+00, -1.455080e-01], [2.093750e+00, -1.906250e+00, 2.093750e+00, 3.703130e+00, 8.906250e-01, -1.357420e-01], [2.953130e+00, -1.359380e+00, 2.984380e+00, 6.992180e-01, 4.250000e+00, -2.578130e+00], [-3.796880e+00, 2.062500e+00, 2.937500e+00, 1.109380e+00, -2.078130e+00, -2.203130e+00]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%1) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[4.906250e+00, 1.718750e+00, 2.187500e+00, 9.750000e+00, 6.187500e+00], [2.765630e+00, 2.812500e+00, 1.050000e+01, 1.050000e+01, 3.421880e+00], [8.593750e-01, 7.625000e+00, 8.750000e+00, 5.000000e+00, -1.609380e+00]]> : tensor<3x5xbf16>}> : () -> tensor<3x5xbf16>
    "func.return"(%0) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

