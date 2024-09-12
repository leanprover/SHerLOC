"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xbf16>
    %4 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<bf16>}> : () -> tensor<bf16>
    %5 = "stablehlo.reduce_window"(%2, %4) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %6 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%6) : (tensor<bf16>) -> ()
    }) : (tensor<4x6xbf16>, tensor<bf16>) -> tensor<3x5xbf16>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5xbf16>, tensor<3x5xbf16>) -> ()
    "func.return"(%5) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.406250e+00, 6.687500e+00, 4.375000e+00, 4.875000e+00, -1.960940e+00, 3.984380e-01], [-2.921880e+00, 1.494140e-01, -5.742190e-01, 3.484380e+00, -2.593750e+00, 1.923830e-01], [1.257810e+00, 3.625000e+00, -4.418950e-02, 3.164060e-01, -3.781250e+00, -2.843750e+00], [-3.500000e+00, 5.781250e-01, -7.773430e-01, 2.281250e+00, 7.937500e+00, -2.812500e-01]]> : tensor<4x6xbf16>}> : () -> tensor<4x6xbf16>
    "func.return"(%1) : (tensor<4x6xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[6.687500e+00, 6.687500e+00, 4.875000e+00, 4.875000e+00, 1.000000e+00], [3.625000e+00, 3.625000e+00, 3.484380e+00, 3.484380e+00, 1.000000e+00], [3.625000e+00, 3.625000e+00, 2.281250e+00, 7.937500e+00, 7.937500e+00]]> : tensor<3x5xbf16>}> : () -> tensor<3x5xbf16>
    "func.return"(%0) : (tensor<3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

