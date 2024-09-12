"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x3xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xbf16>, tensor<2x3xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x3xbf16>
    %5 = "stablehlo.concatenate"(%3#0, %3#1) <{dimension = 0 : i64}> : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> tensor<4x3xbf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x3xbf16>, tensor<4x3xbf16>) -> ()
    "func.return"(%5) : (tensor<4x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xbf16>, tensor<2x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-2.375000e+00, -2.265630e+00, -3.554690e-01], [3.686520e-02, 3.640630e+00, 8.984370e-01]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.750000e+00, 2.203130e+00, -2.156250e+00], [2.906250e+00, 5.343750e+00, 4.437500e+00]]> : tensor<2x3xbf16>}> : () -> tensor<2x3xbf16>
    "func.return"(%1, %2) : (tensor<2x3xbf16>, tensor<2x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x3xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-2.375000e+00, -2.265630e+00, -3.554690e-01], [3.686520e-02, 3.640630e+00, 8.984370e-01], [-2.750000e+00, 2.203130e+00, -2.156250e+00], [2.906250e+00, 5.343750e+00, 4.437500e+00]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    "func.return"(%0) : (tensor<4x3xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

