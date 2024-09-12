"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xbf16>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xbf16>) -> tensor<4x5xbf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5xbf16>, tensor<4x5xbf16>) -> ()
    "func.return"(%4) : (tensor<4x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3.937500e+00, 5.234380e-01, 4.625000e+00, 2.687500e+00, -3.937500e+00], [-3.734380e+00, 5.375000e+00, -5.437500e+00, 4.468750e+00, 3.250000e+00], [-1.656250e+00, 2.593750e+00, -4.531250e+00, 8.906250e-01, 6.601560e-01], [-3.156250e+00, -2.171880e+00, -3.328130e+00, 2.484380e+00, 7.187500e-01]]> : tensor<4x5xbf16>}> : () -> tensor<4x5xbf16>
    "func.return"(%1) : (tensor<4x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.156250e+00, -2.171880e+00, -3.328130e+00, 2.484380e+00, 7.187500e-01], [-1.656250e+00, 2.593750e+00, -4.531250e+00, 8.906250e-01, 6.601560e-01], [-3.734380e+00, 5.375000e+00, -5.437500e+00, 4.468750e+00, 3.250000e+00], [3.937500e+00, 5.234380e-01, 4.625000e+00, 2.687500e+00, -3.937500e+00]]> : tensor<4x5xbf16>}> : () -> tensor<4x5xbf16>
    "func.return"(%0) : (tensor<4x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

