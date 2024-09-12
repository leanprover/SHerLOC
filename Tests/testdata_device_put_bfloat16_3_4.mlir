"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xbf16>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xbf16>, tensor<3x4xbf16>) -> ()
    "func.return"(%2) : (tensor<3x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[3.890630e+00, -5.000000e+00, -3.968750e+00, -1.742190e+00], [6.750000e+00, -1.515630e+00, 5.125000e+00, 5.062500e+00], [-2.328130e+00, -3.859380e+00, -2.437500e+00, -3.609380e+00]]> : tensor<3x4xbf16>}> : () -> tensor<3x4xbf16>
    "func.return"(%1) : (tensor<3x4xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3.890630e+00, -5.000000e+00, -3.968750e+00, -1.742190e+00], [6.750000e+00, -1.515630e+00, 5.125000e+00, 5.062500e+00], [-2.328130e+00, -3.859380e+00, -2.437500e+00, -3.609380e+00]]> : tensor<3x4xbf16>}> : () -> tensor<3x4xbf16>
    "func.return"(%0) : (tensor<3x4xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

