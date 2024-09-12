"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x1xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x4xbf16>, tensor<4x1xbf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x1xbf16>
    %5 = "stablehlo.triangular_solve"(%3#0, %3#1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<4x4xbf16>, tensor<4x1xbf16>) -> tensor<4x1xbf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x1xbf16>, tensor<4x1xbf16>) -> ()
    "func.return"(%5) : (tensor<4x1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x4xbf16>, tensor<4x1xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-1.515630e+00, -1.176760e-01, 1.195310e+00, 2.265630e+00], [-1.982420e-01, 2.734380e+00, 4.000000e+00, 2.734380e+00], [-2.812500e+00, -2.531250e+00, -3.027340e-01, -1.750000e+00], [1.023440e+00, 2.250000e+00, -7.265630e-01, 3.796880e+00]]> : tensor<4x4xbf16>}> : () -> tensor<4x4xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[4.437500e+00], [-4.042970e-01], [-4.375000e+00], [1.398440e+00]]> : tensor<4x1xbf16>}> : () -> tensor<4x1xbf16>
    "func.return"(%1, %2) : (tensor<4x4xbf16>, tensor<4x1xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x1xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[3.984380e+00], [3.468750e+00], [-1.929690e+00], [1.398440e+00]]> : tensor<4x1xbf16>}> : () -> tensor<4x1xbf16>
    "func.return"(%0) : (tensor<4x1xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

