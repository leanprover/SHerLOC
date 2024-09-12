"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x1xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x4xf16>, tensor<4x1xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x1xf16>
    %5 = "stablehlo.triangular_solve"(%3#0, %3#1) <{left_side = true, lower = false, transpose_a = #stablehlo<transpose NO_TRANSPOSE>, unit_diagonal = true}> : (tensor<4x4xf16>, tensor<4x1xf16>) -> tensor<4x1xf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x1xf16>, tensor<4x1xf16>) -> ()
    "func.return"(%5) : (tensor<4x1xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x4xf16>, tensor<4x1xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3.134770e+00, -1.165040e+00, -1.280270e+00, 3.953130e+00], [-4.468750e+00, 2.023440e+00, -2.595210e-01, 2.658200e+00], [7.519530e-02, -1.421880e+00, -1.767580e+00, 4.679690e+00], [4.449220e+00, -1.000980e+00, -9.109370e+00, -2.769530e+00]]> : tensor<4x4xf16>}> : () -> tensor<4x4xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[6.445310e+00], [3.988280e+00], [-1.346440e-01], [-8.625000e+00]]> : tensor<4x1xf16>}> : () -> tensor<4x1xf16>
    "func.return"(%1, %2) : (tensor<4x4xf16>, tensor<4x1xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x1xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.355000e+02], [3.734380e+01], [4.021880e+01], [-8.625000e+00]]> : tensor<4x1xf16>}> : () -> tensor<4x1xf16>
    "func.return"(%0) : (tensor<4x1xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

