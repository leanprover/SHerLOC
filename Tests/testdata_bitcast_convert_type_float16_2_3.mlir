"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui16>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xf16>) -> tensor<2x3xui16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui16>, tensor<2x3xui16>) -> ()
    "func.return"(%4) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[5.152340e+00, -7.021480e-01, 2.080080e+00], [1.206050e+00, 9.287100e-01, 5.363280e+00]]> : tensor<2x3xf16>}> : () -> tensor<2x3xf16>
    "func.return"(%1) : (tensor<2x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[17703, 47518, 16425], [15571, 15214, 17757]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    "func.return"(%0) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

