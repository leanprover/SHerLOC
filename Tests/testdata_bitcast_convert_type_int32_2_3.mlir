"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xi32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui32>
    %4 = "stablehlo.bitcast_convert"(%2) : (tensor<2x3xi32>) -> tensor<2x3xui32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui32>, tensor<2x3xui32>) -> ()
    "func.return"(%4) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 0, -2], [0, 1, 0]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    "func.return"(%1) : (tensor<2x3xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 4294967294], [0, 1, 0]]> : tensor<2x3xui32>}> : () -> tensor<2x3xui32>
    "func.return"(%0) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

