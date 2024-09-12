"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %1 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui8>
    %2 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<2x3xui8>
    "stablehlo.custom_call"(%2, %1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui8>, tensor<2x3xui8>) -> ()
    "func.return"(%2) : (tensor<2x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[0, 0, 0], [1, 1, 1]]> : tensor<2x3xui8>}> : () -> tensor<2x3xui8>
    "func.return"(%0) : (tensor<2x3xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

