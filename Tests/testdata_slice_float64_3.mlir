"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<1xf64>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xf64>) -> tensor<1xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xf64>, tensor<1xf64>) -> ()
    "func.return"(%4) : (tensor<1xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-0.51773568596824759, 1.8601306351761653, 3.3756521433494546]> : tensor<3xf64>}> : () -> tensor<3xf64>
    "func.return"(%1) : (tensor<3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<1.8601306351761653> : tensor<1xf64>}> : () -> tensor<1xf64>
    "func.return"(%0) : (tensor<1xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

