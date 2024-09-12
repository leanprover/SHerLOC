"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xi1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3xi1>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<1xi1>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xi1>) -> tensor<1xi1>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1xi1>, tensor<1xi1>) -> ()
    "func.return"(%4) : (tensor<1xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<3xi1>}> : () -> tensor<3xi1>
    "func.return"(%1) : (tensor<3xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<1xi1>}> : () -> tensor<1xi1>
    "func.return"(%0) : (tensor<1xi1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

