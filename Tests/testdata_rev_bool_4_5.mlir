"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5xi1>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x5xi1>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<4x5xi1>
    %4 = "stablehlo.reverse"(%2) <{dimensions = array<i64: 0>}> : (tensor<4x5xi1>) -> tensor<4x5xi1>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x5xi1>, tensor<4x5xi1>) -> ()
    "func.return"(%4) : (tensor<4x5xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<4x5xi1>}> : () -> tensor<4x5xi1>
    "func.return"(%1) : (tensor<4x5xi1>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5xi1>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<true> : tensor<4x5xi1>}> : () -> tensor<4x5xi1>
    "func.return"(%0) : (tensor<4x5xi1>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

