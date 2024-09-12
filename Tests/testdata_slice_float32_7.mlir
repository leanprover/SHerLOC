"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<7xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 7>, start_indices = array<i64: 4>, strides = array<i64: 1>}> : (tensor<7xf32>) -> tensor<3xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%4) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-2.26640487, 1.58945656, -0.586266816, -5.04116297, 0.919672251, 1.11756611, 0.796692073]> : tensor<7xf32>}> : () -> tensor<7xf32>
    "func.return"(%1) : (tensor<7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.919672251, 1.11756611, 0.796692073]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

