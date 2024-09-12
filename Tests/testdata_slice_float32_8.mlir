"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<8xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xf32>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 6>, start_indices = array<i64: 1>, strides = array<i64: 2>}> : (tensor<8xf32>) -> tensor<3xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3xf32>, tensor<3xf32>) -> ()
    "func.return"(%4) : (tensor<3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<8xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[3.697320e+00, 0.691851497, -4.25815248, -4.68465376, -1.92406702, 3.28017545, 0.282912105, 3.48844242]> : tensor<8xf32>}> : () -> tensor<8xf32>
    "func.return"(%1) : (tensor<8xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0.691851497, -4.68465376, 3.28017545]> : tensor<3xf32>}> : () -> tensor<3xf32>
    "func.return"(%0) : (tensor<3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

