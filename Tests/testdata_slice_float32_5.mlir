"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<5xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2xf32>
    %4 = "stablehlo.slice"(%2) <{limit_indices = array<i64: 5>, start_indices = array<i64: 1>, strides = array<i64: 2>}> : (tensor<5xf32>) -> tensor<2xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf32>, tensor<2xf32>) -> ()
    "func.return"(%4) : (tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[6.68327617, -1.5351733, -7.69604063, 0.212505788, 0.221927553]> : tensor<5xf32>}> : () -> tensor<5xf32>
    "func.return"(%1) : (tensor<5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-1.5351733, 0.212505788]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%0) : (tensor<2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

