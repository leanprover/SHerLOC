"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x4xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<3x4xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x4xf32>
    "stablehlo.custom_call"(%2, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x4xf32>, tensor<3x4xf32>) -> ()
    "func.return"(%2) : (tensor<3x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1.62003875, -1.38775826, 2.65838385, 5.53908825], [-2.93717909, -2.98530722, -5.79372835, -3.51537132], [-1.01081443, -2.62119961, 6.94263887, 2.50051951]]> : tensor<3x4xf32>}> : () -> tensor<3x4xf32>
    "func.return"(%1) : (tensor<3x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1.62003875, -1.38775826, 2.65838385, 5.53908825], [-2.93717909, -2.98530722, -5.79372835, -3.51537132], [-1.01081443, -2.62119961, 6.94263887, 2.50051951]]> : tensor<3x4xf32>}> : () -> tensor<3x4xf32>
    "func.return"(%0) : (tensor<3x4xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

