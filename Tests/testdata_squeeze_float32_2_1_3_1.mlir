"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x1x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x1x3x1xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x1x3xf32>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x1x3x1xf32>) -> tensor<2x1x3xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x1x3xf32>, tensor<2x1x3xf32>) -> ()
    "func.return"(%4) : (tensor<2x1x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1x3x1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[[-1.59504378], [-2.05178547], [3.58855987]]], [[[-0.277097464], [3.28455472], [-1.11533749]]]]> : tensor<2x1x3x1xf32>}> : () -> tensor<2x1x3x1xf32>
    "func.return"(%1) : (tensor<2x1x3x1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-1.59504378, -2.05178547, 3.58855987]], [[-0.277097464, 3.28455472, -1.11533749]]]> : tensor<2x1x3xf32>}> : () -> tensor<2x1x3xf32>
    "func.return"(%0) : (tensor<2x1x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

