"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x4xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x1x4xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<2x4xf32>
    %4 = "stablehlo.reshape"(%2) : (tensor<2x1x4xf32>) -> tensor<2x4xf32>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2x4xf32>, tensor<2x4xf32>) -> ()
    "func.return"(%4) : (tensor<2x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x1x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-3.65913486, -2.68992448, 1.79780495, -2.7662487]], [[-5.10396671, -1.40102684, 1.50747836, -1.04594171]]]> : tensor<2x1x4xf32>}> : () -> tensor<2x1x4xf32>
    "func.return"(%1) : (tensor<2x1x4xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x4xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3.65913486, -2.68992448, 1.79780495, -2.7662487], [-5.10396671, -1.40102684, 1.50747836, -1.04594171]]> : tensor<2x4xf32>}> : () -> tensor<2x4xf32>
    "func.return"(%0) : (tensor<2x4xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

