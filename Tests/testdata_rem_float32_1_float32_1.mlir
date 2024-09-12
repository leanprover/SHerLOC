"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1xf32>, tensor<1xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<1xf32>
    %5 = "stablehlo.remainder"(%3#0, %3#1) : (tensor<1xf32>, tensor<1xf32>) -> tensor<1xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1xf32>, tensor<1xf32>) -> ()
    "func.return"(%5) : (tensor<1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1xf32>, tensor<1xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    %2 = "stablehlo.constant"() <{value = dense<0x7F800000> : tensor<1xf32>}> : () -> tensor<1xf32>
    "func.return"(%1, %2) : (tensor<1xf32>, tensor<1xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0x7FC00000> : tensor<1xf32>}> : () -> tensor<1xf32>
    "func.return"(%0) : (tensor<1xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

