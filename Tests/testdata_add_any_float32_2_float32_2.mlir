"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xf32>, tensor<2xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xf32>
    %5 = "stablehlo.add"(%3#0, %3#1) : (tensor<2xf32>, tensor<2xf32>) -> tensor<2xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<2xf32>, tensor<2xf32>) -> ()
    "func.return"(%5) : (tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xf32>, tensor<2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[-1.14770842, -0.810447216]> : tensor<2xf32>}> : () -> tensor<2xf32>
    %2 = "stablehlo.constant"() <{value = dense<[0.174771667, 3.75413108]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%1, %2) : (tensor<2xf32>, tensor<2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[-0.972936749, 2.94368386]> : tensor<2xf32>}> : () -> tensor<2xf32>
    "func.return"(%0) : (tensor<2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

