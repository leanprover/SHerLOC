"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2xui64>, tensor<2xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<2xui64>
    %5 = "stablehlo.add"(%3#0, %3#1) : (tensor<2xui64>, tensor<2xui64>) -> tensor<2xui64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2xui64>, tensor<2xui64>) -> ()
    "func.return"(%5) : (tensor<2xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xui64>, tensor<2xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[2, 1]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %2 = "stablehlo.constant"() <{value = dense<[4, 2]> : tensor<2xui64>}> : () -> tensor<2xui64>
    "func.return"(%1, %2) : (tensor<2xui64>, tensor<2xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[6, 3]> : tensor<2xui64>}> : () -> tensor<2xui64>
    "func.return"(%0) : (tensor<2xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

