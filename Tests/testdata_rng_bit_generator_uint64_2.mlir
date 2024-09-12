"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<2xui64>, tensor<ui64>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<2xui64>
    %4:2 = "func.call"() <{callee = @expected}> : () -> (tensor<2xui64>, tensor<ui64>)
    %5:2 = "stablehlo.rng_bit_generator"(%3) <{rng_algorithm = #stablehlo<rng_algorithm THREE_FRY>}> : (tensor<2xui64>) -> (tensor<2xui64>, tensor<ui64>)
    "stablehlo.custom_call"(%5#0, %4#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2xui64>, tensor<2xui64>) -> ()
    "stablehlo.custom_call"(%5#1, %4#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<ui64>, tensor<ui64>) -> ()
    "func.return"(%5#0, %5#1) : (tensor<2xui64>, tensor<ui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<3> : tensor<2xui64>}> : () -> tensor<2xui64>
    "func.return"(%2) : (tensor<2xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2xui64>, tensor<ui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[3, 4]> : tensor<2xui64>}> : () -> tensor<2xui64>
    %1 = "stablehlo.constant"() <{value = dense<3349939604322698703> : tensor<ui64>}> : () -> tensor<ui64>
    "func.return"(%0, %1) : (tensor<2xui64>, tensor<ui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

