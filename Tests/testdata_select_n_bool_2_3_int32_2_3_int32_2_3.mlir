"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xi32>
    %6 = "stablehlo.select"(%4#0, %4#2, %4#1) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> tensor<2x3xi32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xi32>, tensor<2x3xi32>) -> ()
    "func.return"(%6) : (tensor<2x3xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<true> : tensor<2x3xi1>}> : () -> tensor<2x3xi1>
    %2 = "stablehlo.constant"() <{value = dense<[[0, -1, 0], [0, -7, 0]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    %3 = "stablehlo.constant"() <{value = dense<[[-1, 1, 3], [-4, -2, 1]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    "func.return"(%1, %2, %3) : (tensor<2x3xi1>, tensor<2x3xi32>, tensor<2x3xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-1, 1, 3], [-4, -2, 1]]> : tensor<2x3xi32>}> : () -> tensor<2x3xi32>
    "func.return"(%0) : (tensor<2x3xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

