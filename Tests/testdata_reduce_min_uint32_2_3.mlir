"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<2x3xui32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3xui32>
    %4 = "stablehlo.constant"() <{value = dense<4294967295> : tensor<ui32>}> : () -> tensor<ui32>
    %5 = "stablehlo.reduce"(%2, %4) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %6 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%6) : (tensor<ui32>) -> ()
    }) : (tensor<2x3xui32>, tensor<ui32>) -> tensor<3xui32>
    "stablehlo.custom_call"(%5, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3xui32>, tensor<3xui32>) -> ()
    "func.return"(%5) : (tensor<3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[0, 4, 3], [2, 3, 0]]> : tensor<2x3xui32>}> : () -> tensor<2x3xui32>
    "func.return"(%1) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[0, 3, 0]> : tensor<3xui32>}> : () -> tensor<3xui32>
    "func.return"(%0) : (tensor<3xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

