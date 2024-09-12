"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6xui32>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xui32>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6xui32>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<ui32>}> : () -> tensor<ui32>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<4x6xui32>, tensor<ui32>) -> tensor<6xui32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xui32>, tensor<6xui32>) -> ()
    "func.return"(%6) : (tensor<6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[4, 3, 1, 2, 1, 1], [0, 1, 2, 0, 0, 3], [1, 4, 4, 2, 3, 4], [1, 2, 0, 2, 0, 3]]> : tensor<4x6xui32>}> : () -> tensor<4x6xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, 2, 0, 0, -3, -4], [2, 0, -1, 1, 0, 0], [0, -4, 1, -1, 6, 2], [-2, 0, 1, -1, 3, 2]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[9, 13, 10, 9, 7, 14]> : tensor<6xui32>}> : () -> tensor<6xui32>
    "func.return"(%0) : (tensor<6xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

