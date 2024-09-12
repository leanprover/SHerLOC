"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xui32>, tensor<3x2x4xui32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xui32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<3x5x4xui32>, tensor<2x1xi64>, tensor<3x2x4xui32>) -> tensor<3x5x4xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5x4xui32>, tensor<3x5x4xui32>) -> ()
    "func.return"(%6) : (tensor<3x5x4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xui32>, tensor<3x2x4xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[1, 3, 6, 3], [4, 2, 4, 1], [7, 0, 4, 4], [0, 1, 1, 2], [0, 3, 1, 0]], [[3, 2, 0, 2], [0, 6, 1, 1], [3, 3, 2, 0], [3, 1, 9, 4], [0, 1, 2, 2]], [[2, 2, 5, 2], [0, 5, 1, 3], [7, 0, 2, 1], [4, 0, 3, 4], [3, 0, 0, 2]]]> : tensor<3x5x4xui32>}> : () -> tensor<3x5x4xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 1, 3, 6], [1, 1, 6, 1]], [[0, 7, 0, 2], [0, 3, 1, 0]], [[1, 1, 0, 2], [0, 3, 6, 4]]]> : tensor<3x2x4xui32>}> : () -> tensor<3x2x4xui32>
    "func.return"(%1, %2) : (tensor<3x5x4xui32>, tensor<3x2x4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[1, 3, 6, 3], [4, 2, 6, 6], [7, 0, 4, 4], [0, 1, 1, 2], [0, 3, 1, 0]], [[3, 2, 0, 2], [0, 7, 1, 2], [3, 3, 2, 0], [3, 1, 9, 4], [0, 1, 2, 2]], [[2, 2, 5, 2], [1, 5, 6, 4], [7, 0, 2, 1], [4, 0, 3, 4], [3, 0, 0, 2]]]> : tensor<3x5x4xui32>}> : () -> tensor<3x5x4xui32>
    "func.return"(%0) : (tensor<3x5x4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

