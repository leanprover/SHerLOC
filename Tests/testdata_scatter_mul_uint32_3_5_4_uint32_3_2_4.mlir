"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xui32>, tensor<3x2x4xui32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xui32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<3x5x4xui32>, tensor<2x1xi64>, tensor<3x2x4xui32>) -> tensor<3x5x4xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5x4xui32>, tensor<3x5x4xui32>) -> ()
    "func.return"(%6) : (tensor<3x5x4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xui32>, tensor<3x2x4xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[0, 1, 1, 0], [3, 0, 3, 2], [3, 1, 0, 1], [3, 1, 0, 0], [4, 4, 2, 1]], [[0, 0, 0, 1], [3, 5, 3, 0], [1, 1, 0, 0], [0, 0, 2, 2], [6, 0, 3, 1]], [[2, 5, 2, 3], [0, 2, 3, 1], [0, 4, 2, 2], [0, 5, 3, 3], [4, 4, 1, 1]]]> : tensor<3x5x4xui32>}> : () -> tensor<3x5x4xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 2, 5, 0], [3, 1, 1, 0]], [[2, 3, 5, 4], [4, 4, 0, 3]], [[1, 4, 0, 2], [1, 1, 0, 0]]]> : tensor<3x2x4xui32>}> : () -> tensor<3x2x4xui32>
    "func.return"(%1, %2) : (tensor<3x5x4xui32>, tensor<3x2x4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[0, 1, 1, 0], [0, 0, 15, 0], [3, 1, 0, 1], [3, 1, 0, 0], [4, 4, 2, 1]], [[0, 0, 0, 1], [24, 60, 0, 0], [1, 1, 0, 0], [0, 0, 2, 2], [6, 0, 3, 1]], [[2, 5, 2, 3], [0, 8, 0, 0], [0, 4, 2, 2], [0, 5, 3, 3], [4, 4, 1, 1]]]> : tensor<3x5x4xui32>}> : () -> tensor<3x5x4xui32>
    "func.return"(%0) : (tensor<3x5x4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

