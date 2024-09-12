"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xui64>, tensor<3x2x4xui64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xui64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%7) : (tensor<ui64>) -> ()
    }) : (tensor<3x5x4xui64>, tensor<2x1xi64>, tensor<3x2x4xui64>) -> tensor<3x5x4xui64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5x4xui64>, tensor<3x5x4xui64>) -> ()
    "func.return"(%6) : (tensor<3x5x4xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xui64>, tensor<3x2x4xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[3, 3, 1, 0], [3, 2, 2, 1], [5, 2, 1, 4], [0, 0, 0, 6], [4, 2, 0, 3]], [[3, 1, 1, 0], [0, 0, 5, 0], [2, 3, 1, 1], [3, 0, 2, 1], [1, 1, 1, 0]], [[1, 1, 3, 3], [5, 4, 2, 1], [2, 1, 0, 2], [4, 7, 4, 0], [0, 1, 3, 5]]]> : tensor<3x5x4xui64>}> : () -> tensor<3x5x4xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[[7, 1, 2, 0], [1, 3, 3, 1]], [[2, 3, 5, 0], [4, 2, 1, 6]], [[0, 4, 2, 0], [2, 0, 0, 2]]]> : tensor<3x2x4xui64>}> : () -> tensor<3x2x4xui64>
    "func.return"(%1, %2) : (tensor<3x5x4xui64>, tensor<3x2x4xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[3, 3, 1, 0], [21, 6, 12, 0], [5, 2, 1, 4], [0, 0, 0, 6], [4, 2, 0, 3]], [[3, 1, 1, 0], [0, 0, 25, 0], [2, 3, 1, 1], [3, 0, 2, 1], [1, 1, 1, 0]], [[1, 1, 3, 3], [0, 0, 0, 0], [2, 1, 0, 2], [4, 7, 4, 0], [0, 1, 3, 5]]]> : tensor<3x5x4xui64>}> : () -> tensor<3x5x4xui64>
    "func.return"(%0) : (tensor<3x5x4xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

