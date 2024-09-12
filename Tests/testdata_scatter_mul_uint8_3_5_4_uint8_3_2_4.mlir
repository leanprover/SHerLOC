"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x4xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x4xui8>, tensor<3x2x4xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x4xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 2], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<3x5x4xui8>, tensor<2x1xi64>, tensor<3x2x4xui8>) -> tensor<3x5x4xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5x4xui8>, tensor<3x5x4xui8>) -> ()
    "func.return"(%6) : (tensor<3x5x4xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x4xui8>, tensor<3x2x4xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[3, 5, 2, 3], [3, 2, 3, 4], [1, 1, 3, 2], [1, 6, 0, 0], [3, 3, 3, 0]], [[2, 2, 5, 1], [1, 4, 4, 0], [0, 0, 2, 4], [3, 0, 4, 0], [2, 6, 3, 4]], [[1, 2, 1, 2], [2, 1, 0, 3], [1, 1, 0, 1], [1, 2, 0, 3], [6, 7, 3, 4]]]> : tensor<3x5x4xui8>}> : () -> tensor<3x5x4xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 0, 1, 1], [1, 2, 4, 3]], [[1, 3, 0, 0], [0, 4, 1, 2]], [[1, 0, 2, 0], [1, 1, 7, 1]]]> : tensor<3x2x4xui8>}> : () -> tensor<3x2x4xui8>
    "func.return"(%1, %2) : (tensor<3x5x4xui8>, tensor<3x2x4xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x4xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[3, 5, 2, 3], [0, 0, 12, 12], [1, 1, 3, 2], [1, 6, 0, 0], [3, 3, 3, 0]], [[2, 2, 5, 1], [0, 48, 0, 0], [0, 0, 2, 4], [3, 0, 4, 0], [2, 6, 3, 4]], [[1, 2, 1, 2], [2, 0, 0, 0], [1, 1, 0, 1], [1, 2, 0, 3], [6, 7, 3, 4]]]> : tensor<3x5x4xui8>}> : () -> tensor<3x5x4xui8>
    "func.return"(%0) : (tensor<3x5x4xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

