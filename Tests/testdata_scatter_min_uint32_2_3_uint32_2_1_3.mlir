"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xui32>, tensor<2x1x3xui32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<2x3xui32>, tensor<1x3x1xi64>, tensor<2x1x3xui32>) -> tensor<2x3xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui32>, tensor<2x3xui32>) -> ()
    "func.return"(%6) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xui32>, tensor<2x1x3xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[1, 0, 0], [5, 1, 3]]> : tensor<2x3xui32>}> : () -> tensor<2x3xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 1, 5]], [[5, 0, 5]]]> : tensor<2x1x3xui32>}> : () -> tensor<2x1x3xui32>
    "func.return"(%1, %2) : (tensor<2x3xui32>, tensor<2x1x3xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[1, 0, 0], [5, 1, 0]]> : tensor<2x3xui32>}> : () -> tensor<2x3xui32>
    "func.return"(%0) : (tensor<2x3xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

