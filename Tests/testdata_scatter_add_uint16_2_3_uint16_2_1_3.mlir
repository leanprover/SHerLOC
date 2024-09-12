"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<2> : tensor<1x3x1xi64>}> : () -> tensor<1x3x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<2x3xui16>, tensor<2x1x3xui16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<2x3xui16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>}> ({
    ^bb0(%arg0: tensor<ui16>, %arg1: tensor<ui16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui16>, tensor<ui16>) -> tensor<ui16>
      "stablehlo.return"(%7) : (tensor<ui16>) -> ()
    }) : (tensor<2x3xui16>, tensor<1x3x1xi64>, tensor<2x1x3xui16>) -> tensor<2x3xui16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<2x3xui16>, tensor<2x3xui16>) -> ()
    "func.return"(%6) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<2x3xui16>, tensor<2x1x3xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[6, 2, 5], [0, 2, 1]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    %2 = "stablehlo.constant"() <{value = dense<[[[1, 5, 7]], [[5, 2, 1]]]> : tensor<2x1x3xui16>}> : () -> tensor<2x1x3xui16>
    "func.return"(%1, %2) : (tensor<2x3xui16>, tensor<2x1x3xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<2x3xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[6, 2, 18], [0, 2, 9]]> : tensor<2x3xui16>}> : () -> tensor<2x3xui16>
    "func.return"(%0) : (tensor<2x3xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

