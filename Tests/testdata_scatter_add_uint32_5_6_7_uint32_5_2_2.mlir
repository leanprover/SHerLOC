"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui32>, tensor<5x2x2xui32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui32>, %arg1: tensor<ui32>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui32>, tensor<ui32>) -> tensor<ui32>
      "stablehlo.return"(%7) : (tensor<ui32>) -> ()
    }) : (tensor<5x6x7xui32>, tensor<2x2x2xi64>, tensor<5x2x2xui32>) -> tensor<5x6x7xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui32>, tensor<5x6x7xui32>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui32>, tensor<5x2x2xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01000000000000000100000001000000050000000000000001000000050000000100000002000000010000000000000000000000030000000000000001000000000000000300000002000000000000000100000002000000000000000200000000000000040000000000000001000000040000000000000002000000030000000100000000000000000000000100000001000000010000000200000001000000030000000000000002000000010000000200000002000000020000000300000002000000020000000100000004000000000000000200000001000000000000000200000000000000020000000200000000000000010000000000000001000000020000000300000001000000030000000300000004000000000000000300000000000000050000000000000002000000000000000200000000000000030000000200000001000000020000000200000004000000070000000000000001000000030000000700000002000000040000000100000001000000010000000200000000000000090000000600000003000000030000000100000000000000050000000200000003000000010000000000000001000000030000000000000000000000000000000000000004000000030000000300000004000000020000000000000001000000030000000400000000000000020000000500000001000000010000000100000007000000010000000000000000000000030000000100000005000000010000000100000002000000040000000400000002000000000000000000000001000000040000000200000001000000020000000100000001000000060000000000000004000000000000000200000000000000010000000200000004000000000000000100000000000000010000000700000004000000000000000200000000000000000000000000000000000000050000000000000000000000010000000300000000000000010000000300000001000000020000000100000002000000000000000200000001000000000000000000000001000000000000000000000001000000050000000200000001000000000000000000000002000000020000000300000000000000030000000000000000000000010000000A000000030000000300000006000000"> : tensor<5x6x7xui32>}> : () -> tensor<5x6x7xui32>
    %2 = "stablehlo.constant"() <{value = dense<[[[3, 3], [1, 1]], [[0, 1], [2, 3]], [[1, 2], [1, 0]], [[2, 0], [2, 2]], [[5, 3], [0, 1]]]> : tensor<5x2x2xui32>}> : () -> tensor<5x2x2xui32>
    "func.return"(%1, %2) : (tensor<5x6x7xui32>, tensor<5x2x2xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01000000030000000100000001000000050000000000000001000000050000000100000003000000010000000000000000000000030000000000000001000000000000000600000002000000000000000100000002000000000000000200000000000000040000000000000001000000050000000000000002000000030000000100000000000000000000000100000001000000010000000200000001000000030000000000000002000000010000000200000002000000020000000300000002000000020000000100000007000000000000000200000001000000000000000200000000000000020000000300000000000000010000000000000001000000020000000300000001000000030000000300000004000000020000000300000000000000050000000000000002000000000000000200000000000000030000000200000001000000020000000200000004000000080000000000000001000000030000000700000002000000040000000100000001000000010000000200000000000000090000000600000003000000030000000300000000000000050000000200000003000000010000000000000001000000030000000000000000000000010000000000000004000000030000000300000004000000020000000000000001000000030000000400000000000000020000000500000001000000030000000100000007000000010000000000000000000000030000000100000007000000010000000100000002000000040000000400000002000000000000000000000001000000040000000200000001000000020000000100000001000000060000000000000004000000020000000200000000000000010000000200000004000000000000000100000000000000010000000700000004000000000000000200000000000000050000000000000000000000050000000000000000000000010000000300000001000000010000000300000001000000020000000100000002000000000000000500000001000000000000000000000001000000000000000000000001000000050000000200000001000000000000000000000002000000020000000300000000000000030000000000000000000000010000000A000000030000000300000006000000"> : tensor<5x6x7xui32>}> : () -> tensor<5x6x7xui32>
    "func.return"(%0) : (tensor<5x6x7xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

