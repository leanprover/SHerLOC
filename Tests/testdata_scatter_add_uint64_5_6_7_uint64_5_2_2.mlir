"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui64>, tensor<5x2x2xui64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui64>, %arg1: tensor<ui64>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui64>, tensor<ui64>) -> tensor<ui64>
      "stablehlo.return"(%7) : (tensor<ui64>) -> ()
    }) : (tensor<5x6x7xui64>, tensor<2x2x2xi64>, tensor<5x2x2xui64>) -> tensor<5x6x7xui64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui64>, tensor<5x6x7xui64>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui64>, tensor<5x2x2xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x050000000000000004000000000000000000000000000000000000000000000002000000000000000000000000000000040000000000000003000000000000000000000000000000000000000000000000000000000000000600000000000000030000000000000000000000000000000100000000000000020000000000000007000000000000000000000000000000000000000000000001000000000000000300000000000000020000000000000005000000000000000400000000000000000000000000000001000000000000000200000000000000000000000000000004000000000000000200000000000000070000000000000001000000000000000100000000000000030000000000000006000000000000000300000000000000060000000000000008000000000000000000000000000000020000000000000005000000000000000400000000000000030000000000000000000000000000000000000000000000020000000000000006000000000000000400000000000000070000000000000000000000000000000300000000000000000000000000000000000000000000000300000000000000020000000000000003000000000000000200000000000000070000000000000001000000000000000500000000000000000000000000000000000000000000000300000000000000060000000000000001000000000000000700000000000000010000000000000002000000000000000500000000000000000000000000000002000000000000000200000000000000020000000000000002000000000000000300000000000000020000000000000005000000000000000200000000000000030000000000000003000000000000000000000000000000030000000000000002000000000000000600000000000000030000000000000000000000000000000300000000000000020000000000000000000000000000000100000000000000010000000000000005000000000000000200000000000000020000000000000003000000000000000100000000000000020000000000000001000000000000000100000000000000010000000000000004000000000000000300000000000000000000000000000002000000000000000000000000000000010000000000000004000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000001000000000000000200000000000000000000000000000000000000000000000100000000000000000000000000000001000000000000000100000000000000000000000000000003000000000000000100000000000000000000000000000001000000000000000100000000000000030000000000000002000000000000000000000000000000030000000000000000000000000000000100000000000000000000000000000002000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000004000000000000000100000000000000030000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000010000000000000000000000000000000300000000000000010000000000000006000000000000000100000000000000030000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000200000000000000020000000000000002000000000000000000000000000000000000000000000000000000000000000500000000000000030000000000000003000000000000000100000000000000000000000000000003000000000000000100000000000000010000000000000002000000000000000100000000000000050000000000000000000000000000000300000000000000000000000000000001000000000000000000000000000000000000000000000003000000000000000100000000000000010000000000000002000000000000000100000000000000000000000000000004000000000000000400000000000000020000000000000003000000000000000000000000000000000000000000000000000000000000000500000000000000050000000000000003000000000000000100000000000000010000000000000004000000000000000100000000000000000000000000000003000000000000000000000000000000030000000000000001000000000000000100000000000000"> : tensor<5x6x7xui64>}> : () -> tensor<5x6x7xui64>
    %2 = "stablehlo.constant"() <{value = dense<[[[0, 2], [3, 2]], [[2, 0], [3, 3]], [[0, 3], [1, 1]], [[0, 3], [1, 4]], [[0, 1], [0, 5]]]> : tensor<5x2x2xui64>}> : () -> tensor<5x2x2xui64>
    "func.return"(%1, %2) : (tensor<5x6x7xui64>, tensor<5x2x2xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0500000000000000040000000000000000000000000000000000000000000000020000000000000000000000000000000400000000000000030000000000000000000000000000000200000000000000000000000000000006000000000000000300000000000000000000000000000001000000000000000200000000000000070000000000000002000000000000000000000000000000010000000000000003000000000000000200000000000000050000000000000004000000000000000000000000000000010000000000000002000000000000000000000000000000070000000000000002000000000000000700000000000000010000000000000001000000000000000300000000000000060000000000000003000000000000000600000000000000080000000000000000000000000000000200000000000000050000000000000004000000000000000300000000000000020000000000000000000000000000000200000000000000060000000000000004000000000000000700000000000000000000000000000003000000000000000300000000000000000000000000000003000000000000000200000000000000030000000000000002000000000000000700000000000000010000000000000005000000000000000000000000000000000000000000000003000000000000000600000000000000010000000000000007000000000000000100000000000000020000000000000005000000000000000000000000000000050000000000000002000000000000000200000000000000020000000000000003000000000000000200000000000000050000000000000002000000000000000300000000000000030000000000000000000000000000000300000000000000020000000000000006000000000000000300000000000000000000000000000003000000000000000200000000000000000000000000000001000000000000000100000000000000050000000000000002000000000000000300000000000000030000000000000001000000000000000200000000000000010000000000000001000000000000000100000000000000040000000000000006000000000000000000000000000000020000000000000000000000000000000100000000000000040000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000020000000000000002000000000000000000000000000000000000000000000001000000000000000000000000000000010000000000000001000000000000000000000000000000030000000000000001000000000000000000000000000000010000000000000001000000000000000300000000000000020000000000000000000000000000000300000000000000000000000000000001000000000000000000000000000000020000000000000001000000000000000400000000000000000000000000000000000000000000000100000000000000040000000000000001000000000000000300000000000000000000000000000003000000000000000400000000000000000000000000000000000000000000000100000000000000000000000000000003000000000000000100000000000000060000000000000001000000000000000300000000000000010000000000000002000000000000000000000000000000000000000000000000000000000000000400000000000000000000000000000002000000000000000200000000000000020000000000000000000000000000000000000000000000000000000000000005000000000000000300000000000000030000000000000001000000000000000000000000000000030000000000000001000000000000000100000000000000020000000000000001000000000000000A0000000000000000000000000000000300000000000000000000000000000001000000000000000000000000000000000000000000000003000000000000000200000000000000010000000000000002000000000000000100000000000000000000000000000004000000000000000400000000000000020000000000000003000000000000000000000000000000000000000000000000000000000000000500000000000000050000000000000003000000000000000100000000000000010000000000000004000000000000000100000000000000000000000000000003000000000000000000000000000000030000000000000001000000000000000100000000000000"> : tensor<5x6x7xui64>}> : () -> tensor<5x6x7xui64>
    "func.return"(%0) : (tensor<5x6x7xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

