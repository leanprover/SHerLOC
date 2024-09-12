"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x40xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x40xi64>, tensor<3x5x2xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x40xi64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i64>, tensor<i64>) -> tensor<i64>
      "stablehlo.return"(%7) : (tensor<i64>) -> ()
    }) : (tensor<3x5x40xi64>, tensor<2x1xi64>, tensor<3x5x2xi64>) -> tensor<3x5x40xi64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5x40xi64>, tensor<3x5x40xi64>) -> ()
    "func.return"(%6) : (tensor<3x5x40xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x40xi64>, tensor<3x5x2xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0200000000000000000000000000000001000000000000000000000000000000FDFFFFFFFFFFFFFF0400000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF01000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF030000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFF0400000000000000FDFFFFFFFFFFFFFF00000000000000000100000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF000000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FAFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF02000000000000000200000000000000FDFFFFFFFFFFFFFF0100000000000000000000000000000001000000000000000000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFF01000000000000000300000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0600000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF000000000000000001000000000000000100000000000000FBFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF020000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF010000000000000000000000000000000200000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF040000000000000003000000000000000000000000000000000000000000000001000000000000000700000000000000FEFFFFFFFFFFFFFF00000000000000000200000000000000FEFFFFFFFFFFFFFF01000000000000000500000000000000010000000000000000000000000000000500000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0400000000000000FCFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFF04000000000000000000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0600000000000000FAFFFFFFFFFFFFFF010000000000000001000000000000000000000000000000080000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000003000000000000000100000000000000030000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000300000000000000040000000000000000000000000000000600000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF00000000000000000100000000000000FAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF030000000000000003000000000000000200000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0500000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF02000000000000000300000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000000000000500000000000000FFFFFFFFFFFFFFFF02000000000000000200000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF01000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000003000000000000000800000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF02000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF02000000000000000300000000000000F9FFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF010000000000000000000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF010000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000200000000000000FBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF03000000000000000200000000000000020000000000000004000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0600000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF03000000000000000000000000000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF000000000000000001000000000000000200000000000000FCFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF0300000000000000FEFFFFFFFFFFFFFF06000000000000000200000000000000FFFFFFFFFFFFFFFF03000000000000000000000000000000010000000000000002000000000000000A0000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFF040000000000000001000000000000000200000000000000FDFFFFFFFFFFFFFF0300000000000000060000000000000004000000000000000300000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000030000000000000004000000000000000100000000000000FFFFFFFFFFFFFFFF020000000000000000000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000100000000000000FBFFFFFFFFFFFFFF03000000000000000500000000000000FDFFFFFFFFFFFFFF03000000000000000100000000000000FFFFFFFFFFFFFFFF050000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF0300000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000200000000000000FCFFFFFFFFFFFFFF0200000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FCFFFFFFFFFFFFFF0700000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF0400000000000000000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF000000000000000001000000000000000300000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000400000000000000010000000000000004000000000000000200000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF0200000000000000010000000000000000000000000000000000000000000000F9FFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000010000000000000000000000000000000100000000000000030000000000000000000000000000000400000000000000000000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFF0100000000000000020000000000000000000000000000000800000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF0000000000000000020000000000000003000000000000000600000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF02000000000000000300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF02000000000000000100000000000000FEFFFFFFFFFFFFFF02000000000000000300000000000000FEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0300000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF020000000000000003000000000000000100000000000000FDFFFFFFFFFFFFFF0100000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0500000000000000000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF04000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000F8FFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF05000000000000000000000000000000010000000000000000000000000000000000000000000000020000000000000001000000000000000500000000000000030000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000020000000000000002000000000000000200000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF05000000000000000500000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF040000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFF0000000000000000000000000000000001000000000000000300000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFF02000000000000000400000000000000F9FFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0300000000000000000000000000000001000000000000000400000000000000000000000000000000000000000000000000000000000000FBFFFFFFFFFFFFFF03000000000000000100000000000000FBFFFFFFFFFFFFFF03000000000000000000000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF00000000000000000200000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF01000000000000000100000000000000FEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFF05000000000000000300000000000000000000000000000001000000000000000100000000000000FAFFFFFFFFFFFFFF0100000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF04000000000000000300000000000000FCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000000000000000000070000000000000000000000000000000800000000000000030000000000000002000000000000000500000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000020000000000000002000000000000000000000000000000FAFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000100000000000000000000000000000006000000000000000200000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000040000000000000001000000000000000000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000"> : tensor<3x5x40xi64>}> : () -> tensor<3x5x40xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3, -1], [6, 2], [-1, 0], [0, 1], [5, 3]], [[-2, 0], [0, 1], [-3, 2], [1, 4], [0, -2]], [[-1, 3], [2, -1], [2, -2], [-2, 2], [1, -2]]]> : tensor<3x5x2xi64>}> : () -> tensor<3x5x2xi64>
    "func.return"(%1, %2) : (tensor<3x5x40xi64>, tensor<3x5x2xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x40xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0200000000000000000000000000000001000000000000000000000000000000FDFFFFFFFFFFFFFF0400000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF01000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF030000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFF0400000000000000FDFFFFFFFFFFFFFF00000000000000000100000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF000000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FAFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF02000000000000000200000000000000FDFFFFFFFFFFFFFF0100000000000000000000000000000001000000000000000000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFF01000000000000000300000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0600000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF000000000000000001000000000000000100000000000000FBFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF02000000000000000000000000000000010000000000000000000000000000000100000000000000FDFFFFFFFFFFFFFF010000000000000000000000000000000200000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF040000000000000003000000000000000000000000000000000000000000000001000000000000000700000000000000FEFFFFFFFFFFFFFF00000000000000000200000000000000FEFFFFFFFFFFFFFF01000000000000000500000000000000010000000000000000000000000000000500000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0400000000000000FCFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFF040000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF0600000000000000FAFFFFFFFFFFFFFF010000000000000001000000000000000000000000000000080000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000003000000000000000100000000000000030000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000300000000000000040000000000000000000000000000000600000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF00000000000000000100000000000000FAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF030000000000000003000000000000001E00000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0500000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF02000000000000000300000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000000000000500000000000000FFFFFFFFFFFFFFFF02000000000000000200000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF01000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000003000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF02000000000000000200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF02000000000000000300000000000000F9FFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF010000000000000000000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF010000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000200000000000000FBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF03000000000000000200000000000000020000000000000004000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF0600000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF03000000000000000000000000000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF000000000000000001000000000000000200000000000000FCFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF0300000000000000FEFFFFFFFFFFFFFFDCFFFFFFFFFFFFFF0200000000000000FFFFFFFFFFFFFFFF03000000000000000000000000000000010000000000000002000000000000000A0000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFF040000000000000001000000000000000200000000000000FDFFFFFFFFFFFFFF0300000000000000060000000000000004000000000000000300000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000030000000000000004000000000000000100000000000000FFFFFFFFFFFFFFFF020000000000000000000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000100000000000000FBFFFFFFFFFFFFFF03000000000000000500000000000000FDFFFFFFFFFFFFFF03000000000000000100000000000000FCFFFFFFFFFFFFFF050000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF0300000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000200000000000000FCFFFFFFFFFFFFFF0200000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FCFFFFFFFFFFFFFF0700000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFF0400000000000000000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF000000000000000001000000000000000300000000000000FEFFFFFFFFFFFFFF000000000000000000000000000000000400000000000000000000000000000004000000000000000200000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFF8FFFFFFFFFFFFFF0200000000000000010000000000000000000000000000000000000000000000F9FFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000010000000000000000000000000000000100000000000000030000000000000000000000000000000400000000000000000000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFF0100000000000000020000000000000000000000000000000800000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF0000000000000000020000000000000003000000000000000600000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF02000000000000000300000000000000FFFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF02000000000000000100000000000000FEFFFFFFFFFFFFFF02000000000000000300000000000000FEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0300000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF020000000000000003000000000000000100000000000000FDFFFFFFFFFFFFFF0100000000000000000000000000000003000000000000000000000000000000FFFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0500000000000000000000000000000000000000000000000000000000000000020000000000000004000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000F8FFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF05000000000000000000000000000000010000000000000000000000000000000000000000000000020000000000000001000000000000000500000000000000030000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000020000000000000002000000000000000200000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF05000000000000000500000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFF0FFFFFFFFFFFFFF00000000000000000000000000000000FCFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFF0000000000000000000000000000000001000000000000000300000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFF02000000000000000400000000000000F9FFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0300000000000000000000000000000001000000000000000400000000000000000000000000000000000000000000000000000000000000FBFFFFFFFFFFFFFF03000000000000000100000000000000FBFFFFFFFFFFFFFF03000000000000000000000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF00000000000000000200000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF01000000000000000100000000000000FEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFF05000000000000000300000000000000000000000000000001000000000000000100000000000000FAFFFFFFFFFFFFFF0100000000000000FBFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF04000000000000000300000000000000FCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000000000000000000007000000000000000000000000000000F0FFFFFFFFFFFFFF030000000000000002000000000000000500000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000020000000000000002000000000000000000000000000000FAFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000100000000000000000000000000000006000000000000000200000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000040000000000000001000000000000000000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000"> : tensor<3x5x40xi64>}> : () -> tensor<3x5x40xi64>
    "func.return"(%0) : (tensor<3x5x40xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

