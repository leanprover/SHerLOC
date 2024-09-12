"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi64>, tensor<4x3xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i64>, %arg1: tensor<i64>):
      "stablehlo.return"(%arg1) : (tensor<i64>) -> ()
    }) : (tensor<4x2x3x5xi64>, tensor<2xi64>, tensor<4x3xi64>) -> tensor<4x2x3x5xi64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi64>, tensor<4x2x3x5xi64>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi64>, tensor<4x3xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0000000000000000FFFFFFFFFFFFFFFF0200000000000000000000000000000001000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF010000000000000001000000000000000500000000000000040000000000000002000000000000000300000000000000030000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF00000000000000000000000000000000020000000000000000000000000000000300000000000000FCFFFFFFFFFFFFFF00000000000000000100000000000000000000000000000000000000000000000000000000000000020000000000000001000000000000000200000000000000000000000000000004000000000000000200000000000000F9FFFFFFFFFFFFFF04000000000000000200000000000000FDFFFFFFFFFFFFFF000000000000000000000000000000000400000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF02000000000000000500000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000000000000000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF0100000000000000000000000000000008000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0300000000000000FEFFFFFFFFFFFFFF00000000000000000200000000000000FDFFFFFFFFFFFFFF0000000000000000010000000000000008000000000000000300000000000000FCFFFFFFFFFFFFFF01000000000000000300000000000000FAFFFFFFFFFFFFFF030000000000000000000000000000000000000000000000010000000000000005000000000000000100000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF010000000000000000000000000000000300000000000000000000000000000004000000000000000200000000000000FFFFFFFFFFFFFFFF030000000000000001000000000000000000000000000000000000000000000000000000000000000100000000000000"> : tensor<4x2x3x5xi64>}> : () -> tensor<4x2x3x5xi64>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 4], [2, 0, 2], [-2, 3, 0], [-2, 1, 0]]> : tensor<4x3xi64>}> : () -> tensor<4x3xi64>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi64>, tensor<4x3xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0000000000000000FFFFFFFFFFFFFFFF0200000000000000000000000000000000000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000100000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF040000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF0200000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF010000000000000001000000000000000500000000000000040000000000000002000000000000000300000000000000030000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF00000000000000000000000000000000020000000000000000000000000000000300000000000000020000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000020000000000000001000000000000000200000000000000000000000000000004000000000000000200000000000000F9FFFFFFFFFFFFFF04000000000000000200000000000000FDFFFFFFFFFFFFFF000000000000000000000000000000000400000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF02000000000000000500000000000000FFFFFFFFFFFFFFFF0300000000000000FDFFFFFFFFFFFFFF01000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF0100000000000000000000000000000008000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0300000000000000FEFFFFFFFFFFFFFF00000000000000000200000000000000FDFFFFFFFFFFFFFF0000000000000000010000000000000008000000000000000300000000000000FEFFFFFFFFFFFFFF01000000000000000300000000000000FAFFFFFFFFFFFFFF0300000000000000010000000000000000000000000000000100000000000000050000000000000001000000000000000000000000000000FDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF010000000000000000000000000000000300000000000000000000000000000004000000000000000200000000000000FFFFFFFFFFFFFFFF030000000000000001000000000000000000000000000000000000000000000000000000000000000100000000000000"> : tensor<4x2x3x5xi64>}> : () -> tensor<4x2x3x5xi64>
    "func.return"(%0) : (tensor<4x2x3x5xi64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

