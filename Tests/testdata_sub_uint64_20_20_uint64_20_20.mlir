"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xui64>, tensor<20x20xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui64>
    %5 = "stablehlo.subtract"(%3#0, %3#1) : (tensor<20x20xui64>, tensor<20x20xui64>) -> tensor<20x20xui64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui64>, tensor<20x20xui64>) -> ()
    "func.return"(%5) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xui64>, tensor<20x20xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0200000000000000020000000000000002000000000000000600000000000000030000000000000003000000000000000000000000000000030000000000000006000000000000000400000000000000020000000000000002000000000000000000000000000000020000000000000003000000000000000200000000000000040000000000000004000000000000000400000000000000050000000000000000000000000000000200000000000000010000000000000002000000000000000000000000000000020000000000000002000000000000000200000000000000050000000000000001000000000000000100000000000000000000000000000002000000000000000400000000000000020000000000000001000000000000000100000000000000050000000000000002000000000000000000000000000000030000000000000000000000000000000000000000000000050000000000000001000000000000000200000000000000030000000000000002000000000000000600000000000000000000000000000000000000000000000100000000000000000000000000000002000000000000000100000000000000020000000000000000000000000000000400000000000000000000000000000002000000000000000100000000000000010000000000000001000000000000000100000000000000020000000000000003000000000000000200000000000000000000000000000002000000000000000200000000000000020000000000000002000000000000000000000000000000000000000000000000000000000000000500000000000000030000000000000001000000000000000000000000000000040000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000040000000000000001000000000000000100000000000000010000000000000001000000000000000000000000000000040000000000000003000000000000000700000000000000030000000000000003000000000000000100000000000000090000000000000007000000000000000100000000000000040000000000000000000000000000000700000000000000000000000000000003000000000000000000000000000000030000000000000003000000000000000300000000000000010000000000000000000000000000000100000000000000030000000000000001000000000000000000000000000000030000000000000000000000000000000000000000000000040000000000000003000000000000000200000000000000010000000000000000000000000000000000000000000000020000000000000001000000000000000100000000000000010000000000000000000000000000000100000000000000030000000000000002000000000000000700000000000000000000000000000004000000000000000300000000000000010000000000000000000000000000000700000000000000040000000000000004000000000000000600000000000000000000000000000006000000000000000000000000000000020000000000000003000000000000000100000000000000070000000000000002000000000000000200000000000000030000000000000002000000000000000100000000000000000000000000000001000000000000000400000000000000010000000000000002000000000000000400000000000000000000000000000001000000000000000400000000000000000000000000000001000000000000000100000000000000000000000000000003000000000000000300000000000000030000000000000001000000000000000100000000000000000000000000000003000000000000000200000000000000000000000000000001000000000000000100000000000000030000000000000001000000000000000200000000000000000000000000000001000000000000000100000000000000020000000000000006000000000000000200000000000000020000000000000001000000000000000200000000000000030000000000000000000000000000000100000000000000050000000000000000000000000000000200000000000000020000000000000002000000000000000100000000000000010000000000000002000000000000000000000000000000000000000000000004000000000000000500000000000000000000000000000004000000000000000300000000000000010000000000000003000000000000000600000000000000000000000000000000000000000000000700000000000000020000000000000002000000000000000300000000000000010000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000200000000000000000000000000000006000000000000000100000000000000010000000000000000000000000000000300000000000000010000000000000000000000000000000100000000000000000000000000000003000000000000000200000000000000030000000000000002000000000000000300000000000000000000000000000001000000000000000600000000000000020000000000000000000000000000000100000000000000010000000000000000000000000000000200000000000000020000000000000002000000000000000200000000000000000000000000000001000000000000000100000000000000040000000000000000000000000000000200000000000000010000000000000002000000000000000000000000000000020000000000000000000000000000000100000000000000050000000000000003000000000000000400000000000000020000000000000001000000000000000100000000000000010000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000020000000000000001000000000000000200000000000000020000000000000003000000000000000200000000000000030000000000000008000000000000000100000000000000000000000000000000000000000000000100000000000000020000000000000004000000000000000100000000000000020000000000000000000000000000000300000000000000010000000000000002000000000000000400000000000000000000000000000004000000000000000400000000000000050000000000000001000000000000000000000000000000020000000000000005000000000000000000000000000000000000000000000003000000000000000200000000000000030000000000000000000000000000000100000000000000030000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000010000000000000004000000000000000000000000000000000000000000000001000000000000000200000000000000000000000000000000000000000000000200000000000000000000000000000002000000000000000200000000000000020000000000000001000000000000000100000000000000000000000000000000000000000000000100000000000000030000000000000003000000000000000000000000000000010000000000000002000000000000000000000000000000020000000000000001000000000000000200000000000000010000000000000000000000000000000200000000000000010000000000000003000000000000000300000000000000090000000000000003000000000000000100000000000000030000000000000001000000000000000300000000000000010000000000000002000000000000000200000000000000020000000000000002000000000000000100000000000000020000000000000005000000000000000100000000000000010000000000000003000000000000000600000000000000020000000000000005000000000000000400000000000000050000000000000002000000000000000100000000000000020000000000000003000000000000000300000000000000020000000000000001000000000000000100000000000000010000000000000004000000000000000100000000000000010000000000000001000000000000000200000000000000020000000000000003000000000000000400000000000000010000000000000001000000000000000400000000000000020000000000000002000000000000000300000000000000040000000000000004000000000000000400000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    %2 = "stablehlo.constant"() <{value = dense<"0x0300000000000000010000000000000004000000000000000200000000000000040000000000000003000000000000000000000000000000000000000000000005000000000000000100000000000000030000000000000000000000000000000300000000000000000000000000000002000000000000000200000000000000060000000000000002000000000000000200000000000000000000000000000003000000000000000100000000000000000000000000000005000000000000000200000000000000000000000000000001000000000000000500000000000000050000000000000002000000000000000500000000000000030000000000000003000000000000000100000000000000060000000000000006000000000000000100000000000000010000000000000002000000000000000600000000000000040000000000000001000000000000000100000000000000010000000000000000000000000000000000000000000000010000000000000003000000000000000100000000000000020000000000000001000000000000000400000000000000000000000000000002000000000000000200000000000000010000000000000008000000000000000100000000000000010000000000000003000000000000000200000000000000000000000000000003000000000000000500000000000000010000000000000002000000000000000100000000000000010000000000000006000000000000000800000000000000000000000000000001000000000000000100000000000000010000000000000004000000000000000200000000000000040000000000000000000000000000000000000000000000000000000000000003000000000000000100000000000000010000000000000002000000000000000300000000000000000000000000000005000000000000000200000000000000010000000000000000000000000000000000000000000000020000000000000002000000000000000300000000000000010000000000000003000000000000000000000000000000010000000000000000000000000000000100000000000000030000000000000001000000000000000500000000000000000000000000000004000000000000000000000000000000010000000000000003000000000000000000000000000000000000000000000000000000000000000200000000000000040000000000000004000000000000000200000000000000020000000000000003000000000000000000000000000000000000000000000004000000000000000100000000000000000000000000000001000000000000000000000000000000020000000000000005000000000000000000000000000000020000000000000003000000000000000200000000000000010000000000000001000000000000000300000000000000010000000000000002000000000000000200000000000000000000000000000002000000000000000200000000000000030000000000000000000000000000000900000000000000000000000000000003000000000000000400000000000000000000000000000000000000000000000000000000000000020000000000000001000000000000000000000000000000030000000000000005000000000000000200000000000000050000000000000000000000000000000100000000000000030000000000000000000000000000000000000000000000030000000000000001000000000000000000000000000000050000000000000001000000000000000300000000000000030000000000000003000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000010000000000000006000000000000000300000000000000010000000000000000000000000000000000000000000000010000000000000002000000000000000100000000000000020000000000000000000000000000000100000000000000010000000000000000000000000000000500000000000000010000000000000002000000000000000200000000000000000000000000000005000000000000000100000000000000030000000000000002000000000000000000000000000000030000000000000002000000000000000200000000000000010000000000000001000000000000000200000000000000020000000000000000000000000000000500000000000000020000000000000000000000000000000200000000000000040000000000000000000000000000000100000000000000050000000000000003000000000000000300000000000000030000000000000003000000000000000500000000000000070000000000000000000000000000000200000000000000000000000000000002000000000000000200000000000000020000000000000000000000000000000200000000000000030000000000000001000000000000000100000000000000030000000000000002000000000000000200000000000000020000000000000002000000000000000000000000000000040000000000000006000000000000000000000000000000020000000000000000000000000000000400000000000000000000000000000001000000000000000200000000000000020000000000000002000000000000000700000000000000000000000000000006000000000000000300000000000000000000000000000002000000000000000300000000000000000000000000000003000000000000000100000000000000000000000000000003000000000000000000000000000000050000000000000002000000000000000200000000000000030000000000000002000000000000000400000000000000040000000000000000000000000000000000000000000000010000000000000000000000000000000300000000000000000000000000000000000000000000000000000000000000030000000000000002000000000000000100000000000000000000000000000004000000000000000300000000000000040000000000000003000000000000000000000000000000040000000000000004000000000000000300000000000000030000000000000001000000000000000200000000000000000000000000000003000000000000000000000000000000020000000000000009000000000000000000000000000000010000000000000004000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000200000000000000000000000000000002000000000000000400000000000000020000000000000000000000000000000100000000000000010000000000000006000000000000000100000000000000010000000000000001000000000000000300000000000000030000000000000006000000000000000300000000000000060000000000000002000000000000000200000000000000030000000000000001000000000000000200000000000000000000000000000001000000000000000000000000000000010000000000000002000000000000000100000000000000020000000000000003000000000000000500000000000000020000000000000001000000000000000000000000000000060000000000000002000000000000000100000000000000050000000000000000000000000000000000000000000000040000000000000000000000000000000100000000000000000000000000000004000000000000000100000000000000030000000000000004000000000000000000000000000000010000000000000001000000000000000200000000000000030000000000000002000000000000000000000000000000020000000000000002000000000000000300000000000000030000000000000000000000000000000100000000000000010000000000000009000000000000000000000000000000030000000000000003000000000000000100000000000000000000000000000001000000000000000500000000000000010000000000000001000000000000000500000000000000030000000000000000000000000000000300000000000000000000000000000002000000000000000100000000000000010000000000000001000000000000000000000000000000000000000000000004000000000000000000000000000000020000000000000003000000000000000100000000000000040000000000000000000000000000000100000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%1, %2) : (tensor<20x20xui64>, tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0400000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000030000000000000001000000000000000300000000000000FFFFFFFFFFFFFFFF0200000000000000FDFFFFFFFFFFFFFF020000000000000001000000000000000000000000000000FEFFFFFFFFFFFFFF020000000000000002000000000000000500000000000000FDFFFFFFFFFFFFFF01000000000000000100000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF02000000000000000100000000000000FDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0300000000000000FCFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF000000000000000004000000000000000000000000000000FAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0400000000000000010000000000000002000000000000000200000000000000FFFFFFFFFFFFFFFF0500000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000F8FFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF010000000000000001000000000000000100000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF02000000000000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0300000000000000FFFFFFFFFFFFFFFF010000000000000000000000000000000400000000000000FDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0400000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000010000000000000000000000000000000200000000000000010000000000000004000000000000000200000000000000000000000000000001000000000000000800000000000000070000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFF02000000000000000000000000000000FFFFFFFFFFFFFFFF000000000000000002000000000000000000000000000000030000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF00000000000000000400000000000000FFFFFFFFFFFFFFFF01000000000000000100000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FCFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF020000000000000001000000000000000400000000000000FFFFFFFFFFFFFFFF020000000000000001000000000000000100000000000000FEFFFFFFFFFFFFFF050000000000000001000000000000000400000000000000FDFFFFFFFFFFFFFF00000000000000000300000000000000FCFFFFFFFFFFFFFF0200000000000000030000000000000001000000000000000500000000000000010000000000000002000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF01000000000000000300000000000000FEFFFFFFFFFFFFFF02000000000000000400000000000000FDFFFFFFFFFFFFFF00000000000000000400000000000000FBFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF000000000000000003000000000000000200000000000000010000000000000000000000000000000000000000000000020000000000000002000000000000000000000000000000000000000000000001000000000000000200000000000000FBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000010000000000000001000000000000000400000000000000010000000000000000000000000000000100000000000000010000000000000002000000000000000000000000000000FCFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFF00000000000000000200000000000000FDFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFF02000000000000000300000000000000FFFFFFFFFFFFFFFF03000000000000000100000000000000FFFFFFFFFFFFFFFF03000000000000000100000000000000FEFFFFFFFFFFFFFF00000000000000000500000000000000FEFFFFFFFFFFFFFF02000000000000000200000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFF9FFFFFFFFFFFFFF0200000000000000000000000000000000000000000000000400000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000100000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF01000000000000000000000000000000010000000000000000000000000000000300000000000000FCFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF060000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FBFFFFFFFFFFFFFF0000000000000000FBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0200000000000000FDFFFFFFFFFFFFFF0100000000000000000000000000000001000000000000000200000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000000000000000000000000000FFFFFFFFFFFFFFFF0100000000000000FDFFFFFFFFFFFFFF020000000000000001000000000000000200000000000000FFFFFFFFFFFFFFFF0100000000000000010000000000000003000000000000000400000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0100000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000100000000000000FFFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF04000000000000000400000000000000FDFFFFFFFFFFFFFF000000000000000001000000000000000500000000000000FFFFFFFFFFFFFFFF0000000000000000020000000000000000000000000000000300000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF01000000000000000100000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000300000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF01000000000000000000000000000000020000000000000000000000000000000100000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF000000000000000001000000000000000000000000000000FBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000000000000000000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0100000000000000FCFFFFFFFFFFFFFF00000000000000000200000000000000FDFFFFFFFFFFFFFF030000000000000002000000000000000900000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFF0300000000000000000000000000000001000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000010000000000000000000000000000000300000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF030000000000000005000000000000000100000000000000FCFFFFFFFFFFFFFF04000000000000000200000000000000FFFFFFFFFFFFFFFF000000000000000002000000000000000200000000000000FEFFFFFFFFFFFFFF01000000000000000000000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0400000000000000FEFFFFFFFFFFFFFF0100000000000000FFFFFFFFFFFFFFFF01000000000000000100000000000000020000000000000004000000000000000100000000000000FDFFFFFFFFFFFFFF04000000000000000000000000000000FFFFFFFFFFFFFFFF0200000000000000000000000000000004000000000000000300000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%0) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

