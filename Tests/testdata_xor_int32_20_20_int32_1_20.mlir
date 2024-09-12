"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xi32>, tensor<1x20xi32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xi32>
    %5 = "stablehlo.broadcast_in_dim"(%3#1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xi32>) -> tensor<20x20xi32>
    %6 = "stablehlo.xor"(%3#0, %5) : (tensor<20x20xi32>, tensor<20x20xi32>) -> tensor<20x20xi32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xi32>, tensor<20x20xi32>) -> ()
    "func.return"(%6) : (tensor<20x20xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xi32>, tensor<1x20xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x03000000FEFFFFFF0200000000000000FEFFFFFFFDFFFFFF000000000600000000000000FDFFFFFF040000000400000003000000000000000000000000000000FFFFFFFFFEFFFFFF0800000002000000010000000200000003000000FAFFFFFF0100000000000000000000000100000000000000000000000600000000000000FEFFFFFFFFFFFFFFF8FFFFFF020000000000000000000000FFFFFFFF06000000FEFFFFFF040000000400000001000000FDFFFFFF01000000010000000000000004000000000000000000000005000000FDFFFFFFFBFFFFFFFBFFFFFFFEFFFFFF0300000000000000FEFFFFFF040000000100000002000000FDFFFFFF000000000000000002000000010000000000000001000000FDFFFFFFFFFFFFFF00000000FEFFFFFFFFFFFFFFFAFFFFFFFEFFFFFF02000000FDFFFFFF03000000050000000000000001000000FCFFFFFF01000000FEFFFFFFFEFFFFFF04000000010000000000000002000000FBFFFFFFFFFFFFFFFAFFFFFF01000000FFFFFFFFF9FFFFFFFEFFFFFF0000000002000000FFFFFFFF00000000FDFFFFFF0400000002000000FDFFFFFF04000000FEFFFFFF01000000FDFFFFFFF9FFFFFF000000000200000001000000FFFFFFFF000000000100000003000000000000000200000006000000FEFFFFFF03000000FEFFFFFF0100000005000000FEFFFFFF0000000000000000FFFFFFFFFEFFFFFF010000000100000001000000020000000100000000000000000000000400000003000000FFFFFFFFFDFFFFFF04000000020000000000000001000000000000000000000002000000FEFFFFFF030000000500000002000000FEFFFFFFFFFFFFFF0000000002000000FDFFFFFF01000000000000000500000004000000FDFFFFFF03000000FDFFFFFF01000000060000000000000002000000FCFFFFFFFFFFFFFF0000000000000000000000000100000000000000FCFFFFFF00000000020000000000000000000000FAFFFFFFFDFFFFFF07000000FEFFFFFF0400000000000000FDFFFFFFFFFFFFFFFDFFFFFFFDFFFFFFFEFFFFFF01000000000000000000000000000000FDFFFFFF0000000003000000FFFFFFFFFDFFFFFF0000000000000000FFFFFFFF01000000FBFFFFFFFDFFFFFF090000000000000002000000F9FFFFFF000000000000000000000000010000000000000007000000FDFFFFFF0200000002000000000000000000000004000000FFFFFFFFF9FFFFFFFEFFFFFF0300000003000000010000000000000003000000FFFFFFFF00000000FBFFFFFFFFFFFFFF0600000002000000010000000100000000000000030000000200000002000000FCFFFFFF000000000100000000000000FFFFFFFF03000000020000000200000000000000FFFFFFFF01000000010000000200000000000000FDFFFFFF000000000000000000000000FDFFFFFF01000000000000000400000000000000FEFFFFFFFEFFFFFFFFFFFFFF010000000000000000000000FCFFFFFF00000000020000000300000002000000FFFFFFFF000000000000000001000000FEFFFFFF00000000FAFFFFFF00000000000000000000000001000000000000000300000003000000FFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF04000000FFFFFFFF020000000700000006000000FEFFFFFF04000000000000000000000000000000FCFFFFFFFDFFFFFF000000000400000000000000000000000100000001000000FCFFFFFF0000000002000000000000000500000001000000000000000000000003000000FEFFFFFFFFFFFFFF000000000500000000000000000000000000000002000000010000000200000001000000FFFFFFFF0100000001000000FFFFFFFF02000000FAFFFFFF06000000FEFFFFFF000000000000000001000000FFFFFFFFFDFFFFFFFFFFFFFF0000000000000000000000000000000001000000FFFFFFFF010000000000000000000000FDFFFFFF01000000FAFFFFFF02000000FFFFFFFF01000000FFFFFFFF02000000010000000500000001000000050000000300000006000000FAFFFFFF00000000040000000400000002000000FFFFFFFF0400000000000000FEFFFFFF02000000FFFFFFFF03000000FFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFFFFFFFF01000000FBFFFFFF0500000003000000FFFFFFFF0200000002000000FDFFFFFF01000000"> : tensor<20x20xi32>}> : () -> tensor<20x20xi32>
    %2 = "stablehlo.constant"() <{value = dense<[[2, 4, 4, -2, 1, 0, 3, 5, 1, -1, 0, 1, 1, -3, -1, 3, 0, -2, -2, 0]]> : tensor<1x20xi32>}> : () -> tensor<1x20xi32>
    "func.return"(%1, %2) : (tensor<20x20xi32>, tensor<1x20xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01000000FAFFFFFF06000000FEFFFFFFFFFFFFFFFDFFFFFF03000000030000000100000002000000040000000500000002000000FDFFFFFFFFFFFFFF03000000FFFFFFFF00000000F6FFFFFF02000000030000000600000007000000040000000000000000000000030000000400000001000000FFFFFFFF0600000001000000FFFFFFFF02000000070000000100000000000000FEFFFFFF0100000006000000FCFFFFFF0000000000000000FFFFFFFFFCFFFFFF01000000020000000500000005000000FFFFFFFF0000000004000000FCFFFFFF0600000004000000FDFFFFFF03000000FEFFFFFF00000000040000000300000006000000F9FFFFFFFEFFFFFF010000000200000002000000050000000000000002000000FFFFFFFF01000000FFFFFFFF0200000005000000FDFFFFFF0200000003000000FDFFFFFF050000000200000005000000F8FFFFFFFFFFFFFFFFFFFFFFFEFFFFFF070000000400000001000000FDFFFFFFFBFFFFFFFEFFFFFFFBFFFFFFFCFFFFFF00000000FAFFFFFFFEFFFFFFFEFFFFFFFCFFFFFFFFFFFFFF02000000F9FFFFFF00000000FCFFFFFFFCFFFFFF04000000FDFFFFFF04000000FCFFFFFF0600000000000000030000000000000002000000FFFFFFFF0200000003000000FEFFFFFFFCFFFFFF06000000FCFFFFFF07000000FAFFFFFFFFFFFFFF04000000FEFFFFFF0300000005000000FEFFFFFF01000000010000000000000000000000FFFFFFFFFEFFFFFF0300000000000000FAFFFFFFFDFFFFFFFFFFFFFFFFFFFFFF0000000006000000FEFFFFFF00000000000000000300000007000000FFFFFFFFFCFFFFFF0500000003000000FFFFFFFF02000000FFFFFFFF01000000FDFFFFFFFFFFFFFFFEFFFFFF0500000006000000F9FFFFFF070000000300000000000000060000000300000007000000FDFFFFFF00000000000000000100000001000000FCFFFFFFFFFFFFFFFFFFFFFF00000000FCFFFFFFFEFFFFFF00000000F8FFFFFFF9FFFFFF03000000000000000500000000000000FEFFFFFFFAFFFFFFFCFFFFFF02000000FEFFFFFF0000000001000000FDFFFFFFFFFFFFFFFEFFFFFF00000000FDFFFFFF01000000FDFFFFFF0200000004000000FBFFFFFFFFFFFFFFFAFFFFFFFDFFFFFF0A000000050000000300000006000000000000000100000001000000FCFFFFFFFFFFFFFF04000000FDFFFFFFFCFFFFFFFCFFFFFF000000000200000000000000FBFFFFFF07000000FFFFFFFF03000000000000000400000001000000FCFFFFFFFFFFFFFF01000000FAFFFFFF02000000F9FFFFFF0100000001000000FFFFFFFFFEFFFFFF030000000000000006000000F8FFFFFFFEFFFFFF0000000000000000FCFFFFFF0600000003000000FDFFFFFF00000000FEFFFFFF00000000FCFFFFFFFDFFFFFF03000000FDFFFFFFFEFFFFFFFEFFFFFF00000000FFFFFFFF0500000004000000FAFFFFFF01000000FEFFFFFFFDFFFFFFFAFFFFFF00000000FFFFFFFF00000000FDFFFFFF01000000FFFFFFFFFCFFFFFF01000000FFFFFFFFFEFFFFFFFEFFFFFF01000000FCFFFFFF04000000FEFFFFFFFEFFFFFF0100000000000000020000000500000002000000FCFFFFFFFFFFFFFFFAFFFFFFFEFFFFFF0200000001000000FCFFFFFFFCFFFFFF01000000FAFFFFFFFFFFFFFF0000000003000000020000000000000005000000000000000300000005000000FDFFFFFF02000000000000000500000001000000FDFFFFFFFEFFFFFF02000000FCFFFFFFFEFFFFFFFCFFFFFF00000000070000000500000004000000FEFFFFFF02000000FEFFFFFFFCFFFFFF0500000004000000FFFFFFFF000000000100000003000000FCFFFFFFFDFFFFFF02000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000FEFFFFFF0200000000000000010000000000000002000000FAFFFFFFFCFFFFFF00000000000000000100000001000000FDFFFFFFFEFFFFFFFCFFFFFF01000000FEFFFFFFFEFFFFFFFDFFFFFF03000000FEFFFFFF060000000100000000000000FFFFFFFF010000000400000004000000FEFFFFFF05000000020000000700000007000000FFFFFFFF0700000004000000FCFFFFFF010000000400000002000000FAFFFFFF060000000100000002000000FFFFFFFF0300000005000000FEFFFFFF00000000FFFFFFFF00000000FAFFFFFFF8FFFFFFFCFFFFFFFCFFFFFF02000000FCFFFFFF0300000001000000"> : tensor<20x20xi32>}> : () -> tensor<20x20xi32>
    "func.return"(%0) : (tensor<20x20xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

