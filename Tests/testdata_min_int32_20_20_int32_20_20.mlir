"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xi32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xi32>, tensor<20x20xi32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xi32>
    %5 = "stablehlo.minimum"(%3#0, %3#1) : (tensor<20x20xi32>, tensor<20x20xi32>) -> tensor<20x20xi32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xi32>, tensor<20x20xi32>) -> ()
    "func.return"(%5) : (tensor<20x20xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xi32>, tensor<20x20xi32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFEFFFFFFFFFFFFFF04000000FFFFFFFFFDFFFFFFFAFFFFFF02000000020000000100000000000000FEFFFFFFFEFFFFFF0000000000000000FFFFFFFFFDFFFFFF0300000000000000FEFFFFFF00000000010000000400000001000000FFFFFFFF0300000002000000FFFFFFFF0300000002000000FBFFFFFF0000000003000000FFFFFFFFFDFFFFFFFFFFFFFFFAFFFFFF0000000004000000010000000500000002000000FFFFFFFF0300000004000000030000000000000002000000FEFFFFFFFEFFFFFFFFFFFFFF000000000000000006000000000000000200000000000000FEFFFFFF02000000FFFFFFFFFFFFFFFFF7FFFFFF0400000001000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF00000000000000000100000001000000FFFFFFFF050000000100000000000000FEFFFFFF00000000FFFFFFFF000000000300000006000000FEFFFFFF01000000FCFFFFFFFDFFFFFF040000000000000000000000FFFFFFFF04000000FCFFFFFF03000000FDFFFFFFFCFFFFFFFFFFFFFFFDFFFFFF04000000FDFFFFFF0000000005000000FFFFFFFFFEFFFFFF0400000000000000070000000000000001000000FEFFFFFFFEFFFFFF02000000FDFFFFFF00000000FCFFFFFFFAFFFFFF010000000000000000000000FEFFFFFFFBFFFFFF0000000000000000FCFFFFFF00000000FDFFFFFF060000000300000003000000FFFFFFFFFDFFFFFF0000000001000000010000000300000001000000FFFFFFFF0100000003000000FDFFFFFFFDFFFFFFFCFFFFFFFFFFFFFF0300000002000000020000000000000000000000010000000100000002000000000000000000000001000000FEFFFFFF02000000FBFFFFFF01000000FCFFFFFFFEFFFFFF0100000005000000FAFFFFFF010000000300000002000000FFFFFFFFFDFFFFFFFCFFFFFFFEFFFFFF03000000FDFFFFFFFFFFFFFF03000000FFFFFFFF00000000FEFFFFFFFFFFFFFF03000000FDFFFFFFFAFFFFFF01000000000000000500000000000000000000000000000000000000FFFFFFFFFDFFFFFF0000000001000000FFFFFFFF01000000FDFFFFFF06000000030000000000000001000000FAFFFFFFFDFFFFFF0200000000000000000000000600000002000000FCFFFFFFFEFFFFFF000000000400000004000000020000000000000000000000FAFFFFFF01000000FDFFFFFFFCFFFFFFFEFFFFFF03000000FCFFFFFF03000000FEFFFFFF01000000000000000100000000000000000000000000000000000000010000000200000003000000FEFFFFFF00000000FEFFFFFFFFFFFFFF0A000000FCFFFFFFFEFFFFFF030000000000000000000000FDFFFFFFFCFFFFFF00000000FFFFFFFF01000000FEFFFFFFFEFFFFFFFEFFFFFF000000000000000003000000FEFFFFFF00000000030000000200000002000000FBFFFFFF05000000FDFFFFFF03000000FFFFFFFF0600000000000000FFFFFFFF0100000000000000FBFFFFFF0800000002000000FEFFFFFF0300000002000000010000000100000001000000FDFFFFFF020000000100000000000000FDFFFFFF040000000000000002000000FEFFFFFF02000000FFFFFFFF02000000FFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFF01000000010000000000000005000000FFFFFFFF00000000000000000000000000000000000000000200000003000000040000000000000000000000FEFFFFFFFFFFFFFF0300000000000000FFFFFFFF000000000400000001000000FDFFFFFF050000000000000002000000FFFFFFFF0300000002000000000000000500000000000000010000000100000002000000FDFFFFFF0000000000000000020000000100000003000000FBFFFFFFFFFFFFFF0300000003000000FEFFFFFF0100000004000000FEFFFFFF0000000000000000FEFFFFFFFDFFFFFFFFFFFFFF0300000003000000FFFFFFFF03000000FDFFFFFF0000000000000000FEFFFFFFFDFFFFFFFDFFFFFF0400000000000000F9FFFFFF030000000100000001000000FFFFFFFFFEFFFFFFFFFFFFFF00000000030000000000000000000000FEFFFFFF00000000FFFFFFFF0000000000000000FCFFFFFF000000000000000000000000FEFFFFFF0100000001000000FFFFFFFF020000000100000000000000FEFFFFFFFFFFFFFF0000000004000000FEFFFFFF01000000FFFFFFFFF8FFFFFF01000000FFFFFFFF0200000003000000FDFFFFFF"> : tensor<20x20xi32>}> : () -> tensor<20x20xi32>
    %2 = "stablehlo.constant"() <{value = dense<"0x04000000FFFFFFFF0100000000000000FFFFFFFFFEFFFFFFFDFFFFFF00000000FCFFFFFF02000000F6FFFFFFFBFFFFFF010000000200000000000000FEFFFFFFFDFFFFFF01000000FEFFFFFF03000000FFFFFFFFFFFFFFFF0400000000000000FBFFFFFFFEFFFFFF0100000000000000060000000000000000000000FEFFFFFF0000000000000000FFFFFFFF00000000FCFFFFFF02000000FEFFFFFF0000000002000000FFFFFFFFFDFFFFFF00000000FEFFFFFF02000000FEFFFFFF030000000100000002000000050000000000000001000000FFFFFFFF000000000000000000000000FDFFFFFF02000000FFFFFFFF000000000100000001000000FFFFFFFF0300000001000000FEFFFFFFFDFFFFFFFDFFFFFF01000000FBFFFFFF000000000200000000000000FFFFFFFF0100000004000000FFFFFFFF00000000000000000000000000000000FCFFFFFFFAFFFFFF06000000FEFFFFFFFFFFFFFF03000000FFFFFFFFFFFFFFFF030000000000000000000000FEFFFFFF040000000000000000000000FCFFFFFF000000000000000002000000FFFFFFFFF9FFFFFFFEFFFFFF01000000FCFFFFFF04000000FFFFFFFFFDFFFFFF020000000300000002000000FDFFFFFF0100000002000000000000000000000000000000FFFFFFFF0000000000000000FCFFFFFF03000000FDFFFFFF0200000000000000FDFFFFFF0000000000000000FBFFFFFFFEFFFFFFF8FFFFFFFFFFFFFFFFFFFFFF0000000004000000FCFFFFFF020000000000000002000000FFFFFFFFF9FFFFFF04000000FFFFFFFF060000000000000001000000FAFFFFFF000000000600000000000000FDFFFFFF0100000005000000FBFFFFFFFFFFFFFF0000000000000000FFFFFFFF00000000FFFFFFFF010000000100000007000000FEFFFFFF0200000003000000FFFFFFFF02000000FDFFFFFFFFFFFFFF0400000006000000000000000200000000000000030000000000000001000000FFFFFFFF010000000000000003000000FFFFFFFF02000000FDFFFFFF00000000FFFFFFFF00000000FFFFFFFFFFFFFFFF0100000003000000FFFFFFFF00000000040000000100000000000000010000000100000001000000FEFFFFFFFBFFFFFF01000000FDFFFFFF00000000FCFFFFFF00000000000000000100000002000000FFFFFFFFFFFFFFFF03000000000000000100000000000000FEFFFFFF000000000000000000000000FEFFFFFF0100000000000000FCFFFFFF00000000000000000000000002000000FFFFFFFF00000000FDFFFFFF010000000000000001000000020000000100000001000000030000000000000003000000FDFFFFFF00000000FBFFFFFF000000000600000002000000FFFFFFFFFEFFFFFF01000000FFFFFFFFFCFFFFFF02000000FEFFFFFF00000000010000000200000001000000FDFFFFFF0000000000000000010000000600000004000000FFFFFFFFFEFFFFFF01000000FDFFFFFF030000000100000000000000FCFFFFFF0100000003000000010000000100000000000000020000000100000004000000FFFFFFFF00000000FEFFFFFF0600000003000000000000000000000000000000FFFFFFFFFDFFFFFF0200000004000000FFFFFFFFFFFFFFFFFCFFFFFF03000000FFFFFFFFFCFFFFFF000000000000000000000000FDFFFFFF03000000FCFFFFFFFFFFFFFFFBFFFFFF000000000100000000000000FFFFFFFF02000000FFFFFFFF0100000004000000020000000100000000000000000000000400000000000000FFFFFFFFFFFFFFFFFEFFFFFF0300000003000000FFFFFFFFFFFFFFFF02000000FFFFFFFF00000000FFFFFFFFFDFFFFFF0300000004000000010000000000000002000000FEFFFFFFFEFFFFFF0100000000000000000000000100000001000000FFFFFFFF04000000FEFFFFFF00000000FDFFFFFF000000000300000000000000FFFFFFFF0200000000000000000000000000000006000000FCFFFFFFFFFFFFFFFDFFFFFF020000000300000002000000060000000100000002000000FAFFFFFFFDFFFFFF00000000FEFFFFFF00000000010000000000000000000000050000000000000000000000FEFFFFFF0300000000000000FFFFFFFF01000000FFFFFFFF0100000000000000FEFFFFFFFCFFFFFF04000000FFFFFFFF0200000002000000FEFFFFFFFFFFFFFF010000000000000000000000010000000400000001000000"> : tensor<20x20xi32>}> : () -> tensor<20x20xi32>
    "func.return"(%1, %2) : (tensor<20x20xi32>, tensor<20x20xi32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xi32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFEFFFFFFFFFFFFFF01000000FFFFFFFFFDFFFFFFFAFFFFFFFDFFFFFF00000000FCFFFFFF00000000F6FFFFFFFBFFFFFF0000000000000000FFFFFFFFFDFFFFFFFDFFFFFF00000000FEFFFFFF00000000FFFFFFFFFFFFFFFF01000000FFFFFFFFFBFFFFFFFEFFFFFFFFFFFFFF0000000002000000FBFFFFFF00000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFAFFFFFFFCFFFFFF02000000FEFFFFFF0000000002000000FFFFFFFFFDFFFFFF00000000FEFFFFFF00000000FEFFFFFFFEFFFFFFFEFFFFFFFFFFFFFF000000000000000001000000FFFFFFFF0000000000000000FEFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFF7FFFFFF0100000001000000FEFFFFFFFFFFFFFFFDFFFFFFFEFFFFFFFDFFFFFFFDFFFFFF00000000FBFFFFFF00000000FFFFFFFF00000000FFFFFFFF00000000FEFFFFFFFFFFFFFFFFFFFFFF000000000000000000000000FCFFFFFFFAFFFFFFFCFFFFFFFDFFFFFFFFFFFFFF00000000FFFFFFFFFFFFFFFF03000000FCFFFFFF00000000FDFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFCFFFFFFFDFFFFFF0000000002000000FFFFFFFFF9FFFFFFFEFFFFFF00000000FCFFFFFF00000000FFFFFFFFFDFFFFFFFEFFFFFF02000000FDFFFFFFFDFFFFFFFCFFFFFFFAFFFFFF000000000000000000000000FEFFFFFFFBFFFFFF00000000FCFFFFFFFCFFFFFFFDFFFFFFFDFFFFFF00000000FDFFFFFF00000000FFFFFFFFFBFFFFFFFEFFFFFFF8FFFFFFFFFFFFFFFFFFFFFF00000000FFFFFFFFFCFFFFFF02000000FDFFFFFFFDFFFFFFFCFFFFFFF9FFFFFF03000000FFFFFFFF020000000000000000000000FAFFFFFF000000000200000000000000FDFFFFFF01000000FEFFFFFFFBFFFFFFFBFFFFFF00000000FCFFFFFFFEFFFFFF00000000FFFFFFFFFAFFFFFF0100000003000000FEFFFFFFFFFFFFFFFDFFFFFFFCFFFFFFFEFFFFFFFDFFFFFFFDFFFFFFFFFFFFFF03000000FFFFFFFF00000000FEFFFFFFFFFFFFFF00000000FDFFFFFFFAFFFFFF010000000000000003000000FFFFFFFF00000000FDFFFFFF00000000FFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF01000000FDFFFFFF00000000030000000000000000000000FAFFFFFFFDFFFFFF01000000FEFFFFFFFBFFFFFF01000000FDFFFFFFFCFFFFFFFCFFFFFF00000000000000000100000002000000FFFFFFFFFFFFFFFFFAFFFFFF00000000FDFFFFFFFCFFFFFFFEFFFFFF00000000FCFFFFFF00000000FEFFFFFF0100000000000000FCFFFFFF00000000000000000000000000000000FFFFFFFF00000000FDFFFFFFFEFFFFFF00000000FEFFFFFFFFFFFFFF01000000FCFFFFFFFEFFFFFF0000000000000000FDFFFFFFFDFFFFFFFBFFFFFF00000000FFFFFFFF01000000FEFFFFFFFEFFFFFFFEFFFFFFFFFFFFFFFCFFFFFF02000000FEFFFFFF00000000010000000200000001000000FBFFFFFF00000000FDFFFFFF01000000FFFFFFFF04000000FFFFFFFFFEFFFFFF01000000FDFFFFFFFBFFFFFF0100000000000000FCFFFFFF0100000002000000010000000100000000000000FDFFFFFF0100000001000000FFFFFFFFFDFFFFFFFEFFFFFF0000000002000000FEFFFFFF00000000FFFFFFFFFFFFFFFFFDFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFF00000000FFFFFFFFFCFFFFFF000000000000000000000000FDFFFFFF00000000FCFFFFFFFFFFFFFFFBFFFFFF0000000000000000FEFFFFFFFFFFFFFF02000000FFFFFFFFFFFFFFFF000000000200000001000000FDFFFFFF000000000000000000000000FFFFFFFFFFFFFFFFFEFFFFFF0000000003000000FFFFFFFFFFFFFFFF01000000FFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF020000000100000001000000FBFFFFFFFFFFFFFFFEFFFFFFFEFFFFFFFEFFFFFF0000000000000000FEFFFFFF00000000FFFFFFFFFEFFFFFFFDFFFFFFFFFFFFFFFDFFFFFF00000000FFFFFFFF00000000FDFFFFFF0000000000000000FEFFFFFFFDFFFFFFFDFFFFFFFCFFFFFFFFFFFFFFF9FFFFFF020000000100000001000000FFFFFFFFFEFFFFFFFFFFFFFFFAFFFFFFFDFFFFFF00000000FEFFFFFFFEFFFFFF00000000FFFFFFFF0000000000000000FCFFFFFF00000000FEFFFFFF00000000FEFFFFFFFFFFFFFF01000000FFFFFFFF0100000000000000FEFFFFFFFCFFFFFFFFFFFFFFFFFFFFFF02000000FEFFFFFFFEFFFFFFFFFFFFFFF8FFFFFF00000000FFFFFFFF0100000003000000FDFFFFFF"> : tensor<20x20xi32>}> : () -> tensor<20x20xi32>
    "func.return"(%0) : (tensor<20x20xi32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

