"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xui32>, tensor<20x20xui32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui32>
    %5 = "stablehlo.shift_left"(%3#0, %3#1) : (tensor<20x20xui32>, tensor<20x20xui32>) -> tensor<20x20xui32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui32>, tensor<20x20xui32>) -> ()
    "func.return"(%5) : (tensor<20x20xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xui32>, tensor<20x20xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x06000000000000000200000001000000020000000000000000000000030000000000000002000000040000000000000001000000010000000300000005000000020000000100000001000000030000000200000000000000000000000000000000000000040000000100000003000000010000000300000000000000000000000000000002000000020000000300000001000000020000000100000004000000010000000200000000000000000000000100000002000000040000000300000000000000030000000000000000000000040000000300000003000000030000000000000004000000030000000300000001000000020000000000000004000000020000000100000001000000050000000500000000000000000000000200000000000000050000000100000002000000000000000000000002000000000000000000000000000000000000000100000006000000050000000400000000000000020000000100000004000000050000000000000006000000020000000200000000000000040000000000000002000000000000000400000002000000000000000000000003000000000000000000000001000000000000000200000000000000010000000000000006000000030000000100000000000000060000000200000007000000010000000300000002000000000000000100000003000000010000000000000003000000010000000100000004000000050000000200000002000000020000000100000003000000000000000400000001000000010000000300000001000000040000000600000001000000020000000000000001000000040000000300000001000000000000000300000003000000020000000000000003000000000000000200000004000000060000000300000002000000010000000300000005000000040000000200000006000000050000000000000001000000040000000300000003000000000000000200000000000000000000000300000001000000010000000000000004000000020000000300000000000000010000000200000000000000000000000000000000000000020000000300000002000000000000000100000004000000010000000600000004000000010000000100000001000000040000000000000002000000030000000000000004000000010000000200000000000000000000000400000000000000030000000100000000000000000000000000000005000000000000000200000001000000000000000400000001000000020000000400000000000000020000000000000001000000020000000100000000000000000000000000000002000000010000000200000001000000000000000200000007000000000000000000000000000000020000000100000001000000010000000200000000000000060000000500000001000000020000000200000000000000010000000100000001000000040000000300000001000000040000000100000000000000020000000000000000000000040000000200000001000000040000000100000003000000010000000100000004000000000000000300000000000000010000000400000000000000000000000700000002000000030000000600000001000000000000000000000002000000000000000300000007000000010000000200000001000000000000000500000002000000010000000400000001000000040000000000000001000000000000000300000000000000040000000200000000000000010000000400000000000000020000000100000000000000010000000500000000000000020000000000000001000000020000000100000005000000000000000500000005000000010000000300000002000000010000000000000002000000020000000000000000000000010000000000000005000000010000000100000004000000010000000200000000000000040000000100000002000000010000000100000002000000030000000000000003000000030000000500000003000000000000000500000000000000000000000400000002000000020000000500000003000000020000000500000000000000010000000200000001000000010000000200000001000000020000000100000004000000030000000100000000000000010000000300000001000000000000000000000001000000"> : tensor<20x20xui32>}> : () -> tensor<20x20xui32>
    %2 = "stablehlo.constant"() <{value = dense<"0x02000000000000000000000001000000010000000200000000000000000000000200000002000000000000000000000002000000000000000000000001000000020000000200000003000000000000000400000001000000030000000200000001000000020000000100000004000000010000000000000004000000050000000500000002000000030000000100000003000000030000000400000002000000030000000200000002000000050000000300000000000000000000000100000005000000030000000000000003000000020000000200000003000000000000000000000000000000010000000000000001000000020000000100000000000000000000000100000001000000020000000000000000000000010000000300000000000000010000000300000003000000020000000300000001000000010000000100000005000000010000000200000001000000030000000000000001000000000000000500000000000000000000000200000006000000020000000100000004000000020000000600000002000000000000000000000003000000020000000100000000000000020000000100000000000000010000000300000001000000000000000000000000000000000000000000000001000000030000000200000002000000010000000500000004000000030000000300000001000000010000000400000004000000010000000A00000003000000000000000000000000000000030000000500000000000000000000000000000001000000010000000700000006000000000000000100000000000000010000000100000000000000000000000000000002000000060000000300000005000000020000000000000000000000000000000000000000000000030000000400000000000000050000000000000003000000020000000000000001000000070000000000000001000000020000000100000000000000040000000200000001000000020000000300000003000000000000000100000000000000000000000100000000000000010000000400000004000000000000000600000000000000020000000300000000000000010000000200000000000000000000000600000006000000000000000300000002000000030000000000000002000000050000000000000002000000020000000200000000000000040000000100000002000000050000000200000001000000020000000300000002000000000000000400000000000000020000000000000001000000010000000000000002000000040000000000000005000000030000000000000001000000020000000100000002000000030000000100000000000000020000000200000000000000010000000000000000000000030000000300000001000000030000000700000000000000010000000100000004000000010000000100000001000000000000000100000001000000060000000400000005000000000000000700000001000000020000000200000001000000020000000400000004000000030000000300000002000000020000000100000002000000020000000000000003000000010000000000000009000000020000000000000002000000000000000200000002000000010000000100000004000000050000000300000004000000020000000000000001000000020000000000000007000000000000000300000003000000030000000200000000000000000000000000000005000000000000000300000004000000010000000100000001000000010000000400000000000000060000000200000003000000010000000600000003000000000000000100000004000000000000000200000004000000020000000000000000000000000000000000000000000000000000000000000003000000030000000000000001000000010000000300000001000000020000000100000000000000010000000100000000000000020000000600000000000000000000000000000000000000030000000400000001000000000000000000000001000000020000000000000005000000000000000000000003000000020000000200000000000000010000000100000004000000010000000200000002000000000000000100000005000000010000000200000000000000010000000000000001000000000000000200000000000000"> : tensor<20x20xui32>}> : () -> tensor<20x20xui32>
    "func.return"(%1, %2) : (tensor<20x20xui32>, tensor<20x20xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x1800000000000000020000000200000004000000000000000000000003000000000000000800000004000000000000000400000001000000030000000A000000080000000400000008000000030000002000000000000000000000000000000000000000100000000200000030000000020000000300000000000000000000000000000008000000100000000600000008000000100000001000000010000000080000000800000000000000000000000800000002000000040000000600000000000000180000000000000000000000100000000C000000180000000300000000000000040000000600000003000000020000000800000000000000040000000200000002000000020000001400000005000000000000000000000010000000000000000A000000080000001000000000000000000000000400000000000000000000000000000000000000040000000C00000028000000040000000000000002000000200000000400000005000000000000008001000008000000040000000000000010000000000000000800000000000000040000001000000000000000000000000300000000000000000000000100000000000000100000000000000001000000000000000600000003000000010000000000000030000000080000001C000000020000006000000020000000000000000800000006000000020000000000000030000000020000000004000020000000050000000200000002000000100000002000000003000000000000000400000002000000020000008001000040000000040000000C0000000100000004000000000000000100000004000000030000000400000000000000180000006000000008000000000000000300000000000000020000000400000030000000300000000200000020000000030000002800000010000000020000000C0000008002000000000000020000001000000006000000030000000000000008000000000000000000000018000000080000000100000000000000040000000200000006000000000000000200000020000000000000000000000000000000000000000800000018000000020000000000000004000000040000000100000080010000000100000100000008000000040000002000000000000000080000006000000000000000100000000400000008000000000000000000000008000000000000006000000004000000000000000000000000000000140000000000000020000000010000000000000004000000020000000400000004000000000000002000000000000000200000001000000001000000000000000000000000000000080000000800000004000000010000000000000008000000070000000000000000000000000000001000000008000000020000000800000000010000000000000C0000000A0000001000000004000000040000000000000001000000020000000200000000010000300000002000000004000000800000000000000008000000000000000000000010000000200000001000000020000000080000000C00000004000000020000001000000000000000030000000000000002000000040000000000000000000000070000000800000003000000180000000400000000000000000000002000000000000000180000007000000004000000020000000200000000000000050000000001000001000000200000000800000020000000000000000100000000000000030000000000000004000000100000000000000002000000080000000000000004000000100000000000000040000000140000000000000004000000000000000800000002000000020000005000000000000000140000005000000004000000030000000200000001000000000000000200000002000000000000000000000008000000000000000A000000020000000800000008000000040000000400000000000000080000000200000002000000040000004000000002000000030000000000000003000000180000005000000006000000000000000500000000000000000000000400000040000000020000000500000018000000080000001400000000000000020000000400000010000000020000000800000004000000020000000200000080000000060000000400000000000000020000000300000002000000000000000000000001000000"> : tensor<20x20xui32>}> : () -> tensor<20x20xui32>
    "func.return"(%0) : (tensor<20x20xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

