"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xui64>, tensor<20x20xui64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui64>
    %5 = "stablehlo.and"(%3#0, %3#1) : (tensor<20x20xui64>, tensor<20x20xui64>) -> tensor<20x20xui64>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui64>, tensor<20x20xui64>) -> ()
    "func.return"(%5) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xui64>, tensor<20x20xui64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02000000000000000400000000000000030000000000000005000000000000000400000000000000010000000000000003000000000000000200000000000000000000000000000002000000000000000000000000000000010000000000000004000000000000000000000000000000020000000000000003000000000000000000000000000000010000000000000007000000000000000600000000000000010000000000000004000000000000000000000000000000040000000000000000000000000000000200000000000000000000000000000001000000000000000000000000000000060000000000000000000000000000000100000000000000030000000000000001000000000000000400000000000000000000000000000003000000000000000C0000000000000000000000000000000400000000000000000000000000000000000000000000000100000000000000060000000000000002000000000000000200000000000000040000000000000001000000000000000200000000000000050000000000000000000000000000000400000000000000010000000000000003000000000000000000000000000000010000000000000002000000000000000300000000000000020000000000000005000000000000000500000000000000030000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000010000000000000004000000000000000300000000000000000000000000000000000000000000000100000000000000020000000000000001000000000000000000000000000000010000000000000002000000000000000300000000000000000000000000000003000000000000000200000000000000030000000000000003000000000000000300000000000000000000000000000003000000000000000400000000000000030000000000000000000000000000000000000000000000020000000000000000000000000000000100000000000000010000000000000000000000000000000100000000000000000000000000000003000000000000000200000000000000000000000000000000000000000000000000000000000000020000000000000000000000000000000400000000000000030000000000000001000000000000000000000000000000070000000000000004000000000000000200000000000000030000000000000007000000000000000000000000000000000000000000000001000000000000000000000000000000030000000000000001000000000000000400000000000000010000000000000007000000000000000000000000000000010000000000000001000000000000000100000000000000020000000000000000000000000000000000000000000000040000000000000001000000000000000300000000000000000000000000000000000000000000000000000000000000050000000000000001000000000000000100000000000000050000000000000002000000000000000300000000000000000000000000000001000000000000000000000000000000040000000000000000000000000000000100000000000000010000000000000002000000000000000400000000000000030000000000000000000000000000000100000000000000000000000000000004000000000000000200000000000000000000000000000005000000000000000200000000000000010000000000000006000000000000000400000000000000000000000000000004000000000000000A00000000000000020000000000000003000000000000000100000000000000000000000000000000000000000000000400000000000000000000000000000001000000000000000200000000000000010000000000000002000000000000000300000000000000040000000000000000000000000000000300000000000000030000000000000002000000000000000500000000000000030000000000000002000000000000000000000000000000040000000000000000000000000000000000000000000000050000000000000004000000000000000400000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000030000000000000004000000000000000000000000000000020000000000000006000000000000000000000000000000030000000000000000000000000000000200000000000000000000000000000000000000000000000400000000000000010000000000000000000000000000000000000000000000000000000000000004000000000000000000000000000000020000000000000002000000000000000300000000000000020000000000000000000000000000000000000000000000020000000000000001000000000000000100000000000000030000000000000000000000000000000400000000000000010000000000000001000000000000000200000000000000010000000000000001000000000000000200000000000000000000000000000001000000000000000100000000000000040000000000000003000000000000000000000000000000010000000000000003000000000000000200000000000000000000000000000001000000000000000000000000000000030000000000000000000000000000000500000000000000070000000000000000000000000000000200000000000000020000000000000002000000000000000000000000000000010000000000000004000000000000000200000000000000030000000000000006000000000000000000000000000000050000000000000001000000000000000000000000000000020000000000000000000000000000000300000000000000020000000000000005000000000000000000000000000000060000000000000001000000000000000400000000000000020000000000000003000000000000000100000000000000000000000000000000000000000000000500000000000000040000000000000003000000000000000300000000000000040000000000000000000000000000000600000000000000030000000000000005000000000000000100000000000000030000000000000001000000000000000100000000000000020000000000000005000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000001000000000000000200000000000000030000000000000000000000000000000100000000000000010000000000000002000000000000000200000000000000050000000000000000000000000000000100000000000000020000000000000006000000000000000600000000000000010000000000000001000000000000000100000000000000010000000000000000000000000000000600000000000000000000000000000000000000000000000100000000000000010000000000000001000000000000000000000000000000010000000000000006000000000000000200000000000000070000000000000001000000000000000200000000000000000000000000000001000000000000000200000000000000010000000000000001000000000000000000000000000000020000000000000002000000000000000100000000000000000000000000000001000000000000000300000000000000000000000000000003000000000000000200000000000000030000000000000005000000000000000300000000000000090000000000000001000000000000000000000000000000040000000000000005000000000000000000000000000000020000000000000000000000000000000200000000000000030000000000000001000000000000000100000000000000050000000000000001000000000000000300000000000000030000000000000001000000000000000100000000000000020000000000000000000000000000000100000000000000030000000000000001000000000000000000000000000000050000000000000000000000000000000100000000000000010000000000000000000000000000000300000000000000010000000000000003000000000000000000000000000000030000000000000001000000000000000500000000000000060000000000000002000000000000000200000000000000030000000000000005000000000000000500000000000000000000000000000001000000000000000300000000000000000000000000000001000000000000000100000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    %2 = "stablehlo.constant"() <{value = dense<"0x0300000000000000040000000000000002000000000000000400000000000000000000000000000002000000000000000000000000000000010000000000000001000000000000000000000000000000010000000000000001000000000000000000000000000000010000000000000001000000000000000300000000000000020000000000000003000000000000000100000000000000040000000000000000000000000000000200000000000000020000000000000000000000000000000100000000000000050000000000000001000000000000000100000000000000010000000000000000000000000000000100000000000000020000000000000004000000000000000200000000000000060000000000000002000000000000000100000000000000010000000000000003000000000000000600000000000000000000000000000001000000000000000000000000000000050000000000000003000000000000000000000000000000010000000000000003000000000000000200000000000000040000000000000003000000000000000000000000000000020000000000000002000000000000000000000000000000040000000000000000000000000000000100000000000000030000000000000001000000000000000200000000000000010000000000000001000000000000000200000000000000030000000000000001000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000003000000000000000100000000000000030000000000000002000000000000000000000000000000030000000000000000000000000000000300000000000000010000000000000000000000000000000200000000000000000000000000000000000000000000000600000000000000010000000000000003000000000000000200000000000000010000000000000001000000000000000000000000000000040000000000000000000000000000000600000000000000000000000000000001000000000000000100000000000000010000000000000001000000000000000200000000000000030000000000000006000000000000000400000000000000000000000000000004000000000000000100000000000000030000000000000001000000000000000100000000000000000000000000000004000000000000000100000000000000020000000000000004000000000000000300000000000000030000000000000004000000000000000400000000000000050000000000000004000000000000000200000000000000010000000000000001000000000000000000000000000000040000000000000001000000000000000300000000000000020000000000000000000000000000000100000000000000040000000000000003000000000000000000000000000000010000000000000000000000000000000300000000000000000000000000000001000000000000000100000000000000000000000000000003000000000000000300000000000000000000000000000003000000000000000300000000000000020000000000000002000000000000000200000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000020000000000000001000000000000000300000000000000030000000000000005000000000000000300000000000000030000000000000000000000000000000200000000000000000000000000000002000000000000000000000000000000020000000000000003000000000000000100000000000000020000000000000001000000000000000000000000000000030000000000000004000000000000000000000000000000010000000000000005000000000000000200000000000000000000000000000001000000000000000100000000000000000000000000000002000000000000000100000000000000000000000000000000000000000000000400000000000000020000000000000002000000000000000300000000000000000000000000000003000000000000000200000000000000060000000000000000000000000000000200000000000000050000000000000002000000000000000200000000000000010000000000000001000000000000000000000000000000080000000000000000000000000000000300000000000000000000000000000004000000000000000100000000000000050000000000000000000000000000000000000000000000010000000000000002000000000000000100000000000000020000000000000000000000000000000500000000000000030000000000000002000000000000000600000000000000010000000000000004000000000000000500000000000000010000000000000001000000000000000100000000000000010000000000000000000000000000000100000000000000040000000000000002000000000000000100000000000000000000000000000000000000000000000100000000000000000000000000000001000000000000000000000000000000010000000000000002000000000000000400000000000000050000000000000002000000000000000400000000000000040000000000000003000000000000000300000000000000050000000000000003000000000000000500000000000000010000000000000000000000000000000000000000000000050000000000000001000000000000000100000000000000020000000000000002000000000000000000000000000000040000000000000001000000000000000300000000000000030000000000000002000000000000000100000000000000000000000000000001000000000000000100000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000070000000000000001000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000300000000000000050000000000000000000000000000000100000000000000040000000000000000000000000000000000000000000000010000000000000000000000000000000100000000000000010000000000000000000000000000000300000000000000020000000000000001000000000000000300000000000000050000000000000001000000000000000000000000000000000000000000000001000000000000000100000000000000010000000000000002000000000000000100000000000000010000000000000001000000000000000200000000000000020000000000000000000000000000000000000000000000010000000000000003000000000000000300000000000000010000000000000004000000000000000000000000000000040000000000000001000000000000000200000000000000010000000000000002000000000000000200000000000000050000000000000004000000000000000000000000000000000000000000000001000000000000000200000000000000040000000000000005000000000000000100000000000000020000000000000000000000000000000000000000000000010000000000000002000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000002000000000000000100000000000000020000000000000003000000000000000100000000000000010000000000000005000000000000000400000000000000030000000000000004000000000000000000000000000000040000000000000002000000000000000500000000000000000000000000000000000000000000000400000000000000050000000000000004000000000000000200000000000000020000000000000003000000000000000400000000000000020000000000000000000000000000000200000000000000020000000000000003000000000000000200000000000000020000000000000001000000000000000500000000000000010000000000000000000000000000000100000000000000000000000000000000000000000000000300000000000000010000000000000002000000000000000000000000000000030000000000000002000000000000000100000000000000030000000000000000000000000000000400000000000000010000000000000002000000000000000000000000000000040000000000000003000000000000000000000000000000010000000000000001000000000000000000000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%1, %2) : (tensor<20x20xui64>, tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0200000000000000040000000000000002000000000000000400000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000300000000000000000000000000000001000000000000000100000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000100000000000000000000000000000000000000000000000400000000000000000000000000000000000000000000000000000000000000040000000000000002000000000000000000000000000000000000000000000001000000000000000200000000000000040000000000000000000000000000000000000000000000000000000000000002000000000000000000000000000000000000000000000000000000000000000100000000000000020000000000000001000000000000000000000000000000010000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000100000000000000020000000000000000000000000000000000000000000000010000000000000000000000000000000300000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000200000000000000000000000000000003000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000001000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000030000000000000001000000000000000000000000000000000000000000000004000000000000000000000000000000020000000000000004000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000010000000000000001000000000000000000000000000000000000000000000001000000000000000100000000000000020000000000000000000000000000000000000000000000040000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000100000000000000000000000000000002000000000000000300000000000000000000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000000000000000000000200000000000000000000000000000005000000000000000200000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020000000000000003000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000200000000000000000000000000000000000000000000000100000000000000000000000000000002000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000020000000000000002000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000100000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000002000000000000000000000000000000010000000000000002000000000000000000000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000500000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000020000000000000001000000000000000100000000000000000000000000000000000000000000000000000000000000000000000000000003000000000000000300000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000000000000000000000100000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000010000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000000000000000000002000000000000000200000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000200000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000200000000000000040000000000000001000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000000000000000000003000000000000000000000000000000010000000000000005000000000000000000000000000000010000000000000000000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000020000000000000000000000000000000000000000000000020000000000000000000000000000000000000000000000020000000000000001000000000000000000000000000000010000000000000000000000000000000100000000000000000000000000000000000000000000000300000000000000010000000000000002000000000000000000000000000000030000000000000000000000000000000100000000000000020000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%0) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

