"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xui64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui64>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xui64>) -> tensor<20x20xui64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui64>, tensor<20x20xui64>) -> ()
    "func.return"(%4) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0200000000000000020000000000000004000000000000000100000000000000060000000000000006000000000000000100000000000000000000000000000004000000000000000400000000000000020000000000000001000000000000000000000000000000000000000000000002000000000000000100000000000000020000000000000001000000000000000200000000000000070000000000000005000000000000000100000000000000050000000000000000000000000000000200000000000000030000000000000003000000000000000300000000000000070000000000000001000000000000000200000000000000030000000000000000000000000000000100000000000000000000000000000002000000000000000100000000000000030000000000000001000000000000000400000000000000010000000000000000000000000000000200000000000000020000000000000000000000000000000000000000000000020000000000000004000000000000000300000000000000000000000000000004000000000000000100000000000000000000000000000001000000000000000200000000000000020000000000000001000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000001000000000000000000000000000000020000000000000000000000000000000300000000000000000000000000000000000000000000000200000000000000010000000000000005000000000000000200000000000000020000000000000001000000000000000100000000000000030000000000000002000000000000000200000000000000020000000000000000000000000000000000000000000000030000000000000001000000000000000300000000000000030000000000000004000000000000000400000000000000010000000000000000000000000000000300000000000000020000000000000000000000000000000900000000000000000000000000000004000000000000000300000000000000010000000000000000000000000000000100000000000000020000000000000001000000000000000000000000000000020000000000000001000000000000000100000000000000030000000000000005000000000000000100000000000000030000000000000003000000000000000000000000000000060000000000000003000000000000000100000000000000010000000000000001000000000000000100000000000000030000000000000003000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000003000000000000000400000000000000010000000000000001000000000000000400000000000000000000000000000001000000000000000300000000000000010000000000000000000000000000000000000000000000020000000000000000000000000000000200000000000000010000000000000003000000000000000400000000000000010000000000000000000000000000000100000000000000010000000000000000000000000000000200000000000000000000000000000001000000000000000200000000000000030000000000000001000000000000000000000000000000030000000000000001000000000000000100000000000000010000000000000001000000000000000100000000000000020000000000000006000000000000000600000000000000030000000000000000000000000000000000000000000000030000000000000004000000000000000500000000000000000000000000000002000000000000000400000000000000040000000000000002000000000000000300000000000000010000000000000000000000000000000300000000000000020000000000000003000000000000000500000000000000060000000000000001000000000000000400000000000000000000000000000003000000000000000300000000000000000000000000000000000000000000000300000000000000000000000000000000000000000000000000000000000000040000000000000002000000000000000600000000000000010000000000000001000000000000000300000000000000010000000000000000000000000000000000000000000000010000000000000001000000000000000100000000000000010000000000000003000000000000000100000000000000000000000000000002000000000000000000000000000000020000000000000003000000000000000200000000000000020000000000000001000000000000000000000000000000040000000000000000000000000000000100000000000000000000000000000001000000000000000500000000000000030000000000000004000000000000000000000000000000010000000000000000000000000000000000000000000000000000000000000000000000000000000500000000000000000000000000000002000000000000000400000000000000030000000000000002000000000000000300000000000000070000000000000001000000000000000400000000000000060000000000000004000000000000000300000000000000020000000000000006000000000000000100000000000000050000000000000003000000000000000200000000000000060000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000030000000000000001000000000000000000000000000000000000000000000004000000000000000300000000000000040000000000000003000000000000000000000000000000000000000000000000000000000000000300000000000000020000000000000001000000000000000600000000000000000000000000000001000000000000000000000000000000010000000000000003000000000000000000000000000000000000000000000001000000000000000200000000000000020000000000000005000000000000000000000000000000070000000000000003000000000000000000000000000000010000000000000000000000000000000000000000000000030000000000000004000000000000000000000000000000020000000000000003000000000000000100000000000000000000000000000001000000000000000000000000000000000000000000000003000000000000000100000000000000030000000000000000000000000000000200000000000000020000000000000001000000000000000400000000000000040000000000000001000000000000000300000000000000000000000000000003000000000000000200000000000000010000000000000003000000000000000700000000000000010000000000000001000000000000000100000000000000040000000000000000000000000000000100000000000000040000000000000002000000000000000000000000000000030000000000000002000000000000000200000000000000000000000000000002000000000000000000000000000000010000000000000000000000000000000600000000000000020000000000000002000000000000000000000000000000040000000000000004000000000000000200000000000000030000000000000005000000000000000400000000000000000000000000000002000000000000000000000000000000010000000000000002000000000000000100000000000000030000000000000006000000000000000200000000000000040000000000000004000000000000000000000000000000000000000000000007000000000000000100000000000000020000000000000003000000000000000100000000000000010000000000000000000000000000000000000000000000010000000000000000000000000000000000000000000000030000000000000001000000000000000000000000000000000000000000000001000000000000000000000000000000040000000000000000000000000000000000000000000000000000000000000000000000000000000100000000000000020000000000000004000000000000000300000000000000030000000000000002000000000000000200000000000000020000000000000002000000000000000100000000000000020000000000000001000000000000000500000000000000020000000000000002000000000000000000000000000000020000000000000002000000000000000100000000000000000000000000000004000000000000000400000000000000"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%1) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000F7FFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000FBFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF000000000000000000000000000000000000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFAFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFBFFFFFFFFFFFFFF0000000000000000F9FFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFF9FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFF0000000000000000FAFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFF0000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFAFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF00000000000000000000000000000000F9FFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF00000000000000000000000000000000FDFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF00000000000000000000000000000000FFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFF0000000000000000000000000000000000000000000000000000000000000000FFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFCFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFDFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFFBFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFF0000000000000000FEFFFFFFFFFFFFFFFEFFFFFFFFFFFFFFFFFFFFFFFFFFFFFF0000000000000000FCFFFFFFFFFFFFFFFCFFFFFFFFFFFFFF"> : tensor<20x20xui64>}> : () -> tensor<20x20xui64>
    "func.return"(%0) : (tensor<20x20xui64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

