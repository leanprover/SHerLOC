"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xui16>, tensor<20x20xui16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui16>
    %5 = "stablehlo.multiply"(%3#0, %3#1) : (tensor<20x20xui16>, tensor<20x20xui16>) -> tensor<20x20xui16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui16>, tensor<20x20xui16>) -> ()
    "func.return"(%5) : (tensor<20x20xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xui16>, tensor<20x20xui16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0400000003000000000003000000040001000100020000000000020000000100030000000200000002000300040000000000010002000000020003000100030004000100020002000200000005000100020002000100030001000300010001000400000002000000020000000300000001000000050002000100000001000100010001000400010004000000020001000500050000000300010001000000040000000200000003000400000001000200000000000000030000000200020002000100020001000000010000000100010004000200010000000600010001000200010002000300010003000300030002000100010003000000020002000200030000000300000002000200030001000000010000000400030002000100010002000000010002000500020000000100000004000000030000000400040001000100020003000500010001000400040000000100030000000100010006000000050002000300000001000100040004000000050005000000010000000300020002000000020001000400020001000400030003000200030001000000000001000200000001000200010004000400030000000100020002000200010005000100030005000100040000000000020003000200010000000000020000000000010001000000020000000000030001000200010003000100020001000300030002000000000004000000030002000000020003000300000002000400030000000700000000000100020002000100030003000000060005000100000003000200020000000200040000000400000000000100000002000000060002000200060000000000000002000300020003000100030002000300000001000000020001000000020000000500000000000000010003000400000001000200010003000000030000000300030000000300010001000500050000000000050001000000030004000400000002000200030000000100020001000100010002000100000004000100030002000200000001000100000000000000000000000000040000000200010001000000060006000300010002000300030002000200010000000000040006000000"> : tensor<20x20xui16>}> : () -> tensor<20x20xui16>
    %2 = "stablehlo.constant"() <{value = dense<"0x010004000100000002000200000000000700020000000200030003000100040004000300040001000100010003000100020003000400040002000200000000000000010003000300030002000200000000000000000000000000010002000200040001000200030007000200010002000300030003000100010002000100020000000100050000000400040001000000010000000300020002000100000003000300010007000100010004000100010004000700030003000100030006000000000002000200000003000100000001000400020001000200030004000400080000000100020001000500000001000100020000000100030001000800000004000200070004000100040000000000010001000300060002000000050008000200060003000200000002000100040000000000010000000000000002000100020000000500000001000000030001000100010000000200020000000400030000000500000000000000030002000100010002000000000004000400030001000100020000000200020004000100000003000000020003000100020003000200030000000000000001000100030004000200000003000000000003000400000007000100040000000200020003000200010001000400000000000000040002000100020000000000010000000100000002000400000005000300050003000200040003000300010005000100010004000000070005000100030002000600020002000100010000000200000005000400020002000100010001000000040002000500020004000400000002000100000003000100040002000A000200010000000000010001000200050000000300070001000100010000000300020000000200020001000500010000000500010000000500050005000200010002000000010000000300050003000600000001000200020004000000050004000200010006000400030004000400010004000000000002000100020006000100000002000300010002000100050002000300030001000100040003000100010003000000030003000000010002000200020001000500040002000200030001000200020000000700"> : tensor<20x20xui16>}> : () -> tensor<20x20xui16>
    "func.return"(%1, %2) : (tensor<20x20xui16>, tensor<20x20xui16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x04000000030000000000060000000000070002000000000000000600000004000C00000008000000020003000C000000000003000800000004000600000000000000010006000600060000000A0000000000000000000000000003000200020010000000040000000E00000003000000030000000F00020001000000010002000000010014000000100000000200000005000000000006000200010000000C00000002000000030004000000010002000000000000000900000006000C000000000004000200000003000000000001001000040001000000120004000400100000000200060001000F0000000300020002000000030000000200100000000C000000150000000200080000000000000001000000180006000000050008000400000003000400000004000000040000000000000000000000000008000100020000000F000000010000000C0004000000010000000000020000001800000000000A0000000000000003000800040000000A0000000000040000000900020002000000000002000800080001000000090000000400090001000000000002000600000000000000010004000C000C000000000006000000000003001400000015000500040000000000000006000600020001000000000000000000000002000100000000000000000000000100000002000C0000000A0003000F0009000400000000000C0000000F0002000000080000001500000002000C00060000000E000000000001000000040000000F000C0000000C00050001000000000008000400000004001000000000000000000000000000020000000C00140004000600000000000000020006000A00000003001500020003000000000000000400000000000400000019000000000000000100000014000000050004000100060000000300000009000F0000001200000001000A000A000000000019000400000003001800100000000800080003000000000000000200010002000C00010000000800030003000400020000000200030000000000000000000000000004000000000003000300000006000C000600020002000F000C0004000400030000000000080000000000"> : tensor<20x20xui16>}> : () -> tensor<20x20xui16>
    "func.return"(%0) : (tensor<20x20xui16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

