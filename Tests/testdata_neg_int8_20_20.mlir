"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xi8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xi8>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xi8>) -> tensor<20x20xi8>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xi8>, tensor<20x20xi8>) -> ()
    "func.return"(%4) : (tensor<20x20xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFD00FD0003FDFE05FFFF0401020103FFFC00FEFBFE06000200FEFF05FEFC0102020300FEFD00000002FD030400FB01FC0501FF0104000301FFFB00050300FEF900F8FFFB01F9FF01FDFD0300FAF800FF0301FFFE01FFFC010402FE0503FFFA02FE020200FF040200FFFF02FB000302FE01FBFDFFFF04030000FE00FFFFFF0302FC02FD03FE0202FDFD030103030500F903FC02040303FD0102FD0502FBFE02FF0101FBFD01FEFB0000000003FF0002030100000100FFFF00060403FC030400FCFC0001FE02FE00FF04FBFDFA00FFFF000200FEFD00FF000301F9FD00000001FD010000040303FD050300FF00FF03FE01000200FDFE000001FE0300000001000101FD00000100FE02FCFF03FDFC02FBFCFE03000700FDFE0002020201FF0100FDFEFD000301FB00FD0200010101FC03FF0003000100FF000104030200020102FEFCFE04020000FD00FC03FE0003FE01030200FF000301FD02020000010300FE000000FF0201020202FDFEFF0002020202010003FE0104FD0300FF01FDFFFC00000000FE00FD0001F800FDFF0001FF02"> : tensor<20x20xi8>}> : () -> tensor<20x20xi8>
    "func.return"(%1) : (tensor<20x20xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0103000300FD0302FB0101FCFFFEFFFD010400020502FA00FE000201FB0204FFFEFEFD000203000000FE03FDFC0005FF04FBFF01FFFC00FDFF010500FBFD00020700080105FF0701FF0303FD0006080001FDFF0102FF0104FFFCFE02FBFD0106FE02FEFE0001FCFE000101FE0500FDFE02FF05030101FCFD00000200010101FDFE04FE03FD02FEFE0303FDFFFDFDFB0007FD04FEFCFDFD03FFFE03FBFE0502FE01FFFF0503FF020500000000FD0100FEFDFF0000FF00010100FAFCFD04FDFC00040400FF02FE020001FC05030600010100FE000203000100FDFF0703000000FF03FF0000FCFDFD03FBFD00010001FD02FF00FE0003020000FF02FD000000FF00FFFF030000FF0002FE0401FD0304FE050402FD00F900030200FEFEFEFF01FF0003020300FDFF050003FE00FFFFFF04FD0100FD00FF000100FFFCFDFE00FEFFFE020402FCFE0000030004FD0200FD02FFFDFE000100FDFF03FEFE0000FFFD000200000001FEFFFEFEFE03020100FEFEFEFEFF00FD02FFFC03FD0001FF0301040000000002000300FF0800030100FF01FE"> : tensor<20x20xi8>}> : () -> tensor<20x20xi8>
    "func.return"(%0) : (tensor<20x20xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

