"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xui8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xui8>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xui8>) -> tensor<20x20xui8>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xui8>, tensor<20x20xui8>) -> ()
    "func.return"(%4) : (tensor<20x20xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x00060202040400040304030002030301030002000101050104000100000301020004020203020000020000040202000202030101030100010001020004030307020000000102000102020004020005020100010003010401040102040505010702000302030203000200020403010300020200040002010300030301000104020500020103000002030001000201070107000303000204040301030107020102040003000202030402000100010300010000070301020102030302030102050102000000070304030501050101050303020502010703000002020300020007030300020202010103000301020001020104020102050100000003010301010106000705020103000004000000020005060700040202030003020102000700000005000205000501030003030202000400000003040A00020300020000010004030001000002030302000102000203030103020101000301000203000204030001030105030000010300010602000006030203020001050100010302000301040402010202010101020407000001020500"> : tensor<20x20xui8>}> : () -> tensor<20x20xui8>
    "func.return"(%1) : (tensor<20x20xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00FAFEFEFCFC00FCFDFCFD00FEFDFDFFFD00FE00FFFFFBFFFC00FF0000FDFFFE00FCFEFEFDFE0000FE0000FCFEFE00FEFEFDFFFFFDFF00FF00FFFE00FCFDFDF9FE000000FFFE00FFFEFE00FCFE00FBFEFF00FF00FDFFFCFFFCFFFEFCFBFBFFF9FE00FDFEFDFEFD00FE00FEFCFDFFFD00FEFE00FC00FEFFFD00FDFDFF00FFFCFEFB00FEFFFD0000FEFD00FF00FEFFF9FFF900FDFD00FEFCFCFDFFFDFFF9FEFFFEFC00FD00FEFEFDFCFE00FF00FFFD00FF0000F9FDFFFEFFFEFDFDFEFDFFFEFBFFFE000000F9FDFCFDFBFFFBFFFFFBFDFDFEFBFEFFF9FD0000FEFEFD00FE00F9FDFD00FEFEFEFFFFFD00FDFFFE00FFFEFFFCFEFFFEFBFF000000FDFFFDFFFFFFFA00F9FBFEFFFD0000FC000000FE00FBFAF900FCFEFEFD00FDFEFFFE00F9000000FB00FEFB00FBFFFD00FDFDFEFE00FC000000FDFCF600FEFD00FE0000FF00FCFD00FF0000FEFDFDFE00FFFE00FEFDFDFFFDFEFFFF00FDFF00FEFD00FEFCFD00FFFDFFFBFD0000FFFD00FFFAFE0000FAFDFEFDFE00FFFBFF00FFFDFE00FDFFFCFCFEFFFEFEFFFFFFFEFCF90000FFFEFB00"> : tensor<20x20xui8>}> : () -> tensor<20x20xui8>
    "func.return"(%0) : (tensor<20x20xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

