"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xi8>, tensor<20x20xi8>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xi8>
    %5 = "stablehlo.xor"(%3#0, %3#1) : (tensor<20x20xi8>, tensor<20x20xi8>) -> tensor<20x20xi8>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x20xi8>, tensor<20x20xi8>) -> ()
    "func.return"(%5) : (tensor<20x20xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xi8>, tensor<20x20xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFEFF00FE020105FEFB00010001FCFE0306FF000202000000000203010003FEF9FD07FC040002050301000001010103FEFC010202FEFD0101FF00030700FFFFFEFE0003000201FCFE02FEFD03FF0000020102FBFF030000050200FC0100FFFF02FF04FE010204FF010000FD000000FD0005010003FE00FEFE00FC00FFFDFFFEFE0103FEFEFEFF0307FF010605FA0000000002FEFE000000FD0202FFFE01FD0004FCFAFCFF010200FC0000FDFBFEF8050103FC02FE00000100FEFB0000000001FF0005000000FFFF030000000204000001FCFFFFFF0000000400FFFEFB000003F90300FB03030001000102FC00020100FD000003FA01FDFFFFFBFCFB0001FCFDFD050202FEFEFFFE0200FEFE0000FEFEFE010000030001FE00000200FE01000204000100FF04030100FEFC0000FF0003FBFFFFFEFEFF00FFFDFF020105040400FFFD0000FE0003FE00FEFCFF0002FD030005FB0302FB020101FEFF0000FDFC00FF0003FD01FD020003FA0101FF02FF050000FDFAFC0202FC00FE0001FFFFFE0000FE01030002FFFD0701FF01FF0003FE00"> : tensor<20x20xi8>}> : () -> tensor<20x20xi8>
    %2 = "stablehlo.constant"() <{value = dense<"0xFF000004000000FEFE00FCFF00030200010000030003FDFF0400FFFD00FFFDFE030000040301FC030003FF020006FE0200000002030101040003000001000401030101FEFF0000020100FF01FA000200020001000106FF03FEFE050201FE0400FCFD01070000040000000100020000FE020000FAFB03010200FEFE000001010000FE03FF070100FBFC01FD00FFFE02FDFBFDFFFF00FDFC000201000005FB04FF0100FE00FFFEFEFB0201FDFC0200000201060100FFFDFF0000000101FF01FEFF0100FB03F900FF07FF00000102FD00FC000003FCFC04FE0400FEFF04FF05010000000104FA06010000FFFCFD00FE00FE0000010003FF00FEFE00FFFF00FF04FF00FC01040400FF040100FF02080506FEFD030104020102010002FE00FD000000FD0001FEFF01030001FD00030003FC01060002FD00FEFDFC03FFFE010100080305FF030201FF01FB05FF06000202FDFF0400FC000002020002FFFE01FCFC04FE010000FD01FFFBFD01FB020101FC00FF000405FDFF01000400F70003FFFF0001FD0200FAFAFAF8040002FD0300010001"> : tensor<20x20xi8>}> : () -> tensor<20x20xi8>
    "func.return"(%1, %2) : (tensor<20x20xi8>, tensor<20x20xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01FF00FA020105000500FDFF01FFFC0307FF00010203FDFF0402FCFC00FC0307FE07FC000303F9000103FF030107FDFCFC010200FDFC0005FF03030701FFFBFFFD0102FEFD01FCFC03FE0202050002020302FAFF0206FF06FCFEF9030101FB0203F9FF060204FB010000FC000200FDFE070100F90503FFFC0002FEFFFDFEFFFE01FDFD01F9FE03FC0300FB0505FE02FDFBFF010100FDFCFD0003FFFE040604FBFDFA02FFFEFCFE0702010007FCF8050302FA03FEFFFDFE00FEFB0101FF01FF000105FB03F9FF0004FF00000306FD00FDFCFFFC03FC04FE00000101FFFF0502F90300FA07F906000001FD00FD02FF0003000002FA0202FF0105FC04FF0103F90205FE03FAFAFF010601FE010208FBF800FC0301070200FC010000FEFEFC000204FD010101FB020200FF010003FF03FFFAF9FFFC03FFFE0201FCFDFF04050408FCF8FF03FC01FCFFFBFB03F90000FFFEFF01FBFF02FB000301FC00FE01010004010103FDFCFCFDFBFEFBFA03FE030305FF00F9FF01FD03FC04FEF701FC00010001030303FAF805050301FDFCFC0002FE01"> : tensor<20x20xi8>}> : () -> tensor<20x20xi8>
    "func.return"(%0) : (tensor<20x20xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

