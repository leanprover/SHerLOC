"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xbf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xbf16>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xbf16>) -> tensor<20x20xbf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xbf16>, tensor<20x20xbf16>) -> ()
    "func.return"(%4) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xDCBFE13F7FBEFA405FBF92C0B03D5740DA40473F973FAB3EF6BED3BDE840C8C02F408FBDEA3FB340024091C099BF30C0C03F08C03DC0D03F2540E740453F06C15DC0B0BF03C18CBEF83F1FC04BC06440DBBE483FD33FA33FAA40953E4A40E43F2540863FB93EC83F7DC0D63F86C0873EC2C06B4031BF424075400A4019BF12401940773F6B4056BF9B40C03D4EC051C091407AC03B408440194088402A3F0EBFF03F19C0E23F223FB3BF40C050C0B5BF4BC0D4BF9C40AA3E07C06E3F96BED6BED33F80C0423CACC0E3BF5B4084C00CC0893F17BF773FAE3E6B3EB1C0863F1FC0C8BF9040C8BFAEBF29BFD9BF0D40E0BD11C0F2BE6EBF8A40DC3F7E3EAA401540E04012BE76403E3F6FC020BFA54071BE953FB64066C067C0B3BF0BC0C23FA040B33D5CC048BF7E3F51C04E3FFE3F103E2AC0A1C0B93F87C0933F07BF9D407040173F56C0AF40E03F88BF04BE8A3EBB3F92C0E23F883EC63FB93F8DBFC8BE81C019C0083FBFC0D6B9894045408DBFF0BEA03F22C01D3F86400F3F7C3F28BF8540B8BF88BFBDBF48BFF8BF5E3E714085C091C094C07BC091BF563F74BF8DBFF23EA73F0A3F3FBEDF40703C83BF55C0A04022C08E3F18416840AC3D75407CBF9440433F913F8ABE974000C04FBE5C4009BF2A40AC40A140193F4BBF7BC0A2C01A3F85C03940D3BF11401FBF26417E40F33FBBBD583F19C04D40443FAD3F914080BF27401DC0DF4061BF21BE4F40B6BFD840953E53C0B44098C0964089BE9FC01BC0B6BF573F98BE0F3C693F63C0B2BE95BF6940A2BF22C0804043C08E3E86C0C53FD04086C0E5BE7D3FE0BFA6C0A73E484010C0923F28C02AC00B3F8B40513E933F2E40B4BFCABF103FF63F0340D8BFCABE8C40AE40A24033C0B7BFB5BFC6C023C0C13F1840AA3F2B3ED93D2EC0244028BE76404B40BF4034404840E53DB2BF6A4054402FC030C03540BEBF7940C83F2ABF9740D03FBB40AF3FB440A0C00340913D4BBF47C07E3F893F873FFBBFA340B43F953F64BFEC3F0740C0BF673F52408B3F0340A9C040C031C053C02D405B4033407C40FE3F863FCE3DDEC01F4093C0634005C02E400FC0B7BFF1BFB5BF39400CC0FDBE5AC0C43F433E26BFC2C07DBF753F"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%1) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xDC3FE1BF7F3EFAC05F3F9240B0BD57C0DAC047BF97BFABBEF63ED33DE8C0C8402FC08F3DEABFB3C002C09140993F3040C0BF08403D40D0BF25C0E7C045BF06415D40B03F03418C3EF8BF1F404B4064C0DB3E48BFD3BFA3BFAAC095BE4AC0E4BF25C086BFB9BEC8BF7D40D6BF864087BEC2406BC0313F42C075C00AC0193F12C019C077BF6BC0563F9BC0C0BD4E40514091C07A403BC084C019C088C02ABF0E3FF0BF1940E2BF22BFB33F40405040B53F4B40D43F9CC0AABE07406EBF963ED63ED3BF804042BCAC40E33F5BC084400C4089BF173F77BFAEBE6BBEB14086BF1F40C83F90C0C83FAE3F293FD93F0DC0E03D1140F23E6E3F8AC0DCBF7EBEAAC015C0E0C0123E76C03EBF6F40203FA5C0713E95BFB6C066406740B33F0B40C2BFA0C0B3BD5C40483F7EBF51404EBFFEBF10BE2A40A140B9BF874093BF073F9DC070C017BF5640AFC0E0BF883F043E8ABEBBBF9240E2BF88BEC6BFB9BF8D3FC83E8140194008BFBF40D63989C045C08D3FF03EA0BF22401DBF86C00FBF7CBF283F85C0B83F883FBD3F483FF83F5EBE71C08540914094407B40913F56BF743F8D3FF2BEA7BF0ABF3F3EDFC070BC833F5540A0C022408EBF18C168C0ACBD75C07C3F94C043BF91BF8A3E97C000404F3E5CC0093F2AC0ACC0A1C019BF4B3F7B40A2401ABF854039C0D33F11C01F3F26C17EC0F3BFBB3D58BF19404DC044BFADBF91C0803F27C01D40DFC0613F213E4FC0B63FD8C095BE5340B4C0984096C0893E9F401B40B63F57BF983E0FBC69BF6340B23E953F69C0A23F224080C043408EBE8640C5BFD0C08640E53E7DBFE03FA640A7BE48C0104092BF28402A400BBF8BC051BE93BF2EC0B43FCA3F10BFF6BF03C0D83FCA3E8CC0AEC0A2C03340B73FB53FC6402340C1BF18C0AABF2BBED9BD2E4024C0283E76C04BC0BFC034C048C0E5BDB23F6AC054C02F40304035C0BE3F79C0C8BF2A3F97C0D0BFBBC0AFBFB4C0A04003C091BD4B3F47407EBF89BF87BFFB3FA3C0B4BF95BF643FECBF07C0C03F67BF52C08BBF03C0A9404040314053402DC05BC033C07CC0FEBF86BFCEBDDE401FC0934063C005402EC00F40B73FF13FB53F39C00C40FD3E5A40C4BF43BE263FC2407D3F75BF"> : tensor<20x20xbf16>}> : () -> tensor<20x20xbf16>
    "func.return"(%0) : (tensor<20x20xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

