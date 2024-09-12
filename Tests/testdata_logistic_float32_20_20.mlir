"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf32>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %5 = "stablehlo.exponential"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %6 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %8 = "stablehlo.add"(%7, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %9 = "stablehlo.constant"() <{value = dense<1.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %11 = "stablehlo.divide"(%10, %8) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%11, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%11) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x2C64053F0BE2853CEF295C40EC70283F6F6812C0C732593E548CB2BF53C0AC4070E579C031AAA940F1432BBFB4E86840130DDCC0266466BFEDDD86C0C5CFD83F978DD5BF522D19C0CE9074401A5EFF3E872E8D40520C5EC0497E4F408FADBDC026DF53406985C13F25B3A0C03DE6983F70323D3FA64044C0091B1FC0184FF33E154AD9404D1A383FAB5F2DC0AEEEC13DC069FFBF93BC753F59119CBED01F6A3F090429C0244EB2BF06DE06C0989278400BBD18BF07D57BC0B2D431BF3CF532C0F4820BC0C2677640F0A1F73FF977C9C02F57754069AA25BFAD9F4D40A2DA3740E09E36408E3968BF0FDB3BBEEC27093FA53407BEF01D0A40995C40C006959A3FA71E1DC0D2BA55409EF48BBE0643A440D5E30BBB5EB493BF6330B3BE8AC78340F301534043BE933E24FBB43F3562A43E600A94BE7C5FBB3F4FE1DA3E84A6C5C0DEDB983FD7848A3F69342D3FC7FA3EC098048EC07A5854C0D2AFA4C0FB64C7C090A78E3FD2CF91C06935B73FA833BC408C52BF4038EFE1C0A556F3BFCC84564033BB0FC0E2D8E83F9B6188C0D41A7640D6C32CC01D3EA3C0D83602BF0E2E90C08FE4823ED0D224C0EF5B3D3F3AE01E412B823A3FC3F3BD40868144C0069F9E4000ABE7BF45DD4140BE7A3FC04D9F9CBF4FEB9CBF58BBF33F2CE73EC0ABBF2E3F21B253C0020385C09C73724090542DBFE4A91C40382E2E401307DA404AFD7B40CEA344C010A242BF5D0D6240594036C0B7EA73BF3FF7CEBF3B1C55C02FE324BFBC7A22BF4A7441C021D946C03AA68EC0C74AD1C03F5D023F9DD5033FEC9AE63F3F6B08C08C8CEABF393427C0A3069B40234FFDBF8EDB6F40CE31C13FB9D80AC0D1E553BF0D24C8BDA002E03E691AC5BF1E33FCBF40B49140F47117C05E90CDC02AB244402EEDB33F4E9295C0BD4ADE4025988BB86E81CD3F4C697F4093221FC01AD9C83F915FB4C0DE76BC3F50928640B49E3E4010BC71406494A7C06CA97FC05F92A0BA4034F0C0BD67D0404F984AC0F15244BF3BC368C0C02543C0831278BFD80C2BC020943B40A8F50740603BFD3F769361C0C0BDF73FD11002C09E6283BE7E6E30BE48ACC63FC4BCA73FB33B16416F546E40C2B21240C7ACB440090C8EC077F01E3EC8831140893DCF40DBF058C0C23A99C0D9EE043FD78B2FC0407CA5BE2291ABC05608CD3EE7B5CA40F2E26B3F3F27CD3EC236063FA55E4440726AEB3E49B90FC0FE660BC0A8AD83C02C0E8C3FCA81843FF24DAA3F69303ABE6A96A8BFCC0CC3BF2402A2BF79B6204048B6FBBD45A4E240016E59C05C1EA6C0518985BFBC974BC0241D35C08F4020408FC1964044C7344012532340152B8BBF74AD853FB171C73E77D21DC0BCB8BABF49C65E404B4946BFD861E13F276D2BC07FAB23C0AC0EE13F198F70BFE52D9BBFF335793FDDCFB9BE84D9CB40EE4EF83FDF3E8D4069C7C7C002CC76400416EF3F71D31FC0587F04402F67A6BF1E560ABFB7E80140FDEC47C0F34B6140231E8940929FDEBFEA78DEBFC1D5A8404FE82ABFEA8BACBFD88E333F7FABBEC0615FF9BFD353B13F733EA83F501E50402674BCBF00CEDEBFA087D43FEAB569BFBD5B683F769D4B3FCE75F53F0EA146C0386DF43F10440E40807CA8BF03013F40F00320BD24776D3FE93F04C07C1C4A3FD5F12C40BB2C863FA4877440FDC897BE43B6E93FA0C129C058CC21C045D95A401AFC7440E0343BBFF189A240943904C0BCCFAC3F100DBDC0316A5EBF2AFC413E49945F3F5A1C9740371E0BC0F38C8E3F2135E2BF651D3240F8B62D4036A84CC08F38AC40E567B83F41BE183DE3BF533EF3CE3EBF2BAC1D3FAFF60240540B743FAED754BF99A90AC02E320C3F24D05E4057AA313E5541F93F6152B340722687409DCF94BCC0156C3FFE910641B3C86E4012C94F40C5C71BC03F17A2C0AD1A334006FAEC3F5D1FA24089D7E33F833CE33F115FE93E5C0DF2BF6A043BC049A4523F6BE429404CFD15C01289D6C074451140B7D0F6BE834E88C008359E3E0FCB3540FB21B63EFB2A21BFED8986C00E4E6AC0C2AA3140FA7C83C0D06CC63FB0CBD6C0013C0341BA25BEBFE1D011404ADE45C0C56C77C0105288BFBF283CBFFEB25140303028BFE60C9A40395468BF0499423EE04B59BFA89A423D54B6A740B281C23F77FAE73E4F6735C040ADCF407D0FEFBFC27FEBBE269C58BF3C0C29400DCA8BBE8FD4AE3E595E153F3BD4D3C041CC1D3FBD625940E480E4BFDC59B440234529404124C53E717D0AC0"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%1) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFE9C203FC30B013F040C783F71A7283FC8BABC3D34860D3F79644B3EE2D87E3FC9CBA13C26BB7E3FA66CAD3E3D72793FC014873A5AFF933E4A9A6E3C2140583F7A72223E665FAB3D9D837A3F78461F3F78EE7C3F6356F73C085F763F8D352E3B4AFC763F47C0513F3899D63B227E443FFF412D3FFF52363D3B609D3D99DA1D3F61B67F3FB9232C3FE0C77F3D4D0E063F811DF53D501D393F4A48D93E1CC4363F684D883DB5B34B3E28FBDD3D21D77A3F82D1B53E53119D3CDA7EAA3E6CA36B3D8005D03D94AA7A3FCBAE5F3FBF3BF13A29947A3F17F1AF3ECB18763F994C723F100C723FB73E933E6B95E83EDD7D213FB21FEF3EDC7B653F9736413D4F17453F8AF1A13D103C773F343ADD3EC07F7E3F0FBAFF3E3886753E8BA7D33E77E67B3F0ADE763F3F57123F04EC4D3F8D5F143FDE3EDB3E2BE74F3F68F31A3FDFDE073B6E7A443F3D353F3FBEB8293F613A453DC1603F3C83340F3DD89ABD3B8EAD003B50C2403F49342A3C889E4E3F96497F3F7F5A7F3F33D4603A131B053EA156773FCF04C43D21475C3F32BA633C49A47A3F1B09813D084EC63BE042C03E0E03333CE845103F7EE0903D134B2D3FCEFC7F3F29AB2C3F47537F3FB9A2353D32367E3F6C06103E0331743F57C4433DBBC1683EFF56683EFCCF5E3FEB73453DF8102A3F6A9D103DE09F7C3C7F557A3F2780AC3EB29F6B3F9D33703F0EB87F3F7C1A7B3FD745353D7C1DA33EE5B9783F8977603D39808E3EEB9A293E26900D3D014BB03EC161B13E3C183E3D53632F3D8AAE3B3CCFFCBC3A91E71F3FC03F203F6DBC5B3F5B3AD93D02430D3ED9F28B3D31007E3F2CAFF83DE21A7A3F82A7513FACF8D13D6FB69B3E4B80F33E20901B3FADCA343E9395FA3DEA547D3F96ABAF3D384DD43A11AE743FD8964D3FE882173C06C17F3FD2FDFF3E8F31553F915A7B3F1E4F9D3D75E1533FE1C9683B323C503FDE3C7C3F6D9B733F85457A3F894DAD3B0E1C943CDAD7FF3E9FFC0F3ADB9E7F3F92D1253DC361A23E4130D23C235B393D6CD68C3E3450843DD808733FFBAB643FE3E5603FB673EA3CEFB45F3FAB45ED3D3055DF3E1D00EA3EB841533F619F493F86FA7F3F48F7793F7B81683F62197F3FCA343F3CEFE9093FB31B683F429B7F3F659B053DAF47073C9081203F97C2773D30FCD63E4E18993BBF4A193FF78B7F3F1C20373F764E193F34CE203FE89F743F55EB1C3F1C0AC43D3657D03DB499833C81C93F3FA6E83C3FFE794A3F4FCAE83EC35F583E2C31373E9B46613E7BC76C3FAC49F03E06C97F3F8E9F043DAB5AB53B3F63853E1B5B233D9644643DA2A66C3FC9B77D3F99A9713F037E6D3FFA17813E4C5C3D3FB79E183F7450A03D352F413EF75A783F7B88A13E80725A3F6F96833D8952933DA95D5A3F8DDA8F3E5FCA6A3ED8CE393FDC0CD23E05907F3FE3D45F3F02F07C3FC349FE3ABCB27A3FF6BF5D3F9CBF9B3DA952633F054D5B3E9A77BC3ED846623F43952C3D6BA4783F88857C3FA5FE183EF725193EA8B27E3FB695AD3E9824533EAB222B3FD2E7283B797BFF3D1DC34C3FB8CA493F1A76763F86123F3E77CF183E421D573F0DA3923EA567363F3961303F55335F3FA5F62F3D0BF85E3F51FC663F5282583E7EAD733F0B00FB3E4F72373FED34E63DA30E303FB6E96F3F438D3D3FD8827A3F4854DA3E2A7C5C3F64D6863D105F973D15E3773F968C7A3F3E5BA63EA06A7E3F2549E63D0A4D4B3F85A5313B184C973E83160C3FF497343F2ABE7D3F662CD13D69B8403FE762153EB416713FE917703F73C4203DF9D37E3FB0FD4E3FE762023FF92F0D3FBDC7A43EB337263FF8B4623F70C6383F22509B3E9683D23DD332223F1B5C783F89130B3F000A603F770F7F3FE24D7C3FC5ACFD3E752A373F6DF17F3FF3017A3FD669763F9F17A53DF386CD3BE64D713F0C425D3F59657E3F120F5B3FBAE85A3F39AD1C3F1946063E702E513DACE0313FC12D6F3F955BB33DC575A03A9206683FC776C33E5540643CBA9E133F26E0713F8E87163FEAF9B13E9506713C2F4FCD3C78FD703F415F843C662F533F7C299F3A0BEE7F3F40063D3EBF35683FECF9313D0B09A83CDE40833E54F0A53E81AF763F3ACEAE3E7AF07D3FCA33933E3A200C3FE16F993E450A033FD2A67E3FC80A523FDF821C3F4B4B633D9F9C7F3F3206093E4724C63EAEB9993E5DF86E3FA644DD3ED6A4153F4F51243F3D9AAE3A013F263F9EB4773FB91C133E0D177F3F79066F3FBB57183F0A06D33D"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%0) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

