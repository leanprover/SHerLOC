"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<20x20xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %5 = "chlo.next_after"(%3#0, %3#1) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%5) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<20x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xB7BCAD3F1A27C1C0374B243FFFAD15404B0A72C08672093F11A67CBFA45118C092A522404D59D5BF682F9FC00F7043BF265C993F3AF7F23FA8A4723FC4E177C001F0ACBF020E6E40B0F36340EABF20BFEBAB7740602C9F3FB76C3340A7BE574081A242406F3EEBC0971A49C0B0985CC0CCA7B6403054BE3FB89D3BC0959B7BBD5850DB3F415D15C0E762473FA018003FE046CCC078B60340AAB98DBF3F1295C09EFB643FDF799540DB7138C02BBE24C0D8123F40F52CB9C00FB35CBF5D64A43F123A11C168D1E03F3397293F2878FBBFE6D136C0D874BD3E1D2C93406E11F83F7CD6C43E652C84C0D1E0B0BF7A26E9BF766C383FBF6807C0C79C8440385440C040AF0A4068F0253F6AB707C01EB3B9C0FB3C8DBC36A988BE014C8FC0EBDBBFBEDD0A303F42D183402074503EDC4E8F40AD3064C0D1B9D8BF004968400EFA63BF5A3131BFD32248C0B47E51C071230F3F716B08C0EAA92140903CD83F7882A2C0BBF12A406F2B80C0A1F51E407BBDF3C075E53D402ACBB5BF977BC1BD24A74D3EB7EC99C0EED7F13F2AC2ADBE40C9513E51872B3F946D74405FC000C0DE672B40B6DB11BF2BD4A3C0D5AE0940DFC6C93F6095A1C0DD62D2405BCEEE3EBAF7FE3EC38BC13F9DEA4A40AFD7AC40AFCF4BC0CAE2D6BF54C799C00C3D223EAD8FD2C02A3F063E2E0EFABE80DE1CC01928E4C09E649D3F659AE3C04933F33FC578844042082BC015B9CDBFB93FD140CEB494BF616795BFED00E8BFB2609DBEF172704055BF6BC080319740C0D501404AC109C01D006A40F577F83F228693C016827DBF0A11874001C0DFBEB70F50C075EE7F400A72903F3011BD3F6D8487408F356E3FF1C145C054D024C0CE16FE3F0C9099BF83B98540ED1F4540C52B0F4050F9EBC01A90933FD7D29CBF071AA9BFE1CC383FC4FEA2C0021D73BEFE374DC0FCD6DCBF12408340FC2FDEBF6B70833EACD75F3F9970EBC01F894EC04CFC7E3F934C3F402EFE0640EDDCF0BF2EC8C7C0945D63C0499BED400410FCBF0CCC64C05A970D40759B0EC01EDB3D40256BCDBE7E3EAF3ED9E60440ECEB2240CAB3ACBF7336CFBFD7814D3F575F934010541C40F9134040BF619AC0697EECBFE6CE4E3F8063BF3EECCD0040ED396BBF0DFB24BF2624C93FC4280CC0CDD15140369085405C7211401AAD8AC0E566F33D72F7DA402DE9E840B39D88405E7AE33EC583A1C07DED0BBFE460B23F549349C085508F3D1576C73FF45AA33FD4E4473F6FAD3ABF97EF89C0EEB8B640C1CB86BE612B11BF49D116409FBF794053E84EBF61E11FBFC7926E40E1B7FC3F48652D40CAD8B3BFCE5FD2BEB8F737BF42A639401C4A114029E2FDBFAF9E16C035E4014030AF8EC05773A94010A71EC00485FD3FE3F4D9BD2C498840691DF9BF35B127C01E8011C021C764BF4DACF6BE8A38A0BEDD6615C0122AA5BF356641C0BDBA97BE11865340174E2FC00954534094DB2F409CACB13F46C1D43F5C7F41BE95B18EC0F50B3340780D91C0F7947EC070C7BABF3D806D3FA5408B406FF0753F2520E9BF02869CBF66B314400C0DB33F09D2C1C05BA2FCBDE9A1C7BF63FF663FA922073F4B7CE2BF21C9C2403A9B62C0E29D03BF1055C0BF7043AF3FFEAFCB3D9E0748BF15B1813FF34577BDDFDB1F4001AD263FAEC01240A40B87404582633EE4BDCB40D5F555405B13F5BFF21914C048A045BBCC6B85406F2B8EBFC2AF13C0B49B9CBFE2FD03401625943F80C551C029BA6EC025DCC4C07476F4BE772EEABF5423213E68E1504038E55140A58B80C023614E404E5D963F8D3D36C04C2B0BBEF4D8A2C0A83CAC3FEE9E4CC0118454C03C0635C0460633BE38FD60BF1750FD3F839AB540A8D6ECBFA5145140384FCA40A9072240C9477FBF48534DC0E73006403E5C0D400CFB3040E14572C073D707405C7B24C0E8D72FC024AF3E3E5E8F8A4080A8DA3FFDB1A23E491FC14003ED7EC06EF7D7BF38934E4058043B4046D120C02AD40A3F02A0C740C4E54540CEC9E43ED0CB593E0470B840A57607C0A94828C0814088BFEB217CBEF2AFA240B78AFDBF1384C03D1EDB1F3FA370C1BFC3600B40E9A61AC05B36153DF6DA8BBF15FB88C02C73353F8098813F0E7D0BC0A1BBA8BF997C75C0BFF625C08EB8543F1FF961C047A3B93FF0A40B40759101C07E131CBF621989BFCF1F13C0D1667C3F747F4F4088B15B3FE7DF4CC002655D4056B736BF44C2C53FA85076C046DFC4C06FB504407CF649C0F38EF6C05632AF40"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %2 = "stablehlo.constant"() <{value = dense<"0xEF664F402FEE98BE90378440607EA2BF9DE6D73EC4C04940BBCCCB3F85147CC00E9B8CC0EA4990405631B0C0F0D167BF714884BFC5787AC01D26FE3E921A1B400CE792C028C67CC0181150BF98F8B340586E76BE67B378C023908B3F095A21407748BE404782EA3F73D660408F2A45C061FBF73F316770BF69268EBFEAA88C3F9958B5401C8C33C0F7889EC0802C943FD3C40C3ED8B55340540CA0BF43D4B0BF54F42D40A1EE50401CDC0140997F77C08F5AE13FA9A8B03FCF096E3FF57F0BC05C56E540D636963E1407313FCC73A7BF880F88C003DE76BF4F2973C0F18EF0BF51B14FC00ED78340AE2787BE520203C1DA224040C1145EC04F345E40C6E4B0C00E98D6BFC7992B40149E633F363A40C01F8DD7401C4688C0C226FD3A7062823F04ABA63F7F1A11C02C38B73E1A2DBE3D7691DE40F267603F6ABF5440DDAB81BF8A4E373F3A5710404FE712407E2680BFEB6DC33F78A3823F499F6FC01988B6BF13A4593F144D95409469203F1D8AA140D2A20AC056C4943E51CBCF40A0ACF53F2E5C2BBFA04C083ED72A40C093E0AD3FCF619EBE8B012D3FAF99663FA503AC3FCD0A5140C6F83DC0E80E573F4B7B78BF0265FD3E193CA2C022731EC09994973F636AA63EC902B4BF126A7040B2DEBC3ED2FF0C409CC32740066C823FDC96BEBF92C4723F4D458240532995C099E02B40F113EA3F27120EC0E49109BF7BF109BF3216BCC0B6A3EF3FC39771405F138A4038921DC079CBB7BF505AD13D9890273EFC37FD3F09972E3EC0C32E3C77791C4087E0FAC0674C42BF6C4A20C043C202C00992DC3F2B4635C0BD4B183F58AA98C096CDD0BE06F78E3FF230E9BD04735C40E9D4BDC02FB64AC0863415C0861A7EC037974EC0092B31C04F848E3EDB707440EDF06340B20F7E3F7F6B99C0308C9E3F8F1B2C3F814DAC3F8545C43FBAB2753F053A6040D48F25C0F05F4140F389CFC0E9C14CC0127DC2BFC3357E3F43095B4073F9D8BEA5FF5AC0E20FC13F86D833402082CB3FD6E4CCBFDDDE6C405FDA383F1C7E6340F6EDAE3E53E18540A3CB52C07BA9A8C05669BB3E28C140C066D85C4034C431407DC5633F326C9EBE0A71FDBEF2FD59BF77430140FEBFF9BFF0153CC016D32CC0094508C0F77DB0BFB18B833F8A46ACC0B6E80FC0CD8F22C0B58A214038B096C0419AAC40726C08405C5EDA3FC85E983FA14403403FF220C06B3C7A3FC0B36C3FC4725F3F70A9B23EF0921BC007216840D004AB40BEB13C40B8319740E258A63FA6DD95BF0C29B73F478051C0A65612405D3FDF3F03B942C0604C18C070F221C07BA8B63FD9D3EEBF13587740B6EC183E1B83A2BF5A3704400C92B43F33C7D5BF0FC95DC0B9FAF5BEC3943C4062814D4086170840B8317640179E7FBFBB979340766C02BF61E601BE3890A03F655CB8C0B1CFCD4026990EC01895033F20E544C095960AC09828CD3FF0822FC03EDD963ED1D992409366F5BF803B7ABF387AFBC0E4C8F83E5B640E408990E4BFA7F41D4096662F40A7A08CC0B624493E8B28FC3F5E104D40335D5EBF2E3EB5C0C45510402BFD6C3F187F5BBFE3310EBF1BF826401709CE40A269C63F1C58D53E87D3263F7A135140560032BFEEFB5EC0E2F2733FBAD592BFF5C33640848B6AC07DB8D8BF084D96C01FBDBD409DD94640DA4C3D40CF2171C072AD90BF2101B4BF1E6E8540C05441C009914740215271C0AFE88AC08857D9BF49BC334060C832BE2484AD40DECBA0BF0E0EE9C036E649C08A46A740B873DDBFD3E537C0D397CBBD34B7B53FECB3493F1D5780C057A959BDF98392408BA75840DD7004402D411CBE538C75407516D0BD3DD0ECBEE10D06C0FEE455C093F282BF2C9B0C409DBC2EBF46015C40F87D2B3F0C316B3F0026F53DDDF5704098CE1D404AAEFCBEF65DD3BF216180C0DBDD103F5C289F3FEBB80C3F9D23A93F42A4083FD67D6AC066E08F40B2F1063F64B4ADBF6670773FA3697E409F3E4E40A4C792C01EC16FC0A2F652C0C85F71BF836DA0408C4588407701AA3E4AA102C00F17B2BF4E798CC0C6DDC43CAFF622C0FEBB8DC0A78191C0B42E5E40D4B5FB403057A5BE0879B3BF55B33FBED76B893F35A4893F5696284044A7854084C46240CA1A6340B3936440CD585FC080C75640A5A76C3F3402873F4B960ABF77B20EC0A384CF3F27AB37C0EAC79EBF560955C0632BAABFD4B655BF83E9C63F3F75B6BFF2D36C40F1CDE13D1237A8400FD16FBFFD4D9240C021C5BF2CE64B40"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%1, %2) : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xB8BCAD3F1927C1C0384B243FFEAD15404A0A72C08772093F10A67CBFA55118C091A522404C59D5BF692F9FC0107043BF255C993F39F7F23FA7A4723FC3E177C002F0ACBF010E6E40AFF36340E9BF20BFEAAB77405F2C9F3FB66C3340A6BE574082A242406E3EEBC0961A49C0AF985CC0CBA7B6402F54BE3FB79D3BC0949B7BBD5950DB3F425D15C0E662473FA118003FDF46CCC079B60340ABB98DBF3E1295C09FFB643FDE799540DA7138C02CBE24C0D7123F40F42CB9C00EB35CBF5C64A43F113A11C167D1E03F3497293F2778FBBFE7D136C0D774BD3E1C2C93406D11F83F7BD6C43E642C84C0D0E0B0BF7B26E9BF776C383FC06807C0C69C8440395440C03FAF0A4069F0253F69B707C01DB3B9C0FA3C8DBC37A988BE004C8FC0EADBBFBEDE0A303F41D183402174503EDB4E8F40AC3064C0D0B9D8BFFF4868400FFA63BF593131BFD22248C0B37E51C070230F3F706B08C0E9A921408F3CD83F7782A2C0BAF12A406E2B80C0A0F51E407ABDF3C074E53D4029CBB5BF967BC1BD25A74D3EB6EC99C0EDD7F13F2BC2ADBE41C9513E50872B3F936D74405EC000C0DD672B40B5DB11BF2AD4A3C0D4AE0940DEC6C93F5F95A1C0DC62D2405ACEEE3EBBF7FE3EC28BC13F9CEA4A40AED7AC40AECF4BC0C9E2D6BF53C799C00D3D223EAC8FD2C02B3F063E2D0EFABE81DE1CC01828E4C09F649D3F649AE3C04833F33FC478844043082BC014B9CDBFB83FD140CDB494BF626795BFEC00E8BFB1609DBEF072704054BF6BC07F319740BFD5014049C109C01C006A40F477F83F218693C017827DBF0911874002C0DFBEB60F50C074EE7F400972903F2F11BD3F6C84874090356E3FF2C145C055D024C0CD16FE3F0D9099BF82B98540EC1F4540C42B0F404FF9EBC01B90933FD6D29CBF081AA9BFE2CC383FC3FEA2C0011D73BEFD374DC0FBD6DCBF11408340FD2FDEBF6C70833EABD75F3F9870EBC01E894EC04BFC7E3F944C3F402DFE0640EEDCF0BF2DC8C7C0935D63C0489BED400310FCBF0BCC64C059970D40749B0EC01DDB3D40246BCDBE7D3EAF3ED8E60440EBEB2240CBB3ACBF7236CFBFD8814D3F565F93400F541C40F8134040BE619AC0687EECBFE5CE4E3F7F63BF3EEBCD0040EE396BBF0EFB24BF2524C93FC5280CC0CCD15140359085405D7211401BAD8AC0E666F33D71F7DA402CE9E840B29D88405F7AE33EC483A1C07CED0BBFE360B23F539349C086508F3D1476C73FF55AA33FD5E4473F6EAD3ABF96EF89C0EDB8B640C2CB86BE602B11BF48D116409EBF794052E84EBF62E11FBFC6926E40E0B7FC3F47652D40CBD8B3BFCD5FD2BEB7F737BF41A639401B4A114028E2FDBFAE9E16C034E401402FAF8EC05673A9400FA71EC00585FD3FE2F4D9BD2B498840681DF9BF34B127C01D8011C020C764BF4EACF6BE8938A0BEDC6615C0112AA5BF366641C0BEBA97BE10865340184E2FC00854534095DB2F409BACB13F45C1D43F5D7F41BE94B18EC0F40B3340770D91C0F6947EC06FC7BABF3C806D3FA4408B4070F0753F2420E9BF01869CBF65B314400D0DB33F08D2C1C05CA2FCBDE8A1C7BF64FF663FAA22073F4A7CE2BF20C9C240399B62C0E19D03BF0F55C0BF6F43AF3FFFAFCB3D9F0748BF16B1813FF44577BDDEDB1F4000AD263FAFC01240A30B87404682633EE3BDCB40D4F555405A13F5BFF11914C049A045BBCB6B8540702B8EBFC3AF13C0B59B9CBFE3FD03401525943F7FC551C028BA6EC026DCC4C07576F4BE762EEABF5323213E67E1504037E55140A48B80C022614E404D5D963F8C3D36C04B2B0BBEF3D8A2C0A93CAC3FED9E4CC0108454C03B0635C0470633BE39FD60BF1650FD3F829AB540A7D6ECBFA4145140374FCA40A8072240C8477FBF47534DC0E83006403F5C0D400BFB3040E04572C072D707405B7B24C0E7D72FC025AF3E3E5D8F8A407FA8DA3FFCB1A23E481FC14002ED7EC06DF7D7BF37934E4059043B4045D120C029D40A3F01A0C740C3E54540CDC9E43ED1CB593E0370B840A47607C0A84828C0824088BFEC217CBEF1AFA240B88AFDBF1284C03D1DDB1F3FA270C1BFC4600B40E8A61AC05A36153DF5DA8BBF14FB88C02D73353F8198813F0D7D0BC0A0BBA8BF987C75C0BEF625C08DB8543F1EF961C046A3B93FEFA40B40749101C07F131CBF611989BFD01F13C0D0667C3F737F4F4087B15B3FE6DF4CC001655D4057B736BF45C2C53FA75076C045DFC4C06EB504407BF649C0F28EF6C05532AF40"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%0) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

