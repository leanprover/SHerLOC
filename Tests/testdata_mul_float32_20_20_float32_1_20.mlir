"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xf32>, tensor<1x20xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf32>
    %5 = "stablehlo.broadcast_in_dim"(%3#1) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xf32>) -> tensor<20x20xf32>
    %6 = "stablehlo.multiply"(%3#0, %5) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> ()
    "func.return"(%6) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xf32>, tensor<1x20xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x3D58BB40C0B49CBE60609A403F9590C0E2108040A3262D406B300CBED80B2C3FC7B40FC0E59C2F409CF063BFBCF13E3F75CCE23FAF1AA0C082EFF43F6C6F7540326BCD3F54518ABFD7995C3EB6D0F03F529D9BC00A93223F9E564F3F7B26A53FFCA17A40D0B67D3F5538CF3FB7A6734079F981BCDBED9BC052EC26BFF98AA5BFB264E53FED1449BFA235E13F0E3F45C063F483C08AABBAC0489942BE9B6625409D70B53E8035123F8B5AABBE976FDABF7C392A40F946D84094C3E03FFC36733FC658CD3F907DFB3FE3D603C09B51343E84288EC0C2809540D7F22A401898BA4066338DBC520D87BEE676FA3EA1D7B9BE7AC7F6C042AC543CA11E933FC222453F6A256D409ACA42C07FDA1CBE5613CDBF51ED874093F8EABF748012C0736C73408DA19CC07F48F2BF26B940C06B7FA2BFCE85504014EF4DC07109DF3F581A89C08846EEBEEB53FEBEF8EB504047241F40405178405D8ED5BF7E9C87BEFBFEA53F4DF7A4BF1F1315BFBE25993F82479E3FF445F6BD368152C052496CBF988F12400B79453E019402402442EE3FD41C04C07A0980C0657DF43F47DB81C0D1FCA33FF033823EB771BC40C95529C0AB808940BEB85740EBAD1AC066764FC0E4ED29BF8B1F0DC08CD0EF3F584F523FFD09BFBF96700A4051FCB63FB3181840884B87BE647423C044DF0CBF9670D43D775110C0C9FDEC3FF0E8733E3C122DC0ADBCEEBF1D7E693E21C92CBDBD66F63EC7A4483FDBD73240CAC0814095C82840213D0AC0149CFD3FDBB0CDBF5F02A0BEC843C64043972D401540253F9B556140F5260140FE733FC02CD6DDBECD9C243F13A337BE9CC7434042D88D3F809B0AC1052D7BC0EE2C1140EC81063F5032043E29763DBF76C9AD3F84A74F40D313B740588ADFBF0D2768C072189C3FA35F323EB74A4BBF7DE66C403F21123FF5048BC04E455EBFDAB141C0223B6340969D224050E1D93F0781DD3F779B3AC009FC44BFBA341340B54BBFBF6F5316C05DA019C0D97C1B40BC15CBBF8DC8A1BF86AE42BE28809DBE6DFE69C09FD74AC07B3CA8C065003E404065F63EB63072C093696440F896F4BFB04853C02DFAB03FC8048B409BFA053FD85E0D40A4E1543FBECE09BFB654493F8E4F99BF4248B93FDB2395C08B7B9940A8C1A3BF294847C0065D1B40FFE1CE409026BA4090A3E63F1C9A81406D82F0BF217E6F404A4EE74090A0753E82A28FBE6645A54015D097C0F91E49C0B47A45C0616FCE40BE1B2F40A331D53F15590A408DB830401E6CCB3FF8D38EC02E0C29405735A1BFFA26A2C08023F7BF5B2E15BEC71FCEBF555DB3BF6F1A3ABF9D2435BE8EE84DC0653CD0BF4D6A353ED2593BBFC90111405D4D8640BACEB2BF9E37E03FE5F557BE57321F3E9FEF923F449E0FC04FD830C0A6F1AEBE3A918E3E404C2AC07813464053BA00C0103BA840DDBD3F40B95CA240D894284074A389C0AA7AA0C05AD097C01CFA6ABD60885140AA13DABF9A1852BFCD536840F9E30BBF5E0C44C08604F23F13DABEBEA62C8540608030BFBA198640A4BF3AC00006213B8E6A8EC0D385DA3F624A5B4027C0CC40FAF161C0666946C04040E33C86A83C40CFAA68BE6A937FBDF17107413161B4C0BA108C40FD78ADBF5B4DBDBF79D924C09C07703F8E86D6BD0B9141400F5311406039B73F42F73040FA0622BF124AFB3F808F5240C78BA5C0FA148A3F122ACE3F8E24C3BF8DE810C02C373A4054A4B8BE914E193F7FFA234194B5F1BF0BA295BFDB2BD13ECB7E8B407DDE2D40450B533F9A69E8BFA6AB183F84FF0A417BAA3A406FC8D840079C803F6AAAA43F8F0193409A46F73F1F99DDC0EB7D87BF6648E2400A3CEFBD41373E40D596AE3F6E3A13417FAEA0407F4E88405A0BA9409B8B7A3F389849401EEB0A3EE2EF514066C2B940E1B229BFAFF5E63C76877D4099679C3FE47D1040ACFE253FFC223C3FDAFC73BFB4BFEE3FB228FD3FB34273BFE91E77405F3509407EB6854007C8A4BFA7FB84BFB4D98240AC29D340CD9041C014E4A1BFA912C4BF171AE53E0B721FBED48470C0122567C0CBC7424032FCA83D8A279140C1A7F13E7CEC4840C176BCC064A8DBBF929EAD407CCAABC039180B3E0E857DBFA28B0EC08F51D6BFDE550240790403C09084543FBB49FCC000E60F40BFC66C409E1903C01B8A17C0CF12E63FDC4330C061A6E43F568E3E40C495ACBFEE61BE40E6D64ABFA49ECABF110BF23FA91D9F406E0F46C0927025C0BA87CFC0439251C02239FEBF"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[0.651655078, -5.39799786, 1.90161359, -0.145493269, 2.73951817, 5.649080e+00, -0.829076707, -2.49241328, -4.88056612, -3.85774946, -3.16361904, 4.79692364, 0.97927159, 0.584053695, 2.23244119, -3.43535352, 3.98992825, 1.40671241, -1.8466115, 1.29623652]]> : tensor<1x20xf32>}> : () -> tensor<1x20xf32>
    "func.return"(%1, %2) : (tensor<20x20xf32>, tensor<1x20xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x112B74408879D33F3EC812416D49283F646B2F412D8974418174E83DBB67D6BF84572F411C5E29C15B4734407DFC6440F518DE3FD9043BC079B3884009CA52C1C9E6CC40BE92C2BF9FAECBBEA8131C4062D04AC0DF645BC08323C53FE03940BE3BA72B410A28B34023CDABBFE4D117C159969E3D45629641320504402D86C6C06DA3E03F94E2EABE2C627B4017672941549F83C1AA4B03C195ACB33E08665640F3786C3E174F45C09BEC22BF643F7E3EA82AE9408EB81842B958BABF468C17C0588DFAC0FC8BF2C0798BD0406D3E583F28368BC0B1A22E40FCD0BE401241A0C162D88CBDB8FABDBE594167BF44E5F0BE9FD0A0C040808FBDE2E10B409774E5BDA36A22418D8C89C1280B023E05917F40A7D9A5C1649DE240E7BCE740E8F59141646299C0A3818DBF241FD7C02D8F8B4065FF4F414AD890C065EE4DC0C1B7B1C001469BBE8D9B2B40ECA4C6407D3BB9BE47112A41A8CC16C145DD603E87DD4EC02A48C940F1C50F40044072C03FD0BD401C2BF1BD63E4F5BFB6DF03C089BEFBC0C1F9443F97AF37403EFC5BC0CB3F2BC038DF26C02DF824C1DEEFF6C069DF3EBEB958323F291105424C640C404E5B2BC1009B83C1B62D154144152441D0C84BC0AD320AC08B108C3F96C0EA3F5712A44059170A413EB400407A6E8CC0D75FAFBE3808D5BF5C1B3E403CFD493E88FAA73E7F4FA240BB3BAC3F447D0F4001C2944070728EBF15A4263E46E1C2BF3B9E7040D4222F40C79017404F66BC401673ED409AF8FC4087AC10C0CBBC133FC07F0041243EE23F56015FC0DE3FD640805396BE501F03C187A51CC0F57908BF83D9E43E04E16EC1E6CC88C02340DB41E39B96C18F2A0E40851E9D3E7C8F933E84B722407059AD40130E9240550929C170E110C08C4817C099A6D2C04C99A93EEB9EEC3D8A3F2241F45F4E40F3836640647F0A40C2556C416A265BC10A9D00C1F0A402419FE9D83F33FAD9BFD0E0DBBF2EDAFCC066D0BEC0207753C01BD88D407F8C4940605784BF9D53DA40CF1AB9BE7E52373DF24120C1E93B8FC1137B8B40FCC7ECC0965116C0CD93694107A734C1E7A812C184E74EC07BBA4E3FEE2C1B41EB21E6BFB8030D4141BB953F227A7E3F7A7C823FBDCF47BFCE09FAC0ABCD0DC13DA532BF924E60C04CB88CC1E3CE00C0CCE880C15021E3C1D16FDEC073014DC1943610C144876A40531887404A16093FD9B7763FDEDAA441968ED5C038B2B940DAFA7FC043868640044F6CC1CAB44A409407A1BEA810F240C5A40F41ABD46C4010ABD2C06FB2C440BD629C419176C3400EE732BFFCD9C9BF608451BF77BBCFBF94921B3FD4634DC1CA7612C07180A7BEDDD972BF33FDBC3FA23DB5C1EC022AC0087D82BE2EE813BF21D4603F78A473BF5DFAB240A0C65741F4B8A83FA98361BFF8394CC162F84140315E96BF58C83B41D3AC24C111F4A14145256D40312AFE40D904D0C04EDC45C0F88C9E3EA239C74065D47D3EF9E30FC0F10DA441BFF5E73E1451F440DDA513C18D10B83F13A852C180AA53C0205283407524DABFD8BCB33B03A07441F7F8D940483D9A40230C3DC1717092C0C74B01C0695619BEAA60B340F367073DE6092FBEF7483F426F8C9540E48C2EC140A9D340F191B64058610241F0EC8F402E14D2BD471BE23FD236A240235C9DC030853041FEEC63BF530468C0D8778840FFC157C06D57BAC0BF054440BD22633E567DC6C0487E83411215993E550DBFBFB51348C20C1DE940DDB06C403ED8FA3F919A88400819CB3F5A92EB3FD39AC7403D491840C68743418859ACC040800C41389E273F5C37DEC03EC60B4169E88FBE9DC497C1E159BFC0169BBBC05D11953E1A1768C15E61A8C015E3E8C1C2B1C041307B854042764540F9D40B401E232DC192910A3F18A9934040832BC13FF85BBF8D81963C9811ABC1ECB514403E2EA8BE765FE33F91D98440D6484A3FE4C394C0E4711AC10C9C6A40F07243C16E8B2441F3F08240797B40BF3A7014C03AC260C18FA1D241452588C08E7915402728FEBFA14B953EED2B573FDDAFE4C02285063FB166054113A7EE3E37B070C0839396BFCA2775C1F9C2B54177BAAD40B335D041E13AA8C03A7AA23DF27D0DC0D4D8F44069C7D5C02E58374039F0714097BC893F9F67A4C0DC3042C1EA20E140EA97983E9492CFC08276224124231240F7788EC05B8168C18C72A640109396C1574073C0726BC6BFC25D8D3FC09B31410D1A2A41EE0525C1B7F711C1987FC14045C424C0"> : tensor<20x20xf32>}> : () -> tensor<20x20xf32>
    "func.return"(%0) : (tensor<20x20xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

