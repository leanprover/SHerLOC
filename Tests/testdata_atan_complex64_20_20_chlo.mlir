"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %4 = "chlo.atan"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x9A14D9BFC509AE3F539B83C0D5399B407A673E40C1CFF0C0014B943F76D8B8BFE7DA52BFAAC5433FE04502BFA28994BF74E629C07FC0023E01CF8F40DF7311407E7AC03F49511E40CDAC3ABFBB2406C0AEA018BF9885FD3D713FE840A136C23E90CA56C01B830B3F0B95EBBDF50BA3BF18959D3FF3EDDF3F8E463940EEA196C07EC3A2BF17CD9CBFB15A38C0C16AA740B91CBABF0B5D1040EE0AA3BF3A7D95BE83C2FCBF2D971FC0C872FB3F5C358BC07D6D2040ACA8D53F11DFCE40777884C0BAA571C0CEB8643E2F08183FDAAB753EEE9EA7C005F34F40D71452C014A4A4C03D2E4D406F5234403C5934C04EF530C0FD3C57C00884C6400D681E40EC0B7DC08D2346C072F6ED3F2BEA39C0D84E494059C0093E9A68D93FAE4E2040E9BE03405CBBA4409B811240A6A37FC0FC49603F2E87804093FB15C0D920983FFB42AD3F12EED840A0022840B997EBBFFA9D07C002196E40B00CBB3F897E44400CB4E73F76A5883F9106823DB89F15C0AD0B7A40C74E93BFAFB1B63FC843C23FAB77463F510292C0E139C4BF448D90BF6D05C040604E0440AE78AC3F99A585BE20F501C1DB00C2C034885ABC3EC338C00C2D793F175A3FC022AA1BC00745FD3EC658CF3E2714DFBFAADB07C0C2A4CB3EF0C0F3BF34B66C3B4B2BE9BF487845C08298213FFD8FC1BF6B7C36BF3CE59ABF090DFF3F2F7FBD40178986BD373D21C1C2FF3DBF268B583F42F9B33D8C8221402F01023F85E33C40EF61A53FBFEF1C404E1149409473E84079D3153ED3A38DC0261040401CB014C02049A8BF0A0B64406358844043C1FC3E22708540F61180C0A41659406C0EB5BFFE9E623F30B5D1BF85FCC73F8E8065403DF06A3FF06718405A138EC0415E10C09740F1BF078BEB3EC4AE81BF258D9FC022C718BF51A81C401717F2BEBABE893EBF46B2BFA1DB95BFE147843FE8D1B5BFB9783AC0945E7EBFAA7300C01ACB7F4052456740A9F00341101E1CC0FA32D33F6002B83FD4611A40CA146340BA2DD93E2E09703F89D77FBF15B6C3BFC7C1A03E35F35A40B365F53FC2D2ECBFF4C50B405146C83FE8C378C08DDA78C06556944028997540096FFCBF2FDE1D40DD51C4C04FD6A8C051943C3FC704C2BE7D81EEBF4E0206409B8D2140117ADC3F6C970DC0787E26C198EF50408BCA0CC0963E3FC021CC2B40974AA43FC934A5405D6AD93FB8DE39C03461EE3F25CA3340AECDF7BF121B8940448D7D40C81234BF7ECF8BBF73ABF43F00F4C43D5C0225403156AB4068B4DABF763D24BF39E700C06F6BA2BF1D7CAA3F5EF377409C2C94BFD78AC63FEF3D3A40E5060DC0BC8799C06C1105C01C587DC095AB2FBE83CB42C0CC0274C07022403F086DA3408C923EC0D7D1C1BF062D8A405906AEBF00CCF9BEF696AB3FB1CDC7405E9E1B3ED97E403FD81F343F17514040809808C1B2826C3FD76754BFF56674BEFCB0F3BF94048BC029343840E191D6BFB96139C0A83E723EEF660940A66FBB3FCECF13C0D4B99FC06E395B40A5E45D3F72F139C0EBE896C09621853F4CC636C0458CD2BF2A11E8BFE1F035C0823AD6BFF5FE27403016C4BF71A35640972668C06AFAA540C2412140AD1E2940BCD8EBC0B7F78D3FF178D2BFB6158E3F20C2DABF1EB4E3BF82BBBA4048FCC43D32AFFDBF601B9BBCD4F6F0BEACD7E03FF593A33F38F699BF922F12BF5D95BF3F8C3D2FC0BA4E26C0770C714025F96340D8C338C0D6F81A406B5B42C053B59CBF08A422C08B5942408C6A9640A5FEB3C0993981407A7D353FB61143C0A188843F6DCC4D3F2EE69540E712E6BF737F3F3E750B3CC01DDE5640896BD8402000C740098A2E403700854035150AC041E6253EBD3337BF33842E3F0542DC3FFE76D7BEC0F190BF89CA0D40854FAF3F5B1D51C03F712E3E677800C047890240516CB740E3DAF53FE4CEC93F434CF2BD49E3EB3F79F9F13FBAB321C08D7239408C8360BF54631A40753D11C0288FDABF7AD4BDC090B1843FFE816440A7578E3FF62623C0C583B13F62DCE53E41464FBF89863FC06E15584000CAB2BF60B789405ACD92BFF2FD07C006A9C63FBB418EC0EFDA1FC0619D91C03BDA8EC07517E8BF965B3840E25A33C0D8640040A9CC0AC0E05DE53F15B4C83F67FB09BF8DC0C9BFABD697406FD9A43DD6916F40AA0E0440ADA9AEC07BC73A40FB11DEBB0D15204058CB0EC165A1D8C0C8FB88BFEE726DC0005145404424D53CEABF62C0BC48D7407B37DCC08FA13640BC95663FA9E98140BAAF09BF158835C0AE9D433F5FEB7C40150AF1C0DD0580BF013903C0839C43406FFE973F44D316C0A0A9A13F4DE461BFAB958240A56208C18DAC413FE571F6BCCF67E2BF05C85C409F45273FD63E07C0EF542F40087925BFCC873740D55D1D404DB0CF3F8A0F16C087A3173FCB5E9F3FCA0C8A3EA03ECEBF1886303FFDE6FE408FF8C63FCFF2D440E0236DBF76EBAE408D9045BE9269E73E50D9E73FCD1230C0955E2E40B50CD93FC0D330C04D06D4BF206F96BF0B0F4DC0F89927C0292B3EC0076F2340C0430B40505C46C0277DF1BF8329DCBE12031CC0CE1083C0346D1F3F085035C074CA9C401B57B63FFBAA4AC05239F7BF1599CE3CE326B5BE6D8EAE3F01F5EFBE027922C01288013E3CA688C0BC4E13C059F35C3FA51341C04ADB87BFB5ED03C007085DBFC59335405F15AD406AF4A03FD46203C08BDEB13F38DA9EBF42F684BF14A29BC013A0F3BEE8259D3EE981D3BF03CE514095D08EBF4D9CC83F15FFF23FB9E049C0E7F761BF95148FC01DB081C0AB7A1E3F708580BF04FA803FBD45E7C039190AC08993833FFCCB6BC087C81D40D235FE3F2ED3A2BD0161633E36A25B405CE8B7BF6DEA0FC06517703F1B89EB3FD949B2BF2F85544069A80B413C5A0C3D0BC676BEF12B1EBD57F84FBFAC6E9BC03C2D08BFB26B363FE0CA733FC633A73EC7BF62BF07CB494048AFB6BF9CADD4C0B8A7CCBF99867CBFE9820F4033A4F3BF28A0834047F11FC0240FC53F32C3293F3C4050BF8ADD34BF4A22563FC2B3B4BDCB83BFBF7FCB863FEBA17AC08010E13F9D148DBF7A5595BFC7422C4026657440750A56BFF171D4BD5D8B85C06FE9753F3363733FBA09BBBF478EA33E9CCC56407A4DED3F160C66C09D299D3F66ED84BE6781463FF3E207C06A929440FD9671C0382D7FBFAAE0843E7D69804017369ABD3ECAB4BE6AB924BF900A1B3E028913C03F1087BEE9737B4006754D3E25EF1FBF40EC33409117B93F9C9756C077A1F83EAA78EC3FD0C7BA3FBA5CDF40DA83B73D722A6F405960EABFEEF7EE3EE9CFC33F976302C0470991C0A098DF3E68635ABFA6FE21BF3BAC0DC0523AD3BF3C7412BF4EB8B240C82A6AC0102A2340F749D23F6F19FBBFA05ABF3ECC8FADC06D524C3FFFAD5040E0EE19BD91AC66BFAF95F9BF6B3F08C1F4C60840C321113F866730C03A1B56C050A201407C58A7C030FAD9BFA087823F59AAF9C0E89A5E401BE0F4C03E54A63FE11579C04ACBE7BFC19E2E400991C2C076724A401CD966C0D6942DC0D9E4AF3F92B70C406C5B02407B2247BF92AC0FC05B851AC01D4B453F90FF6CBFAC4389C0EF7B924000B803401F1E0D405BAFBB40CEB76DC0BEF1A5C02FFC423F9899F03E7AA5DFBFA555763FA4F10240FDA94FC0ED1D7FC0F5D9D83F1B2DB13EEBB4503F1AB536C04F1B0CC0B2C9A9C0163F6CC0B507DB3F0FEC6FC0E69E81407E437BC087C8AF3F3863C8BF7590A9BFE3C6CABF3D907CC06498F9BF576D6CC0CEB86F3FBC619840E33A1BBD2E87F7BEA0098E4059838FBFFA5BFDBF56BAEA4046193CC0C49C0ABF92E7A740DB7A83C0F63659C090E7AA407CD0F3BE61584D3F4CA4973F7C49B5C065F344404684B0BF281695C0101A8A3F3FDB3EC06C96703FC213EABF944C28C0795A3F409910A3BF5A4B3640492B0C4084F9F53DFB27844068240EC067055EC038A095BF9A0E9D3F781573409C107C3F6D29F3C0E7D292C00C7456C0570DA3BF469742C014E320412515DB3FADBD01BF51FB67C0274CB33FE6A739BFC48F3C3FD785D5BF0118C3BEF049063F1A108CBF249E8EBF7E85A24007B1D0C06857ABC057DF9CC0B5508040D6B765BF64521B3F67142D3EF83F023FABBC483EBDBDA1409870BC3D456B9E3F653DF5BFF12C24BFF2FBB2C074FF04C029C9F8BEBB3131C0CB4A4F40B87E1B402DD690BF950C1040B1A994BFADE2E33E4AE75340DB855540059F6CBF52DC0BC0CE4609C072290B40293966C03010AEBF016FC6BFF9316BC0CCE1F5BF951B93400F3F18BD8C547DC016837F3FD77BDDBF43FFB7C081448340A96706BF1957B33F5743AC3F9C0B06400B3D54BF7CD29240025995BE39092140EFC69E3FA68931C0D4090CC0B7FC3DC0E6E6FC3E7800BE40C82FF73FDF415BC06F4BDDBC3007BABEDEAB6640D0FC4F405EB67BC0DF31D2BEDF4CE0BEF9F410406DABA93FDD388AC03D10B0C00E2B0C403F6ACEBF62EF8DC06680AABF049506402BCEE5BF61004240602C2B3F8933753F5B4916C0"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0ECA99BFEF39843E91E7BBBF513BF43D472CC33F6AE8EBBDA849983F0781C7BEF7575DBFA242D23E3AE286BFD67536BFF4089BBF17C3813CE270B23FAE58B23D1C4CB03FCB28963E034BB2BFA9D7E1BE3CE90ABFA0FFBA3D0D95B73F66F8E63B7ED3A4BFF929323D92D7B2BF0B5B82BFD878A23F1FBBBC3EACACBC3F15A21DBEF2F790BFBC75AFBEEC85BEBF6A45163E9800ADBFE04F9E3E4EBF6BBFB95AE0BD9AE6AEBF9C4377BE4EA4BD3F546544BEF17BA53F7068303E0D02BB3FC7608EBD8506A8BFF0286F3C1F120E3FA36F353EA666B7BF4D40AC3D2898BDBF6C3A0DBEC43EB23FC9AD1A3EB876B1BFBA7D31BE8B4BC0BF3948FF3D8E14BA3F5E9F39BEF4A9AABF5FE40A3EC05CB4BFC5E42C3EC212C03F4DD22A3F6D09AA3FED943F3EE660B43FBB22903D8AF9AABF3A274B3D8E4AB13FD340D7BD53F7943F1621C13E40B1B83F2E53483D789CA9BF830285BE008DAB3FE011B23D0F23AA3FB58E0A3ED4A9513F5DE7F23C662BBABFB0B9403E719E97BF1F79C83E5D64883FCB27573EFB11B0BF1A9B82BD4515C5BFCC18263EA1659D3F46E94C3E598CC8BF4029FDBD2D23B4BF9F47B9B9C318A2BF4C3FC43D66FCAEBF3D5022BE781B043FBEA7A73EC5BCA9BFB9938BBEEA40B73FAE370BBFCDDCC83F1DEC1DBFE652A2BF94DD6D3D4ACF86BF2D404ABEF416A9BF3A51B93EBDA6B33F52B4EEBA0D78BCBF21E2EBBBF74D343FF09B513D22449A3F1C30893D561DA53FA3A5EC3D9BB0B43F0129483ED98EB73F0546323B3B32B5BF5092D23D19889FBF06DA2BBEB689B93F15F40C3E6544C53F12A2763EC64DB6BFFA4DF83DE18B88BF7CDD813EFC199EBF501C903E0C2AA83F1484803D68B7BC3F9E4F33BE31EDA6BF2C7253BEB479693F702640BFCC0FB0BFB89CBABCA6D1983FF3FC86BDEECAA93F7F794FBFD8CA87BF7ADFB13E3D7AB6BF12D28EBE9DA6ABBF2D4BCEBEAC48B73F505AFB3D0FD5BA3FFD7E05BD81349B3F237B8B3E68C2B73FA849443EBCF6513FA811443F01C69BBF34A3E0BE8358C53F01AA983E1724A53F989E7CBE68DEA13F6E4E4D3EAD6CB8BF2C3E02BE1296B83F0A42D63D0A8FAEBFBDE1773EC506BDBF4CEFA3BDEA332E3F01D177BEDB32A9BF9784833E7138A63F1A5F313EEF89C6BFA3C2BCBDA0FAAD3F6FEE0BBE2DFEB0BF0700263ED00FC33F11813C3E72B5B43F911083BE27F9B23FEC1A7B3E2773BDBFB963473E8F4AAA3F25FB27BD0802A8BF4131C83EFDE2C63FC62AD13E6882B33FD48D57BD3A62B2BFEF70F1BE6DCC94BFEC82B43EB924AB3F707789BD6A94B53F8C47893ECCA8BEBFBBDC30BE6144BBBF971D4BBE6B6AC6BFF006AEBE4455A9BF1C803F3D915BB63FA537ABBD1E6ABFBF840D553EDBB779BF7F9B27BE5BC1C43FF5D61D3ECBDCA23E084C663F7497BE3F7627A63E464EBABF45924A3CED9235BF41AF0FBEB1D7BDBFD7BB46BE6AE4A73F73BF11BE00BF9EBF4954CD3CAA07A03F48784B3E9409BFBFD07F29BE98A3A63F3560843D65B0BCBF15301DBE4C3BB93F2A3AA1BE42C6A3BF380893BE55A6A7BF313C14BE1EBFA43F09131EBE1633B73F406316BE3925B53FE6C1973D9B77C33F4AF3F6BD4DA19F3F74F2CDBEA2CCA13F9978CCBE73CBC2BFAB82213E57E0C43FF0A30DBFF62BC7BC04B702BF829D983FBD33763E17DA70BF053060BE0FF0B33FB50F90BE23BEB8BFCC3E373E442FB33F957109BE38DEB33F84624BBE2267B3BF65CCA4BE7368BC3F415F193E7502BABF0087AA3D57D4BE3F7FCFA3BE16CD733FB60EAE3E2F66B13F99898DBD54F1C53FD898B4BE146DC13F283CF33D18D8B73F642E6E3DB2DDB03F7E07C3BD01AD9C3EECA054BF644AA83F0708053F928184BFD81E4CBFC8399F3FAD363C3E8122A3BF4E846E3CE177A8BFDE43743E3C18B53F3C48523DA9F1803F8CAD0ABDBDA9A53F5E06843E2F91B2BF063F453EFC98B5BF7B46C03EAD84A4BF69DB4ABEAD4AB4BF611FE43CF1E1A83F5D1E993D278AA2BFA4511B3EA0DC343FB6AC2BBF3CDEB5BF8AA4273E44F1BFBFF36D593E48B0ACBF2E82BABEB4BEBF3F520D4FBE72E3BCBF050B2DBE1B9BB0BFF8819ABD99CEB13F2DAF2DBEED69AA3FB18575BEEE2A9F3F01A4823E4E21A6BFB0341ABF6A7BAE3F2045603BE8E1AE3F60BFDE3D31D8B6BF6D2E993D9BE5C8BFEDC7D83E69F2BFBFA9145CBD804ABFBFFA6381BEEDEAA03F694E223BE61EC1BF427DEE3D153BB9BF35934F3DAF03C23FDD3C743E07BEBFBFC68EB4BE34A6C23FA74A7E3EC473B8BF818B8BBCE8D7B4BFF05F653ECFCEB03F9C34AEBEF368843F44A194BEFE29C33F19A5C3BD0EE0253F41B99CBCD440B9BFD1BC6B3E2A55B43FB4E5E6BECA369E3F586895BDAAF3AE3FEAFA2A3E3BBAAD3F494F91BED9258E3F3ABE243F740FB53F51292FBFD9A8C73F0F3F003EFFB6C43F56BF123E7514C5BFF7E1373E7AF370BECBB8EC3E8881B23F8BB980BE1A23A73F45911E3E94FBA6BF358619BEEB27BBBFE0E98EBE8323B3BF48B13EBE067EAB3F1A153E3EF6E6AABF48C00BBE0D6DBEBF1031D6BE940AABBF71A80C3D0F81BDBF00981C3E1FECB83F979187BEDFEA8BBFFBADAE3B7F30A1BF8A5D463FFB90BEBFD013CCBE9A1FC83F361874BE87B599BFB219003E2800A4BF78F5C3BD90C995BF23C717BEAE23BF3FF8A1143E8726AA3F4720B3BEBB0B933F0532A0BE4979C3BF6FA24BBE4F68F3BE2AE2803E9F7DB8BF43057A3E6BDF9CBF150CCE3EEE43B63FFDA66DBECC5CC3BF517E5FBE3CBDAABFA29E0E3D82B282BFBEF2CD3E5CDDB8BF8DF518BD3A7BBF3F190B83BE4C0DA93F062E413E52D0AABDE8A6653EB4A5A93F91CCC8BD6AC599BF58FD0D3ED0F29B3FF40172BE4324C43F355ACD3DD9ED143DF4657BBE1EADE5BDADD68FBF4F5CAFBF0048AFBC114E6D3FF4B9083FF5ED333FB9F05ABF85B1A73FBEF4E4BD0BF5B6BF576B09BDD803B1BF1456C23EB1BFBCBFE7F44D3E16E4A3BFAEEC293EABD6523F23EE03BF60F45ABFA088FF3E8E0FC0BF25E04CBF5863C03FEA8077BE7AD3943F605F62BEC702B6BFB528A23EF8A1A93F381653BDF140C8BFF1107ABE60127D3F91CED03EC1527CBF71C7CC3D34E8AB3FB977F63D5E9AA9BF5B63A43DCF5C00BFF8AF533F2852BEBF6717373E2CDEA9BF00A27CBD80DFC63FCEBA813EC695AFBD3998BBBE904514BFCB12DB3DE42E95BF94A529BD423CA93F588D473C1634BEBF8873B33E3752BA3F73DF81BEE591B23F0CCB0A3FC451C53F4E550D3E882DC83F2F568C3E83B38BBFB41DD43DF386A83FF7179CBEA780ADBF46B3A43CD6A852BF3454AFBE0C5AA3BF52774EBEE8A9C6BFFF50373EAB67B1BFDB20FF3D97D6A63F967C93BE1C63C73F91063EBE2E31BF3FF49F973E7AFF43BE30DEB8BF43C0C5BFF652E5BDB9C6933FE0A7C73D64D4B5BF3A3334BEEC98C03F771E2BBE0B1B92BF5ACC603EB665BBBF1E2C413D15E2B8BF9DE8AD3C492DAEBFC32AC1BD320FC13F62860CBE6927B73FA0AF1EBE4BD1A3BFA0CD0C3E9BD6A83F23D75D3E7760B4BF6537D2BED2809ABF9EC2D73DEA97C2BF7D6767BE0AE8B13F0C48A23DCAB7C13FF39E193E4238BDBFCE1B02BE4E6E383F3432943E54B591BF7BC1503E2481B63FD0DB60BED70EAEBF2B57B13D2D68223F5A62463F0C85ACBFE56726BEC2C4B8BF3582B2BD848FBB3FE05463BE9B9CB83F70D1F9BD8FFC9C3F5A2FACBEE36D9DBF342EB2BE1019AFBF3671C6BD9B0FA9BFE348783D9291AE3FFECCD1BAF6CBC5BFF1BA673E6684A9BF2791C2BEDB0FBA3F284B3EBDC97AC6BFDC5C433E266AB6BFF1C1F0BDD78BB13F3B2B83BC1889893FD409053F43A1B7BF731C953DE04AC1BF71074CBEDC17BA3FA4729ABE9C4FA73FFFDBDDBEAF48B3BFA7E73D3E60E6B6BFD84B973E5D5B923FDB68A93C2711B13F4A61C8BD6E6AA8BF4CB9A7BD06A3BE3F514C773E02E1C63FDD3D05BE89CCB6BFD30FD1BDF7E3B8BF84E590BED7B9BC3FF987853C12D7C3BF11DA8DBEB8C7833F35A162BE88EBA43F4E0703BF1EA9EBBE7E8DF23EB9C089BF4F99C5BE757ABF3F81FBC2BD9DFDBBBF0E13BDBD8223AB3F30364EBDD3000E3F68DCFC3DE567F73EEBB81F3E7311B03F9712633BDBFEA63FC8AEB8BEA663C6BFF08836BE29D091BF0B05B5BDB614B5BF3A3B343E386F9E3F3E6011BEACAC9C3F35F925BE1E78C33F144F9C3E9120A63F9DC692BD7C48AABF442561BE66BAB83FF50F50BEBC7D9CBF9198ADBEB2CBADBF1285DBBDACA5AD3FAB18DCBAEE1CABBF996E683DA7C3C2BFD74F24BE9AE7AA3FAFD9EDBCD479963FF46AA33E3AE8953FF2660FBE3AB2AD3F18F157BC8599A03F0F20133EBF41ACBF546A2CBE646AA0BF46D7493D5EABB53F9872463D1DB6A4BF3EFE0ABBE93AC5BF8934903E45CDB83FB16E19BE7EB7E8BEA4D8C3BE84169F3F080F333E24A6BDBFB171E5BD00A3A23F85474FBE65DDAEBFC23474BD2DF3A43FC85362BE64E3A13F5F70813DA15CB33FB8A2BEBE"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

