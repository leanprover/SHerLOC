"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %4 = "stablehlo.real"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xf32>
    %5 = "stablehlo.imag"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xf32>
    %6 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %8 = "stablehlo.compare"(%4, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xi1>
    %9 = "stablehlo.sine"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %10 = "stablehlo.cosine"(%4) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %11 = "stablehlo.exponential_minus_one"(%5) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %12 = "stablehlo.negate"(%5) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %13 = "stablehlo.exponential_minus_one"(%12) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %14 = "stablehlo.subtract"(%11, %13) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %15 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %16 = "stablehlo.broadcast_in_dim"(%15) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %17 = "stablehlo.divide"(%14, %16) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %18 = "stablehlo.add"(%11, %13) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %19 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %20 = "stablehlo.broadcast_in_dim"(%19) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %21 = "stablehlo.add"(%18, %20) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %22 = "stablehlo.constant"() <{value = dense<2.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %23 = "stablehlo.broadcast_in_dim"(%22) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %24 = "stablehlo.divide"(%21, %23) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %25 = "stablehlo.multiply"(%10, %24) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %26 = "stablehlo.negate"(%9) : (tensor<20x20xf32>) -> tensor<20x20xf32>
    %27 = "stablehlo.multiply"(%26, %17) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xf32>
    %28 = "stablehlo.constant"() <{value = dense<0.000000e+00> : tensor<f32>}> : () -> tensor<f32>
    %29 = "stablehlo.broadcast_in_dim"(%28) <{broadcast_dimensions = array<i64>}> : (tensor<f32>) -> tensor<20x20xf32>
    %30 = "stablehlo.complex"(%25, %29) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xcomplex<f32>>
    %31 = "stablehlo.complex"(%25, %27) : (tensor<20x20xf32>, tensor<20x20xf32>) -> tensor<20x20xcomplex<f32>>
    %32 = "stablehlo.select"(%8, %30, %31) : (tensor<20x20xi1>, tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%32, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%32) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xA72BB33FBBAD3540F89D994083DBE03FEEC14640CF46EF3FD4DC044067F7A6BED05172C05606EE3E47B3C53FAE5D5040C3BE554047DE943F070D124044B36EBF5B09C440FF0F07C041B7533F221D26C07E9D154069392EBF5439C83E068F83C03A781D4032121FC07C3105C00E431940137B33C00F975DC01C96DC3E323B8B403A7D703F4CCD9CC08775AB3F4B7C1BBEE72883BE3B61353F117A37C0F13DBD3F1D3B923E6F37A2BF1EE58CC058ED48BF85ECC540B007963D317E263FBC77ED3F03479240250F1DC008C5054069479540D95554C0DA5459BF44096F40E684B8C0A18795C0C5853AC0F3604CC0F50D594086648D3E1A46F43F13C13E40E4A42E3F8C419BC07ADB4040D284844085998BC06B97553FB9C557C036056CBEA6336040D1E20C40187A2640F3369EC0621E02C06AF2853EF931F940E5B630C00E3E16C046FF1E401E5066C08DF6BA3C224A944085BD8CC0C74540C032FA1F40BFAF0D401A5183C03482233DFDC5D2C0AC6B21BFBC9F3F401FC94AC00D69A5403CB768C08D857B3F0C02A83CEA248E3EA6D2DC3EDC6C09414D3ACCC0CBD2AE3C0B0828C0AACB41C0CF0B0A3EC709803F2147DBBF5CE6EE3F79DB14406D4F8D3F0181BE3FF84D82C079A79840684EE2BF59AAC73F0E4759C0A633E9BF1B816DC07D071040E7D1EB3F2EBB1ABF2E518CC00CA5013E9575BC3F3EF18AC0B4152F40525B10C166D400C0D00392BEF8E164C014AF8EC008FEE23C2F676BC08CDD173F14369740DEFBE83F8F5061C011362EC060005BC09F74933E5E638740CBA425C0E5235CC0C0E02D40D1473FC0F1FB924029E2FC3F1E6077C0C5CF5740410C953F249B52C0239E933E3F38AFC060B7813D21B56840B24E85C0798A05C1BC5541C0AD9F1440DD8DE640309858BF784AF6C0549688BF875625C0870F2740B5AC0140DB2B8ABF660346C0778380C040332ABF316EA93FCE6DEB3FE1EE05C0DFFF35C0D6601241E01591BFA669C8BF19E296BE9F4C9C3EC82BA94054262DC011FB813F9CC341BFB84FB03FA0E42FC093F3B3C055E57C3F3440193F39ACE3C0B0B464404FDFC03F9AB35440AE5F60BE1829B04053681E4075FD91BE1ACE4A3E3E783AC02F7E04C0BDB9C3C0AD359E409B6B01400C0009BFCD6CE53E19EE29BE8895C13EED19E740ACEBAABF79F1464046D3BAC0D149EA3FCCC09A3FF5074ABF896953C01A8CDA403111F43F5B10E23DC5C652402BECCFBF56965A403DD81C409073EB3F10509B4004FE9DC0EDAB104005085A40DD7949C01930DA3E49A857403072463F6C8869BF2AFB17BFE76FA13F0DE492BE720EAC3FF585B140EBB523C0C8F7DDBFBB5F133F2F3A9DBF6233C53F31BA9F3FA65B9F40E5BB72BF39BE06BF4EAD18BF5D0377C0BA0A97BE149C4EC041BBC5BF8991BBC077E488C0F2360A40F8E44040230D1B4072884DBF62EDB64071DFD2BF56EB88BF0B4315406A9D6EBB1766123F0EAEAC3F78A02BC07FC92C3FB083ACBF90EE80C0531758C0CBCE17405E2501BE4E8AFCBE3CAB7BC0E7A50BC02DEFC0BD4C8D733F37FFF0BF906B64C043562DC05F2EF9C07B91AE3F2D623A40FDAAAFBF58F81E403038F8C00F22E1BF9B933340EFDC28C0EAE30A40BEEC933F501F483FC3A7384093AD3FC0782F30BEE2C0C2401255CB3FC0ED4B4081B166C010D35340B764C3408AC86B3F689A03C0AF7331402C2144BF540549BFE273303F6AE648BEF52B75403F24313F5EBBF9C072488E3F68389540E5DA0640A6EC07C1FE8D9ABFE25B7EC0A33C5EC0291FADBF44DF08C0F1358040AD174DC003377D3FE9A1E83FBEF3723E36DFC6BDDC9A38C0FDCA16406B41BA3F972442C01EC5EDBE342486C0CF11A63FF013FA3F6B52AD3FD35F6B3E686E22402928E6BF6783523F219F45BE0068B13E85DE20BF123B563E340652BEBE6088C0F7A1064012CB0AC0852D774071AE6B3F88B6663F8386813FB7C69540669681404F26EBBFA831873B2100044044399CC02CE3883E1BA505C03DE77CC0B7B7193E4F2ECABFA61C39C07439EEBD5B332C3F7A1003BF716835C04200F43FBB2F4FC0F2606B3F2B4C03C005C3EEBF800C21C08C72DCC033750BC0CFAAA83F0F5E00C0591F2CBE7BFEAC3F18F550C03DD721BF7F93FE3F01C2E83F36F138C0DA807FC0C8406940BBEC88BF23F1AA3F3BDC4040C20E5B3DD62664C0A130B83E96747D402C2B923F503712BFDFA05ABFEC12A43FE456B83F8F0006BF6647873FF4335D40253B0CBF0A4E933FA6A8B9C0FCFFBCC0AD020CBF2E5D0740D05464404283C73F10FD78400927AFC0BEFF84BF1161D0BEE0468C40096821C0D5DF4D40236CAE3E374BBB3F8E31DF3FE35463BF4D340DC0C7D2B23F17A9553FAF72A6BF7B0264C075B9343F22A9FB3FB813AF3F1DC91240021D7140D0DCAE3F5F24BEC0453CAD3F403ACBBFB4026E40DCCC03C01FB729401C78D63F04E9B23F36B4933FC63A8FBF6FFB74C0279E9B3E2DD31140B5A084C0ED2F5F3F28329D404F59CA3F7408913F6295DD3F276C93C02FEE4EBF20EBECBFF9CB56C06FDE19C0C80E3DC092146A408C09B13EC4A099409CA686C098C501416CF031C0C5236240FED44D3F91EB393EC6882DC0B099BABFAAAFB33F5514D83FE12B84C0A37D5840EDB20FC075FD39C0E29329401CE814409BB8D1BFF0BE38408FD22D3FDFC70440027398BF3F62AABFFACC503FB0BE1740A3EB4A40FF6197BF91CBB44001A424C0502C133E775BD6BFC6A40140A17018C0D0BB76402EC65C40D19F2D40E587864098C080403E384EC00E43813ED47E7240762675409D61C73F27C298BE62D8ABBE66C79FBF72160FC0E323AF3F32876DBF3FC78ABF833179BD41119FBDAB163F40447A21BF87E32440B969D53FC72C1D40B58928BE3153123F48A0293E04F64740978D78C03609514053863CC0C24DA8BF864F943F35865DC0C53E4140197083C073FDC73E2A5BB640764399C0D540FE3E7CEA7E40BD23A2BF8B9F1EC08F801D40E8380CC03A42943F71A4ACBFBA450840804D78402ECDA13FA613564007032ABFA67D224096BA9C3FF0129E404740B83DA8E50F40F9F1F83F3E2118BFA795D43FF0FFF43E2A1B913E44705EBE971BC33FBD22D43F5774193F8E3B853F45AF9FC0174588C0FE74E93EC846B23F1CE9A53F0B97FDBE46A53A3F3D219640EDA37640CFBDC9BF2044C140E551C2BFA85017C0C7BC933FBC9D4C408080E93FA09198C0778887BFADA40941227290C0123D8AC0A8DAEFC06EDBC0BFEE1241BE2A9313402F0E8B3FDF1119C003B23ABFF5F823C0AB530DC090C2A940AC912440A4024D3FF4A47E3DBF53B4BFE09F19C069C7873F2E4F3EBD9273C040D44104C06912E3C01E26263E0D3E0040085F98C07665FCBEDDACB5BF8BAE04C02B6C4C3F65542540A9FB28C01D699ABF741F86C015A4763E118F4640AD70A73F3982B9BD547D7940B440B73FD130BBBF784E6BBFF17827BE1743CBC055ADA33F5A822A40F9DE1740110A07BFABF8B940B13E87409AF870C02EC0AD40907B93C0EF34B4C04DB66B3F885C67C08C865FC031A0134092650C4057DAED3FF11C59C0AC81CF406B26B33E47CAEBBCB1612E40C7EE76C086F18E40502640C0B143613F691B4EC05A001E3FA13545BF4563883F8F466E402BDCE4BF0C08E4BE183B563F93B91540C68CDBBDBA1E0840E4B02D405EB6EB3FB5564840D049B1BF293200C09AAA5B3F66D2F73F7285AB3DC2A993BED2A1873F9BE4E9BF9F4C2DC0B4947EC061B980BFDAE8B7C02A24153F278831407775364062A4AEBFF315A140735B89C0A88CE8BE8472874016854AC0AF5C85C0A9DF03BF96BCF7BFBD002CC036773640B6200CC10235AE3D690B8F3F0129C03F4C5251C0E91E61C0BE0D293F8EA4DCBFD6BA44C0DF6411402CDDA6BF7E787740D9A3ED3F5C75A93D20DACFBFB3F863BFD93C26C059D9BE3FA60008BF85CE413F63441D40371AE9BFC48529C0EE9015409D35B540D0B3AB3EC2FDB4404F1215BE8F35A3C09AF198BD43AAF4BFE9F46DBFCB3699BFE08D53BF6E2478BE98DA5CC0F9F84840D535DEBE3F5148404F1DAE4068583DC0F06456C0BFF5723E67EB844066CD133FDF7ECE3D356BD53F7BD5CEBF460692BFD57A6E408346B23D76574B40F55C8340EE58A83FABC3D0BF9CA726BFC438C63F41EAC2BE005AB3C0C2793AC0AFB0A5BFCC8885C0718E223E7D87AB3FFDEB313FD7C4E0BE416580C00291A9C064911AC0345F8F40AF244A402083B4BE084983C03ECBD9BEC18977C030E36040450F98BF728204BF639F73BF21643940CAEA96BF626AD2BE2C03D53F1B795BC037EED93F4E6D2840E0B19ABF37E1D7BF1003A340950975BF39174C3F4A1115C0A8B5DA3F02C2A13F695F5BBF698D6ABE6C3BA4BD024DAE3FA8EDFF3FB0384B3F53FEF3C0BF08833E25A0C4BFD78270BF9D3552409A3196BE56848C401C70993FF36022C00AFC923E3A19173FAE3AEF40299E093FBDB6BB3E19280DC0C8D21FC0"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x1FD5BA3FBC4B06C12372863EC3293340A94954C02B55E9BDBB8A02BFAAB6943EF02263BFE83D94BEE1A0AE3E3C204FC1F960DCBF7883913EA42C75BF1831503F134F8440E4C022BF4F039240CEF19C4085665CBFC55B073F20B7E14113063A41714796C0694170405ED32CC005D09740700071C114A8A8C025F50C428E8881C11A8D1E4226CD58429E6D6D3E93ED173E67399C3F299A473E2A1D0EC00A69103F455EEB3F98D4EB3EF571CEBE20A6533F3A747F3F703BEB3BE7CC26405FACF1BF712C53BF7EF3B6C0B6C7D2C18A4F38C2B139AEBFE84E2B3E026004C3498CB2C21856BBBE7FF41241D0A06DC12DC544BF862454408A2166BFD8E69CBFE7D6F1BD8B69B53F1AE220C1F0ADA9C11CFA03C294A81C4196632C41B57B81418DA27240FBB87FC0063FADC0A675643FF6CA6940C37E914484D39BC3CACF9CC0854CF6BFBFA567C14B8032416BD34D42425196BF5B0348C01417194124566DC0353D2DC0888112BF532C06BDBB3F933FB5004EBEAC723CC1C127DF3F23F605411D2388C123180E3F2CC68BBCF8B9863F55A9F9BD563D46C3E5415B43EB16DE40521B163E515580BF60FE7A3C04EDC53FCA83104037AAC0BFAA2C9BC01D16863FDE42F0BF0AFC0CC2DB2E3DC2193AF9BEF0B70E40B49044C03E36413F785481C081F121C0FB2FA3BEF54D1E3F9F02A6BE8B2AF6BDE2CC7140A7F118426EE46DC5C878CC44E10DE4BEAEC485BE5BBB1CC2448291411B569E4174400C3F0A053B42A726FCC1A59A85C0CEE98241590260C1C366C7C000EB0342B23F1CC1716854C10DF502C1F52211C10F9F824014A2DFBE74B860401BE52EC1F4291AC1A21EAA40B11F4541E7F1E442C9CB074280919741C79799BF06E788C47B0AE144E28BA3C0FCB61B3F3688553FFDDE413F2F72823E69FFA1BFCB9DB9C0C75065402ACE38BF70A2953FF4BEDDC15706A9BFE2A6CA3F26D7893FE3BA8BBF59517640CD6E8CC51771AC44C263873FB6AF04C0CF3B803F8A6BB83DC19A82406C17C8C058822F3F10D3343F10D8C03FC433F44019A49A3FD07635BF4807FE432148AD43CE9209C099EA653F4FEC80BF6BA123BD895E884078858440507F7A3F8FAE653D81037BC0E0AD61BFE1658A4227593AC1A80C00BF4C3F013F0AE9693FD0EB933D130D1F44EBB37CC340792740ACCE2D41FBC338408459A7BFF1A4F03EB22F513F6430E4C31BEA94C20BD5A9BEA0DBD5BDB5C626C0B3D5BCBE52F5B3C06035C73FD01E88C1782B77C204508A3F88EA93C0B72534C1EBF242C0ADFF5341A306C0C0692F843F20113B3F5847CA3FFA75683F496BFB3F4377013F2A119A4057C589C0B4DD42BE1E94193F32D5513FED3A064095A8B84195F089C260732A3FD92AE5BEB7169D41901755C18B784141416C6AC03337924032912FC380A0ECBFB03E78C0375BB4C0EA3F36BF470FD342238FDA424D6DFFBD0D15A4BFE5A030BF2DB92C3B6B5DDD3F92FE78BF01D58DBFDB4EA53ED9FDC6402B35DBC1BB3EA8C0833A9EBFF1BE8E3F404784BD39094AC0291247400B96BD3F5A35D43D5268AEC06DF986C191A588C44DCCFCC38CB1F23FF09D0FC18144983FCBBCBA407D66943E596C33C04C74D4C01BC01240B7127CBFDD3397BF08FBCB40D43AC9C0678180BFF3F0CEBC4D1020400448EB3ED60693C1739352BF231E5DC33E2716425AD0194035E943408F2C9CBF37B29B3E15C9613FBCDD063F5AF2B44145B78F40A1E06B44D3744344C8E4BB4178F63DC27B379CC481660345FF5A174199E5C6C1B5B9F9BF640C163F03896CC17246B9414880C3BFD85195BD618A80BEF4D06DBEB0EB0E41125E5DBF9277CCBFB25DB7BF10398DBFB6B254BDE7597ABFE90ABCBF119745BF5C9AD6BF855FC640065EB7BF95A39CBEE201653F9A3D853F77D78A3D41A5533F19B6FD3D2BEE0A42451EE7C0F60810C0DAEE6D40F8E78BBFA26B323FFF6A773F079A6FBFE72B6ABF5650E541FABE86BEB46D823B7D19F9C14F856842D5E67C408A5B863F0D8032BF5B83DFBDA4F2A1BDE9D60FC197FF9C3F1F37AC3DC53FEE4072F684C0FD5586C081104041516C194050B64240C892E7BF3B5FBCC018756B4037F91EC0CF7E723F1D116240880C02401D5F9A3EA44499BFD925AA3DF955A4BF52752FC039F0D1C16BF3D7C04EABB6BF04AA1EBF1A3F18400DFD1DC11E408D41E793713F1A84C44192B713C1E195F83EE9520C3FEF2AA33FF777A03FB418183ED7020B3F8A95F9408C9D5CC12EFABD3F8E9C3D3F69CB2243A50FAA4252E265405FF40740F29D10C0354C703F4D8CAEC2A53AA2C251B30C3F5D9BB8BE297C01C06A5DBBC0E41D87BFB1BAD53CAFF7A13E9E5E30C0FF9839405CBE5EC08154723EA3BD6BBF9AD09640B1CA87C117593140306111C0192C813F32B899C0C092D8BF8864893FB003F93F1EBC1ABFBBA0B2BE0AC2A4413F0956C01455C74083D165BEA0CBF1BFBA9A2F3FBA0EA03F3E444FBF454348BE201CA4C1B1B7BF4121F92E42AE2550C202238DBC8D1CB2BF4BE7FFC07DC44542162C10400FAB0FC07176AEC0B0A4953F9F7098C11227674048DA644296DDA4C1503949C47931B6C46D3B80C1C87EC140FD8C343FFFA006BE7CBC03C055E658BF8D0CEE3E30DB24C028A901C1679344C12D41B7C099D0E3C045D791C0555B19C0A7A61BBF3FB10E41CF674940707C1DC04D28403F244BD1BFF3E96C402E5677C0FB5AE4BFB9D02FBDC2EAA9407AA076C026F52E4012D2BC3ED27E19C023439A40F4A03EC1D8B824414EA7F3C1A8A15EC14084FFC032A81AC11C54AB41E99BB0C026F8F4BF4E95B83F1A4D813FCDDECDBD82E8BF3FFA508CC07094963E43C3853FB6DCEF3EED675CBD5D561E415C29443FD8E3AA400FBB7640C5CD10BFE144B8C0EB76933FCD69CA3D00D333416994EFBFBCC81AC183800DC1A2A6FABF6032ACBE2A61CC40394D69417C6AF1C1F4EF6C4085FD0943A84563C28D96B13DE30504BF4261A3BFC1EB9BBF51CF94C0161B6540652082BFAD7E953FD75C703F0C61814003F5B4BFD0478C3FDC099ABF51A913BE74E1C2BFF28761BF46D0673EBEBFB33DC7270FC034C52AC0D1981040DDCDB53FFF6E6C3F7C6D07BE684A1640CB54F13E10B7D1BDAE3322BFDC911442D4977D428545F8BEA028D9BE1419B23E8702D5BF51E68F3F1C90C13E6EA5FBBEC28DBC4157998EBFEED95143FA9F903E6872A8C0A2A49E400FA932C1A0056CC109C8634277C0A644DF601445D636EDC097641342AE9B543F208E00C0F73E9F405B4F6E3FA5472440C67E99401B929B40496089C01DA86FC275CCA142BE2F90BFD4C1F5BED1770A401AB9F43DB4B698BF35B95B3F595C4C434B071841228C8FC32EBA04C4BD7C6E40749416BF8A78623D7D36033F44F61B3FF98077C093A99440EAE296C0FE54CCBFBB173BBF172603BF3BFC57BE8FDBFDBF474A89BD898AC44116B20E404079A13EE97201403F571D3F47C005BEBB1FF73F1B1DE93D29CD99C05D611CC02D5E1043F333A842E86621C1C69B98C11BF80342874017C28E09943F680C24BF81426AC1C592EF40D10643C0F21552C0A60887C0DDE063413A2B853FC64992BDE2FAF44031C55F3ED05803C237D3E4C1DB15B3BF05D90D3EFEAF98BFE1F553BD2422953F96FF633F3A5024C07B3BCCBF78609E3FCBE3CE3EA5FC32BF03209E3D00FC7FC0A223CCC0152B44C0D50A30C18561323F3FDA64C022101440315424C095E4843FDA63C83CB5AEC73F4806294017E3C1C1727F33C155C9A742244B04C3ADE4D640B37C8CC0AA9DFFBF3F2F063FC2F438412EC30AC25CBAF7411ACD71410C1D01C27A2E3C3FDBEB444069DBD5BF2FAAF9C04414724041E549BF35BE573D2E01843FAE5DF5BF6CCC85C1BFAA0A40592D1240FD0FD53F05709CC01E1DA63E71F0C940F747B841D2D290BE92C5A2BDDF919ABD3C7A81BF1689FFBFD5B18B3FB3858F3FFAACD73E11381DC05A6BF33F444193C084391B40251E5C3F0F674B3EB679513F5698AFBD8C33C23E69B68D3DDC6EFABE8F0081BFB385FE3E7D235CBFB41775413BD871C0CD3E8CBFD7EC233AEDADE6C2A8C1ABBFA97760C1BBD325C08C9CF7415964EFC0AB98573FBFCB61BD1D1481BEE5F419404F8D0A41BEE29641D4713F418B3185BF66DD91BFE9B1B53F6A3296BDC25A32BF7C16C33CB59BC73E822FE6406276B8409FB40D417FB7F9C160F4004084F78FBE98DF573FAE9B943ED0EE80C2D10E99422CFB03C2AB7DEA41C00488BF6042C7BB780F20BF7BFBB73EAB0749C1D83632C16774D93EE97A00BFAFB7A8407151EB40D93AD43EC2FEC7BE1549B8BFBA7F754173F46ABF4148DBC066617D3F58121CC0987E0E3FDCDF83BFCEC8674058E068409C7186BE3F68CEBF4B092C3F2CD632BE2CAA0440E490153E3ABA0DBFBDDE4CBF404C703E6201813EFD51513D05A98ABF682F84BF459E2DBDFA3212BF9E0CB73F4F625BBF165D293EB04C37446D94F5C36AD66A3FC89244BE441968C016759BC0"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

