"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    %5 = "stablehlo.exponential"(%4) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    %6 = "stablehlo.constant"() <{value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<complex<f32>>) -> tensor<20x20xcomplex<f32>>
    %8 = "stablehlo.add"(%7, %5) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    %9 = "stablehlo.constant"() <{value = dense<(1.000000e+00,0.000000e+00)> : tensor<complex<f32>>}> : () -> tensor<complex<f32>>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<complex<f32>>) -> tensor<20x20xcomplex<f32>>
    %11 = "stablehlo.divide"(%10, %8) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%11, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%11) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x4FA50B4089B972BFC957373FCB8323404384A5BEB0000040D82FA140E81D69C0FE4C29402F3B5A40788F40BFC559074010CCA2BF7D8607C086CC153F6BA1B3BE4073E4BF7AC2143F76419340AC759B3FB02C063F19F5214084BB143F14EE5BC08506B0BB20266BC078A289C0B72AADC072F8B6C0A87E9D3F1595D3401F049740FCACB5C0207735BF1CD89340FBB7603FEFD33CBF0AD0463FEC606AC017A50FC0FC0817C0DE73B0403CC9C340256D893F499A523F6423813E593982BFC78194BF8F0DED3F1717FCBFD8A81540A4E9414013DD0CBE6C2BD83FEC59763FC5CE1140ADC20AC0A278EB3F15E7E53FA1B39540A57C50BF691EACBF2D66933F09422E3F7DD392BAF395A0BE07B312C0F2E313C087D30140F728F8BFF85F7DC0F6B0953E1C43BC3FD971D13FA73819C0F4091B3F3B3D2EC0AEA15BBFCD82FA3F6064EB3FA92583C03D5DEEBF9425893FB484D9BEF9D630BFC5B0264022F769C0915B90C0287D72C0573B92BF1877EE3FA7755D40FB1BC83FFF39304058CF5E3F25A316C08FD581408E7525C01D966340CA3556BE2CEAC93F35D1823F90C0E1BE67344D3FD2FF71C09E789B40477188BE74424A40BC702640C5C008C0573132BF6411C13FB8392940183A88BF58696FC0124176C04947CB404C85C73FF16502C06EA99140A247DB3FEA85393F9E982240C34409C0F01994401DC57E400CF7A93FDF7D403F43462D401D7952C09289B4BF832697C0037C48C00DDE37C0DDB34BC04826E23FDF84323F135C9DBF0E5CC4BF4AEAA63F22AF9C3F049357C08637BBBF0834EF3EB6CBEEBFBAD29E407D4D1C407C44FDBF936BCD3FC36CCD3FF4788040E4F091BF38B8933F48D637C010977E3F6C3994BF7F55B5C0FD6AB3BF787B023F1C6A8E3ECB0620407BC88A40F258C5BEDA740840945A5B406A344FC03C1B9E40AF2090C015B5DE3FDCF91140179651BFC146B73FE6FC34C0295AD0BF531362BFF7F0A3BF81AC64C095B73DBF5D31C53FF4D612BF118A8E409ADEA8BC3865823C8C2BAEC0D99B653F221A67C06D479F40A1FBFDBF2BAA8FC06C81E33E9D1BB6C0A5E1DB406CA5B83FB5BD6A3E951A55BF6E2DB8BF25E122BF158DCD4065F11BBF555D84BF37156F3F6FC85740648F89C0AC982040D0CE1240120F5540DF5D9440E58002C053D0B4BFE04FDFC01AC51FBF1C26DF3EC30D433FDA02623E28165F3E3308B53F35A549BDF9C68FC0AF7803C0C20F4440C03835C04C1A413F1FCA30C0D72B1F40B061C3BF181587BFFBACA7C0960DB63EA1969EC0DFB208C061AF60C0710CB640E39B033F97A664C083360540467DDB400C470340468788BF0CA09640C4A13B3F519D07C03B8C77BE0B16C23E5BD51440EA39FE3E4D804DC0B0D97D3F46DE873F5B0359400BE4A9BF60238D3F2EE72F402CE95A40D5C8F3BF23624EBF0ABB7ABEED93CD3FD1AF4BBFF0E96FBFF62A813F8774833F6C9F933F34E9373F9B404DC038C5AEC0C8CA6F409FF995BFE354D3C0BFFF684040A463C088DAEBC0D2539E40F70CCDC0695BC3BF0E2550C06EADA53EFF5A0740E3A0D4C0D00512BE90B65BC070E8493F1A9F0C40BAF0A23FDC9D05BF77DF59BEA905FCBE54691840A1F80140B1ECC9BF8D4EC5C0B6F7DCBE86B53540424C86C0063A0F40EB759940893041C04D21283F0360664052DE5A3F1C0B2440F8AEAA404F917640475111C1E558BEC0CC5EEE3FA29761409EAA1840A21FFFBF9F9206402BAA83C009638B3FCADE8E4046374C40481F104043741E3F28ED263F3C0E0140F46B07BE47755D3F389BF93ECEE2D9BF02C1ABC05C5521C0B8035F40B00C8BC02CBFBCBFF722D4403611D83FE8F61A3F88399ABF1C6C563E5DD3C5407D316DBCDB62D2409954A54016BA10C03BCBDC3F54CD76C0F3431B4014A48CBF6A3B6B40DDAD86C0F7BA13BFF25E9FBFC6C8E2BFB45C84BE3E3C6E3FB312543F629983BE90BDA9C0E78E8DBFD4671DC1A6C404C0FF42033F15BCCC401F14663D29C08DBFE6945B40CE3315BF80B28B4059C1DEBF64024B3F5A609FBF1E61923F87BC0B40FD6E3A40383CF140AD42E8BF1511303E9405234013D79B3F6C42B0BF5D5D7DC09F7D2D40C927833E0F1200C09BB8CA403798A93E419ECBBFAE3840407FBF57BFB7F4C0402B892D3D5BF6534093812EC0DE2CEDBF79028BBE5B0141C0F4A1B7BED96FF2BFCB77F4BE19DD9F400B48774065F19E407C0B8DC0E4CD9AC05D28223FF20505BFB15DE5BF5D12CC3C3BC8803D9AF8FEBF7A4FB83EA4DEBCBD3E0316404B47983FA3B5B4C0BD20783F3AD00BC095BDEE40BEC6333F8934B53FDD674B3F4FC2B1BDFE303A40632471C090EC0EC0FD87E1BF570C59BFF9E12D3F6B6E88BF26F2833FA2CF97C06B2AEABEB61D91C0C11FE33F7A2934C0732736C08E975BBF8A08F53F8C216B4014C242BFE1CE9140BBD14140200CA83FA8E052408BDE2DC03DEF37405CB853C09A8AB03FA5193640AF20E13FB5E91A3F565012C06FA303C0EA7AA4BF0450A4C00A721D3F7F38F0BC96E2434019558340784CAE4052C75740F24DEB3F97E7A6C0BA76BBC0920D9DBE6FB6BBBF55CA4F408DA46F3F717B3A40F271733F074B0B40C006183FF252AABE8EB9C73F56A22CBD0948913F75517DC0F14A6D4064D612C09AF114415E4E23C0E5653240B5BEEFBF3B4EDFBF4B8FBC40773CA6C095ACFC3F7C7F59C0D919A53EA20F44C0FE0E16BDC9007940E576C63F88CC4840CBB1903F012401C0ACE9B7BF1F1F0A3F5A7171C056BD3340BC571B3D20244640BB031340564F4BBF2803CC3F85E9F5BFBD441A3E5AC3183C16978CBFD59C82BFDF622DC0C944A940B91C33BF2AE61640F67F2E40E82DC83FFC411CC047E37FC00D296BC0349FCB40B4E936BFABA02FBF17F5C63F5DF86A4051F0A24005E7223F26178EC0B50D87BEF4E96A40B0AAC6BF7ECE2F40E69823C0ED0FDF40431B0BC08981A2C0069B66BFD787F23F2FDE3940F4279340B3181ABF2B9D9E40D4D4BBC04862A9C04C9D203D0C3574BFE8696A407F62F03F6108594014FF00BF7AD966BF769BA1C00D3F15C08F7CA53F9352DF3FAF592540B136C8BE16E4863F569516C08EFAC1BF8AB091C05F188C3F1D941BC0BC7B1D40533FB2C0ABCAACC0581D0ABEC07E04C02BC0A73FEE2838BF2F9B73BFD29763406E0043C027288ABF3D613DC0E73367C0A2FDA8C035FB27C0DB90CCBF3C8BDAC0CD130640D1B28ABF4009953FB9F1D44090A8B4BFFBD2C33F7C9215402824A73FE8B7D0BF8A215540603F5E40E13618403F8EC3C00C7DDC40650591C08433233F2E08B53E21BB3940B8233CBD926907C064EB62401E47323D4BACA8C0EC8915BF257509C0349AE840B56F45BFAC2954C0B42783C0D311B7C05A5193C027A9E3BF862766C00AD012409E81733E0BB0D7C0A84CD33F367C9B40692C0CBF51EE11400A48CF3FA713B3C017D5EBBF148A05BF2311824092BA423F588C2140028655C01B0E8440C294B33F0FF117C008A4ACBF0663C0C0120C733F2B3D1840559BCEBF192624403B2C1BBFFB71C6BF1416E6BF3B1BD33F5811923FD44978C075C4323FBF10A2BFAB97163F4DFFA03C4F98C6BF810C4140EB865B4043589BBE8A721BC0F2BEFCBFA3583D40F27602402B3517C015C4813FA142AFBFDE21E33E2D8A51407539504051CE454059012AC038CD6FC06FE4E43F40C8A93F43EF95403A369DC0814E02BF50CE2F404890CA3F0AE008C011493D40F952C1C0DF1093BF5CAFC73E26A7943F85D115C10558413F8D8995C0572373C0A176A33FBBF7AFBFCC6B524054A9B1BF0C4AA1BF0B8F35C0E75D3D404C5484407388CDBE2E2508C12F5FD3C0985A9E3F108C57C0430DC33C3CBDE03F74E0F9BF42DDD9BF94DAF93F23F49CC02BD9AC3EB673B0BFC1B44FC04944EEBFBB4E3640990FCBBF51F3EBBF070982BF402C2BC064A5873D1DD9304084D238C0485478BFC928B93FF9E4B0BF4AFCC740194E0BC05C280AC038DCB03F8B6ADBC063A5EBBF754A62C0D190CBC0360B21C038D264BF7EA6E03F1E0873BEA795DE3FA75453BFE55A08404DB08CBF7DB4CE3EFCEFEDBC1D1FC0C06C3F4C40B1460540C01E54BC470582C0F32FADC0E77B4DC01ECE613FF7872AC0F3FE9DBF316DD3BFBDD348C0811B43C03A198C403855B23F275699C06BA5AEBF8C82A7C0D72033C09F7414400EAE00C079A54A40F0458440DDAC0740A7CC89403A0605BF48706F3EB899C5BEA6ACFB3F93A1A63DC3BC2B3E04E223C0CB715A4076F0D43FC8FA08BFDC36CE3E77311A41FF269DC09A7142C02DCC64BFE1133ABEA0CD87409EF6E84017DC4E40C502CB3FCE7992BF094A563C6B215640770A42C0F12C6EBFE051B1C0BFF2263F397E0AC0488014BEAD047BBFA96330408134AFBF3CADFFBF2E9D553FCB79EE3E699A4E409DAE064033E10840C21118C0B4D32E400E00583F980DFE3F322A3B40CFD402C05C0F70BEA32D9CC00E2C2A3DCF50783FC24393403EA98040671CF2BF"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xE36E6E3FA605A4BD54A8B23F76F6223F586A773E4CD9363F5DBB803F30A94E3B3059893F098AB1BC775FF3BC919C0C3F86C8AEBD23C29BBE1C77253FD961A6BD40CE033E3A2C903DB5157F3F2EFE183C8431AD3F9270643F1364F53F5C9D2B3FA012F63EF897EA3F6F620F3CA608273CAEEA903A2B9A4A3B48FF7F3F3F2EB0BAA8262A3B2B4111BB6E627E3F443EF53BFDAD983EA09C333ED2E581BC39B8A9BC0C5C893DD7FE6ABD01BB7F3FCE2AFD3A318C323F80B45B3D496B463E73296FBE3225853F8D0924BEC18D8D3F9688573C1F62D83E3B8B0E3FC22B943F4DE8E43E8DC994BCA72AED3D257E7A3FEB1E27BECDEB583E3E729EBE9A65483F6F07003E60DAFF3E4AEAA1BDCEBC87BD63DDAEBDD929843F7E3308BE273D933C82CEAD3B328D763FDA89653E655E933D57B1373D8CAC313D26C13ABD0C5E823FB096133E31A294BBD37883BC43E1403FAE06A6BDEC9DEBBE0357283F16CC94BBBD75D13CAF411F3CA05FA5BCE1A3953F169188BD03439D3FC259F73D99469A3F460102BF99E5813F381C1ABC3C09793F95DEB7BB5167613F7B650F3E2F09BF3EC3004D3E0AB8793B306EB7BCA76650C0327585BEA6BD843FA4948ABDB8795A3EFE5AC23E0994763F4FFF6CBDBB9C94BC00D2823C6EFE7F3FB75CE43AEBCE80BB801A07BE14145F3F97EFBB3D5516853F5E3E93BD93D5803FC769F1BB27A8513F2366FE3D4407893F758C373CC66E6F3DADA26A3E692233BD879F4EBC8552D3BB7F09293D81E03C3FC6E797BE6673B63D559B373EB6DEB13F10D4033E6CC5353E3D9C913D4E24683DFD7E0BBE3FAC833FDA32AEBD30A4773FDF47483EBD047E3FEC4E84BCF246B53F71B82EBE1B114D3F0B0672BE9B901B3A9A365FBBAF84203F5B41863D6928833FC029A5BD606B0B3E1798473F2546843F62C9593BED2E803F34D9E53BAF588D3F22F8293E49C3463EF1D6A93E36820939BC9B72BD406C523E6C2390BE62F8A83CFEF394BC67DE563F8460A9BDDF0E7D3F049475B9D738013FC3BD643E3F92B93F1935D73EAA59803F3DEFCFBB15C5243CB6F79A3B51D3373B8F2AF43ACCB54F3FB514113D2791413E135BA9BE4AD9B03E1332023D16A49E3E043781BE3510CB3FF8C96BBE287434BC3577063C02058E3FCAA9BDBC818D803FBEEE0EBCD8F32F3E6DE8DEBDA62DAF3ED71BCD3D5D072F3F1C9F443D79D3173FAE5BD63E1CE9EF3E6A6E1F3F0AE315BE8875573CB27F2E3DE6B4173D63F450BDB6372E3D2195FA3D8E7B19BE3124A23BA07EEF3AB9BA75BB9557C4BB1407C83CE96F82BCD6B3D73F8A106C3FECE0663F4E2D633DCF596E3F9B1DCABDB1487E3F4B5DC33B252DD73DF5D9BCBC1F19803FAA8C703F5C102140405D8A3E5B9E493F43335D3EC9AB7D3F549903BD659BB23F0E0B833E8942813FF02901BD24CB9C3E33A156BD04FC5C3FBC35DEBD6A0F6C3E9725583EF3A84E3F6F466C3E054EF83FA6C9F63DB04F65BB53721FBB08296D3E83B96DBD0822833FC90C363C3D4F1A39F57220BA5FB79F386ED6D7BA217A113DF0813B3C01B4653FE2E50CBD64DB5DBF93A12F4073618B3F77E50B3FE26E4B3FF0D0B6BD2B2DE33EE02AFEBD0C7D843F580EB73D0C982E3E7451883CCB898BBF292E883F2C9417BC1642453CFE0E813FC22287BAB026CA3F6D4B2DBFAADCAF3F8A05FE3EEB77803F72E54FBBFB17E138F79D1C3820DB943F819FA1BD9C03843F004DB8BD38DA873FA49BEB3DFAB1773F4B2EB0BE2543833F82CD093D49592A3F9BEC1B3E2D1A623F836660BCA2D0363F273CD23DF957EB3D22DBEB3DDADBA5BD928501BD1DD4B83A0CEB52BC0205803F0707AC3A3836353FE18E9ABEF0620D3FB009CDBC6A1BFE3E813E963DD976803F212391BB9A42913FCACE1C3E0FB5743FF82794BDEE84813FE437BA3CA356973EEDA6A2BE13CA113EA42403BDDB03403F9831303E2631D63ED980853EC4E6C4BE961D873E07BECF3D9D6E4C3D3C937F3FE260C3389346DFBECEE54ABEE4F1053E733410BF6FD7F63D897DC83D84AA1C3EBD0B4B3E91A88F3F06D0033D3404803FC76807BA20B4763FA25AC03FE631613FEE8177BEB50590BC0639073C9FD3353FFAE53CBF47927F3FCFA7163AA0AC80BEDF2B343D5FB3983E73E55ABD1881F03FFC7AB0C0B7EA6BBC72FE84BD707623C05C93B2BF077F823EDE6522BFEA0EA53EB219B0BECF8C7E3FCD95A4BC053FDA3ABF6C453C48BB293FEAFEF4BDD13C123EDAE0473B1F940D3F23F044BF16D7163F82FFB6BC2D5D753FC3CCA83D151E033B03D63D3BDBF14F3D34B1C13D920D453F1534B33EBD69303F647798BC07C4853FBA180F3D1AD215BCAE02DEBDDC028F3E01F0143E4B994D3E73674D3EBC5BFE3BD1F677BB463207BBD9DE2C3CC48377BDB9B8A0BC1B95293D31A6E43E2C4E7B3F6F2A8ABC6059813F2BB09E3ACDD0AD3F15B59BBDC6A98ABD6145A33CD101063C9AE5103DDAF9803FFA376E3D9A328D3FD55335BFAB29453DB4D0E6BDC9369D3B37875C3B69AC5DC05EB82E418B297D3FB94C44BC4207813F86FB093D0C7AA13B6546113B4FBBBA3E721BDDBE71FE793FC7F2F43CC8B8773F58BC293DCF36693FE281583DB6A6AE3E4D12F03E5E6AF83E2716A33E935085BC005D2DBCC28FE3BD9BAB6B3CB7AC9FBD99C4003D78B147BBA3B11FBECAAB7F3F39D31F3B8A9E933F1B85403DA92E5D40BF0B34BF240BDD3E3267A2BF1077A23F3511B83AC57A853FD9ADB4BEE21A363E11C7AA3DE376B5BC49BA003C0A623441E4504F41FC136E3FA5377FBD500A843F42A758BEB19F093F91E6173BFC48483EF35247BEA10F1B3D8F3354BD741246BECD4C253F58CB7E3FB555853D809062BD4FB9963DC30CCA3CC3BDFC3A76839E3EA4FC1EBEB6EB993F64D620BEBEBC7E3F26086D3BCC91383CF1FA44BB67B67F3F0D39D0BCBE09873FDEBC21BD6F11803F44FE4ABAD71F7E3B24CA9EBB6BA6953F9950463DAEE27D3F91B2B7BBEA5B7E3F21E7363BAEB9A33BF5844C3938FBC5BEF279CCBED4F7953FD7B854BDC01FB33EFC8265BE30C390BB1C6099BB85BA783FA7308D3E9F1F6F3F66D6CDBCC737993F13F0C8BE35AA5F3CE603633E70FC9D3F8FE2B8BE65CF6F3FB344493D2E0D923BEAD51DBA4855373D79FEE63D1DAC933E2A5061BE47BE833F59663BBB2D6CFABEBE430DBE00BC723C9FE1B43C63D9553B7B2794BD03BA0DBA7FB0753A1AF5373ED049663E7FF27F3F68A3A6BA15B5913FFB05563EC8ED713F840685BEF076843FFF1E4CBC8C866A3F114F5A3CF405803F1C44833AEA9B283F2C0BA53DBDAF723F4A7F14BBAFBFF8BDD0CA74BDDEA3033FD5CA8D3E6C3F1D3C3C472ABF40DF7F3FAD6EFEB9AE66A7BC2D96FD3CEBE1B4B94B9F553BA4002EBE0470D13D3309693F6D5EA13C8336C3B823829A3AA94F7E3FF19E82BB4E987E3FE415D13DF2FF80BA72D26ABB7B2217BC89853EBF4FA1AD3F0DCE173F29A89CBCCD6BFCBC171A953F756777BE12614F3E26A0353D6B2F9C3FDEEDE73E8D9238BEEC781D3EBB7B6F3ECE8DD3BE39B26C3CDCE8283ED5509B3F9CD6AE3E6E503E3F0D2A9DBE1E9A243FAED6933BA49787BE719C2D3DC54D783F3D2515BC9375EABC0EDDB0BDB6C5823F677C463DFABC583DB2B2933D70C7453EBBAC923D1C00853F878F95BB7F4C853F0584BCBCB0F595BBE889BE3CF8A7703F10A080BE0B6CD13BF5C967BB69247F3FAFF8823D317305BE24BFE13C0A20803A08EC0DBBE2B4223F2C339F3E41FD82384F9C76385C12F3BB37E5BD3BF07F633F17136CBEBDEF7D3F402814BDD77CB4BE419521BE2077833F823A3CBD4753823D98DA56BF5415E939066CA73A646A083DA7FC483AB0E4843F153436BE37D41DBD940D413E54BCE33BA7A31E3BA860AABE7A303E3D30B22FBE58267D3D114C68BC5AA158BEEA1DCBBE6145ACBED65A743F741B2240C1E1043D048630BD1BB6693F07FD4EBEFF23803F038CD0BAD6CE073DADCDDB3D50C392B9240085BA6845E73C61F00CBB152E533D94C468BD79E75A3FF5C7F3BCE4BC613F145CD0BDFA77703F361EC1BD7B7F193F7E83E4BBA50722BBDA7901B908A2633F6B35A7BAF12E363C04CB523C0515D43C3940F23CC701D93C061980BD9DDE72BE991E8DBA669860BCD1F63CBD30176D3F172E663EE483153E65C42B3E4CDE27BDD45B463DC73E1EBEDCC890BB210E813F18B8633CF9067D3F0359D6BB39750F3F2947C5BD02A2603F80890F3CAB12793F222AC5BFC840803F9000073D2782BA3EDE37C23D28FF7F3F5E2C86388939F93C400310BDFD34AD3E96244ABF8116803F7FEB82B80F3E653F58941DBE9A40503FB5869BC009AEF03C416D15BD05004C3BE4011B3B216FD13D803E5BBC87B9EDBEDF16A23E7E8342BDA0D48ABE79E2343F5E49CB3D9B79823FC663113DF3C68A3F9511C6BD32D9743FE9AB363D21AA933F3790213D125EE63D4637C2BCBDB6F63B28DBA239B661673F7DF5B5BEB7B0803FC2078DBC"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

