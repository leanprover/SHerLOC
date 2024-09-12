"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %5 = "stablehlo.subtract"(%3#0, %3#1) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%5) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x78DE723F8293F83E2D0640BFB5C58E40001A00C02DB0ABC0BC38B040429EAB3F4BCCD5BF6A4B27C0EDC394C06B1DCDBFF227BCBF80FEE93FF968D8C082784DC027AA1D4021E2683F927EB9BEF6BBCBC0628FE0BFF5B09BBF28E711BF9043B9BFB8F22BBFE38DBB3F17F26AC037B4394087D26ABF19E007C06E8710C065148040A5695840D846C73F1475563FB1254FBE9C9FDB3F545A10C0A9343ABE8FF9B64043A08C4081D99BBF38ED803F9B4470C0C4490840F092BABC1DE022C04D5C8040CEB3E53E58254DC03FF0714047CE47BF6AEC5A40CC828D4062FE613E9DA70140135C8C3ECA9CA7BFFB89FA3F83CFCAC00D85803F6DF2C9C041AE37C0821FF1BEE7E51A40D94EA6C08A1FC13FE614BBC08FFB5D40DAE6F93FCAC41DBFDFB6674020DB35C0A72970C0E84446BFD06376401EE4FFBF5E0A723FFA3244C00ED6F53FCD4A993F7EC81A400BECC13F2126DDC02578903F4A905B40997BAEBF2FEDEA3E8CCFE1C09C5A943FE2532FC089D7B8BF2532C23F16D9F8BF89E810C063055440A8D520BFA59D14C0BD5618C0015D80BF69A351C0FFD41D40E39A11BE0AD938402D35AD3EBED594BFA92675BFB36E3540559B94BF55A6EB3B76473E3F032E903ED658EE3FBABFE53F7D835C40B2B19C3F91DF80BF4B26CE3FD7769640391025BF8817ABBEF9231E3CA1542BBF47FA9EBFD8998FC0220B764042FAAFBFAADE51BFAF08B240D8D9A04034AA0FC0203BF53E116EC6BF54499E40C01D963E18251140528F85BEB37D0140E8590940D1A50BC0C1A017406250973F450AA53E2282E240805D26BF0C263F40943D18BF5FF4533F984E0140C0A814C047ECF8BF07B04F40E7AFC7BF111201BF9C5274BD69AA09C0F6DD82BF951E02C0FACE3BC0FBFA99C0391D6940E4AB9CC0B0AC0EC07C076240CB4EA4BF4AB016C0E099BE3EDB19A5C0885F5A409FA66EC0E08F9040D94CEE4006C20CBF19948D40EA31833F64995540DD0C913F11898B4032D1493F844878407524DCBE8EE984C00EF58AC0F615693FAE176AC0CB8600406652BD3FB9F994405A0E7740B917793F89142ABF058D0640369858C0EC0CD73FA6EF6F3E88A65A3F01EB76C03ED389C08C110140E2B255BF6E98903EC6247BC058D70E40550B3BC09F65A740B5215CC045CDBEBFB6DE2C4097438BC0C4D03C40410616BF12053DBED9812DC09192ED3F509A0E4178FEE73FBB5ED13F7E440F40F87D853F4EBD5D402CBBBBBE735E4B3F324A0CC014D26DC01C8C0F4063D9793F155430BFC3CDFFBFB24831BF536B563F6C398B404274F53FF686AF40FDC9DD3F2214CCC03DC315BF955BD2BF33C0FA40FCA84EC0327F1C3DE32C3B400640C8C00D470D412E3700C0728A5A3F39B5D63E45FD46C00F6E883F3615E4BF69621B405AAC2040D11ACA3FAEF42B403DD718C02D6B00406338894096DD3340F325E83F52042ABF30B81CC0DC571BBEB1F1A63F46973EC05E917BBF950CC8BF99848D408D2478BFBBE5123FCD5761BF9A2551BF4437F3BFDF89F93E6F20DB3EEDAB66BFE200514056F29C408CC4AC3DFA05EA3F58FE28C0E06820C06D27EA3F357911C09C5A9B3F1A9E03C0E02DFA3FD196A8BF230371BFDC2A753FAAD3DCC00C6B5140B7F80ABFE28F60BFFE1C19C07E69EF40EE1706C030DA00BF92EA7A40DB95213F68EFBEC09BDA4B40F8E101C04F508AC00BFF07BE1590CF4073D3A43F17429E40568296BF8C6B09C04A9A14C0B287923DDAB0C6C0B0EC783F11F4C83E5DBE4CC010A393C07222C13FF799E8BF5D0BF93F1EF4A43FC40FF4BD3E9B7B40F2D4D2BF04ECE0BF94787C4003698ABF7AC9A7C0ECF19FBF835E553F3354E5BF9638C93F2C70863F5E0D63409FB39C3F4CF07C4038C77840EBCB9AC04B2B5FC009DA0FC051275EC0A9BD84C0F4B51EC0FC626E40B5418B409E8E5240FE44963EC1902840F271E4BFE495D53E916F86C06E3CFBBFE62C1D409EC08FC04FE5AC40ADCC16BE5D4DBE3F16E20340B16DC640F2D3883D5A0996C00A8A59C04798E6BF629921C0CB51F2BFA204EC3E044E0740E5ADF13FB3768A40D4D275C0F314FFBF9A7D133FD1FE54BE614C05418D2735C0751527BE51DFA940F3E2CCBE5BA7D3C03702B2C0286B99BFF3F174C058E7974018408B40B95B0140F061C9C003750A4019DEADC0F8B7A7C0C9F356BF3D19363E58F1D340BA7CC33F4F645CC0E849D640A9B688BFD5AF26BFE6E57B3E5A5E2B40360FB840DA0A1C3F631ABE40729BD84050D3B3BF8C110840AD50AC3F9F58C43F1FD50840388395BE70EC34405C429BBEE3D4CFBF70B2223F8A9B84BED962BD3D3605ABBF91636AC00B2656403E9D4FC034290FBF866F1D40AC813CBFC9E0263F5670D8C0396CAF3DA429F23F61F4C1BF01B956C038073440C8957DC0245586C01C9803C0B2D542C0D6DF1C3B2DAA36404A04AE3F8AFEB2BFDDC5E83D98F250BF3CD0A6BFDC1DC3BFC54771C06B321AC0BC2E243F67A09F40FC7C19BF56A523C07A1C60C05F6F193F8F4189C09C26FBC0AFAF673FE3478740261F9740795D71409210C440991A32BFB7E781BF4FB5B2C03574AD3FFC3E79C0EE73AA3E8818D1BFFE77B5BD95F086BF31A144C0E7B507C05DA4233F550A82BF84E748BE3CA44FBFF52E1F3F2BA8D03EAE3EA24020184F4029733740E72107C0F3B578C074A55E406AD07E3E259C35405C67253F20786A40444B2440010827C02DF8A9C0DCF7B440E6733F402F201CBF6BDB953F1F3A5DC0D140843FFDEFC13F4E978F3FBADA9B401A2945C0D7FC0FC02366253F8A60CDBF5B609840C12724C033D0A7BFDE61843F74941E3E383E1C40A5DFDDBF768F52C0CF1027BDBF2A83C00D2C23BDCA0292C0389B48C0C84C0BC09E331840F9422DC0BF3BD83FF207B73FCAED4D4066694F4096C0903F81950EC030130741BB3978BEF58B9C3F3E48C7BE3758B23F7A76543EE65964BF9DDD08404F4BCCBE28F53D40FAAE06BF2F739F3F5C2AA8C0F29576C0F3DA57C09484464097C3E33F1FEF1C40BF5B29409F5BF73F3813B63FBA0891BF1D5B01BF6A465C3EBBA2063F7671BD3F104F484064D18BBF3FCD1540467403418EB0BAC0384A4EC0F6C2203FEB0DDF3F09124FBE946241C060F21CBF1DA60F404702733F43CB2C4070783C3FE9A38BC0552B9D3E252C99C015A38D3FCEFE4EBE68DA30C023436E40FE5810C0D4EEF23E12A706C0C60091BFC214443E50FCC13EFB81C24051C801C0909343C0C5D6DBBF746B3940DF6028BFD52C1DC0421246C0BA2BDC3F5898373F131043C00D5F063F1BD2ADC0FBC6DFBEEA824D3F6C2D1BC07AB363BF860B8F40BB9F9E40A51D0F40589DFFBF88FF953F680F323FD043A4BFBA405F401C3627C05288DEBE2ED03F40E73580C09F5C1E40F60B84C0294914C001C631C0F8B9A740885286C045E09C40F213B3C0957857BE867F273FF2102040F8D38FC062CDACBFBF0062C01DE78AC0C95DAE3FF66299BFF5188140C24C2DC03073CD3FE4F3BF3D04F464407EB8F63F580FDCBFDAB6DF3F380225C099E706C1B161164080620340E74B8B3F91B9443B3EE2B4BDCFE8CD3CFCFDA640CF0DDEBE03429C3E591C2F40B4F9E2C0D42BE63F81B6934015DD78BE44D62F407EF6C4BE3F1D74C0994474C06FDD8DBE7603384012F1F73EDCB3FFBFB5953BC0A7B314C0C21E84C062B862BEEB5EB0C01D3202C06C84D0BF99D66B3E41CE2B4014D20EBFE3BD45C0F521343E9326F6BE98524F408ADF3F40E8D382BF62F7D13E31F8E6BFE65105C05DEC85C0967FBC4099B9873F2DF95C402DB8D3BF16D422BF43CE11C0064236C0C9F847C0C2F102BE1A7696BFC727A1BF8C04D0BF8CCB65403146C23EABE31EBF03508F40D9B588C01CB026BF3319D53F4B45783F3C3130404B636F3F60D699C0D107B03F3F6D1340FB680D3FECD0A3C0D92B8740CA2BB23F2B7E4BBF711EA2C036921A40CA39C7BF8D4FDD3D11074C405CB2C9BFD602573FCEA44DC03487EA40B840C4BF7B886B3EE88C83BEA52AD5BF52E0AFBFC23DE2BF7F8CC6BF61D7E63ECB50E53D31F752400A580540CD1F3A402BD8FE3FD5C737C03E8649BE7F7124400C8069BFC848253F0C19573FF15D213F45DB214069E60FBEC1544E40C03852C0F97601BF266AE13D19759FC075CFF04043D8B94015F3A340E97B92C08B9B3E4006016F3F909BA3402B4FA840130322BFF03CF53F4BE4BABFDD2B25C0342F7F3FFD67433FF4F09AC0C80ECDC0A471B0406FED3940FC4D0B3F55CEE33F943D73BECD5DB3C020A41C404BF94FC0FD2E66402015AF409E0BF53F5834AABFA76FC53F9D0708C0681C943EE43E2240D73D81C00A1443C08F5B3AC0CC12FA3EE55BC93EF252A8C0D4CC59406A8A1DC0089F3640C848493E0F1D6C3EA5F737BFC4DDCE3F56831840470342C0D1059E3FAE86123F2E8B06C01CA1CABF22A6474050C240BF144DEA406EC21CC08F5D7E3EC37F8CBF2AD75A406266123EF4E1173F984A293F7E61D73FADBE59BF"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    %2 = "stablehlo.constant"() <{value = dense<"0xFAEA90404246FD3FC7C4563F59014B40F1E884C0AECEDEBF07CBCEBF85A9493F9973543FF3DC41C0BBB7FA3FAAAA66C0F17A60C09CA6D0C04F59083F17B24B402D40D140D0C918406A76933FBA676A409F08BF3E61BB29C0AF3052C0307C43C07C6D21BFC59C5440D622A740BB0820C0C3E216C1FEF59D4034E56FC0A9793CBF3CAD16408E20A8BF6C0A4A3E994846C05816EAC0420246C0BB22FA3FC8D9A740CCF1A8BCA6C841C0C7CAE2C06648424019025EBFE4FC243EAD51B5C075118BBF6519183D40DF5A40C3A974C09FA8B43FA52A3D40FDC70F3FA0D5AD3F13E7E8BE37D85AC0FAF3E340DC3DACBE30A4AEBF19CE833F49B001C0A85A91BF227687BF7EAF3940AE355940690B43C07704703F40A4F8BD3DA8AF40DCD78BBF5DED75C0B2EAF8BF2F70913EC38A9FC059C8CA3FB1C9C8BF25F0743FE0C4FC3E71EFB3BFB777A53FB4F9CF3F2C238BBFD47A2EC09E48253FB4C7C03E4B461840A2C021406219933E8FC6993A4476C2BF2DDD9A3F1FFDC93E832A5FBFD4889240F681354000574440A90D20C0C66D9840F09F613F52621140F35E363FCD9302C05578513FFCE971BFDCB012BF753A0A407F2513406919EB40B4868740C3149640E8499DBFF0F20640097061C0A27A904036BF7440E64DBC3F8B54C33F49A729C1CC5349C0FC9FF1BE0D4A3EC04F119840DD0F48C08A737DC009DB55C0C03B0840F320663FC9D8D84092200BBF74F802407EC5743FAC18853F3BB208C01584A13F4E8B7F3E322B92C015C6C640CE1EB3BF93DB6BBF3F3D8F4083B3AB402F0E52C00F7A03BFF9149F3EC06C8540A38EEEBFC9E88D404D0700C01494AF3F9953493F00927A4052796CC0551E9D400D9507C00E3D613F9207C0BE074F88BF7CBF09C0DFC4BBBF7EBDC73E764940BD9A60A9BE31B9A43DA05B6040AC637CBF73C39EBF8C9F8BC07CFE3B3F9E85E5BFED1B83C0E2BF694064D37D3E0C7C1F402AA665BF18378F3E70A7C240844A1ABFCBDD90C099B1033F6A90C1403E739640774023C054E7693FB87C7BBFFEC3CB40D4259AC0DB3C134006E599BFC220593F433D123FFFED773F876FDAC0936F2D404357A0C060367F407893874063E55EC0B120ED3F1A58483FF9A907C0766EA0C065C8F4BF784091C02DE103BF44F6C0BF727FF63F3FBAB43EAF510D40011A4F40885181C0E360EA3FBC6A48407C5A9DBD9DA349BE631BE03FCA51334008E9093DCE35FCBF1F630640DAA6B0400FFAD2C00B8D9D403E8E793F8F63A83EDFE974C06DE896406803A8C01083A5BF326DBE3F140612C01F7ED1C02DABAA3F75893E3E6407DEBE944D20C06F33B73E315D1940560F25BFAEE263BE9DB08A3F33A8CCBFFA038C401E2CEBBFAA28A3BEEFEE6E400FA588C0CE49683E45F1CDBF1486553FD33BACC023DFD13F315D723FAD89C1BE6585893F4BE3D33F74ECC9C0EDC29240C5E29DBF07BAE54083F900BFCA7DAA40CB4BBD3F191F23C01D1B0BC00FD502C010F5FD3F562EF0BFAD790640801F453F79A64BC0E51EEABFCCACA0C0887551BFCA5FF2C0C49F7A3F971D8EBF30DDA43F3C2CD8BEE0A41BBF9B6AADBEC82E06BD104FA9BF76428F3F8566C43E325598BEF60DBCC099D233BE5E921240AA5E603F2011394005A10CC01DC303C124B914BFE1A4913FC4F38A3F5F01B1BE5214B6BF5C5C88BDCB49284105E382C0DFA8D5BFF02EDE3F0D2D11BF841A72C0D1DC01C0DDF9823F114600BE3C28F2BF092AAC400D8CABBE8D1569C00BE76E3E28651C40E44F90C0AA961F3F5D79D8C0BDA9743F5C1535BE2EB466C03116D0BF01150C3F72AD243E61343EBF758E59BF3870B4401C02383F86E284C0C682A0BF7E6AD43FE75B814029ABF03F8F088B3F7FDED2BF363ABEBEF9318DBF4F09D23F9569A03D994E244080878940DDEE8FC0E22F854063B7C13F8551EA3F2407D63FC4E727C03DCBA9C0B34259BEDF0AA53FEE41AB40A23BA0BF58A30A40B48F093F4A588EC08AE3EB3F3D809CBF409D1340A86D40C0996B0F3CD03E75C0381E4740ABFB103F3CFB89C072CAE0BFA58DDF3D523649BF5B21F2BF3836BC3F1865BF3F30111AC0B66700BF6739D0BF9FEA3340D00A364028FC18BEF0E560BF15AC0A41F051F63F897AD63F6EA33B409FBF9140260657405DF1D8BFAD6A043FE3775140C2BDB33FA7EC3BC07C0A85BF28FFB13F5251D540579316C1938608C0E768A74028F1B7BF1BE8073F4C526B408FAA793F991538C0E341A04094495540107EB5BFBD507D40908DC6C04C1093BFB3DA86C08EC4E6BF920822BF7EE897C0DE2A374021AA3B3FE4221240930A48C02FFF993FD728024022A2F14018BDB240BABE71C0DCDF49C05A58044186DA83BFE8F9C83F56C1B83FE99B28C05F8E67C01995D83E10E5D33F8B4C9E4002CFDD3EBC88AD40B8C02E3E5FE38CC0AF04AFC0AE50F2BF6D7A224004BED6BE94CCAA40A9118B40C85E9D404A8F17C0BE247EC00910D5C0E4A9124069A2D3BFCE7779C058C26A40ED55193F5EAAC13DE73FE0BFB19DADC06BA2394040400D3F373D0C3F8AE0623FD52CE140097D98C06B1725BF7CDFAABFCFDB7DC0B3D33BC09A97D53E53B145BF4E54634094681040847DE83EE8791FBEBDDBBD3F98B822BF3B448E3F80CCF33EBF70503EBE2991BF08DB99C01507923F84495DC06379ECBF96E7C8BF946000C02C43D33E4F0049BF6B345D4049D3F73F991568C05052A63FF613803ED479A2BF9AC480C0496B74BF0F42A4BF049A41C0003C8FC0B7E3333F9F2B30C08E97A140829AF8C07C3A8B3FFB242DBF35CB7ABF1CEB02C07BA82CC001FC45C0CD0F4140DA58E83FEF5F1840C8CE7BC00F518C40AE7F0340D0CE8AC08950EABED055D0BFF2123B4045406DC0D91C96400F744BC09972B23F5E20ED3DFBAD9940496C8EC03DECD9BF1F69294061C01D3F21BDD43F8587E5C0FF151F40C3C74DBF8F5E8FBCE6F26340C82E5040951DA34081AB0BC0A1F172C067028BC03B3EAA3FBC1D0DC058FB9EBFEDA880C0A262A8BF9A4DCCBE780A89BE642C933F7FB485405EF8563EF7B1A4401BAE333FB7E675BFC06DC53F1A3D53405B2904C02BAAB5404C02ABBF6570B7BF20D4B53E95F647C0ED41C5BF29133140FC249840EC8FAEBEDF433140CB44A5BFD6DFCC3F4CA20B3F3D2BDD3FA286F33FE91477BEB72130C0EBBC07C08D3E81C0617E1940EC8A33C04E076DBE2D7D34BF1FF2173F0CE89FBF20E0213E8D808DC0613E4AC006314EC0664BB33F5368DEBF6C444B3F985B45C0ECDE5AC0CAAF95BFDC1D5140F1E0C33F62EDE4BF212771BF8AF78FBF5E9B69C04576823EB870F43F428F53C0F1AC1C3F7D95D2BEAF5C893F139976BF48F9FCBEB6EA17BF364940400EC2803F51AE1D4080121FC017E81CC007DE843F3D1FA53FF477813F1AD9934001E41FC05F2D34BFD5DCB24070E425C053BE813FF1282A405FBF95BF6127B0BF4BBA3BC0D80DAFBD374E883F838881407C1604BF0B1388404FC0BAC01DD9CCBFFFB927C01A462CC0150A1ABFC61A8AC093455DC07ECB8ABF705F87C0BA5A08415C89643E5E96AEBCC1141740DA650F3F0A4E154048CD2BC0ED902F3FF9CFE9C007702BBF05968DC0F864833F122916C0F2483940372CBAC0110E89BE7270A240C4E18C3D4561BC3EA6B5554090182740AC888840BD162840ADE971BFFA3B6D3FD9BBFD3F64596040796B9640AB13F23FA1BAB3403F85D03F878D4BBFD97BB5BFCAE0C33F943C88C08F8637BFD3370DBE7D088740BE9327C0CEA32C400A8704BF41D865C086A69D3F48512C407ABEE3BFBEFE58C01D8505C0847B87BFEAAAC43EC07F45407257C53E3A047C3FC72C6CC07B4BDE3CF0B8D9C08CC283BDCEAD7ABFAE82BEC04247AA3D10818DBF0CA79A408657ACBEA14011C01B202440E9377340EA18D13F34780B40CADBFAC019C6C53ED1D40FC0222BE03E886122C05C0B9E3FB420F9BF4B1718C08449DCBF2C4AD53FA64D8D4024B3A5C0F488463EB70A26BE57490CBEEC688AC028E83840B578E7BF62CA7ABF82591840B072CD3FDD075DBFD98989C08A2CD2BCCF59AE3FC646B040C01BFDC0369FE940D62A0BBF946F06BFC13AA0BF52F5E83F12062440BF1AA2C0DED643C0D4C412BE1E20B73EA32CBD40E7EA0640761527BD39381540A0FD8FC0C190B6401B33A03E53EA8EBEDC376C4034E943C02C04A7C0437D8740D8176DBFB6634F3F057FBE3E0F1AD9BFC4B4A9403DA372404F5C7940204682BF7A5788C0415F2F40E4716F40C27147C04D3768BFB73A7F3FD0631BC15A7ABB3F868049C04259E13FB0029D3FD0B405406CB4673FAEB7D9BE7576784001C24640D9B9E2BF3CBF17414F370CC053DC56BF24808B409938944034AB13C07D849CBF5B95553F42C4944033F3D9BF8E8798C0F55C1B3FFFE43D3F01FE4B3FFCF10540F796CB3EC6302A404ADECC3F91EFBA3EDB250FC06A183EC05C1B68C040A358C0FDF6D6BFFF312640423C9ABF968EB43F2BA3DCBE01A6E5BF4BFFC23D"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1, %2) : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x561E65C06221BFBF7A65CBBF2214A53FE2B7094003F967C07EEBE340FF920D3F0C0320C0488CD43EDC71D3C0F41B0040F86602401E9305412374E9C04C95CCC01A6B82C09022BDBF0ED6C1BFEA7720C1C52808C0CDC5B73FE5B62D40D0B4CD3FC05328BDA7ABEDBFF14D0EC179DEAC409B3508410AE6E1C08CBBBE3F9AA39740D278833FB3B3374079F2233F3E563940207F1041B89F563FA8B408C070FCF13E35498D40CBB7E73F0A8301418046D9C04ACA3F40424F3CBE3DC34740AA20A340A1B0D23E4C02D4C0014DF340E1470CC0280EEE3E99137740D49591BF7FC41E40B9636C4096ED06C1B9CC124077269FC00043D2BC481A89C0DA01DEBF835C163FB84CF6BED87409C197CD91407515D9C0B1C065400D5D62C0DCD5F33E1ED2EE401C9765BFD62B81C026C28640A4FF1040B469DCBEC07139BC96CB63C0C0E25440A0CEC2BD902E4B3F9C872640B7E885C0584FF73E5477434018846FC0FC6204C02201EBC02A34943F80319CBF5BDA29C0DDB28F3FD44389BF18FDDAC0681BF43E6A8C6CC04000373E2499E4C0F92CF1BFDE82B1C0847AE03F3EF4F23FF57A04404942A43FA0FA16BF1F8447C0D024093F1F2008C1CA4B87C0A8977CC06955C13F50687CBEF327AA408EE388BF5D6626C0BC961EC0001CAD3DB4E27441BE0F2040E8100D3E31E83E40E37BADC07325F13F980007BF16F3E540E13860C0CEFFDBBF68409BBFEA3DB240545189C0DC4FF4BE5EC325C072A2E2404AF977BF632C01403DD289403C0786C04FE96240D85DA1BFBDD906C06ADF85C078AF664064F1F240FCE775BFE86697BFD96FA23F7AD466C0F2AA8040CA726CC00ACB2EC0E4872BBF5EA108409740ADC0C3C30340ACF941C023B825BF46DC77BFF83D48BF861356C089255040512B9BC03A01F3BFB2E15C40834199C0BE2EAFBFEB69CE3F78D24BBFE95F2B40A0C7F7BFE6D50941D0D97240DF364CBF4C58F73FFF04F63F81B2434039649EC062D29E40F117AA401E5C5740B152CFC066AE0DC14A53E5BF005E51BB80382BC098808BC06E7AC94097B616406E00A240B8B7FF3DE6289EBF0A23913FD8465C403AD283BFC0D6A740BE8C48C07C8401C1640453BF3813283E7E05CFBF07BD19404C708B3FC59D844036EBCE3FC5E1B740264DF7BF5CA65AC06E4716406EECD1C0E84992BE80215D40C30001C04AF6BAC03968F73FDEC01141A0627C3DD94495BFDA1C0D40E3D940405EB4AE3F8D62BCC0DD65EC4024B2E3C0D21A96C054FFF43F1CB09940F0F2ACC0EE1F50406EBD193F116F26BF763CD440986D07412BDC8440CEF8C53FAC33BEC08AB9F53F381400C09A11AE40266525C03D81853E29A9EB3FF91595C0208A8E40F0112ABE640F963F481854C0B299943FAAC9563F881F31BEC801CC3F0092FC40408A78BDC4BADE3F07A600C0EAA16E3F207F2840A0ED1141E07231C038C1113F100B1AC11847B43E5EC180C0969E8EC08375C83F4A531C3F20EFCE40AB033DC09AD01C40A0CF3EC08D22CBBFAE15A43FAE401440D35EAE4028B3A9BD1E702D41BB3C7B40E0E9983F94510A3FD0F80DC050FFF2BF0AC10A407A600FC0D6542240553F4BC03F14C93F848182BF92ED9D40C10F913F6C0E13C1625319404ECF5BC019FAA83FBBF7BA40510001415EEA4EC0DC60CBBF5F858840A06F0340F7CDBCC048A6EAC012E403402ECC29C0D12EEFBFB7B5E1401F42A2408030DF401ABE0CC02B6701C06031DCBEEADFA9C019F8BBC05CA893401701233EC291B4C000CBD4BD3AAE623FDF529E40FD6C7D3FCA96BB3FB0135F402BD3B140B96F0CC0B281F5BFD6029640440E6DBED91C2EC1FAF2FBBF568E9F40DAA209BF801EB3BDB87F3FC0936FD53F80580D3EC62FB3403F478840DAFE6EC0F917A4C056DD14C0F53AC1C0942207C1C627014040E6DFBEB8A73540B7CBBA3FE475B0BF423CA840815D61409F1B213F49B2AFC00A11EAC0B74A6D404A12D5C058B39B40E5A18940B458B6BE34225240223E794048B44440105196C030A6DD3E2E359DC04DD845C092CD1A40CDE50D409751004087242B400AFFC640F8F6A9C0063D5FC096F03E400450963E8E531F411689B4C0277C40C032A7AE40EDE8F43EC27F74C1B396EFC0D8F237C0B04AD8C020F7443E28E87D3F68D46D4046EFD9C0C0058EBF8ACDDAC0498313C0BC844C3E003C9BBF00FD2FBDEE022F4178BBA7BF0484BB3FFCE9BC3EF84B97BFEE935BC06CE7D93F010D0A4188C08CC032EB26407BFD0241329DABC02B4B05417CB01F40DBF0B74066377C40EC8DAE3EB65EF2402A934AC0FAD416C090ECD2BF2277374001298EBF72AB57C0F56933C125540FC0F085083F8F152640F1F8B9C0C066963E07136BBF565003C14B172E409851B040A719F8BFC455A0C0DE9108C0D4A78CC0F0EE19C128840EC018E2AD3F4B18AF4042E9974090F096BF129E7ABF7D29A7C0FC2FA5C0D712C7C07001583F90CF4D3ED4F687406A3CD3BF0189D4408F185340D733C7C0FB3883C0133A013F2A6322C0D6111BC0FE6CFFBFB63F6B407F97854056A5384018E268BFB63982400670BDBE70FD87C0F54AAA4024AD75BFB08EACBDBD7F5CBF0E0069C0DEE053C0E2B061C0917CFBBF1D1358BF24B8C2BE2C61A7BF3EC5A4BF8A25D63EC953C53FDB0C1E4196140640565ECA40AC2987BE284214C00483AF40EEB527BE39DC674094DA33C0F71CDD3F6E30C64029317AC06CF9B1C05196DD408D7EE0403496B03EBD0E1D40D800DDBE344CB04043FC4F3F46F77740809A37BEF5059640159A55C08F45A93FDFF51FBFE9D5D940A00B083ECF27E43FBCBDFDBF4C86D4BF4092773DF6DE0C40CA98F5C0F11B06C02082743E07EBD53EACDA3BC015D7C1C0FAE6C33F140614C0B088F13E9824973EEC35A83F58DCCABFFC20F6406A56354050FF9BC0546EFA4058C4F3BF4155064107FF37C00C9E0C404C62663EB0848EC056A28EBF4AE2AFC054D0A440E245514033DFB240EBB9D2C06CF0D2BF475D08C037EBE3401C134640D27836400E7D3A40765E483F625F30C0C6E7ABBF5BDDB4C00139F9BEB944BE3F40897FBDA0E02EBEA402793F178755C090D4184175D48CC0BC0465C052277040EC2752404A043EC046D6F8C0D4548BBE087706BFF7620F40B0B68C3F9058433EB8EEC2C0CD3BCCBF7E7391C042F37640FC99F53F6445A33F8489A93FB8C70C3F3EB9343F8E0FB3BFD6F9DCBFA46AB83F8018623E4401284120EC903F60D7293E169147C0CF4F9440A6D2B9BF0CBB203F5065A63EC2ED3840C63723C0468092C0740E144037AD8FC0960B303F0C7E8E40357C2BC03A2533C027D3F8401D0A8B4055702940047D44C009A608400646983FEA9C30BF20BCF73E239767C05B7F39C05771AF406E07C7BF37DBB73FC553ADC0230555C01ABCECC0F8ABF740B8995FC080E42FBF744340C066AD9CBF104900C0A2F06A40409447C034A7CA3F50885CC0ABFAACC022E22BC070AF2EBFC0425FBEDC33484026264D409EB92D400F9DC840C4DE2140E02D26408090A640F238BFBFC26F86C09C84C5C0D433EA3F40068E3F93E316C0220226BF38B213C0A0E4FC40EA4B8FBF1994F3405BF859405EC72AC0B88D453F0ACBDE40C3D648C0AC0B0941B4A1EFBD893F0EC1A7AB78C05A1F25BF8091EDBE6E1A08C0A375C8C039D6B1C07872B0BF41C6A1C073090DC0CE4510C18884D7C00C4C61C0EC5BACC04317873FCCED723EEDFFD5BF8B5CADBF56B471403C347D4007B3484077BDA7C0AAD24140F30F90C04760C8BFE40118BFF4159540F7E8D0BF356CA7404F45DE3F2FA0B93F02219CBF63D74EC044BCC6C02A6803BF1C3C0AC0E4981B40BA7DD3BF5B4F2641D436E33E4694B73E58E92641F65E8BC008A4E83E7EC14AC08738A73FEEB8A040908ED0BF2AB909C1644484BE60A1FE3D754406414E2DB0C04216CF400342743FFA03DF3F48A1C9C048918B4098E9513F7D1EEA3FF6C3C23F3DBABFC07F93C0405D0D5AC08AB7EF408DB7B2BF30C59140C55949C08070123E84ECC9BE32BC84C098FF49C0C739A83F1C1F8D408A9B54408AAC383FBF6D26C0E5681E4190C122C18D92B13E640D4640ECEAAD3EEE5096BF9E7FDCBF7D46B64012D9B240C09A373BBD703740822413C1A54827C0707A1A3E3611EAC08A66404180E0D13DE3EF9940448D89C0447136BF76A97F40DE4F2541A047833F8A29963E158B8D3F0C84EABF567B62BFDECE89C03EC941C08ECF0BC1407DACC08F641C41E0E2283E659E4CC076AC9C40E8672B3F2445D3C0D88C42413CDB96C0C2D7D7409F7D6D40DC11303FFCCE5AC0E22A233F4EA1D9BFE8F265C0740C12BFC21E11C03E8448C1009138BFDCF2A93FCBD47DC0C6451EC104BCB64057909EBFB1390140FC798EC0D576F73F998881404A2F813F2C14D23FC70275C04EBC5BBFCAEC323EFA5D98C0B3BF4BC0304830408EEABD3FA4AC2441DCB1963F1989684074EE143FAC94523F0E89AC3F383B51BF17CE8B3FC0835E40961E72BF"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

