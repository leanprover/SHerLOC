"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f64>>
    %4 = "stablehlo.negate"(%2) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x6910C132E236EF3F7EDF290266CF0140444987F2BF67084090E4B533A2C30B406CCDDED336000640F60D8A2CF31DEC3FEBD7880B2A1BE73F9CFA6967EA0E0D40938B066B3364F3BF42EB5599E32F14C0CEFBA5319E73F4BFE0F34CF6AEFF0DC022DD2C0DB75BB63F2CFFA44299BF074054874768D48BFF3F528E64B9F2BFF0BF327992448557C73F9E684F7025B6EBBF5793E58352E20040E257AEDC3B46E83F5FAC87507A9F10401A9F89012F6A02C0D53C278DA34FF3BFD88144DAA28AF0BF255D479D2F40E13F946B271CC28CF1BF2D92DEE6F0BFFEBF221237FA762511C05F90F51641BDE2BFA251F208A6C6EEBF54D0202DCAD512C0FEE6C18143BD0AC0769388941FA30EC0708A67A8352610C07365043DC9B6FDBF5E29ADD17279FFBFA440A3643193FBBFAAF5F902B0E9DDBF72110798CE3DE13F6036879C8B2206409045A785C0C0F13F61AA49B53CFC07C0B84EB13E7FD507C0A7506E553A00F9BF48CFC78F7D96F73FE88B29F0895E16404C9759B29CC716403818822B727803404560093BB6E707C0A72644DB4D810740A4F60966DDDAE03FECA06587DD5318C07412360C6E618CBF647CE4357DAC05403AF5C2D7556210C03CF365ADF83B04C02269D5A21C26D6BFDA8ED7F224B6F9BFE41F8A97A4A6D73F6B45B7C132E6E43F95B4AD6168560EC08C6D4949C33811C0C02651DA47D0E3BF4C84168C029ED7BF3917DD9BB22410C002F5A579652088BF06F6EFA18F05E13F7DA2494A397E13C088636A171E8ED6BF067AD214BFD607407CEF54BF7E83FFBF4E8A0454E86EF0BF6296D92F804EF3BF6649AD36837CFB3FDF9B793D3149F0BF02658BA0EF0A1040224ADAE58E550EC09D5540D578B9F2BF21A4CD2AF58EE6BF249848FCED7106C09CF6447BFCBAF73FFE43FE9281C50CC051330F6EB6C9F63FD634B2FD213BD6BF514203FC2B88ECBFF8E72B92ECA4174001C3E40DE18CEA3F943F4F3639D0CA3F98C4359551C01040DFAC69D5556FD4BF1CB7D15AE01D1240EA68F8259499D73F65B290723D4A13C01FA47AD72E8216C035EE7777B471FBBF72721E59141ED83FAF9EB2338B9A18403DEAFFE3A14D0AC03418CC259D80EABFCE9AF783A8DA0140C87E61F48DA4F03FC612A8C7D079E6BFAFD9BD43508BD5BF4092998240FDF33FC01E5D07B34A10400EBA39CFF03505C01406D8131FB2124004AE89A0B51ADEBF4C059C249445FA3FFA43CAA63116F6BFF623B0DBD58EE4BFE68129C78D9010402AD6B34CA8EFFE3F765A28E5281EC03FCC77175DFAD1F2BFE2BA7175912FD8BFA4C8D9B80CF805403CD8028EE8F10540DFF8A4BFF355F1BF810FB6A486C20640AEBAE08F6FE91540D43E86D445DD0240F00919B99698B13F0099CA23ABB20340DE0B8436181804C0D67354B652370D4024F77F60B7080240E238E62E8A6C174095A6228A6C8202C065FAFA5D96E505C08A167BF9C8190AC031888FA3565EB5BF1440FF474E9EEA3F66A86136DFE105C080639E664EB9FFBFBD68FCA4307BF43F6CE449DB39DDF33FE74A7BBCF55DC1BF922DCF29611811C077AD5D9F5C7E1440F2DCE4F38F590240625C50D1C8AD09C0E6A72668C880F93F134D45A26D770840E87788752F760A4011074DDD4C24FC3F377AC5EA0BF2F1BF2C01895FA8BB09C0093C2E3E6607BD3FE129520C76EC0640A9F4EE178D30C93FD7ABBB82A1F70AC062A1DF5996EA02C0A511DDD828F3D13FF74BE61430AF10C098EC495015E018C0EE987D6681E702C07B40ACED3EE2B5BF4EFBAD984A8209C0F7C088131E09E83FD71A6DE440D91A4026B304BF29691440C9BC03AD0CD5014089BD2CEE1FBC0240560B7255735ED33FDA6F0A49E51DF93FF44B150752F3F63F3DB4F5A2CDBC0BC00D1C5B662E71F43F96A3124706B414C0B24F8F2F9F1A134040807CE60466DF3F0C67667FDFC7E43F5E9358EF5AF3C9BF9228F531F97605C06A74DD1A932A0940D0205331C97B15C08CD9AF98C8D5D23F40814F1832DDEABF765E448930750CC0FCF87ECFAC07F6BF9EB700F434DB07C0DFB5F7ED1CA0F03FA4D4B64693F7F1BF046FA12AA2B21AC07E46AE0B41081B4034216FA85C64F3BF29945DD50DFC11C05C514EC3F34A0B40306D67C1DF091240B81CEB62A58E09C0630F044DE94B07C0DAEFB5AE97AF0FC016C1A3BC373F0640A09FCF0C3F9F094026C2E791C1310F40374A18DB4D4D1440B736B86BC2C7014008B971B1065CF1BFF8EF913811DE913FC6D0351FA89200C0E24DC9913E1FD23FDEEABF77165B10C0BAC9762C865E12C0BC033F635C03CE3F9BF6C9ED28F5034021948517CD0B0740E75C4F5D0ABE1140DE37FD2FAAD0F63FA8CF2B295AA00B40E5DDCDB3AB3CA1BF72E752B73BE5FEBF0CCAF0E650700640D03CF35118AB0AC0FDFFA39F49751340EF20D88DD53F07C0EA931DB6CD3503401711DE94002BBB3F8E60F0AF85D81140632816906269D43FAAA27827AD2108C0FACF4955A11FEF3FB0EA06EA0E03FD3F6298D17DB59A094089D468823E630B4075975300172FFA3F2F5EBD50C11FF53F1CCEF9E7B0C2DEBF83B601050DE80BC0ACBF2A6CF5B8104093CFB4A69E450BC07258ED55DE530BC0BA2BA7CD595CE1BFAA456B1C461FB4BF2751FC1F0017E33F825203E1F59206C04AE4CB601B620C40CA869DB833C40240CEDBB1D6722701C09EEF063A208405406892CB10BC401B405466AED42C9119405F4BD8A4AB58C9BFDF568F12C766BEBF8A7566282050F23FC2835BAFCE9F12C080465EE60696FD3F767948990716F6BFB888A23DFBF705C0BED5C1D8D70F09C06AC668D92A22F5BF14EF1854253BC83FFAC23D13DDB519C09C64CB5027A0ECBFC86FB3892595FEBF223098FBB1D4FA3FA872C5873D2D1EC06A0AAF30EEF4E73F052B43295B2D1D40B8F165D983A3014024F0C0DF55640040A8575AF6163719C0C2F875225118FABFA8B812B6A7D605C0E11B867A28A610C0CAD624FEF23C0DC0680C40029C8DF53F49E29A338E1AF5BFD6BA1974EF8414403A556916A84DFB3FDEB9F1A4F14D0240B6590CFADAF5F93F3817DB472DF3FE3FC7448A5B4B100CC024E1510E5723CCBFB89B7F6B8FCD05C08E765AD843D00240C1855FD4B884C23FFE42877C122202C0683B2BF6EAB2DABF5AA8C781020612C0F5F6006C882BF3BFF60357FC54B500C0B6A25100BA69D33FEBD8AA8F9016D63FC9AF25218512E43FD4F364C66B1CDEBFBE56B2FB6663E13F34DC5ED14909EB3FF0F2E03398370140FE9E944F6D030EC02FE7A2CB2575E13F2A6F5294E43E18405C1619176BC0E8BFA7451373690116C008489FF8C16702C02A1CF4EDF4AA18C08E9BCA18CFF3E33F64BAE6A0059815408603685A94F6FEBFCEB9DD91B6A4074013F9DDA99340E7BFB0066069AF1F0240124A0DB3FF3BE6BF4EBD8B1FA01016C0F8CE1C9EA3931BC0E433B016169EB6BF5DBA52F40BB11540EA590CEA429B07C01CD6709208050DC0C20583B90EFD0240AE4A7CEF5EBA014072F1D6F6B76DFD3FD0347570AACDD93FFC9382A0FE5D23C00B532090DB38C93F92CCED57CD230140DF08256EA9BED73F9A912C53783202C05C7A4DBC2481DABF459CB7BF03F9FC3FE207A3F6354F06400174359B2C2393BFF0F6EAF9141323C07C83ACCEC44A06C09AD1BF6574DEE13F499A2BF5FFCB17C07D08DDCA7353DA3F3BEC019775DEE53FCCAA597E9BC4E8BF8678B8FCB77E1040FACD898403D0FA3F52EEAE515DA01440BC707A088967A63FBC775A1B5847FEBF667FA4DD1ACF134014B90F365F45F2BFB1ED8BD198CF1340F04C748EA5C0094072535352BB2CFABFDE1DAEAC3F91FA3F8252D567D602094001318F421D52DD3FEB9FFF157B6713C0C45B4D80772402C0989050BCD06015400021F43EF1631740B8BDF099EBB1FDBF4348F994171B0240D757CA4AF649F43F7A08DD6286B305C0238A3DC95F9DE2BF21ED4C1139A583BFC46CFA335EE612C0101B4D10F91D0FC0D0ABB38936D410402E20AB37A499E03F00B3546F02E403C066F54C53128915C0C91CAADE17C00740B370E9D1002902C0686604EDED910240AE1DD7A400C113C0A13A8FF78D84F63F8CAEA49FDEC9FB3FD26BF3A8070B18401C785AF0CF511B404AFE53E33EAB06408670E921E9D3FBBF69CC667D120310C0C1F26865D60A23400A390C9699EF1540C1C336AC70B7FEBF44D336A86B471140D61F8C8902F6EEBFF2A21D013BFEFCBF06B3BC79541E1B4049EBAAF3217309C097F1652CD7C305405125F28D38D806C0083E7AB4E4F8F13F2C33F52F1D4D14403CF64F5E40CDE5BFA8C0F159E760F43FE1667CA46109FDBF58C2D504A663EABF11CEEFAD60780940184B869E4AE2DF3F38F7519214E50140B1FE051091550AC0C13572F100DAE63F0DEF764B540516C0451390A10C83EC3FC06BB7AD6BB302C046FA78FA55F40AC0B8FD0CC2971919C086E8415E1A45FB3F17928E5A277505C0E60B95DFE6B813C08F2F7B6AEFAAE83FCC2E106BDA72E53F2B0F406FA42AE33F4F08513644ECA5BFF6F68FE3DA320340C8AB6512E9B10440F1C44EDD247501C0C181F6715CC2C53FBE50053E9D4C19C076CA22F8B2100240507268D5C03109408C6ED973256CD5BF4B4059FC5C530A4039677FFEEE67D2BF0A4716728D6205C01ECC6329911CEABF42AE42CEC42012403A406BD991281940565E2A42FB1E184095173A33556D0240C817D39C06FF0BC089FCB7F3BF35EBBF62D24FEACD2A0840FF657CB812B61B409DCB4DE197160440CAD6DCD85CFA13C0668E00888C4C0C403C7FE0DFFE020540589D0490BE12D3BFBD8296C6248FE1BF9964F3FFFAEE1640A2BECABBEB40FEBFDDF821BB1B7412C05ADFAE684ACFFE3F4FD00D6889DDE1BFFE0FB2935A23E8BFDE68352399AE10C00CB71362D5D3FABF201975A58CDD18408AD37B9A485FE2BF802530D54CF40FC03E0712CF798802C02E2C51B1F86907401090A785875919C0EFFEF7C365A0EC3F87C682EDB7A9E53F425FD548EF5710C0C49FAA49D52603C09C254C27F57CFA3FCCC8F8B90CAE06403ACBB953C43FAA3F6EC183DA240513C0BA8F264EB89004C0243CB411595DE5BFF6C86EB0908CF5BFC251D3EA9531DB3F6E3702ADE659F5BFF0D8C2B68EB4F73F34787E9B54B901C09C0DA05AB9C010408FB87674F8CCF53F34D95427CED8184034EA8ADB075AD7BF3A3CCF74A52BCF3F587AA7E0AD1603C058AFEB5C5C2D03C0D752F54CB08BE7BF0E18A64160FFD6BFAAB6398E508800C055C869F2140CCABF8A762B893832C53FEFA86EA2349EFA3F7263FFBE8FE7AA3FAC7E8765FBB7F3BFF8188A87B29F17C0EA0F626530250A40C7D9519DDE15F23FED55533A556DFE3F83C3DD41BF4200C0A2085BAEF8E11BC006CE6602B157C9BF8EB57A5878380340387BEDD4CFA1C2BF00007C29598DFB3FB3FCBE5B582D014038D273AFE57C0A40895859E56600D0BFF43C26550755E43F1C7599EFDD30AB3FBA61D3CC65BEF6BF0A1660F6B94BD43FF830C48A879004402816321238F8BFBFB8DFE5530B53E1BF584BF8B2C80D08C0A897D9FFD03A004092BE9D51078510403B44015A257012408258DFC618130BC088B1DBDEC03003C0D002BC064AA2D0BF5A294CBE0753E3BFC720E4B5DC6114C0E6AA44EC13E60C40668380469B8BF13F5DFA1FE0C75E18C064D4D3EA531A0CC0E732A9154AB0E73FCE91D95B48121040E517BB0D22510540C67052228393E5BFCE407C62903B0440A50289C5375516406C399DF5E82B0CC0D46D6A6701038B3F489A04B022B3DB3FAE8369E98E7FF53F3410C6BD21D2FEBF211E62C7BA0CC7BF305F701B2C82D33F890B55A8A4AA10C040A8A9540A68B8BFE42BD86BED530D40183B1E66C15C18404A77BDFA7538F1BF2AC8885C1C14FCBF6F924895178F0F40D88327E40AD804C0863D802CD10FF0BF4E9D6BD6BA6BE63FE7251A03CD4710405CD805F35080FCBF603DE9CF68C20540CAF797489849B13FF89869CA0135D73F6CD9B53D42DB014016A3330BD39C07C07D2CBD5BB90FD1BF79D014F4651C0FC0B21D1622FA31E6BF1AFEFA2C5D0AEBBF101C03599B80F83F55BC2CFFF0A407C0522EC2E3CC20DC3F5C24E437CC6106C07D95F68DAD37D0BF2AD8C02B8695F53F9C0F5F57F84506C034AF44067366CBBFBAE1F757737F0CC07F77EEF3DCD01140104F38AB75610E4021F7D9E2412FE0BF848D474594910CC05AD823381A30D03FCE48EB3FE30A0C4086E4363B259916C00586D271E58911C032628EDCBEF50DC0E6F59910B273F4BF4B429195DA0BF5BFBDCE449138F90DC0C7092784ADD512C00379B25D24D0EE3F813BD4E9BA3C18C0BE05E0FE935E11C0CEED503EBF7AFF3F64C5DDE5B1BDE6BF93A530EC865AE53F9D742084919D00C039731F3CDDB21340459874E3E136B23FE53B4F440199024092A200A957B1DABFE77C5D097ECA10C0789B389821420EC01071DC7E6F73DDBF90614A6B013D16C0C200BA72D11C1040CAC515554400F63FCCD548C77FC0FF3F75B868B1661F16408A42C2BDF14F05C00AD3E23BCAED19C09EEE50285DA4EBBF4E40F99F46A306404803BAE98ED71940BD5D3AB454ECF0BF249F753FF43FFC3F004CCEC1685B05C0B899FB6C2A9F04C0FE1783CEDF46F8BF1D577503BF7EFDBFC024FDCA60B4CC3F09F43BFD9C1F0940A8A240D19D290EC0F0D22A3F9C6DE63FE1B7E9083620D1BF76FA90212B30CABF084BBFD1170E1040146E468033891640F3625751D61A12C0108BA903B4F319C0A0E668444A8B08402AF0EAD9D94308C0844B6E2A1D1CFA3F5E291F9068EDF4BFAC65DB6BF7AE78BF066B84A3D6C0F53F56C858A6CB8DFDBFAB4AA368EF9811C0E670CDE9E02704C0CB18010E0B6DFF3F611DDEC9360413C0F4DA1AE7845519C04241AE4A75F7DBBF9D494D3E608700C0EA04A4987D35ECBF853227C031A015C01C4AB75AC6F6D13F25BAFEE8FC56FB3F6E856D56A72B11C070CE4A368F9A0240448A1A60E18BFEBFA8863B2F34CEE3BFEFEA4F8BBBAAE4BFC16BC243582312C0D8F1251F69BA17C06E73ACDA8270F13F6E8F0C61AB2E17C05C2107F523BC06C03A0A3C99B032FDBFD405D4B36DFAFABF103BA1E230A4E93FBD8D7C8EEA34F1BF4C5BF8233E17EFBFFF23D3D2DB5801C0BCF2AC49B6E500C054514A7183D1DD3FEE231012C49015C0300E7C5DF001CDBF344E92254BE6F8BF8E155692A930DA3FB49481923696F3BFF8D5805D085309C05479FA4FE96D02C00905FAA119E2114032B61E48FCAC2740D9891343AABF15C018F9749318EFFB3F2A5243B65DBAFB3FE612F7739CB70EC0DA6E6F56192AF93F88CFEF335DA702C0D0CE9AC6D3A710C0FEC6E66D8D5603407BD41BCAB2F40C40021CA7A10660F43F1324E1328B72F33F46D3300AF92EDB3F7B47F1457E8E10C0AC1B2F12355611C0FA70A5998BD8114038B085509CC7F53FB850EBC71CCD10400064F8B9A8EC17C048B90613407FFFBF71973CC7AED0CCBFB012CD27DA75ED3FCC1DC2848174F8BFE4D526CC0A4B1EC0B4FD41228BF111400E5DC47F7A2306409EF7851FA826F3BF72E931D00EA1AEBF47A938DF7FF12040CCB9290F248DEBBF9C4717B164840940AA3B702DEDDBFFBF56DD0C384D4AE43F0A9209282C2604C0040E578FAE5613401B4BE649DCB6D13F59551849577814C09FE120DFB972F53FF2755325BE68F0BF74794B58BD550AC047153C0354D2EFBF27FA87FAAD8BE13F7F22C58BA363FBBF2A3DBA5D175101403E1F5BFF75FAF63F199B4915D966EF3F7D7C79DE56901540EE6D22E0DFF8EC3F81111BE5DFCEE73F6E7F9058269E12C03F714027CEE3FABF06EC5C12C3B50E4040300DA1CD7ED53F382DD3EE358D094005AE4434002DFC3F3355118A7B00D3BF9862196B9A8D04C06436A9694E0ACA3FD012446FAD580C4041A6DA83AD3D05C08E22D0DAE1D410C0EF3DE2F071CC0240E44C9150AAE30A40FB84DDA18B2A02401D47C58A969FC2BF2EFA71405FD212C0DE1587F6138FF53FFCC62F18E66ED23F7C0470D61F98CABF72CA275A5EE0F83FF3A2C109BDC3C73FF51971E3100DEE3F841E604887C217C085B6B37230FFD3BF8F10081FD5671140B9A356F0EA36F73FB6B83AFFE12E14C0B642F0F777F3A43FBE5C60FF1C4108403884D1F6CEA8FBBFFD7E6F3960071340834B63775EEF04C0A0A9433331C419C0800F8D22F0400A4013E19AC00F1404C01CC77B5F5C8BB3BF2C41333465B1AC3FF8960B75E3B1FA3FB5202D068845104063B7F085EECE15C02EE3CCEC35B7064056A6A7ACB12CF13FCE70A08B07C11A406CB6C02CADC206C08776E7FEDCF31DC0B22FEFF71EF1024036929E73022300402E281EBA3881F2BF472748AFF56DEBBF44548BD772C9F33FB8826632BD8713C024EEAA6577AFF33F7019A3A968F2FBBFC601568C745ED4BFC19DE49F8831014005CA67AA9C400640972CF4D2E1A30F403CB89849D30809C052C76914077691BF8336E9090E93F33F9068E19B872514C0DE28F6E77031EC3F12B26531AA8502C04E458B02C8BBF73FF42B82961D0AFFBF88D964790E13E53F42A5DEAE1D2806409264E2BC75AA1340124CF0843F8EFF3F867F761E6211E03FF2CF0E5C254210C054FA00566EADE83FC028D3EBD2B6F33F18C748D91945EEBF96F421B4535505C0BC08D75B6CB910C0ECFC70A3CECFE9BF058BCB82E3CBE53F69C0A1BA0DA70240DCBCF64B124AFE3FA486E4EB21C4FABF38A3E6E613A3E43F6A86A4E1649606C06699BE079319E43F407CC03641831640EBDC4198833007C05E5B4C0F3FB10040DF222D670CD50E40F75A73FE9029D0BFADF4C7E2F04CF8BFD06346269210E53FA614EF4E0F17EC3FF49053D37789F43F68A31FE442200440F877B9DC5BB0F5BF0E7ED391023921C0F8C51199F5CBEBBF22CA22C19AD7F7BFD49761A775ECE23F1083A702812C89BF0C466CD4ECE201C0C8D31A8C96E010C0C56165B063E2B23F92B9F046A9321340"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%1) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x6910C132E236EFBF7EDF290266CF01C0444987F2BF6708C090E4B533A2C30BC06CCDDED3360006C0F60D8A2CF31DECBFEBD7880B2A1BE7BF9CFA6967EA0E0DC0938B066B3364F33F42EB5599E32F1440CEFBA5319E73F43FE0F34CF6AEFF0D4022DD2C0DB75BB6BF2CFFA44299BF07C054874768D48BFFBF528E64B9F2BFF03F327992448557C7BF9E684F7025B6EB3F5793E58352E200C0E257AEDC3B46E8BF5FAC87507A9F10C01A9F89012F6A0240D53C278DA34FF33FD88144DAA28AF03F255D479D2F40E1BF946B271CC28CF13F2D92DEE6F0BFFE3F221237FA762511405F90F51641BDE23FA251F208A6C6EE3F54D0202DCAD51240FEE6C18143BD0A40769388941FA30E40708A67A8352610407365043DC9B6FD3F5E29ADD17279FF3FA440A3643193FB3FAAF5F902B0E9DD3F72110798CE3DE1BF6036879C8B2206C09045A785C0C0F1BF61AA49B53CFC0740B84EB13E7FD50740A7506E553A00F93F48CFC78F7D96F7BFE88B29F0895E16C04C9759B29CC716C03818822B727803C04560093BB6E70740A72644DB4D8107C0A4F60966DDDAE0BFECA06587DD5318407412360C6E618C3F647CE4357DAC05C03AF5C2D7556210403CF365ADF83B04402269D5A21C26D63FDA8ED7F224B6F93FE41F8A97A4A6D7BF6B45B7C132E6E4BF95B4AD6168560E408C6D4949C3381140C02651DA47D0E33F4C84168C029ED73F3917DD9BB224104002F5A5796520883F06F6EFA18F05E1BF7DA2494A397E134088636A171E8ED63F067AD214BFD607C07CEF54BF7E83FF3F4E8A0454E86EF03F6296D92F804EF33F6649AD36837CFBBFDF9B793D3149F03F02658BA0EF0A10C0224ADAE58E550E409D5540D578B9F23F21A4CD2AF58EE63F249848FCED7106409CF6447BFCBAF7BFFE43FE9281C50C4051330F6EB6C9F6BFD634B2FD213BD63F514203FC2B88EC3FF8E72B92ECA417C001C3E40DE18CEABF943F4F3639D0CABF98C4359551C010C0DFAC69D5556FD43F1CB7D15AE01D12C0EA68F8259499D7BF65B290723D4A13401FA47AD72E82164035EE7777B471FB3F72721E59141ED8BFAF9EB2338B9A18C03DEAFFE3A14D0A403418CC259D80EA3FCE9AF783A8DA01C0C87E61F48DA4F0BFC612A8C7D079E63FAFD9BD43508BD53F4092998240FDF3BFC01E5D07B34A10C00EBA39CFF03505401406D8131FB212C004AE89A0B51ADE3F4C059C249445FABFFA43CAA63116F63FF623B0DBD58EE43FE68129C78D9010C02AD6B34CA8EFFEBF765A28E5281EC0BFCC77175DFAD1F23FE2BA7175912FD83FA4C8D9B80CF805C03CD8028EE8F105C0DFF8A4BFF355F13F810FB6A486C206C0AEBAE08F6FE915C0D43E86D445DD02C0F00919B99698B1BF0099CA23ABB203C0DE0B843618180440D67354B652370DC024F77F60B70802C0E238E62E8A6C17C095A6228A6C82024065FAFA5D96E505408A167BF9C8190A4031888FA3565EB53F1440FF474E9EEABF66A86136DFE1054080639E664EB9FF3FBD68FCA4307BF4BF6CE449DB39DDF3BFE74A7BBCF55DC13F922DCF296118114077AD5D9F5C7E14C0F2DCE4F38F5902C0625C50D1C8AD0940E6A72668C880F9BF134D45A26D7708C0E87788752F760AC011074DDD4C24FCBF377AC5EA0BF2F13F2C01895FA8BB0940093C2E3E6607BDBFE129520C76EC06C0A9F4EE178D30C9BFD7ABBB82A1F70A4062A1DF5996EA0240A511DDD828F3D1BFF74BE61430AF104098EC495015E01840EE987D6681E702407B40ACED3EE2B53F4EFBAD984A820940F7C088131E09E8BFD71A6DE440D91AC026B304BF296914C0C9BC03AD0CD501C089BD2CEE1FBC02C0560B7255735ED3BFDA6F0A49E51DF9BFF44B150752F3F6BF3DB4F5A2CDBC0B400D1C5B662E71F4BF96A3124706B41440B24F8F2F9F1A13C040807CE60466DFBF0C67667FDFC7E4BF5E9358EF5AF3C93F9228F531F97605406A74DD1A932A09C0D0205331C97B15408CD9AF98C8D5D2BF40814F1832DDEA3F765E448930750C40FCF87ECFAC07F63F9EB700F434DB0740DFB5F7ED1CA0F0BFA4D4B64693F7F13F046FA12AA2B21A407E46AE0B41081BC034216FA85C64F33F29945DD50DFC11405C514EC3F34A0BC0306D67C1DF0912C0B81CEB62A58E0940630F044DE94B0740DAEFB5AE97AF0F4016C1A3BC373F06C0A09FCF0C3F9F09C026C2E791C1310FC0374A18DB4D4D14C0B736B86BC2C701C008B971B1065CF13FF8EF913811DE91BFC6D0351FA8920040E24DC9913E1FD2BFDEEABF77165B1040BAC9762C865E1240BC033F635C03CEBF9BF6C9ED28F503C021948517CD0B07C0E75C4F5D0ABE11C0DE37FD2FAAD0F6BFA8CF2B295AA00BC0E5DDCDB3AB3CA13F72E752B73BE5FE3F0CCAF0E6507006C0D03CF35118AB0A40FDFFA39F497513C0EF20D88DD53F0740EA931DB6CD3503C01711DE94002BBBBF8E60F0AF85D811C0632816906269D4BFAAA27827AD210840FACF4955A11FEFBFB0EA06EA0E03FDBF6298D17DB59A09C089D468823E630BC075975300172FFABF2F5EBD50C11FF5BF1CCEF9E7B0C2DE3F83B601050DE80B40ACBF2A6CF5B810C093CFB4A69E450B407258ED55DE530B40BA2BA7CD595CE13FAA456B1C461FB43F2751FC1F0017E3BF825203E1F59206404AE4CB601B620CC0CA869DB833C402C0CEDBB1D6722701409EEF063A208405C06892CB10BC401BC05466AED42C9119C05F4BD8A4AB58C93FDF568F12C766BE3F8A7566282050F2BFC2835BAFCE9F124080465EE60696FDBF767948990716F63FB888A23DFBF70540BED5C1D8D70F09406AC668D92A22F53F14EF1854253BC8BFFAC23D13DDB519409C64CB5027A0EC3FC86FB3892595FE3F223098FBB1D4FABFA872C5873D2D1E406A0AAF30EEF4E7BF052B43295B2D1DC0B8F165D983A301C024F0C0DF556400C0A8575AF616371940C2F875225118FA3FA8B812B6A7D60540E11B867A28A61040CAD624FEF23C0D40680C40029C8DF5BF49E29A338E1AF53FD6BA1974EF8414C03A556916A84DFBBFDEB9F1A4F14D02C0B6590CFADAF5F9BF3817DB472DF3FEBFC7448A5B4B100C4024E1510E5723CC3FB89B7F6B8FCD05408E765AD843D002C0C1855FD4B884C2BFFE42877C12220240683B2BF6EAB2DA3F5AA8C78102061240F5F6006C882BF33FF60357FC54B50040B6A25100BA69D3BFEBD8AA8F9016D6BFC9AF25218512E4BFD4F364C66B1CDE3FBE56B2FB6663E1BF34DC5ED14909EBBFF0F2E033983701C0FE9E944F6D030E402FE7A2CB2575E1BF2A6F5294E43E18C05C1619176BC0E83FA74513736901164008489FF8C16702402A1CF4EDF4AA18408E9BCA18CFF3E3BF64BAE6A0059815C08603685A94F6FE3FCEB9DD91B6A407C013F9DDA99340E73FB0066069AF1F02C0124A0DB3FF3BE63F4EBD8B1FA0101640F8CE1C9EA3931B40E433B016169EB63F5DBA52F40BB115C0EA590CEA429B07401CD6709208050D40C20583B90EFD02C0AE4A7CEF5EBA01C072F1D6F6B76DFDBFD0347570AACDD9BFFC9382A0FE5D23400B532090DB38C9BF92CCED57CD2301C0DF08256EA9BED7BF9A912C53783202405C7A4DBC2481DA3F459CB7BF03F9FCBFE207A3F6354F06C00174359B2C23933FF0F6EAF9141323407C83ACCEC44A06409AD1BF6574DEE1BF499A2BF5FFCB17407D08DDCA7353DABF3BEC019775DEE5BFCCAA597E9BC4E83F8678B8FCB77E10C0FACD898403D0FABF52EEAE515DA014C0BC707A088967A6BFBC775A1B5847FE3F667FA4DD1ACF13C014B90F365F45F23FB1ED8BD198CF13C0F04C748EA5C009C072535352BB2CFA3FDE1DAEAC3F91FABF8252D567D60209C001318F421D52DDBFEB9FFF157B671340C45B4D8077240240989050BCD06015C00021F43EF16317C0B8BDF099EBB1FD3F4348F994171B02C0D757CA4AF649F4BF7A08DD6286B30540238A3DC95F9DE23F21ED4C1139A5833FC46CFA335EE61240101B4D10F91D0F40D0ABB38936D410C02E20AB37A499E0BF00B3546F02E4034066F54C5312891540C91CAADE17C007C0B370E9D100290240686604EDED9102C0AE1DD7A400C11340A13A8FF78D84F6BF8CAEA49FDEC9FBBFD26BF3A8070B18C01C785AF0CF511BC04AFE53E33EAB06C08670E921E9D3FB3F69CC667D12031040C1F26865D60A23C00A390C9699EF15C0C1C336AC70B7FE3F44D336A86B4711C0D61F8C8902F6EE3FF2A21D013BFEFC3F06B3BC79541E1BC049EBAAF32173094097F1652CD7C305C05125F28D38D80640083E7AB4E4F8F1BF2C33F52F1D4D14C03CF64F5E40CDE53FA8C0F159E760F4BFE1667CA46109FD3F58C2D504A663EA3F11CEEFAD607809C0184B869E4AE2DFBF38F7519214E501C0B1FE051091550A40C13572F100DAE6BF0DEF764B54051640451390A10C83ECBFC06BB7AD6BB3024046FA78FA55F40A40B8FD0CC29719194086E8415E1A45FBBF17928E5A27750540E60B95DFE6B813408F2F7B6AEFAAE8BFCC2E106BDA72E5BF2B0F406FA42AE3BF4F08513644ECA53FF6F68FE3DA3203C0C8AB6512E9B104C0F1C44EDD24750140C181F6715CC2C5BFBE50053E9D4C194076CA22F8B21002C0507268D5C03109C08C6ED973256CD53F4B4059FC5C530AC039677FFEEE67D23F0A4716728D6205401ECC6329911CEA3F42AE42CEC42012C03A406BD9912819C0565E2A42FB1E18C095173A33556D02C0C817D39C06FF0B4089FCB7F3BF35EB3F62D24FEACD2A08C0FF657CB812B61BC09DCB4DE1971604C0CAD6DCD85CFA1340668E00888C4C0CC03C7FE0DFFE0205C0589D0490BE12D33FBD8296C6248FE13F9964F3FFFAEE16C0A2BECABBEB40FE3FDDF821BB1B7412405ADFAE684ACFFEBF4FD00D6889DDE13FFE0FB2935A23E83FDE68352399AE10400CB71362D5D3FA3F201975A58CDD18C08AD37B9A485FE23F802530D54CF40F403E0712CF798802402E2C51B1F86907C01090A78587591940EFFEF7C365A0ECBF87C682EDB7A9E5BF425FD548EF571040C49FAA49D52603409C254C27F57CFABFCCC8F8B90CAE06C03ACBB953C43FAABF6EC183DA24051340BA8F264EB8900440243CB411595DE53FF6C86EB0908CF53FC251D3EA9531DBBF6E3702ADE659F53FF0D8C2B68EB4F7BF34787E9B54B901409C0DA05AB9C010C08FB87674F8CCF5BF34D95427CED818C034EA8ADB075AD73F3A3CCF74A52BCFBF587AA7E0AD16034058AFEB5C5C2D0340D752F54CB08BE73F0E18A64160FFD63FAAB6398E5088004055C869F2140CCA3F8A762B893832C5BFEFA86EA2349EFABF7263FFBE8FE7AABFAC7E8765FBB7F33FF8188A87B29F1740EA0F626530250AC0C7D9519DDE15F2BFED55533A556DFEBF83C3DD41BF420040A2085BAEF8E11B4006CE6602B157C93F8EB57A58783803C0387BEDD4CFA1C23F00007C29598DFBBFB3FCBE5B582D01C038D273AFE57C0AC0895859E56600D03FF43C26550755E4BF1C7599EFDD30ABBFBA61D3CC65BEF63F0A1660F6B94BD4BFF830C48A879004C02816321238F8BF3FB8DFE5530B53E13F584BF8B2C80D0840A897D9FFD03A00C092BE9D51078510C03B44015A257012C08258DFC618130B4088B1DBDEC0300340D002BC064AA2D03F5A294CBE0753E33FC720E4B5DC611440E6AA44EC13E60CC0668380469B8BF1BF5DFA1FE0C75E184064D4D3EA531A0C40E732A9154AB0E7BFCE91D95B481210C0E517BB0D225105C0C67052228393E53FCE407C62903B04C0A50289C5375516C06C399DF5E82B0C40D46D6A6701038BBF489A04B022B3DBBFAE8369E98E7FF5BF3410C6BD21D2FE3F211E62C7BA0CC73F305F701B2C82D3BF890B55A8A4AA104040A8A9540A68B83FE42BD86BED530DC0183B1E66C15C18C04A77BDFA7538F13F2AC8885C1C14FC3F6F924895178F0FC0D88327E40AD80440863D802CD10FF03F4E9D6BD6BA6BE6BFE7251A03CD4710C05CD805F35080FC3F603DE9CF68C205C0CAF797489849B1BFF89869CA0135D7BF6CD9B53D42DB01C016A3330BD39C07407D2CBD5BB90FD13F79D014F4651C0F40B21D1622FA31E63F1AFEFA2C5D0AEB3F101C03599B80F8BF55BC2CFFF0A40740522EC2E3CC20DCBF5C24E437CC6106407D95F68DAD37D03F2AD8C02B8695F5BF9C0F5F57F845064034AF44067366CB3FBAE1F757737F0C407F77EEF3DCD011C0104F38AB75610EC021F7D9E2412FE03F848D474594910C405AD823381A30D0BFCE48EB3FE30A0CC086E4363B259916400586D271E589114032628EDCBEF50D40E6F59910B273F43F4B429195DA0BF53FBDCE449138F90D40C7092784ADD512400379B25D24D0EEBF813BD4E9BA3C1840BE05E0FE935E1140CEED503EBF7AFFBF64C5DDE5B1BDE63F93A530EC865AE5BF9D742084919D004039731F3CDDB213C0459874E3E136B2BFE53B4F44019902C092A200A957B1DA3FE77C5D097ECA1040789B389821420E401071DC7E6F73DD3F90614A6B013D1640C200BA72D11C10C0CAC515554400F6BFCCD548C77FC0FFBF75B868B1661F16C08A42C2BDF14F05400AD3E23BCAED19409EEE50285DA4EB3F4E40F99F46A306C04803BAE98ED719C0BD5D3AB454ECF03F249F753FF43FFCBF004CCEC1685B0540B899FB6C2A9F0440FE1783CEDF46F83F1D577503BF7EFD3FC024FDCA60B4CCBF09F43BFD9C1F09C0A8A240D19D290E40F0D22A3F9C6DE6BFE1B7E9083620D13F76FA90212B30CA3F084BBFD1170E10C0146E4680338916C0F3625751D61A1240108BA903B4F31940A0E668444A8B08C02AF0EAD9D9430840844B6E2A1D1CFABF5E291F9068EDF43FAC65DB6BF7AE783F066B84A3D6C0F5BF56C858A6CB8DFD3FAB4AA368EF981140E670CDE9E0270440CB18010E0B6DFFBF611DDEC936041340F4DA1AE7845519404241AE4A75F7DB3F9D494D3E60870040EA04A4987D35EC3F853227C031A015401C4AB75AC6F6D1BF25BAFEE8FC56FBBF6E856D56A72B114070CE4A368F9A02C0448A1A60E18BFE3FA8863B2F34CEE33FEFEA4F8BBBAAE43FC16BC24358231240D8F1251F69BA17406E73ACDA8270F1BF6E8F0C61AB2E17405C2107F523BC06403A0A3C99B032FD3FD405D4B36DFAFA3F103BA1E230A4E9BFBD8D7C8EEA34F13F4C5BF8233E17EF3FFF23D3D2DB580140BCF2AC49B6E5004054514A7183D1DDBFEE231012C4901540300E7C5DF001CD3F344E92254BE6F83F8E155692A930DABFB49481923696F33FF8D5805D085309405479FA4FE96D02400905FAA119E211C032B61E48FCAC27C0D9891343AABF154018F9749318EFFBBF2A5243B65DBAFBBFE612F7739CB70E40DA6E6F56192AF9BF88CFEF335DA70240D0CE9AC6D3A71040FEC6E66D8D5603C07BD41BCAB2F40CC0021CA7A10660F4BF1324E1328B72F3BF46D3300AF92EDBBF7B47F1457E8E1040AC1B2F1235561140FA70A5998BD811C038B085509CC7F5BFB850EBC71CCD10C00064F8B9A8EC174048B90613407FFF3F71973CC7AED0CC3FB012CD27DA75EDBFCC1DC2848174F83FE4D526CC0A4B1E40B4FD41228BF111C00E5DC47F7A2306C09EF7851FA826F33F72E931D00EA1AE3F47A938DF7FF120C0CCB9290F248DEB3F9C4717B1648409C0AA3B702DEDDBFF3F56DD0C384D4AE4BF0A9209282C260440040E578FAE5613C01B4BE649DCB6D1BF59551849577814409FE120DFB972F5BFF2755325BE68F03F74794B58BD550A4047153C0354D2EF3F27FA87FAAD8BE1BF7F22C58BA363FB3F2A3DBA5D175101C03E1F5BFF75FAF6BF199B4915D966EFBF7D7C79DE569015C0EE6D22E0DFF8ECBF81111BE5DFCEE7BF6E7F9058269E12403F714027CEE3FA3F06EC5C12C3B50EC040300DA1CD7ED5BF382DD3EE358D09C005AE4434002DFCBF3355118A7B00D33F9862196B9A8D04406436A9694E0ACABFD012446FAD580CC041A6DA83AD3D05408E22D0DAE1D41040EF3DE2F071CC02C0E44C9150AAE30AC0FB84DDA18B2A02C01D47C58A969FC23F2EFA71405FD21240DE1587F6138FF5BFFCC62F18E66ED2BF7C0470D61F98CA3F72CA275A5EE0F8BFF3A2C109BDC3C7BFF51971E3100DEEBF841E604887C2174085B6B37230FFD33F8F10081FD56711C0B9A356F0EA36F7BFB6B83AFFE12E1440B642F0F777F3A4BFBE5C60FF1C4108C03884D1F6CEA8FB3FFD7E6F39600713C0834B63775EEF0440A0A9433331C41940800F8D22F0400AC013E19AC00F1404401CC77B5F5C8BB33F2C41333465B1ACBFF8960B75E3B1FABFB5202D06884510C063B7F085EECE15402EE3CCEC35B706C056A6A7ACB12CF1BFCE70A08B07C11AC06CB6C02CADC206408776E7FEDCF31D40B22FEFF71EF102C036929E73022300C02E281EBA3881F23F472748AFF56DEB3F44548BD772C9F3BFB8826632BD87134024EEAA6577AFF3BF7019A3A968F2FB3FC601568C745ED43FC19DE49F883101C005CA67AA9C4006C0972CF4D2E1A30FC03CB89849D308094052C769140776913F8336E9090E93F3BF9068E19B87251440DE28F6E77031ECBF12B26531AA8502404E458B02C8BBF7BFF42B82961D0AFF3F88D964790E13E5BF42A5DEAE1D2806C09264E2BC75AA13C0124CF0843F8EFFBF867F761E6211E0BFF2CF0E5C2542104054FA00566EADE8BFC028D3EBD2B6F3BF18C748D91945EE3F96F421B453550540BC08D75B6CB91040ECFC70A3CECFE93F058BCB82E3CBE5BF69C0A1BA0DA702C0DCBCF64B124AFEBFA486E4EB21C4FA3F38A3E6E613A3E4BF6A86A4E1649606406699BE079319E4BF407CC036418316C0EBDC4198833007405E5B4C0F3FB100C0DF222D670CD50EC0F75A73FE9029D03FADF4C7E2F04CF83FD06346269210E5BFA614EF4E0F17ECBFF49053D37789F4BF68A31FE4422004C0F877B9DC5BB0F53F0E7ED39102392140F8C51199F5CBEB3F22CA22C19AD7F73FD49761A775ECE2BF1083A702812C893F0C466CD4ECE20140C8D31A8C96E01040C56165B063E2B2BF92B9F046A93213C0"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%0) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

