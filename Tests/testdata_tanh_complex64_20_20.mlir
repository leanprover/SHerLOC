"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %4 = "stablehlo.tanh"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x5CFFD2BF5F886CBEF6C723C0925E59C0C03BBBC0944121BF062E8240BDAF11C02306303E56F725C086E67B403B4C4D40156BD63F2ABD21405AFC77C06C50E6BFA71C76BF200E20C069EAA1C03AC9C9BF613F8C40FC9147C08EA8B2BF649A7EC0EBDD9DC0DD468CC0C5AE2FC0ED8EADBF963F4F4047291A40B3A9243FF4E90740D580A1402BD50740C28805C01634783F26C84040B5955E3F048A45402A3E21C0DDF468407D3C2CBF9DEED0BF288DAE408248B93FD6F21C40AD86CABE5FEE8FC09D044BC051D2344090D112C023615540DD4F5440CCA77BC08ED52C406CBB98C095BB9CC0C705D5C04CDF913F35802DC090AF184082898A3F14EDD4BEC26CB8BF202F8B3FBE9620403D7801C013A6213F700A2DBFE66ACB3F86BC46C0045A24BF6780273F26959BBFD0D6F43EB6EA1FC0885DF9BF51F399C05210734088B14840C9B810401B2941C0A848E9BFA98146BFE807BCC0F50ED1C0023217401BBF0DBD2E079440A5EEAF3F0B9410C0CF3C513EEB7548C05934153F1F6FA93F18E51BBE9AE6BCC02EF8A3BF2A09A23F621B8ABC5C7C2C4005460740EF7F08C0BB1B07403928C43E079841BF1277553EDD6721BF3C18E03F70F3863F492F4E40F0AA61BFC036A5BEF7CF7540E057773D9642983F1F64994019C64E4082D7973F762A51C0F3217F3F26587CC000F7EA3FF00B22401478873FCC2613C090E76BC0E20345BFB02C72C066335A40FF88ACBCD54E4940029CD23FBF67F0C095EC4C402FCF3C40B950C640D998ACC023235CBFDAAD40C034BC664085358C3FEE04A4BF53D0F3BFAE08403F8A6B1EC00B774A40DEF239C0E1097C40928C9A3E7D406AC057563C3F9308B5C0ADF699BF7EABD840C74686BFFC46D9BF2B93583F8A081EC022714DBF7D5901BFD0B2914049A946C0F92EB440AFC789BF44D045C0C5BC16C03CA033C0EE49D03FF36C82BF12551DC07874BBBF6FEFB3BFE895F63F126E4BC00BE194C0B7FFBDBED3B810BFFC316940A6EBA240D40526BFC9C54CC0F1A4AEBD350E68C0E98504C0BA10BFBFC41D62403A73D23FAFCFAE3F64A144406BF3A03FA6E2AD400F96D23EA4D556C00DB6C2BD625CB23FAEC72DBDF182CCBF89509D3F3F0B64C0881F1AC030D505C0F4848CC0BEC026405450794043B364C05C8556408CCA0A402800C7BF86B7323F327FFCBF5C3660C0B9288540DF8EB83FC7A5BCBC6D6130BD731083404456853D506243BF53BE503F34900EC115D54EC0F15755BC0D0FA3BD9BE0783F7B266FBE05F86FC071FE643F271F2A3F7F1D38C0B247253FC4478EC0E104D63E0FC2533F1B2BDDBFD97DEBBFEBA4E8C015EBC63F930B06BF00530740A6281A3EFE44464096EEC4BE061FA43FC350A4C0E2C07740FB3FC33FAFC08BBF264899BE1D7B9BBFDC6E193E73B276C00D61F8BEE44B72406F658DBF41C84840313E3C3F158A1A404A7F8C3F0E2964C033940EBF0AE795C072C5A7BE6325BC3FE8DAE13F296114BF457FB23F5FE2DE3F3CC2D23F16C25240C68D82BF10F8EC3F2221CEBCB1EB85403F469C40C4355740BD760EC0C695BC3F8109993F018C13C04D03E6BE414788405C14B3BF79A5C83FF0D7943EF5523540232E58C0817ACDC008336E409A7265C07F3F7F4020E59E40B05D10C09D215BBF9B8787BF28604740E1D599BE3C7B07401ABF3D3E1EBC823F5607E73F74D82BC020123E4060AEB23F303DDA3E638385BF0D0235400E8772C0470B1ABD1A58AA3E3BB5A33EECC241C07F842440AF5DB5BF07DB1AC0650BC8BFAF0BB83E211DCC4075224840DB188A40B3322F405700E93F6BE41540979C3D3F9CA506C07041BDC0F0436EBFB3D8FAC0D12104C08AB587C0CBAC053E6CFE9E3F9EA69B402F5996409B599AC0B18BB83F87D44540144A44C0F721E0C049465840C9D132C052D7A63F6758EEBFA7F99040F78B31C0C03723C00032DA3D4CA68E40B6321AC0033687C037E8633F43618AC0D0AAABC0C2131A3FC55FA3BEB3B3293F86E931406212A03F4A6CB1BEB2F88F3F21B3D0BFF41CC2C0DB7EAF3F5D5F4CBF4A57C9C0CE76ABC056813B3E8DF655C0A3E439C089394F3FDAB5763F26F668BF8013F3C06009A9C02472803F96033340A3F488402B01453F9ABEF2C0615E6FC030F5E83F0DDA8F409BFDA9400AE23CC0283E7E4065087C400FF1C13F50BF4FC0932847C0E202184053244740C82B2F40C5A7ACBF765E8340DC7686C034D144402E4AD7BFC14368C09F39EEBE7DF717C0EF44463E9852DF3F301C5B40F6AB13C0E1940E400F8896BF868F6CC089D69EBEFCCDE43ED03A744046E1BD3E5C8DDC3FAC750F40B2EA3B40481CB64046C025BF87DE8FC032EBCFC0ABE486C06BF962C0B302EEBF96785340EBDFFA3D84F75AC054F697BF0455F23FE1CA54C02D6211BCDABA9E40960593C01944124032BB8E3F4DAB9BC041BC0F3F7801BF3F019649402BF6A840C166ACC06FED85BFFA1681BF4FE72B40C91D10C0A7C928C0A5C073BD546C4240B0D7C93FCBEF72C0937180405C0F88C0EF99633F5458AB3F664C48C05A9E9C3F878E26401CCB86406F85F9BEB3E383C00BB5C6C0015864BF4813CABF3402EAC0150ACF3FF95E00C04B83D7BF0E17CA3F4DFF1540CCDAAE3E6F9EB6BF836808407891FDBF892A55BF2BBC3A40FEB3ECBEDA6F52C08E138540D933DB3FC46B8E40258A0340C57F604098EA4D3F98E05A40051BA93FA1E813C0732B3140A562EA3F29DEA33F41D7C9407A684B40F2D75A3F741A773F82D9F13D2156F4BE155B2CC00953C4BF88D23CC0F62A9DBE33580FBEBBB9573FB88235BFDDBE7340E6BD4AC032071CC06415CDBFB4A259C03600D8C0C5313E4043397640786C46BFD3342AC07E202A3F02C40CC0A8E40BC161C0E53FA2E570C0C6DB5B40F0797E405E9A663FD58451C004242C40CDF126BD9469923EA4665D4015FB70C06112254017DB0BC015F2F840E2785CBFD33094C0D6F363BFBAC7B440F290BEBF34460540F9BCE8BFB6A3CABD05C9DD40505BE13F244D5CBFED4E9BC0020504408732FBBFD6BA16C0C976C5BE9E0DD73FE393BD3C37B489C0C9FC14BE129C194070A02DC066834CC0223B1C3E18554D406350F4BEE54F3D40C0E5733FAE4D114135BF7840E6BB97BF69F92CC0BFE2D3BFCA91B0BF0F0513C0CA0E7640B5FF15BDE45E83C0A03D30BF06D2F9C03670A2BED3B3F2BFA37DFBBF341E37C0436A6C3E38DCB440C8E10C3F2A8DF1BFBE5CC73F3CF533C0A88B37C08F6F97C080836FBF25C36D40FC91C33F719C903ED71CB04012D4A23FB774FFC08ED4A73F0EEB82C071C035C0A413F4BFB968ACC0C78B423F55815DC0EB19AD3FD67F193FE01129405651E0BF8030183D4D9465BF29BA803F515C573FB2FDB3BE3A2C583FB530003FEFFF3F407B0EA8BE18E1F43E9C2C20BFD3906EC0E5A08D3FE6ABD3BFBCAD11403D352CC0917F3E3F660C3C40952ABA3F0603C03E77B915C1F63034BF227FB2C0835DBE3FEB82B43F0F36A33FDB68EC3E8A58EFBFE73486C00633E040E937F63FE1B685C0DD31AA3E3BDA80C0C942463F8AE39CBF1B35DA40C8FE88C0992BACC0C102AFC03A70A0BF181A514066C205C0AEAEF33F953139C0B7E85C3F630F0440F31CDD4064A56240C09745BEBA5C013F5AC494BEC85D933E9137E63F1BC6B3BF9819FBC01AA7653F376845401C4F134091D8823FD4AF41BDBCB4744023F983C02D1A723F45CB4540CB94D43FF6E15AC0842B0F414FAD973E745B6E40D47733C079F008C07BC21740D1FE97C048E297BFF2EDF8BF296729BFFE99E2BCDD436BC0AF25F0BDF7531FC05467CEBF013AEE3F9BD34040FB6C683DA93AF63F64D83FC0F8945A3DC61A99C035E5D0C014F5CF3FEEF7A23F2CB2B040E67C9B3FD0F73E3E062FB7BFCE394C400C6B16409B22CC400B3403403E64363F78AE20BF8CCECDBE40B18BBF0EF1D8BF888DB4BFE3CA523F9528D2BF7D693340E34E963FD0B01B3F75BBD13F0FE41740B5873AC093FE7EC05CA646C01A803BBFDE553FC0FBA671C04FD0C4BF412D83BE8161EABF66CFA14036C541BF13E1643E8F57E93FF7D9814042115FC084C749C072BCEF3FC54AC3BFC563FC3F396904C103046C3E297A60C0D61CC83F436D02401A8A18C0A70A11407FC699BFF18F54C0211D0F3F630C8440E991CF3F845000C0225C89C054F9B84024EDDD3FAD6BF4BEB7122EC027E30F400D208B4031285AC08B3A84C07BFE884019A66CC0F279884071E457BF923538C079E4F83E52D071C0FD2E6540E95E153FA6B2FABF780A0EC0850D333FD86833405122094057A031C09DFE8D404E308440FE48083F410229C088F3A4BF3A794E403FDF68402C2587408C5305C05C135D40CE599C40BB836A3F26B236407DC14D3FF5D054C03F81B8BF9FAC15C02C0B6D40211718C0D23F0040E1DEB9BE7A53CEBFF4DB9B3F537405C007B5D53FD21DC03FA21E0BC08B6CA6BF1D9445C00242833F8214B8BF3BE89BBF40F889BF9B1EBF3F396BD13F"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x90756FBF6216FDBC72557DBF6F6CBDBBABFF7FBF253284B70C03803F368A173A9FCB6C3EE239163F79CE7F3F96EDD2384D82793FC61384BD591980BF3892C7391BC362BFE4E0813EA40280BF35AB743593EB7F3FA7E27337ACA580BF5D17FCBDB10280BF52FA84B8A2F680BFF21963BB87EA7F3F069048BBBEB38E3FC59518BF3901803FCFBD9AB8C85F81BF5C15EE3C2B1A803FA66A9C3BEDA77F3F003D813BC4EB7F3FA511B0BA4E7880BF8EBE9CBD8E30793FDF11D9BD1F7205C0B86E7DBF68467FBF4FBC07BBF0357BBFB3A9F53BC400803F40362CBBD226813F3706903AA6FA7FBF24C59EB81E765D3F650B0B3EBE34813FF129683CA0FF14C0258B24BF64AD693F00374DBEE7287DBFB517073DB539D9BFD7DF0DBD18B57FBF432E7CBB54DBB23F1D0608BF59FF1F3FE4720A3F193385BF33BC073C2ABE7F3F21F941B711A97A3FF35DAA3B82627FBF139055BD18FF7FBFF73BFDB60B827B3F08089EBACF05803FEE019A385FEA7ABFC0420B3C609D7FBFC4E0643BBB665F3F5F9418BD690080BF300B09B7D14D5A3FE7C196BB9B8A803F68C704BCF6A981BF87A9CDBC78021E3FBD043BBF74B69D3ED4042FBF21D6833FC4B25A3DC513803FD0D74CBB941AFFBEC8D3353F9619DA3E1A8C1B4027F77F3FC05FCC378864553F55EE20BD53C8783F4F168ABEBE447B3F972F40BDBA937F3FC23B783E58FD7FBFCCB1A4BAC5C57FBF9D240A3A1D83ACBC2FCD7B3BC475873F17DE4BBD08377F3FA89AA1BA0E00803FB4ED07378FCF33BFDADC893D1F1C803F00C49D3AF5C98FBF9F05DD3D7D9B523F3B2DC03EA02F7F3F0F79D43A09D77F3F08DEE13957F77FBF47A3AC3A980080BFA65389B70B00803FDB7E18B660C180BFEC56893DB40C80BFDBBF6ABC38B9FBBF9D5B043F37BA7FBFF54B7EBB60E94ABF58F49A3CC2657CBF068B343C5135843FB40692BD63D881BFF97C4DBBE0C28BBF4C27B0BD097180BF0E26DAB91E92F2BEB31707BF9120803F2BC476BAC66C12BF292520BDADFDDDBDB46D05BF701584BF1654A7BB6237803F6D0983B9FFF0603FE4F981BC13FA803F00A026BE1D7DCF3E897D3CBE844E18C043438A40E37686C19B892A41A7A7623FD088E9BD280A81BFCDC2663C63F67FBFB53E8DB9CADC7F3FE8B124BA601D803F47AF15BBED187BBFACECB03DFC9978BF1BD6CBBC680F803F36060239D7F6BCBC5D6530BD03DC7F3F08C5963888DB6EBF6F85D93EFFFF7FBF32A2E0B174B056BC5760A3BDEB85443F3BD1CEBDD60780BFA1B88D3AA1E61B3FEF5B333EC09EC43F3067EDBE9A603A3FB93E463F400F87BFF7EE0B3D080080BF155B0B3356D98CBF659E47BF2D4A193E3AD42EBD796EE5BF29BD933F7EFF7FBF657F90383FBF863F234AA7BD3612BEBFCE8DC3BFACEF823EB4FA54BF32D321BFCF91093F1C5B4DBF2C38CCBACD285B3F9A34D3BE62FA593F054513BE3B89FCBF4109A6BDF35939C0DF4E533F38B4793F6F0257BDBA9D8F3FCAB03DBD836A6E3F7953A93C2AF29DBF3EDD31BE8F0BCBBDAA3CDB3F35F97F3FAFA64D38B1F582BF9769973BB73B803F39623C3E94BCA0BF0B44773FB09B90BF7E2E6B3A94329E3E10DB94BE536D7FBF76FF25BA8BCF7F3F19A16DBAB113803F48B3AEB9A45D80BF49AEB2BC7D1149BF457F25BC620761BF579A9CBF03DC1D3FB06CB93FFA71773FEFA9293D73A2803FFBA2ED3A8287893FF16A79BF40827F3F2F01DCBBBD532CBD12A4B03EB439A03E17EDD33D446F813F15E96ABBAA0A82BF650386B9F9EDB13E1C71AC3D5758803F7520343BC1F1803F416884BBD0887F3F9788963CF15F7ABF6871A03CA615AFBFC69456BC236682BF8C7ED9BCB74C8B3F94282140E803803FCFF15F36180480BF26600A38C2F47E3F2D49203AE8FF7FBFBAA24A3574D380BF248F7B3B06E385BF9743933C6A3D7FBF8682EB3B72A5B73FD77A5040D92781BF6BEE5CBC8DDB9D3F557499BEF8FE7FBF67702B38806CF0BE04E72A3F92CA803FE8B2973BA14A9ABFE625A03F42AA6EBFCEDCF63C7FBA7E3FE9CE03BE100080BF9516DD36CCEA403E121F4ABEE00880BF8C6EC43BCCF7833FFA1B98BE020080BFF13FF634691D4D3F1EC90EBE34FF7F3F0EE1C839FDFF7FBF53CB00B57F36863FD3CCBD3C0BFD7F3FCD8797378200803FCDC3393AEEDA683F9AA295BCA1F57FBF8FC581BB9A4C7F3FAF913BBBC31285BFCD01073EEAE27FBF705F78B86FDA75BF2F9761BD4B8B36BFF95B2A3F55173B40F0A120C0B806803FED9E0A3BC117823F0A5189BC58BE7FBF77893BBAF0321E3F4EA7173F778D1B4051DF65BFA8BD7A3F1BD210BC97FF7F3FB106B8B72BF17FBF3ED0D8B84CEC7FBF1E82A5B9417774BFB9BF723C4EF3063E01BC90BEA09093BF183C05BE99567FBF912E40B82203803FC2C1C0B7B49D813F49BA873CA0FC7FBFF8FBE038E45B673F09ACC43A5F00803F5D41543802B68ABF4CCC7ABECA3D803F042F153CFC687DBF1DC29DBAF096803F24326CB8B30580BF334E823ABC0280BFB904D03952125F3F71383C3B673F693F6C5C0FBEDBEF7F3F6956BEB9E7DD7FBFBBB7A13822ACB3BF58C5FB3B070080BFB52FB2B3049884BF31A1093C229E7F3F1F2FAEBDB3572640E13880BF5674823FA20AAC3CFA4B33BF7F95F3BD1111E1BEA2C9F4BD5F0F803FB7530FB90E05803F87826BB92E02803F0A31EB3A9C3D803FD095863A344F7CBFAFAB57BC4C8F853FDB1EF13C91FF7F3F35A1023560C0883F2787BF3EAAFC173E431002BFAE2C81BFEC3533BA0ADB7EBF45C94DBBCFF99CBE5E73893FBB8E4DBF3EA1CD3ED0D97FBF9105653B30606EBF7F8219BDD5FF7FBF032C75B5BDFE7F3F1BAB6EBA94637FBF3C761B3C8BA17EBF1A02C43C732C7B3FE95452BD9D06803FEF67073B8408393FB43383BD6CA87D3F450843BA8BA89C3EBF129A3EA5E17FBF52957DBAAE3D83BFC1137C3B7565B6BF779EAEBDFB4F5CBF99D991BE7F7286BFEC82BBBD7A1873BF934421BC0F00803FCAFB3DB53BFEB3BF6E14143EF6ED823F5116BF3C65B27CBFFEFD4ABC23D56E3F6EB7443B0AE97FBF4708DCB8113D7D3FD9A7493CB82E7FBFA99C833A0E847F3FA66F2EBBA839803FE72CA73B0000803F8696DD32CDF460BF2073023EA9EA88BF261FEFBC301A7FBF99D2A23C65BBE5BD5B89B7BFE28ED5BF4A3BADBDFA65D2BFB6F7BB3FA89077BF79A9A53C0877AD3E8CC72CBF4048C63F8D6A2F3F994F6E3F96C34F3D16D480BF811A8839254559BF81CE793E81206C3F89E13E3DE600803F81839C37020080BF8123F833D6E17FBF6F42A6394E1E81BFF109323D553A2E3F2EED3EBE1A04723F3F6CF23D2438813F47AF6E3BA6D7C23D24AD9FBFA5737E3FB3CE893E6D422BBFA40C5F3F90ABF03E6B72E4BD9878C8BED87EE83E193D34BF8509CFBECCD69E3F1FB4393DBFA37C3F54B4853CFAC2253FA590F9BD94B96B3FEEA28C3D000080BFED747CB2EC0080BF89869F36D1FD8C3F9665963DBE09D73FE163643F4BFC7FBF53CAEC39C791823F244C1CBD4876313FA15273BF845AAA3F2416C1BE0D00803F497EF3B5140080BF20B33138875C5ABFC1920E3DED1783BF98A29EBC731E80BF8EB9C63BA1507D3F8789F83C1F9A7F3F8E6B25BADB20FF3EC20D6BBE11E60E40AFD0CEBF786A90BFBB4AFEBAB24C373F389CE4BC9227813F0364933C2652A0BD12124F3F670580BFD23B023A9D85803F27E942BABFAE7FBFA26AE4BA0612CC3E1F5E153F466480BF38CFDA3BA23E823FDC14AE3A358191BF3BD6143E526E14BF0F8096BC4BAE7FBF1C109CB976C481BF0976983AA448743FA28940BCF25FED3E673629C02DBC7EBFBE6C0A3AECF77FBF1EC689B8385D883F08253E3DCB00803FE84DAF37E9D65F40AFB221C01F01803F31B85DBB1C00803F500C9EB603CF473F55F8C1BE002F95BFCBB088BF766E88BF13B1B9BC94F8BB3F957DAA3DEAA8803F3769AC3BC33DE93FAAB923BE84077C3F0BB9F53BC6D27FBFFE875A38882622BF9195BD3D5E2280BF364892B8CD6E00C06BC2EE3FB3FF7F3F78C6A9B891C802403AC808C0BEE17F3F22C9C6B98E6280BFCD6508BB7C6088BFD6AA94BDFEFF7FBF9EBD7733ED3A80BF1EF5DF3767617F3FCE940A3D9E09823F7B1971BC20B57FBF567C193BFD10803F16EB5DB8DF1683BF0BA4E2BC9800803FD3DDCCB66E5E03BFF456B13E8E21823F918B773C8F1C80BF22AC03BBE7F47F3F40BEB3B97F01803FF7B6CDB900167FBF3797AA3B4AD47FBF70D4523A0233B13F89102B3F08E67EBF92DABD3C1E63803F99D0DBBB73D980BF11F7853B97EF7F3F4ABEEC394E1A81BF21BDB3BB2A8F7F3F10F52C3B4E07803F6863C039A03D803FC38D32BAA1B2403F168009BE10C02D3F0A6FD0BD73667FBF2F4EE53DBDFC7F3F11E29E3A4710793FF264C0BC79CE87BFE0E0603D2AEF83BF9FDACCBB880A843FD2A4C43D1AC85CBF0D36653C05D8A33F7DD0B3BD1A6D8BBFB60124BE60798D3F5CA46EBC"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

