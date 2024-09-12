"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f64>>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64: 0, 1>}> : (tensor<1x20xcomplex<f64>>) -> tensor<20x20xcomplex<f64>>
    %6 = "stablehlo.real"(%5) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xf64>
    %7 = "stablehlo.real"(%3#1) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xf64>
    %8 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction EQ>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %9 = "stablehlo.compare"(%6, %7) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %10 = "stablehlo.imag"(%5) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xf64>
    %11 = "stablehlo.imag"(%3#1) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xf64>
    %12 = "stablehlo.compare"(%10, %11) <{compare_type = #stablehlo<comparison_type FLOAT>, comparison_direction = #stablehlo<comparison_direction GT>}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> tensor<20x20xi1>
    %13 = "stablehlo.select"(%8, %12, %9) : (tensor<20x20xi1>, tensor<20x20xi1>, tensor<20x20xi1>) -> tensor<20x20xi1>
    %14 = "stablehlo.select"(%13, %5, %3#1) : (tensor<20x20xi1>, tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> tensor<20x20xcomplex<f64>>
    "stablehlo.custom_call"(%14, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
    "func.return"(%14) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[(-1.2753961338137398,-5.4528584600751469), (-2.1204584560497288,-0.93194455875625537), (0.080909560640167813,0.6153384391806016), (-3.8527688761429757,-1.5494024479634807), (7.1787169629158374,5.4941270244167599), (-0.31654105907433183,-2.7553474887259655), (2.6327530347947694,-3.0826583072326117), (0.91824514243881805,-1.1196332462529559), (0.57640232277742598,-1.9319617874736363), (-2.2075401692840164,-0.030861113459365278), (-7.0805500712326275,-6.1360347672524966), (-1.4147616710016973,0.50107579904914323), (-0.094221292349402546,-0.85955741934592411), (-1.9901516276176219,2.0176260903831764), (-1.0664657056454483,1.9085965319548044), (-4.3689899523570839,-6.6365100667579817), (1.7553012343783068,-1.9870097637192634), (-2.2888538049911427,2.6599978580308723), (6.1201936501459615,0.42607688296962593), (-6.3560276808241412,4.3597023174432179)]]> : tensor<1x20xcomplex<f64>>}> : () -> tensor<1x20xcomplex<f64>>
    %2 = "stablehlo.constant"() <{value = dense<"0x1AB502177C28124030C0DF5EC41C0A40224EBD2D456C06C08ADFB31118ABF1BF74F17B09015A0FC0BA4D7D2488EAED3F25A214F6D8DBD83FCB19C88D24F7054074D045192671E53F9871BEFA300707C0C3A003EA1A5D1340A48EC74D8F6907C0D586F54C69D5F73F20100CFA3D8E0E4016927FA9718101C008CD7B79581FF6BFC36956A15B00ECBF580F30045A920940A45C8EC47C15EF3FECD886CE0B77CBBF64E5B5DBD90C03C0D25CB63C33B506C02F9F6368B97E04403400514E75410140086810D42974ECBF206A7AADBC9D07C06F7AE223F94B10402ECF83B1F24E01C03E86BF1C471C00C073ED7AC6238916C0D96AA18038BA0FC0B82EBCC53E2E1AC08BA0BD07417DF2BFDACEB36996A4D03FBB9B01F17CE4BB3F3E9A6CBCAF37F13FC0D4F52626BCFD3FE89710595FB513400F3F52DBC131F73FF62DCF25387107C0168262E21A8DFCBF90880AEA1B09F73FBA490F886CDC0AC0BE6137BCB477D63F0C4C35BCDBFAE9BF64AEFCB20F86F83F789DE944BB25E8BFD4FA47F19FDCFE3F74D5379C2B1A07405AAFAE736B5607408E91DF3C5C7501C0C502FB056F43E4BF72182B1691EFD63F3EF1EF4EB5D306409E64BD0D44A504408DABAE791735F63F1E9AE11578B1FE3FA6FDAC896ABCF5BFBB804A9E05E9F9BF0EEE90A9B1F100C01CD13225A0DC05C053704390ABA11B407444C5DCDFC6144053E5F37CC381D03F956E19DF8AA5F8BF41A3DB63A1BE17405D128AFB463C04405F0F4E3957F5F0BF788F2115C791FFBFE09813759ADED7BF5AB68CC4CE8A0FC0B0015F21671CEABF3B4DA02EBAD412403DB7CBCBB2F91040D85EEC71208202407CAB8C2FB9BA1040BC763D7FCBABFA3FE751A69A3DC113407E812665A18DFF3F962E3559F78BE1BFB8C37ACC984500C02F10DC26F0ED0040FE0F8A7B03C721C01E8B5F556C7E16C0062DD52EC51800C04BCF1FF5CA2BF33FE6BEAD47497B0E4094BA7F7855CF05C05CECF7792A62EA3F93D0D16EAC1AF0BF07319EF49343FBBFF6DCFE5C2D58F1BF5C955290929E06C0A6E6FE9B64D3F4BFC2AD63167DCE0840B1F88D887277D2BF5C4960EEF62A03C0CACD52D4CDA5FABFD81B38ACDEAFF6BFAFC3D35CD34BA5BFC8397482D2640CC033E198AEC43906C06245228FC025E9BFE1AF50A80494EF3F451ECD9D56BBFCBFA2D8D2CFE76103403F8B947CB8B3E5BF4ACF321B48AE11C0BACF2542E796F33F5C80A2FC837E07C02A10C8D5012817C01F3E35D2A22FF0BF888C548EFC6602C07068C776FDE2D8BF66E5FD2503FF0040DC25DBC829F61140C2E4C658CE2AE93FCB63D6F71C68E63F7A53B4883FBEE73FF709E840093CC83FEC3491EAB31715C0F9ABF4D449300D4056002656D20305402ADF5898653C1BC0519CDDCC796400C0CC7266182266F43F8B0D891EF8C516C0AF5F0756016AF7BFB27976045EA0094091D341303F3D07C00F62ECD50A6304407A9223B3AEE1F2BFDD10703BD6CC06C0214A8DB19F5AEF3F1B07D76C25D201C0C226AD069D1A08C063A9B3859D9EF13F6EF5A5AC147FF53F7656BA719850E0BF1445847797270BC0DE7D7F838839D93F7B17764087AB07405AC709295BAB13C0A84C0ACF2C20F6BFADF24647104604C0F3ADBC78F668144079D36A395FA80240839BD8DCE2B005404F5CAEA87D39E4BF802BB4FAC9DA0CC0B0EC5A2C00D8FCBF72E69107C5B5D9BF628928927582D43F35476B768FB702C0B27F917E47E5EF3FF0153C26EA50CCBF8668EFB87F5610C00FD2FD4ED0C408407689FE043D310E4007B506F931C2F6BF57287BFC8C36E03FB46FB4E8949110C0842909EAF04D0FC0BC8417E9A1A308C0842A61F5FE61EFBFD4F38CBB7B6F1CC0ED9C8B5C014A1140538A76D9FA1712C02DBCD35BFF45CA3F0BC2F97F29B30740398013F16FCA07C0B8AC5DD0FD56F03FB0B00663E7E1FABF5F372C58DCE717406EF51D372274F6BF7512687BDC9D00C0647410CE8F600140728F3F57BB27F8BFFA5D25E739D3DEBFEE935174C81FEABF39E3977153D8D43F05241D88E1480240A2A395245A5AFA3F046052542C46F1BF523509453760F03F3E8AEAC64816EB3F5E24C2DB62D813401AA403303A310D403BF8266A12BA05C0972F9ABBB92CE83F3807EE56F58B0BC0B6065303E65108C0D493E057FCD70D40C8626A85D0F4E13F82FF99DC279EFDBFB1701A2C5EBDABBFCB5B5F5FCA73F6BF9FFEAB036D471040C216FFC0965DF83FF8A1C4798497E1BF528D776B2C3002402BE2C2980D47FB3FB7582CCF289B03408E720D6A7D01F93F22E858EC3524194051A4901A12380E4023AF33253F6E05C033E1B74B9AE916C02E290528BCCA054001EFBC38CC39B7BF6E8C1D0C7EABD9BF3E2FA52D68CAFF3F157092FC3EDFFEBF551D04E0F9A40640968B6A5DA1C415409A0D3F8A83910440F5DCBFA3ECBF0DC07DC93BBF6CECE23F0E0CA9F9F91804C09322329C17DB0C40C11E02AD23221140CC0CFD97B5FF034008445DD5788ED7BF7446418B4AC11B40F3874735F0D6ECBF9473BEB267A3CD3F8E310F17A1AF1040F8656BB14E3C04409823E09F4250EF3FB65576CEE31105C0D7FF66C3EAC422C076B593E2A2B10EC032823CE7A9260740F55A65C94D7DF5BF74527772109611C0A07A247E16D0EF3F543CFD225E61C43FA614B3D2BE3C903F67614AAA932AF53FAC44FF6B3EEFF93F0A98D93E06431840B1E733D7CD31F6BFE9158BFF87EB1A40EFF13C377274F8BF52F869A8F1F7D33FF096B76053FDDCBFE46705B0CD391140FE3B1E65A36605C05D76BB6AAD57D6BF22ED5FAE540D12C0325346BCDCEEEABFB64D2E86C45F1040F3CA0464EB6D1140D2C505A33E8D04400237D09B53610AC0F8786A372407FE3F8EF681BB669CEF3FF62AF69AFF6607C0DD910AE2A955024097BC2B1A388001C0B23BB7A1E53DE6BF2AB78BD776E80240CB2D5801A1170CC09084C7194EA9E53F8BA86364A3D510C09730FBD8622DE13FEB55AC4E1340F1BF2167E5F2ED8F1BC03F2A13EC915D1140BA6594EB3D8F0040A69456061A5EE43FDD7E25A91E4111400559141ED6B4FD3F6ACCB408CD51F4BFA239D7AD2040FDBF40DA31BCA64312405EB1CF032E66D4BF18DC05372E3CEBBF34AEFEEA82C909C00E6EBD1CF14213C06D03E42DA343D73FD39E933FA0810AC0E8B465A81AD5D93FDFDD0077993414C07931C1E9287EF53F37DDB02C13BA07C027D66E783A96C13F08E907461B4D12C04EA4AC348E6FE13FD3EF1ACD81981740296DF0C26EAD1840B29A8BF09D1C0DC0480E40A9323B0AC0FA28E963FB010EC0DA038ABDADA0C8BF2E2DF41DFEA602C0CEFD265AE6ED8BBFA689EA0EB703EBBF76E36AFC3BA20C4079B8528E9DC609C03E972AFA4C08FA3F9E58C15ACDA6134078A58B6EC550BABF142FDB4C666CE8BF4CBA57CE7D37074064A5C2B11983F93F230A0B8D919901405E14DC380EBEF9BF193922C8566AEA3F7087E70C1CBF134006E1379E07ACF73F8463DD5D436906C0C2F3121BB3E80F40AE9B3E225ECC12C0389D367465341140483197561E8AF43FE2E0A781E39405C0000020E8977DE33F78653142B594EC3FC5455B4822ADD33F84B3633782D412C0A94EC6E5BBF6F43F0EFBFAA39A1CFBBF743FA2413CD6FB3F9686176BD127FABF86581A8139071C4023C848D586DDF7BF54CFF7D2ADF3104000A348ABEB800940F6160BF6C4CE10C020DAC23096CCFF3F3E9B3C82CF6100C0EEEBBF4908B507C0BE1C3D781A041440DEB5817ACD32004026BE27053E00DABF44ABE694089F05C0CBC6D9270F2A05C0EB20AFD77C0906C08DF3AB6F28EA0AC0C668D4868A4902C050905F18DF18B23FD22AAEAD1E3D14400CF133E6D837F83F88C04EB4D338FB3FE42F27099574BB3F9CE8E9815806F9BF6084AEE39F72F8BFE60D048F2867DFBFB88FECC30145E1BF966FC42C23BAD9BF29F7594E01E9F13F85CD779D6CABE9BFC039DFB02A5FF2BF5AC6E0742B5913C0E213B87A995119C05BF5D2ADA73C0AC030C129D33A8D0D40B6475AC83699F4BFE59F26513EB3F4BF3706C6BD2A0210C08E1ED8B3298E0C40E80C15C033250D401B6941260581014036F0154CF6F2DEBFF4EB4478A356F8BF285C0127F024F33FA77AE0ECEA720F4064A4C41DA738C93F57753F30173BA23FF43897A55C11F9BF7400D3A27C6C064004923AB12D4914C0405F6864D8380040C64A486C94B1DD3F22F9C468D4540AC09803DC3830E2F1BF3FD913DC9F45F83F95E7FAD2E442FBBF7ADB73593BBCF93F2D233C63E84FDA3F802A83410D55FDBFF49DD4B3B35405407F7592714C6A0AC0C6125009A45611C0AE947F557C3BF23F7402625F4C3DECBF987A9B9D9C31F73F445958B50AF30D400A390656F74116C0F9B4E6FA8ACE13C0332316E9B4B0ADBF259640E0D1AC1AC0BE870EC06C34014021CB490DFAE4E0BF7C7F54C898010AC05BA8ABD230B810409AB7826CDD7BFCBF4495FBBD5DB6EDBFAE2535E689880C4008F8F257A521F43FE1FB460BA48710C0782A3B5ECA9709C04E829B05B1A0CBBFD3008129139B16C07E4BFF450C79FCBF2C901F3A7B4516C0BCC703796F0EB13F968D2EA36A980A4046E399694D1DE33F57F97ECCDC141740599B88DFF6E7C63FE159C0250279F23FE0286C8FC40ACCBFBEF26F9AE87AF13F86CA44D4E65DE9BF1F3EB3D5C6B1F93F38CE80C4EED3DE3F3133D78767D40840604F2D1CAC79D23FB0EC21BEF3C80340190AC45BAA61C23F96F4A6B441FF04C00EC0E3A130F1FD3F98DCFB5F251B0840ABB3850EDCBB02C040F62BBF6E3700C04CF42272AECCF73FF3FF9AA7984A0540A9AB2AC5E79F0B40BA911545D67AEE3F51FF6F237D4605C069925C0C725DC23FB84C7BD62E6709C0CD5DB81AE458E5BFE4E45EF1ACC79F3F466A0CABDF0FF9BF54731F1B87ECF5BFA2018135C084FC3F1A19FDACDC210BC07AC96824DDCFEA3F9A0DB0BFB4E6B83F4434D2DA87B610C0CCB9E207ECBFE23F82B936EB6D72D4BF2282BEC67D130940239711FA112A0240551C7F64C9FDF13F08F6F6F454650E407A850FE6DA34F4BF90DA0C7DBC880DC00B1FF202DA6AE0BF696E639B057F06C054F666BB7F58174054D679BF86770CC00F68669A23D90EC023D6D024C4EA01C0CEEA1AF57EDDDB3F8E66B7076E95D13F2E98D5BA310C0340204EA039978DDBBF3D666CFCF7F7F73FB00FB70845D81940DBC599EA202AC73FC6DB892D412A0040C36873B204FA0040787561279D8BE9BF23CFF54C4CA8DB3F45341CD35347E0BFB7B4D9D605051040C23847337FA8EBBF89C532A1C403E43F794F007D4A2E0240C4ADD293B55223C0E04C01C86234E7BF27517595F71600C0E0A7298F7D0002C0F8371C7BE04502C0DD3320828972F4BF91F34D3A064FF33F74CD508FF6F9C33FF8CB3A8903A4F13FF1AF6336B1D209C073989CDAB8A4D1BF7CE6E15FEA37E8BF57C60FA2ED7E1540E3042F1090A812C07834F42091D7F93F7A6FE5BD142CF03FFE70C73177C7F4BF1C5E555AECD307C04844511737791CC00EB8D872CAF8E13F2445DC1FAF7604C0AFF25B16609112C082EA38A63F3B1240FE9C076B7EA9FC3F499895158421FFBFF4AF94993332F5BFCEB8FDFD22B108C02E58F0E4354700400FAFCD26D83600C0244C42FD7A2307400A4822A2B6D90DC0108B17C5A19BB43FDFD773CAAC130AC041A1963803FEE0BFA3613C500FFEF13F0E9428B90FE2E23FE796B323C404E5BF63A1CDEED0DEE9BF95A49635821D06404DFE068A14FF02C0F108E6183D1A094011F9B0C99FB8D0BF4F197FA3A7EFD63F8CC56AE68EA310C0A044276A6B19F93F262609FF270B09407C26363D71240140B5AB16BB11DE00C076158E565CDFDA3F2A5D392EC65AF73F6079FEC57DCA0CC0822D350811620BC07FF83904A6E100C0BAB1D5957E5802C0FFF7E4FBF290144070B015EF25C3E8BF366A1844EB2903401B7AF89BE7BCECBF983BD02369B2F9BF1EA0EEAFD83008403E407E4C3CB2F43FD91CBC4A5D0EF0BFCE30D8610B34ECBF2659BDA89D8C0EC03EE9E56D7AA90B4071D02F237FF305C04E0170A14BDC05C0A485EDBF99CB0CC04C1FAA634DA6F33F3EAEAE609B24FFBF137CCF36913905C05A9D87D3BAC2EABF07820D7C730607C0B6CE80C57862E9BFAF6CC36D8461F8BFCE7DADD1E114EFBF5AC611343E28FABF4864D8F8D9390940805F8A5A26B50F40FC8F7AE5000F07C002861DD7B71913C06486562207CECD3F8F658D4E3BBE05C0483979072A35E5BF08286D7FE066FCBF5A017AC3A258EFBFC0C575D3CBCDFFBF96933A0DD43FF93F0BEDAFFDD37FF73FE2D668C6C35E18405407714D9269D8BFC628DA345B221240CEE2870195F818C0DC84C612C41DE1BF20E2E4E92EED0A401ED0388D2E2507C06C4C11535B3F07407AB869A89241CD3F52C59C514349EF3FD6AFA353B793034059ABCC886EA9E8BF9C2F774ACA7C0DC0FC6003B255E60CC068FCE2C68D15F63FD2DCC5107F7FF43F401889CD65AF1840787C428B064FD9BFD286378C12170F403659FC78C915FF3F2B9E25CB2D54D53F60C4075C10D31EC0F00878920F6FF5BFE4F94D932AACCF3F3AA70B8BAB170FC039A558EA00CAF43F2A004A90A1F8F0BFC0094202439418C09CB46429ECBB0BC08E6791858F6509C070EDF2400B48F3BF0C8043E4682203C05D5AD7A811F21240037CA40092A700C0D66CC975E3607FBF9926E8AFC71EB4BF7BB6F776168D11C0CD67C2738ACAFE3FBCDE7D54FCD00CC07417C3E9ABF9E63F4E3C998FB606ED3FA6DECFB486F1E13F143A13CD3BBC09C0C2247FF5AFD5CC3F40793EF29787D33F2C31E4DF15BD12C0184BC249D5510CC0B8D8E461A584CA3FC8AD24FAFB470AC04BD795BF8426E93FD60250A5FAAD1540CA895F9EBAC405400EAB116A6DFCE6BFE433EE501CA6A53F4A2626ADE65F07407837BB2EB8C7FBBF6E5C438C7E3CE7BF110402976D9F05C0925397C9558FEE3FFE4841F97CF112C09EBF0127D447E43FEA90E6D8A54A14404661CF3D1E491740D6699364A76B11406A20BED264A2E13F9090B398CCB2FFBF3A353745F29700407136FD169196F23F345BF08C688205C011423EA40BCC12C0386A20C8CC2FDABF1FD6504B1353ED3F8BC66D661EE2EB3F9A9E2CD5210AE9BFC78B67B9615F10C0A80543F0CA8701C02DC090C83D8FF23F468FF7F1D7FF0940D0EC38F959B60540F8295C49874E11C030F1BAF1E5060340B14F0B71316400C0965E84E03590FC3FA7C31BF2125213C0ACC7754AA04100401616C4EAB4C8A73F3A38F4F00FDF01C081B73BE7A198C3BF2AA7AC4DB4F20240BBC5F9AF98420340B013C79A805BD93FB0A1676C01DBFB3F0FC371B9EFC8FFBF84FECEFB1B28FEBF16BDEA6698AFFF3F68C08EE9FA32DFBF9D13C01DC11EEF3F98B5D12C29050140C358EC71823ECFBF9A636E83F408EBBFE5E41AFA0B09F2BFAB454D45486012C05B02E98C7E28D2BF9496B6A04F4CF0BFC4486810726AE7BFEBFACDE3110010C0882CD41F06A00240FBBAEDAA2C31D73F6CA00C994193FD3FA5A8005193FBD23F64B301B60C9C03405CDF2F2FAAD3F1BF0CE1A3DDBACE1640FED2CFAED967E8BF449DB5C5365B0E4040B8C3DA92750BC0A0D72DB5C394FE3FB3723027E4360C406824F0A8E22EFE3F0E39C9EE3051E1BF963DAA8B410E0440860DAF49028313C0A2C96007600610401110EFE74640F23F0263383CF93A08408618EF94DA56EC3F35579C55BE3002C0F26C13E17149F03FD0F3458E1FC40CC06037EDAD905EDDBF1DE4B9A5990C0DC0032909AEC7060B4074932FCC1CF7ECBF056611A48D280AC0C0A8FA06A3A71C40F96B0DDD568E19C00869EF8E43BC0CC094933926055EF13FAEF775DEDC170AC0ACE60EB354B0EFBF89B797B7CD4F0440C2E34CC770C6F93F4C12BEB6DFCDF8BFB4E90E0D7A4A0B40F01BC9A25D0F0CC092B76BC6CE40F73FC1C00B54FC7B1B409E3C53AC88A01040C36E709FEA5305C0306E575EAE7709C0162823F9348503C051677C94EC0610C0F4AD5152605312C0E071DFAF1B7818409F699D785E7F0E409C277AFB28F301409435ED0F4F2EE83F14AE176552A6CABF2651F0C4FFEDF8BFEE6E7D3913B7DE3F442918DB37930F4032D591068FFFFFBFE47157852BDCEFBF286D20EF07A9E83FF7EF891E392EF5BFBCDF0EDF418EB8BF62A36E4896D21740B60F583C03DA01406C9D74DA2DBBF93F5E258D9E751EF9BF86E4F8C5B239C73F7F03B2707E8103C00E4F7494DEFEC53FAE227B0FE8F2D73F9F264297E8E0F03F7CFFFE69049005C0EF98D8D183E10C4030CB560F8651F03FB1C3932ACBDFF23FC222C365543CE53F7D057C29BEC9FFBFD92FCF453D590340D8EAF1442310F1BF76202F8A9BF401400CABF3C69B681040861633B9EF380BC0D6059CD0FB8C114036E260D6052A0B40680672694CD5014058023082850622C03B26CB2DE36E0D400C9ED9F27C8EF0BF19C54F175D9ADABFED6BAC2D4A9410C072908BDA47380940ED34C0BC603CFE3FAF83ACBEEDB60A402EBA5D2EE5C505C0F00CD3183576004055138EF8A715E53F7C825799A9220640A891C02AD447E73F60C23A55B3EF0CC0ACC9FCC4BF4DE03F5858670EB048E8BF70DCDE595997F5BF0CBD205271941BC006164F35B7BE11C0729DC527960BACBF5E719EDC853FDFBF46DDCCEF05C000C0DE53E091211C1240874CE06DD189F23F226EB8FE063E0C402E6104B089F60B40DC0A0361264822401100E11E8C6BC8BFC0926DC1A0C10AC0ACB1CB442BBEF1BF22234FAD193E18408E97933D4592EABFE4298E7D766409C0548520827F1DCEBF0E14B577F66EFF3FA5973FF293CBF2BF4C825B721C12094034C4F97934B402C0C4556602F5B80E402C3E14BDE371A63F92E4F5E169CB11406E0938AC0FC4F73F78091E06E63DE73FC3CEF0F0B462F03FCC83B347EE10F1BFE666A11B1991EC3F956E6DFB3BB906C0709A52A9CC7610C0C5A62850F5761440"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%1, %2) : (tensor<1x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x1AB502177C28124030C0DF5EC41C0A40F8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33F25A214F6D8DBD83FCB19C88D24F705401ADB5B9401B71C4055C2476FFCF91540C3A003EA1A5D1340A48EC74D8F6907C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF005CB548E371E23FB22265C350E9FEBFA45C8EC47C15EF3FECD886CE0B77CBBF64E5B5DBD90C03C0D25CB63C33B506C02F9F6368B97E04403400514E75410140FE3F3AF9E21EB8BF32FFA38F7E81EBBF6F7AE223F94B10402ECF83B1F24E01C01ED800583E10F1BF161460849C89FE3FD96AA18038BA0FC0B82EBCC53E2E1AC03A8844BFB615FC3F5C1C00C0CACAFFBFBB9B01F17CE4BB3F3E9A6CBCAF37F13F4D41520B147B184017EA7BF9D744DB3F0F3F52DBC131F73FF62DCF25387107C032CCC2C60568F4BF6EF5CE20BACF15C0F8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33F789DE944BB25E8BFD4FA47F19FDCFE3F1ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C09E64BD0D44A504408DABAE791735F63F1E9AE11578B1FE3FA6FDAC896ABCF5BFBB804A9E05E9F9BF0EEE90A9B1F100C01CD13225A0DC05C053704390ABA11B407444C5DCDFC6144053E5F37CC381D03FFE3F3AF9E21EB8BF32FFA38F7E81EBBF5D128AFB463C04405F0F4E3957F5F0BF1ED800583E10F1BF161460849C89FE3F5AB68CC4CE8A0FC0B0015F21671CEABF3B4DA02EBAD412403DB7CBCBB2F91040D85EEC71208202407CAB8C2FB9BA10404D41520B147B184017EA7BF9D744DB3F7E812665A18DFF3F962E3559F78BE1BF32CCC2C60568F4BF6EF5CE20BACF15C0F8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33FE6BEAD47497B0E4094BA7F7855CF05C01ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0C2AD63167DCE0840B1F88D887277D2BF005CB548E371E23FB22265C350E9FEBFD81B38ACDEAFF6BFAFC3D35CD34BA5BFC8397482D2640CC033E198AEC43906C06245228FC025E9BFE1AF50A80494EF3FFE3F3AF9E21EB8BF32FFA38F7E81EBBF3F8B947CB8B3E5BF4ACF321B48AE11C0BACF2542E796F33F5C80A2FC837E07C081B48780D87911C08433814BC98B1AC03A8844BFB615FC3F5C1C00C0CACAFFBF66E5FD2503FF0040DC25DBC829F611404D41520B147B184017EA7BF9D744DB3F7A53B4883FBEE73FF709E840093CC83F32CCC2C60568F4BF6EF5CE20BACF15C056002656D20305402ADF5898653C1BC0A51CE22C7DB6B43F6813083DDAB0E33F63A5107D78D20EC02287A5385ACAF8BF1ADB5B9401B71C4055C2476FFCF915400F62ECD50A6304407A9223B3AEE1F2BF351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF63A9B3859D9EF13F6EF5A5AC147FF53F7656BA719850E0BF1445847797270BC0DE7D7F838839D93F7B17764087AB0740B8624922DDA2F6BFDE75371DD008E03FFE3F3AF9E21EB8BF32FFA38F7E81EBBF79D36A395FA80240839BD8DCE2B005404F5CAEA87D39E4BF802BB4FAC9DA0CC0B0EC5A2C00D8FCBF72E69107C5B5D9BF3A8844BFB615FC3F5C1C00C0CACAFFBFB27F917E47E5EF3FF0153C26EA50CCBF4D41520B147B184017EA7BF9D744DB3F7689FE043D310E4007B506F931C2F6BF57287BFC8C36E03FB46FB4E8949110C0F8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33FED9C8B5C014A1140538A76D9FA1712C01ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF647410CE8F600140728F3F57BB27F8BFFA5D25E739D3DEBFEE935174C81FEABF39E3977153D8D43F05241D88E1480240A2A395245A5AFA3F046052542C46F1BF523509453760F03F3E8AEAC64816EB3F5E24C2DB62D813401AA403303A310D401ED800583E10F1BF161460849C89FE3F3807EE56F58B0BC0B6065303E65108C0D493E057FCD70D40C8626A85D0F4E13F82FF99DC279EFDBFB1701A2C5EBDABBF4D41520B147B184017EA7BF9D744DB3FC216FFC0965DF83FF8A1C4798497E1BF528D776B2C3002402BE2C2980D47FB3FB7582CCF289B03408E720D6A7D01F93F22E858EC3524194051A4901A12380E4023AF33253F6E05C033E1B74B9AE916C01ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0968B6A5DA1C415409A0D3F8A83910440005CB548E371E23FB22265C350E9FEBF0173FDD10AA901C004BF1A440E9A9FBFC11E02AD23221140CC0CFD97B5FF034008445DD5788ED7BF7446418B4AC11B40FE3F3AF9E21EB8BF32FFA38F7E81EBBF8E310F17A1AF1040F8656BB14E3C04409823E09F4250EF3FB65576CEE31105C081B48780D87911C08433814BC98B1AC032823CE7A9260740F55A65C94D7DF5BFD2186E95924F02C09561FDF4AC4705404D41520B147B184017EA7BF9D744DB3F67614AAA932AF53FAC44FF6B3EEFF93F0A98D93E06431840B1E733D7CD31F6BFE9158BFF87EB1A40EFF13C377274F8BF52F869A8F1F7D33FF096B76053FDDCBFE46705B0CD391140FE3B1E65A36605C01ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0F3CA0464EB6D1140D2C505A33E8D0440EA890FA34362ED3F6FBB028D04EAF1BF8EF681BB669CEF3FF62AF69AFF6607C0DD910AE2A955024097BC2B1A388001C0B23BB7A1E53DE6BF2AB78BD776E80240B8624922DDA2F6BFDE75371DD008E03FFE3F3AF9E21EB8BF32FFA38F7E81EBBFEB55AC4E1340F1BF2167E5F2ED8F1BC03F2A13EC915D1140BA6594EB3D8F0040A69456061A5EE43FDD7E25A91E4111400559141ED6B4FD3F6ACCB408CD51F4BFA239D7AD2040FDBF40DA31BCA64312404D41520B147B184017EA7BF9D744DB3F34AEFEEA82C909C00E6EBD1CF14213C06D03E42DA343D73FD39E933FA0810AC0E8B465A81AD5D93FDFDD0077993414C07931C1E9287EF53F37DDB02C13BA07C027D66E783A96C13F08E907461B4D12C01ADB5B9401B71C4055C2476FFCF91540296DF0C26EAD1840B29A8BF09D1C0DC0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF005CB548E371E23FB22265C350E9FEBF76E36AFC3BA20C4079B8528E9DC609C03E972AFA4C08FA3F9E58C15ACDA6134078A58B6EC550BABF142FDB4C666CE8BF4CBA57CE7D37074064A5C2B11983F93F230A0B8D919901405E14DC380EBEF9BF193922C8566AEA3F7087E70C1CBF134006E1379E07ACF73F8463DD5D436906C0C2F3121BB3E80F40AE9B3E225ECC12C0389D367465341140483197561E8AF43F4D41520B147B184017EA7BF9D744DB3F78653142B594EC3FC5455B4822ADD33F32CCC2C60568F4BF6EF5CE20BACF15C00EFBFAA39A1CFBBF743FA2413CD6FB3FA51CE22C7DB6B43F6813083DDAB0E33F23C848D586DDF7BF54CFF7D2ADF310401ADB5B9401B71C4055C2476FFCF9154020DAC23096CCFF3F3E9B3C82CF6100C0351BB7D2E00F05404D7432C248A908C0DEB5817ACD32004026BE27053E00DABF005CB548E371E23FB22265C350E9FEBF0173FDD10AA901C004BF1A440E9A9FBFC668D4868A4902C050905F18DF18B23FD22AAEAD1E3D14400CF133E6D837F83F88C04EB4D338FB3FE42F27099574BB3F9CE8E9815806F9BF6084AEE39F72F8BFE60D048F2867DFBFB88FECC30145E1BF966FC42C23BAD9BF29F7594E01E9F13F3A8844BFB615FC3F5C1C00C0CACAFFBFD2186E95924F02C09561FDF4AC4705404D41520B147B184017EA7BF9D744DB3FB6475AC83699F4BFE59F26513EB3F4BF32CCC2C60568F4BF6EF5CE20BACF15C0E80C15C033250D401B69412605810140A51CE22C7DB6B43F6813083DDAB0E33F285C0127F024F33FA77AE0ECEA720F401ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF005CB548E371E23FB22265C350E9FEBF95E7FAD2E442FBBF7ADB73593BBCF93F2D233C63E84FDA3F802A83410D55FDBFF49DD4B3B35405407F7592714C6A0AC0FE3F3AF9E21EB8BF32FFA38F7E81EBBF7402625F4C3DECBF987A9B9D9C31F73F445958B50AF30D400A390656F74116C081B48780D87911C08433814BC98B1AC03A8844BFB615FC3F5C1C00C0CACAFFBF21CB490DFAE4E0BF7C7F54C898010AC04D41520B147B184017EA7BF9D744DB3F4495FBBD5DB6EDBFAE2535E689880C4008F8F257A521F43FE1FB460BA48710C0F8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33F63A5107D78D20EC02287A5385ACAF8BF1ADB5B9401B71C4055C2476FFCF9154057F97ECCDC141740599B88DFF6E7C63F351BB7D2E00F05404D7432C248A908C0BEF26F9AE87AF13F86CA44D4E65DE9BF1F3EB3D5C6B1F93F38CE80C4EED3DE3F3133D78767D40840604F2D1CAC79D23FB0EC21BEF3C80340190AC45BAA61C23FB8624922DDA2F6BFDE75371DD008E03F98DCFB5F251B0840ABB3850EDCBB02C0842EAB3BA9D7FFBF4404CE2519240040F3FF9AA7984A0540A9AB2AC5E79F0B40BA911545D67AEE3F51FF6F237D4605C03A8844BFB615FC3F5C1C00C0CACAFFBFCD5DB81AE458E5BFE4E45EF1ACC79F3F4D41520B147B184017EA7BF9D744DB3FA2018135C084FC3F1A19FDACDC210BC07AC96824DDCFEA3F9A0DB0BFB4E6B83FF8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33F239711FA112A0240551C7F64C9FDF13F1ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF005CB548E371E23FB22265C350E9FEBF8E66B7076E95D13F2E98D5BA310C0340204EA039978DDBBF3D666CFCF7F7F73FB00FB70845D81940DBC599EA202AC73FC6DB892D412A0040C36873B204FA0040787561279D8BE9BF23CFF54C4CA8DB3F45341CD35347E0BFB7B4D9D605051040C23847337FA8EBBF89C532A1C403E43F794F007D4A2E0240C4ADD293B55223C0E04C01C86234E7BF27517595F71600C04D41520B147B184017EA7BF9D744DB3FDD3320828972F4BF91F34D3A064FF33F74CD508FF6F9C33FF8CB3A8903A4F13FF8144AECB2F600C0726231657DD2EDBFA51CE22C7DB6B43F6813083DDAB0E33F63A5107D78D20EC02287A5385ACAF8BF1ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BFFE9C076B7EA9FC3F499895158421FFBFF4AF94993332F5BFCEB8FDFD22B108C02E58F0E4354700400FAFCD26D83600C0244C42FD7A2307400A4822A2B6D90DC0108B17C5A19BB43FDFD773CAAC130AC041A1963803FEE0BFA3613C500FFEF13F0E9428B90FE2E23FE796B323C404E5BF63A1CDEED0DEE9BF95A49635821D06403A8844BFB615FC3F5C1C00C0CACAFFBF11F9B0C99FB8D0BF4F197FA3A7EFD63F4D41520B147B184017EA7BF9D744DB3F262609FF270B09407C26363D7124014032CCC2C60568F4BF6EF5CE20BACF15C02A5D392EC65AF73F6079FEC57DCA0CC0A51CE22C7DB6B43F6813083DDAB0E33FBAB1D5957E5802C0FFF7E4FBF29014401ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C01EA0EEAFD83008403E407E4C3CB2F43FEA890FA34362ED3F6FBB028D04EAF1BF005CB548E371E23FB22265C350E9FEBF0173FDD10AA901C004BF1A440E9A9FBFA485EDBF99CB0CC04C1FAA634DA6F33FB8624922DDA2F6BFDE75371DD008E03FFE3F3AF9E21EB8BF32FFA38F7E81EBBFB6CE80C57862E9BFAF6CC36D8461F8BFCE7DADD1E114EFBF5AC611343E28FABF4864D8F8D9390940805F8A5A26B50F403A8844BFB615FC3F5C1C00C0CACAFFBF6486562207CECD3F8F658D4E3BBE05C04D41520B147B184017EA7BF9D744DB3F5A017AC3A258EFBFC0C575D3CBCDFFBF96933A0DD43FF93F0BEDAFFDD37FF73FE2D668C6C35E18405407714D9269D8BFC628DA345B221240CEE2870195F818C0DC84C612C41DE1BF20E2E4E92EED0A401ADB5B9401B71C4055C2476FFCF915407AB869A89241CD3F52C59C514349EF3F351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF68FCE2C68D15F63FD2DCC5107F7FF43F401889CD65AF1840787C428B064FD9BFD286378C12170F403659FC78C915FF3F2B9E25CB2D54D53F60C4075C10D31EC0FE3F3AF9E21EB8BF32FFA38F7E81EBBF842EAB3BA9D7FFBF4404CE25192400402A004A90A1F8F0BFC0094202439418C09CB46429ECBB0BC08E6791858F6509C03A8844BFB615FC3F5C1C00C0CACAFFBF5D5AD7A811F21240037CA40092A700C04D41520B147B184017EA7BF9D744DB3F7BB6F776168D11C0CD67C2738ACAFE3F32CCC2C60568F4BF6EF5CE20BACF15C04E3C998FB606ED3FA6DECFB486F1E13FA51CE22C7DB6B43F6813083DDAB0E33F40793EF29787D33F2C31E4DF15BD12C01ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0D60250A5FAAD1540CA895F9EBAC40540EA890FA34362ED3F6FBB028D04EAF1BF4A2626ADE65F07407837BB2EB8C7FBBF6E5C438C7E3CE7BF110402976D9F05C0925397C9558FEE3FFE4841F97CF112C09EBF0127D447E43FEA90E6D8A54A14404661CF3D1E491740D6699364A76B11406A20BED264A2E13F9090B398CCB2FFBF3A353745F29700407136FD169196F23F345BF08C688205C011423EA40BCC12C03A8844BFB615FC3F5C1C00C0CACAFFBF8BC66D661EE2EB3F9A9E2CD5210AE9BF4D41520B147B184017EA7BF9D744DB3F2DC090C83D8FF23F468FF7F1D7FF0940D0EC38F959B60540F8295C49874E11C030F1BAF1E5060340B14F0B71316400C0965E84E03590FC3FA7C31BF2125213C0ACC7754AA04100401616C4EAB4C8A73F1ADB5B9401B71C4055C2476FFCF915402AA7AC4DB4F20240BBC5F9AF98420340351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF16BDEA6698AFFF3F68C08EE9FA32DFBF9D13C01DC11EEF3F98B5D12C29050140C358EC71823ECFBF9A636E83F408EBBFE5E41AFA0B09F2BFAB454D45486012C0FE3F3AF9E21EB8BF32FFA38F7E81EBBFC4486810726AE7BFEBFACDE3110010C0882CD41F06A00240FBBAEDAA2C31D73F6CA00C994193FD3FA5A8005193FBD23F64B301B60C9C03405CDF2F2FAAD3F1BF0CE1A3DDBACE1640FED2CFAED967E8BF4D41520B147B184017EA7BF9D744DB3FA0D72DB5C394FE3FB3723027E4360C406824F0A8E22EFE3F0E39C9EE3051E1BF963DAA8B410E0440860DAF49028313C0A2C96007600610401110EFE74640F23F0263383CF93A08408618EF94DA56EC3F1ADB5B9401B71C4055C2476FFCF915407C30246E3542D4BFCD8CC99FF30A06C0351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BFC0A8FA06A3A71C40F96B0DDD568E19C00173FDD10AA901C004BF1A440E9A9FBFAEF775DEDC170AC0ACE60EB354B0EFBF89B797B7CD4F0440C2E34CC770C6F93FFE3F3AF9E21EB8BF32FFA38F7E81EBBF842EAB3BA9D7FFBF4404CE2519240040C1C00B54FC7B1B409E3C53AC88A01040C36E709FEA5305C0306E575EAE7709C03A8844BFB615FC3F5C1C00C0CACAFFBFD2186E95924F02C09561FDF4AC4705404D41520B147B184017EA7BF9D744DB3F9435ED0F4F2EE83F14AE176552A6CABF32CCC2C60568F4BF6EF5CE20BACF15C0442918DB37930F4032D591068FFFFFBFA51CE22C7DB6B43F6813083DDAB0E33FF7EF891E392EF5BFBCDF0EDF418EB8BF1ADB5B9401B71C4055C2476FFCF915406C9D74DA2DBBF93F5E258D9E751EF9BF351BB7D2E00F05404D7432C248A908C0EA890FA34362ED3F6FBB028D04EAF1BF9F264297E8E0F03F7CFFFE69049005C0EF98D8D183E10C4030CB560F8651F03FB1C3932ACBDFF23FC222C365543CE53FB8624922DDA2F6BFDE75371DD008E03FFE3F3AF9E21EB8BF32FFA38F7E81EBBF0CABF3C69B681040861633B9EF380BC0D6059CD0FB8C114036E260D6052A0B40680672694CD5014058023082850622C03B26CB2DE36E0D400C9ED9F27C8EF0BF19C54F175D9ADABFED6BAC2D4A9410C04D41520B147B184017EA7BF9D744DB3FAF83ACBEEDB60A402EBA5D2EE5C505C0F00CD3183576004055138EF8A715E53F7C825799A9220640A891C02AD447E73FA51CE22C7DB6B43F6813083DDAB0E33F5858670EB048E8BF70DCDE595997F5BF1ADB5B9401B71C4055C2476FFCF91540729DC527960BACBF5E719EDC853FDFBF351BB7D2E00F05404D7432C248A908C0874CE06DD189F23F226EB8FE063E0C402E6104B089F60B40DC0A0361264822401100E11E8C6BC8BFC0926DC1A0C10AC0ACB1CB442BBEF1BF22234FAD193E18408E97933D4592EABFE4298E7D766409C0FE3F3AF9E21EB8BF32FFA38F7E81EBBFA5973FF293CBF2BF4C825B721C1209401ED800583E10F1BF161460849C89FE3F2C3E14BDE371A63F92E4F5E169CB11403A8844BFB615FC3F5C1C00C0CACAFFBFC3CEF0F0B462F03FCC83B347EE10F1BF4D41520B147B184017EA7BF9D744DB3F709A52A9CC7610C0C5A62850F5761440"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%0) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

