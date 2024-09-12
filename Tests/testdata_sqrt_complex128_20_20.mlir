"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f64>>
    %4 = "stablehlo.sqrt"(%2) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x9179C538E92C14C0986ADF5A3FC6D93F5EA10C7952C5ABBFD1A3D1468587E7BF062E5820DC93E93F469C1B5DA4E3D0BF2535B6E2BEE510400BDF2C4624FB07C030F62B03AB2BF43F96AE38C7B6E50AC0AAE1FC5EEC7EC53F8C6D3B30E73AEBBFFE9864DC1404D53FE20FC620BF3D0540265F01012D96E6BFFC13A6475731C63FE708D54737881240D4167A0C822B0BC0C8FFBC5F59B00340A3052EE39142024010635AAC8AE5E53F7C8044C942CEF8BF8CBB663BA35605C09072903B57B809C01612CAEABB7DF93FD1B3B2967BB7D43FD8228828A5EF0E40E109688F2462E33F981F6D4810EA0A40A8B0A1093A21BABFD71191F8DA5C03C04879AD3063A31FC08A28CDF03DBE1C4016BDC0DC6E5CCB3FD0254B597B85FE3F4E41A3C100580940DE308D49BE290C40B679C4D6AA1002C0BFBDDC32921F06C03464E5E5D27307C05C07E474F217FF3FCB788E0378881A4037B42188B62607C0F53DD2E64CC4EABF943EEB073EDDFEBFF973A8CBB79807C00610D60F2F31D4BFEA19C26122540740DE86AEF792270BC0B89B01AA2B2BE1BF0CA16701A4DE16C0F27D82F9642E0940B63FA17EB9CF0640E5CB0C528B560540FCBE9E1DE61602402EB6D89830ACC7BFED4A9620804D16C08E3D04E9DB910EC096CA8C10FF530AC03CCE265BCABBF2BF61590C99F7F601403A5668B9D3E8E1BF91DCD4E1768314C0F2F640D04EBEF93FFDFC6DB3080204C070A6DE24D1E70A4048433100EAEAF1BFF0AB245CAE130440E6AEAE4BFB5E09C09A2B3324A16B1440387FCCD44DF908404EDA78B9DD3DE93FA22B0DA99D630DC0944AE7831C3A02C09F5E4B90185B0940B226AB8DEAFFDF3F0D1B0BD360380840634377EF535A0D4082719D72522005C01C73384F9C21D1BF175B92BEA70F0CC0C37DF7F0E91B1C40DA43B5641931883F0228907C09230640DB87F3999B6AC4BF2662E8AC47F908C0B8EDF78EF54AA83FC340E028BFEAB7BFD02001CF469E0F40EE628EB650E9CBBFA72C2564CB5A02C05762C91112121140B24B8E91A4E9F6BF03E8951F387B18C094347BAB6E9BCA3F6694C7B3D60406C0EE5F0C343C0C20C0E2D70A95E36FF93F0B0AA95052E9E4BFA5A893E2C637B43F4B16BF57D5DD114062F3733B09B4DC3F61E7389BAC1EF3BF278DD861D0DCF6BF08A605BBD0C0E9BF86F7CD3D2603F43FC808FE5A07E1F5BF2EDA09F07CE50D40DE5559F473780CC023026AB7423D05409A103CB840B411C0D16F84FE8A98014093912887AF220A401688755576091B40379D45908AE2DBBFE01444858FC2E3BFBAE81139D7F502407DBB6AFCE8ECDABFF06488C773450840A0157B7531AEFBBF7ECD15DB79920E40BEB32622C1FC0C40EEDFEA0F264CCEBF245F0CD5B58DF5BF62BEDD24917DE43FC72C8426F0990CC0414881F698CDE73FE476DE6BA675C0BF72026121F2B40E40AB839113095010401A4D72423313084078CFDEE7899A0EC0DC5F24561635FBBFF55D32D9FAEF0140E3D6C65C823601401618E7789A0406C0FC7A5C35718709C0D1892D43D4510D40FEACF20EC27A0CC07504813C015AF43F305958509D49F83F0280DF9DFBFD034066815E56C0FFF93F38F26666562C00C0042B5B6D80A605400C2D7161A705D4BF2D4429E462CEE0BF270A86E7B63F16405CA16D1803AD11C0B5A89C58B3B9124031D64FD205A8E3BF4689AF367E2D02C03F2AB6EF2D91194058B2976CD421B43FA11BC3936600B6BF8E7E699180FBE43F4BFEF004B44E11C02885222DDB2513400B6ADEF6CA240A4020FDDE744242DEBFFDC0C792BB9901C0DAC43F32E106F33F3D4CD90CD9D909C0B1A84EDC50D9C33FF1921BA50CB205C0FF8D1DE60E3BF3BFE0237122CE7CFEBFD3E8E6F376CBEA3F96B8B6BCA0920A40AC6387B3E540F33F98205FBDAEE50840321C3BBAB5E5E3BF660E87F3500D124037F1B70E4A490040D36866D64EC6FABFAD8856608902D73FBEC3684F78C210406381621CA3860840636326D6E6310340EFF4F9DF8350D93FA251EE8CADD6FDBF9BB196D589DBE23F6C122224976F08408A5D162FCD45164062B5ABCD81B210C083BFA8B6017408C002667D7E6FAF004036F7AAFC6753F1BF746BF7D8FEA4F2BFE0F38F0DB1ADDF3FB973ADC64150F83FC80A0D76358D00C05070ED349F3313403DBA17C50A6802C00C586A02194801409E815C6F9CE5DD3FB2890695DE43E13F7A53BC8072C7FEBFE07EF33D8DF20140C28977A7C6C506C0AEFDFB9CB14206405AE1A6C7B5DF1040A7B98493689DE6BFA0DAD963E77605C05D86664AFC3F0AC0DA268D483468E0BFD2C2B14B61C9F33FB22515800DCD09C04B5BA66DA340D13F04D1D53A741708409084A0746F5312405AFE6EBEE16C11C09A1288AF98F5EC3F8EB9D147E0C703402CF109CCDB60E6BFBAC2F36BA7A91E40AB12DBFD08FBE33FDF52B92DCA6A104089D5F55BACB8D3BFB1061A5CFC6A09C0D8AFFB1D0584DDBF60196904542718C00FAD573FEE23F23F0683C58A0E7A1AC03426F3E0F610F6BF94688990B3BA1240846232896763E5BF0563D1AFAC5013C0037A913DB795134003FC16A10269E1BF9E1D2367BBACBE3F6E8B65273F958F3F3BC88130110D164064BE342C1CE00C4037F726CB7ADD00C072A4DE6D39BDA93FDEA8D50274BD14C0E9F6B086C601EA3F925F94CBDEA604C05C73BFD1126F0BC0589D5CE9BF580BC04A19A0D9BCCB01C01F1027291D67F23FBEEDC630C76598BFBFA297C9496BF8BFD8D07B76246915C066A7882AEA060CC088FD0D05B997CBBFC85EFF9F074504C004628ED3D9EFE33F5482D3E7AA4306400E175EBE8BE0F53F447497868D83C7BF31BB33A4C1551240A2510C956B0EA83FEC85827495D2FC3F0D8C6FA0AFADD6BF641501FB68A206C0CB614E7C3BCD00C0603FE32AD75309C01EDD3E4258091BC09B6761B04D680440E8906185D1DDDFBF16DBCD0FE82FFFBF15AE1AB322D1FA3F2630382A85A3FCBFF336CA0400D0EBBFCA237AC7AAB40C40DA8BC77DC2B20B406D781EEBBA85F1BF1EBF866078790A4062D9408CE40E07C03C0C182CD0BEE03FD9EAE1CDD3DD07C0B6FBE3C7C9031240629CF0BBE4D588BFA9C665A3A63708401E0684724FD0F83F6F4C267F99C1F0BF395D3B4AFDACFBBF2FD9C649B00810C084F10C9D7212F93F5D052BA973921440ACEECAE7E1A2EF3FAFB83500F4A80540F54C4EE8C0FED2BF9A4B6815163D0640C4106B682B320640472AA4C7ADA4F33F4008209BA6F404C096BA5F0BCFD203C0D022AD499B901240A284D27B6C6116C050BAA7044681F43FBAC29D58548E0AC0D99A98077BAC0340904469C8D93FF1BF92363F8A5378F23F6A25AADFB60F01C07C50CA169B6D08409845D006F6D90F406B40CADCEA8CB43F6020874325E2E1BF123668C6CFBFE9BF21E1CA250B07EFBF4534708B22F2C1BFAEAAA1DDD319F83F7ED1EAAEA9B305C09C8A92CBA5210DC01639D72E6C3B1740C081037F30BA0AC0D330A18D643E044044FD8A85B6A40C40AEB30DB6176114C040ADACB86ED9FB3F136F20803B101CC0047122F5FA4F03C017800A031238FA3F8E576E4CAA58F0BF9C9684A7E7510C4069051152305711C036A405C0BE68F7BF884668404DFCFE3FB4A0980DA6F7E33F5038C38585C5E7BF28E00521D38DD23F222E27BE9199F8BFCE6DC186723009C036D00130010616C017248F12CF3A114074145BE525C00A404BAC73D7AA6C0BC02A33C7DD7F4B0AC018B6F0FBC40BE33F077C79F2254BF7BF0675F68C77BE01C07CBCD9F84D3AF1BFBA22214D08420540FFFFAF70B675E6BF9C5676E0987B04405601312D53881040F05B112055FFEB3F6897FF30FAF07FBFC74359B32671C33F10EF1DA6B28DD63F393962B4B3B20940E1A7DF1CAAE4FBBFD18D594F0FF9E53F0B923412A94610C0DE62199559E416C0AD67D16403EA0640F88A9C9BC3C2E83F7475F30FEAF50EC0B033ABB9EAB51EC0C4DF7F845CDC09C0117290092B4AF1BF80693CC54AFF18403E53A05452210540C081BAE35CEFF83F12CFEE04C0FB164008F791A2177CEE3F94AD74F6C10AF0BF2BDC0D7B1969E33F1AE69CBF0C4FF43F70F9476DD3C2F53F258AB7C9ADFEF6BF45E00DE0BB81F63F23F8F82415BDF73F9A8264072D07D23F8B78D9C8C3FCDABF76A782FC72260140A8474DFD19B608C06719E99FD80DF1BFA87E78A03E5AE9BF66546A93CA36F73FAA962DA0980DF8BFD87042F7E2E3F83F4E9B962F3E89EF3F5D9C9B4D34CCE7BF38082C97E366FE3FB8F19970EB7AF7BFA0A1FE964B081040DEA51B9EAC9BF8BFB599FCE22A341B408C077C5C4B181BC0099A241C17A51440742CB2FAF67871BF40ACCF0F89D5D73F02CA3CDACAF90C40020B05C35C80E4BFFDC1DD7DF0E504405A972BBD9FCB0C4006719F3559D503C07C8FB225B64411409B0538360BC1F13FDF6C53CC711C14C0680751B0734913C0AAD76DDB5A7A17C0BFF082DD05860EC06C2EB1A924C10AC0F654E2E09DF8D43F876004BC01981240A41237D96E2C06408ADDEE1354370540DC0E02672F2011401B5F8A3B93E7D33F80E5827632E4F6BF98FD6550EFD303C0FD87D8434C240840482AB9A04E4F0CC05A99DF6B2BB5F43F8A2150F803CF10C0E7BF4F428347F1BF145B37E3E5800CC0AE5DF6382024E43FC24DF76A34E91340A43834B943FAB6BFC00A2C146175F1BFE784B1BD8FB10FC09A95CBC4D2C7F83F788BC180CB980A40D70FCA3C7C48F23F35B1E7737941F6BFA236634D2D3CFBBF4CC8625785F816404FE868FE68AF00407BFCA4209E37FA3FA60C2461B928D5BFD687ADAA712FDFBF9AC4B4C1301008C02F8A87B8E55B22407026652F32AAFBBFD706CBA42BDE06C018A15780D0181F40D626E76F2068FBBFF02453821DABEB3F323CC7C43A96EABF73C87216718102404A3B6B4DF694E3BF2129591074E5F9BF5F44DF0D4C2D17C0B1217F0CF2A303401DC77F8BEF05C73F74DC399C3C6218C0B0F848B68C64F03F24080FF825D5FEBF703D86E96EF9FF3F2595786719140440662586FF55BB00408E0148E396DA0940FEDFAA2FB86E0140D3F6CB7F417122401677AEF49382DC3F3C9587C03EE00240C3EDA8726A5BFC3FAC5A5B3780FC0BC0EEBE03CC34C80F40BB5FAD916F1C0AC01917ADF5DA82F3BFC1E60D9363CA04C00AC434C43875FA3FE67D5D024B0106C00CD2545B1335EE3FA83EE6B2111D903FC0C28FEE5E371A40D699F4581C22EEBF01574E92BD51E5BF14DC100F3704EE3FCC48B4F6C890F1BF6676E475F27FF1BFBAEC44FC2EC3E6BF5E813F799E50C0BF83CBF8C8CA75E23F7B2AABA2D8820AC017F62B89F91D14C0D72FA02DB49C10401F0D8B8105B0F4BF3E1B9DEF242FCEBFCC020BE82D9113405FCD5326C57D0340F2E34CDF6EE2044099940BC672FA12C08A9C4CE226B21340265EE29AD8E81040380613B3F306CBBFF68B96A7604EDB3F625C6D4BB427B23F8AE8C823940F054062A34DEA44A518C006210CCA7E8CF7BFD108D3AA410B16C09A2F46FC549018C0FF947B011A9316C08AC0A604182C00402F2FC9569912EBBF15228DCCF23617C0DFB2F3318081F23F8DEE2A931F4EA93F6C9D1F6A55A6D23F75DECD4B07CDF43F566CAB69042F16C0D3AEC0E9D82507400E410BC739620740614486C0789603C0D55EA009558100C0226366939039E1BF7CFF2D73A35708C0AB7E720D3C381440634400C3AFAAE53F24F48406C265E4BF3687B2AD8FD11540C853F5069283F83F50AEDCD3FE4A0740BD6F4F8E70E801404642738E80B710409CF3EE9F42CA0140C3E8E92E9EE2E0BF07638B2E8211F7BF5227E1E06FF118C0E1E245F87DE1F23F2D952CFEC1F614C0F8CAEE0B85732240FD72289DEA6DC63F08869711539100C0BD601C804E4504C0F06849AD640613C0A0E2B30D932CF33FABC7D81939520B405813D5F16667DD3F7A2AAF0590ECC0BF5ED97FB5FA73F9BF0CFF358260A809C0400151F56513034070961E2EC0C8EEBF9370C552AE721540E28344DB051719C0843B81BA40AB07406BF0A4324C47E33F8EF4B10D186811C0F6C1126F1355E4BF7260E5099AC81040D1122FCDBFC7ECBF04C7D514D11310C098F3B22291A8D73F3A0CA8578E93FBBF06B838BD48C7EABF00291A8DB8450BC09AA6C5FC26781AC0140CA96D7515F5BF9496654C82C8E03F53406BCE8D9102C096368DBEC396F7BFDF45D597F29BF4BF541FC0A40C4FE83F9C4EEEE7C9FA0DC07C788FB47D811140E731A36F69E006403B444286973C02402DA83C24F0B510C0D6603495E0801440068CA9A961520040DC45555938C7E93F4E387484C967DC3F56C9FEC1392DD8BFD4BFF1ED87AFF03F928BCC8428F7154070A315D93D82F7BFEA9D13B7DF46F53F56CC599CD16DD9BFE48B260BEDE91040AE81CA89F0ECB3BFEB47AE1D7D11D0BF0DE29B92AE3610C08259F35EB8CDD53FA591F0CC5D86F7BF06C87B7C089A05C09EF63E617AA901C0D675467D20A414C00C7BD93863571B403F3F7707A157EC3F5E5B27FA8F370BC0B63A164114B7D9BF76CA770BEF8AFC3F82D144307A08E53F25B72CD78C8C024066019815FCEA0440488B509691060B40A68832816080F33F8865E880179E16C015D450359138F13FC74378C09C3604C066F6D429F4101340DA7F67D8D4F5B43F2E2DBDC2549003C0D8F9DA2F8D2303C0B4A954186A070B40EF9C9A59DA1921C0EF7BE5996B9D02409E4B8E562B110040FAD7A695EFFBB33F1C2B26E2C11ADEBF6E0EB85A2AAA12C0A705D93E19E2FBBF1348F7D0B6C01240F078076C13E90CC0F6AD702C7455AABF8CBC3BA27A63F8BFE4FAAC232F830140318684ED82220C40C8843AD5673D0140D0EAA4DD87DCE93F563394B0BCD0ECBF095F3E0AFC2A13C0CD6B45EECDA30B4084BE3B509FDC06C0E3671572734180BFBDBC09A6EEA40E404E66FE56E8D9F33F8683E80A98DDE8BF92C19B79EFA914C0246E97F912F8B2BF2A93825E3A3D0FC04B426215A6EF074097B1E054C58911C0AF37A986C291FF3F011EF18DC2880C4052B8837451750DC0411663C89A48064084063EA1170F1A40D6A26C7C464BF9BFA6D59E04B783C33FC4252E11E532E53FB2A2415F4CD20440FE46033C6F2001404E6B05879DB8094044E70F14AB2E0E403AF34041876D12C022313570647106C0545104D444D30C40E4BFE2930EBA00C0B80730DCA953F7BF96C2752500251B40ADF70D51970B1AC078CE21914826B83F457EF559D2F9D0BF28D56A1AE78511C037DD7C6D5F1EFA3F00DCBF73AF1C1240BE365A4CA31F0FC02E19D7E5FA2A17404B71CD7B9656E5BFF4ECAE9C9904FC3FDC4A6B3A4A11044075EB7F325E3EE43F90E47C8A86AA0D40C1F8B5D4559907C03A7D5CC7572102C003DB293EFA6A10C0EC99D74493C0D63FD855A536C4491AC00C72980FE0841340FC44450C9A8000C09C0347C9D506FD3F65C9F4295D2AD73F26AED7554B1E04C04D1B2FF01D40F4BFB566C2938F6B04C09275AD08A5C60440B520502A5E81E63F0621E1C5751BEE3F7F418CB146310340F8BC0AA50C251740F842BC85E5F9FBBF612D9EA44705F73F41DC8F42427BF03F176E6064022EFC3FE8C90CDEE4080A408DC4CE4CA6E610400EB528FF895D1340696BB5B96E2DFD3F201CBB85D5D3D7BF504F1BAE7440FE3FD135953860DB19C09792313CF509E1BF84BAEE79EF3C02C08EBADD770945FE3FC3770210DCB9F03FD0B03541D4D710C018F126FE16CCF2BF14C947806F68D9BF9077165D5BFB00403ECBD7C1135600C095A282B67C82F23F50D2173A1527E83F7CB44016D9A304403CE3DDBFC3C7134051C3F1C5B1A010404F0C073FC11110401F2A9888C3E70DC0C6AEC4D45C9CFDBFE5F1269D44C7D03F1801CD5638070240C22A125667A6FCBF73392409DD9E01C0FADD95A3E0EAE1BFD2CC463F94CC08C098B0DF0A54B503C02DB1B0F43131F63F430A979ABE7807401A1F08CCED0AE1BF02FE73DBAB03B4BF2CCF020967550740B6EC35BFD894E33F5E91C9E9EEAAF03F373E468C6570F83F76CC12ED04D3F33F04AFC72F594CC03F94378869BDCD08C02A7E81A16A6A1040EE920A9EEA27FF3F2D2F1B730FB411C0188B0F3A3383F7BF2EEB7A32855FFFBFFE00BADF2776F93F2ADBE373434AFF3F12D741A4877DEFBFFDDEDFFA401105C017FBDC5C478806C061CD0BFE3D1511C07250E1CC427600C0DC3B264AC615C7BFBB9B6E37968202C0F0C3677E6D990DC0F759ECF555CFE53FE664F67D8EA9F8BF8227A91AD442F23F36CC70C0F4F2EDBFBEAC10A6967202C0A21333ED26CC0D408CF50A6E9C9C0D40689D92A4A30E0DC03EED1A15CEF907C0281D71C712DA05407C2E221C1900FE3FB324EB94E243FB3F3FD0E4BB57BC06C0D25B796BCAEF0240AA8FFD9A0BAC0D4010045735D2E302401C7B938324F01340B634DA7CA74EDA3F18D61F85B2B002C0C7F9A51129E208C0F508765C841FF9BF2B8919859C7CC03FB5FAEC371E13F53F68539F0CCE810BC0E16A77C780250A407756CC56446D00C0BE8A0C58694AFB3F180739B8A315D73FFE69412662A500C0C179583D65B605C016E1EBF7C096EB3F8A9F3E1D5243F13F2452278D71F802408C51E7CAA14B0740C2912763304BE4BF2CF0BB6C9857104020307B8A85595ABF0E4F37BF75CD05C0FC12AED1D9B8F9BF0EE7F32AF62B07C0306AB7329717FB3F12EA307BA51CE23FA4339339708308402AA368E849D3164059599831C87CF63FF9ABE4B5B26E1AC06F1066927B17F0BF5C976A793746E23F6948F998C7630640C73C6DFB5A7671BF1F2276AC23FB09402A695E4C85FF11C0B0C830E4306D12C071C759B1737505C059A3346604AF07C0E0A573E8F34B0940010EC21A68281340AC180FA22696F93F6E95E283F862EFBF6643DFFF2C5CF8BF885563160215F0BFE48EAC6E3A5DFEBF8926402E43610740A6690AC4C566E63F509D549B453FEF3FE0D89728D64D05C0C28D3E630E6BEFBF"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%1) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xE82A368E44EFB63FF8677A8E2AFB0140B1213C816BB3E23F92C51FE19321E4BF1BF9423B83FCEC3F5001515328A5C2BFD3CABBBFF4580140895E72F9461EE6BFA28916B261EBF83FDF2D3F9D1945F1BF35BA27F6F005E73F9FBD5D856FECE2BF886BE202B19BF33FC155BC972055F13FCDA4C06EAF38BA3F3E0ADD425F15EB3F4353B9741C3902404A0DC8FDFBDAE7BF40B70E16E349FB3F62408751A269E53F3743695C3173F13F364BA41F99BEE6BFAEC491395ACEEB3FE6208FC07199FDBF281001CB634CF43F485AF0366A54C03F0C532438308FFF3F94EFDC2E6EA7C33F8166011BBF59FD3FEA33885A0D7D9CBFE612D915FC5DFB3F4F844212407F02C0815FE6238A720540A222B25E6169A43FE1D069AB50C9FA3F7F312D60CE46EE3FE9C165625F66FF3F8F9A63870D69E2BFCDD22A3F3872E93FDE795DF5257EFDBF9EA7F3D36DD5004084FA4B8B1038F93FC7A07862E426CF3FB59672CFD87EFBBF383B58BE7694EC3FC1624608976BFABF4B3BC6A6EF4DF23FEAFB44AE4564F43F274392F77994C23F02BBF09ABA91FDBF6E0F9C1EBE5AE43F831949894ECB0340A4B9BBEA1E68FD3F3988F76E4438E73F9EC23FFC6214F83F91626CC58775AFBFB99BCF02649EE83F357C0CDB1EDE03C0FE13B064DB57D43F224152EAD077FDBFB1A230D2BC28F83FE1691D45D6B8C7BF2526F0F69A77D63F0645FFFA47550240BE04B992D46AED3FD85B87808844FD3F8019820F9CDFEC3F03E0A5763940F63FFCC3EC0ABD0FF33F0AD0EC160424014077F3F99FA07DFC3F09397BEBC959CC3F4217BA1D263BE23FE53908CE2DFEFFBFB54E9601A292FC3F7156C66041EBC13FB332EBF4E190FF3FEAF4B99CA7C1ED3F382C86E19C0EB53F2142AC2BB808FABF385AB2A93A96F73F430FD8224F1103408AC4176532DCF23FB4E62244A1C7F23FAFBE9EE4497CF33FEB7621CEAB81F4BFD8B183932DA8D13F9D44DF992EACC5BF50EBAA5116D2FF3F7E6D0E6B9611ACBF498C0FB2E511F23F560455E6F43AFE3F9D3EE25AD2ECF83FB142D2DB266EFFBFDF844160C97DF33F1A2E1D672613F2BFCB158C488EDFD13F117225BC63C50640FF55CC7005F7A83F8AB421484DEAE93F90901AF2C3ED0040DBE075B5E520BB3FEA578F087B7DE23F64BEA65E96C8F3BF0606CBDE95B1E23FC9673816EC20F13F5A41A6F9DE48F23F0B8BD6E33029FA3F6404969DA83DE53FFD35617C66FFFF3FB95B881F9342E03FC46CF0B97F50014031A8909D73910240F09DE7FB264CF73F11A91C97FC99D93FB02B6CE5D3B2E8BF619FA35F5CBAF83F4FBE13FFFD6BC1BFAA5A505129E7FC3F137739197FA5DEBFC8FDB88C970D0140E80E2B967E32EB3FC6946D7A7510E83F9ABFD09E4AA9ECBFCD8852EE9862F73F8EEF8D28A291F3BF515B5F8378B3EB3F74F4089A9103B3BFF27D4F0B19610140FE4314C53709EE3F07B327E49AC0FF3F2758FFE5A3D7EEBFF83C8B3AC9E0E73FD2104773E709F83F85A577F673E1FA3F42F94ECF1236EABFE8C822202A3AED3F62B2EA6FF40C0040E583F240173FD53F3FB4403CF7A6FE3FA0B9493034D8F73F4598994177D4EA3FC318C875DA3CF73FA4DA55138B45E6BFBC189182785DFA3F73671F16224DB8BFC714CB783974F93F69F684CA7AF8FB3FBA3BFCD60113F03F2C3A29FA8EA30240E48031C0EDD7ED3F8470D7B2C27DF3BFFF57D216D839044014EF1C9F01DA8F3FD8E80C0ADE29E13F559CF9035C8FE33F8800FE50DA7EF03F6FB6293B9B92024092356C2ECCFFFC3F3332A338EAB1C0BF57D4E3192ED2D83F69A97D3CAF87F83FE8455265D013A63FEF7517F717C5FC3FF2C927257AD3D63FE3BD3CFAA8F5FABF2DDF1D1165FAD23F2F723D3C0E97F63FC398A8B0D39EFD3F43440455DDCCD43F02CAFC3E7C5DFC3F34FDBAFC6F72C6BFE3B5CAAA07670140C67F370193F2DD3F9FF5E9BFD2AFC13F77FDEEFEBFD0F43FA1B83C429F530140F80A09A7E6A5E63FC7169FB907DEF83FDD39714EA949C03F1158A4A68349CB3F44F65491501DF63FF77AD8053D590140FFD7E0D8928AF43FCA3DDCE90E9FE63F863E039CAF4B01C05C237084EED3F73F97A91B5D9944D7BFA9754DBE3EBBCC3FB1A5572723A4F13FE5DC2B0A04DFF63F64282B3B7528E7BF4501B9A1B5FF014078A9F5F3BE5CE0BF6735049CD6A6F73F7173703A8B39C43FA3FFEE78A505F23F6B4C64A36153EBBFD1A05785E167FB3FDE0DBA720F97EABFCBA8AD13A5ABFF3FDECAAE70A70CF13F94B632EBE744F03F563D0670FE1BF5BF5139EB5A770FC23FA07FBD240D12FDBF82810C75DF80F83F6DEF1D52E7D8F0BF3570EFF0DF87F43F185A781669C6F23F00A65F81DDAD0240FECF2B591CDAEDBF93680EACD247F53F1F7629C7A6BEED3FF5DCE501ECEDFD3FDDB2352A5C64004081A29D848CB9F83F0C5DE2C0723FF53FED073AD29C36F33F935567D9B92AF5BF3A2D03E4BDC2FA3FB247A612EBE1FCBF5E33343A82B2FF3F6B41670DC9BAFABF0556B756192CF53FB688D555C44EFC3F05B25E1A3734F73F1DFCE53A13A3FABF05F04766A7BA01405EE6B85AA36CBFBFA23A655E1533D63FF18AE40553C3963F6540044ED6AD034020A514A3227AE73F96BBCAF1E1B9913FC20A4EE9963BF73FBE116B89EBC5C63F3452D502A345024022FE793F9C98ED3FC68183EE80A9FDBF700660BD2B62E23F47B0635A24FAFEBF6EEFC3010A29F13FD8E171E776BF86BFAEDB570E60BDF63FF8B1F20B3721FEBF2305543B3B78AD3FA7E142CA3BF6FDBFA68FB3F599DDC83FC0AF2D603BA8F93F40DE147D7671FB3F191BE8DE7B82D93FECC72592EBBDF73F4227821C62B6F83F91E2954DB4C4EE3FC88DF9B6F0F9ED3FDBB1D0C8C4E0F13F9CFD9BD4BB41F4BF914D4B1B717EED3F5CF0751CC57AFBBFAE7042EB08DFDE3FB10710F05227054055D2F7C735D7EB3FBA03A1AE57ECF1BF4D047A6BB6FCF63FE70AC17F09EFE3BF7CFB44432D02F33F16B38D3E9529F83F938B0F100022FE3FCCCC237EB09BD2BFA56DA1D9CB63FF3FF8C1E6D4A281E7BF1E4A274FB852F53FDA07AC7890E8F1BF4011B0DF40FA0040BF66F4C6CF6767BF9ECA9FB96FAFFC3F3460906658AEDB3F0DFD7B902957E63FCA55E5A530D2F3BFF3445349B198D83F45A49CA7334F0040DD9023DDB5390240D788C834F6C5CB3F90476482D65DFA3F7A4A8B36B90DB7BFE52FC8A7024DFD3F3C94C2C09B3DE83FD7935E984AF7F63FFF3EE4EFEA32EDBFF8A5C16104E0F23FEEE92BE75E79FF3FB345C7E4E939D13F978771769C0B0340E50F512AD662E43FF454388DC1E1FE3F785CDB2FD405E03F94A335719B71F23F119E1BF4858CEC3F486D1C978F61FB3FCF908AA75FEDFF3F6E71CE14E898943F5A800EDE995BDD3FE0F403F61111ECBF00D2C6107E2DB23F564BDFD48197EFBFCCF4B17C164AF83FCABE884F4A97ECBF9BC9BDE72D48F43F81A9F39DCB5302404565BC34FBDCE43F4061A1C3C10CFF3F007AD6C02DB701401DB714E9E067F2BF12D0DC7E01F10040F89806DA0181FABF953FD5B8E50EE03F7AA99E27BF1FFA3FF5618D1E3E76F23F47CC6402228BF83F3BDAF5D84F2ED63F6FB8F848D0E200C09A45AC30C18BF63FF1C473341857CC3F447C3BF26B24C53F7E826F711F15EC3FAFBC49B516BBEF3FCF3AD9EB2667F9BF2E7267E04A92EB3F6FD9939E46FF03400B5A2BC0D021004042A8B2672E33EBBF1554F95706EDC43F7F16269D0F20FD3FA88E8C0B45C2E83FE8B77A4AFCEEF6BF8F2D0186CC46EE3F6ECE8E96CF77F63FFA5AF4470C9FEF3F95ADF6F065BAF43F0DD2FD609B5A0040F57786673764CB3FD491523AC030D13F9231F1157C18D23F551A0161146BF53FD2EFAA317532F33F7C6587BCCD56D03FF57034625384F53F050FB5342971F33F101B2B13D4D602C08F38A7BAF951FB3F320835118E00CD3F362ACC409B99F83F826BA4C76FF903C080FBA258F9F9D23FD64B508AD827FDBF0BCC061F316C04405AA7DFB7E78DE03F34CA5ED02602FF3F3C0877F9E6B7F73F66FE40B26A4AF13F48C0C89867B0DDBF295A9705AB0DF03F4C07B236C23DE43F36C3D95B38ACF43FA436D64B17CCE1BFAAA4EF4D9204F53FDAFFD8673912E23F49C3DC15AC18E43FA609C98B817CD5BFD4BED7B1537DFB3F61848DF013C4ECBF8765FFA43A2CD73F245A83604A81F1BFAD8E9BD57A49F53F23623C133E14E2BF04DEDD5F4DDAF43F87D82EB67A32D83F181AE3B371C4E93F98842338B6E0F23F4059ECCF24EFF23F2D8B842F8918FB3F89EC578FFD5FFA3FA15885D7B68000401AC66D4EE4DDED3F20F4C9FD931E06405E9EF2C88B75DB3F6C0AB3F386C6DB3F2C00E3157C91FE3FDE3EEFDC2C76C5BFD15214A5890FFE3F6844558D30A7EE3FF921BE3DAFE2F13F7F57D1675CE5FE3FC10D1825A34DFC3FBC7B5FD6E3BCF6BF21289A092DD8F23F61F7112C22EF03C09766697A9761E93F49A88E179DDD00C049C6EEF05744F93F5572C9D86E8CF73F60E88A342215FD3F9A60A9862D58E73F96F0521D6F900040AF899295033AB33F14410A9F9011EB3F858CFE77AC70F7BF89EC42CA7254FF3F80C5FE6152EAECBF353BD9D07EFDFA3FB9A8DE08B8EDF3BF1D3D1932AB64F23F2A379E2E76CBF8BF3B9D343A7DE2FA3F624E50E908B3F73F1C20F1C554AFE63F884A93D2A4A0E8BF4D54521B1D74D83F6C473274C5360040511F4FA8E496FD3F1FCA5DD0CAC5D33F111ADBDF8A54E43F6A3AD008236FF5BF755014A19B7903404F6241FF7A6ADB3FEBCEABAA8395F43F1754B1266D72C0BFDC7C9DEEF419F23FD3F038900D45F5BFE725397F2E5808402947027CA82ED2BF373683F5BB59FA3F20E13882CBE10240332D3B4D5A88D43F935BDB0E808FF53F4C5456FBD1DCEC3F58FADC496984F43FD33F50BFF5EDE73F64241E98A650F1BF249D7ECAF4F4DF3F8D1E91A0BBAA0340FAFDA4FB3C59FC3F494584214386FBBF7C065211C142F43F9DF1E2004059E8BFED5AE16CCFD0F93FE69578C472E3E83F9E32D94B7993FB3FBC9A23D66200EE3F0C00AEF9C04F0340FB4432D84F8FFE3F54D1B438A816F33F67EA559DC8A4EF3FE8D3E7892BFFFA3FDF0CD2D42396F0BF58391787F81301405411AAF07E76E8BFEAD47D349513ED3F31EDAAB387E1F6BF3FCE387208F3F83F670B760C4139ECBF34A383DC8117EF3FC62AB3618F95803F92C05009838804404E0F271A057BC7BFBFF6CA286B7EDF3FA1AFBCC7B77FEE3FF5ED74297B6BDE3FB124342CA868F2BFCEEA8CD68244B33FFF7659F39918EBBF049251957375F63F8F25316F0EE3F2BF23052C8720A6EB3F585A48D5F1390340632F34DBB06FBA3F79D3834BB044F2BF68A5662A6E3402403A7BACFA7321E13FDAE6F927A106004080A135F498F2F2BF237213B5771C03408DCEA0224450EC3F38902170B94AD73F33907424EEC1E23F01722B99119BF23FC9DB08AB7C1CF23F77C224788AD7D23F272833A535FF03C00350C61D6DBAF23F5645D51548FC04C043FC4025F5D0DA3FEDA57DD9664C03406B209BFC9F58F93FF8E8F2F5124FFDBF096678DC1736F13F8DBA11493486973F2ED2415640D5EC3FDD96E983EC15E73FA4B040FF000FE33F795B9066D76E0340A8EC0C4FFC5CFD3FF02A6594C358E5BF9F4144B723C9C73F1123B302642CF7BF5BC64C83C41FF33F4E789370A6EA0040B6DAE4F12BAFEC3F1D1D79F05CC1D6BF115C4E7B32DD0240A66272BCC6CAD43F7282270DE207FD3F4CD9828753BDE33F83F7D53177E3004004C0BEF0A6DAE03FB68BA600B1B6E63F64DB433FF93FF0BF52F31A52461CCE3F8AD1A52CD0100440E1EBCCCE5037FA3FE3FAC57FA5850640A30E5B66EDFBF03FBCD1E2D82537EFBF501DF2C2061EF33FA05F7EE670D8FFBFBE061DDB2CD5F83F67CCBC06759AF13F1AA5B5A4BCE8E53FA6DD63200CB8B8BFE1DAC46CF8E9EF3F7B5AC78816BAF9BF5B12EC75952FF93F882D5D23758ED3BF1BBCF38445DF0440C3F4E9E6B43BF3BF5A109E2D6FA9FB3F22AAB21B4A4DC63FF2ECDB2F2971C33F053BA29390BB00C0B24C2B73DB7A0040BD3E987531F1CBBF35362FA09A93B73FE2041AED390E004022DEDFBBD9D9D33F4F0810BE6595F5BFD4F8CDFCC1B9F63F041F2C49C2A202C085C4496BDAB2CC3F97C81A40C2B6F23FCC68B594FEA0DD3FA88174A0077AF9BFBBCB57C2089BD43F42B9F44C1FE0F23F45D08E82860EF03F135DB49BA7710140B42994D894E1FC3F9251E77DBE34E43FB44B292889A7F13FE33B3948FE940240EB8419577447F73FAC280DD7B7B7D13F895B28CC57EDE63F5EC3E0023CDFD0BF07BD17C2F922FD3FCDDB93F0C41FF83F6418FF6E9132E03F31254AC67204F53FE3498D6AC732F63F6145EA0FCC61F83F6D83224D4977D33F4E9D48DC246ADABFB2C68729F3A3B53FA975EA18E21E0040B6C3DF5EE4A8EC3F1D90A7A88D1EF8BFB0DFBBD16EE0F43F8CB859F98FA3FFBFB632ACA991F50440DAB4D063D8A2C53F5BC81FFFBED5BB3FB86F02B12190FDBFD113B411EAB7F53F2B1DD6A782FDCE3FE73173513E47FB3F1114B351E689E83F84AB17093DDEFD3FF4B38B1BB7E4D43FA901FE23D9D6CC3FF0009DBEBD1B03404B8D02161529F33F7A5EDCE4B3D7FF3FDA172670B2FDF13FDFF5F66E1366F1BFC366A2F771E9ED3F2039ECCB72EAFC3F6AAF7E7B4C3CD93F99510EFCCD9A07403D56699ED7ADF63FBB37CD188C329C3F9F69251C643DF73F76D5CF9D48B3F9BF1CEA485F8E6CF43F7D76695BAA61FD3F1248B86559B48B3FB6EC84FBB26AFEBF3C04D3EC7531E83FB3A1259CD729F73F64D8237C4F46FF3F5C5130E7C3A3E13F4028DA268612F03FEE0158B586AFDCBF41CCCDCB63E6E73FD5E5BCB4F280024070CA2164633B633F95823C6E370CFBBF1DE3E35972B6FF3F2D3FAC24F307D43FBEE5CC12B9DBF73F2D1BC4EA23B7FBBFBBC104DD4C25F63FCDD63D38E691F6BF69236CFE214C0040C5064880D737F1BF9C8F53EC86D3FB3F5B00294E3568F03F4A7AEB74C1E0E53FF3BCB3A4F24B00404561F9FB19910440A0D9B84075ADD3BFD3511CA96EA4E43FE1E0DE766C6EE03FFFE29FB755A6FB3F160CECB241D2E33F02E4798AD02B004032DEB634E4DCED3F37417E049D12E43FEDD79734AAE301C00EE9C432DF88FF3FCBDBBEDB3EF9E0BFED2E0E31F97DFA3FEC652371E064004071F88D0E70ED923FF4B839D7136A044071F924FB63F9F63FC55969154568F8BF22AAF26F42B9FC3FCF76192B902DF43F44979E74A4E0F33F9791D8CF05A60240EB41E56AE4DBE83F09752A768208F23F84EDA17FDA89F93FBE473EEB8F5DC93FE9AAE394CE700040C6F49F6B69F7E6BF6069496EBB9BF13F42D072A929D6FDBF40D3F37E6FCCFD3F64A00BACF03AFCBFB6884CAEF40B024008EB799EF742DDBF39333A7410A8F53F138C0BBE661DC13F7C4869B322CFD83F133BC329B91EFABF1C51AEE88A9CE73F95962EF43528FC3F71ECC0991B02EF3F59210BA7F911DF3FEC3B27CC51A60040E41C1642D93DF63FF98BA65A7B40E03F9CBA3325F1A9F63F1C398D3F1CD3F33F3684111041BEE63F76AE1C7AC1930040492617F90150F03F9B97A107D8E60140EC0645EAF013DA3F9B82A071C436EC3FCD3CA5E4D827F13FE8D08DB5B4C8BA3FFD537FF86A5B04C033A7C1D257B2E23F353CE50E61E7F93F12674DF26A40FA3F9628F08EFF87F4BFD1302996571FC73F988931CBDD94F1BFB6F3AC056078F93F2DCEFCE43B86E4BFB3450E134906F23FE17AE74C9670D53FF4E140152A280040EA872CBF9796F33F66E76DE536D50140F03F001BE2D5EC3F2A44862EACC7DD3FBD1BD81C76D1FFBF843B4420ABFFF13FEB660489B606F03F8216581EF129E73FB6AA34319857F8BF789904D74935F23F6B965EF8AACAF5BF7A65C495FE49DB3F7DBD5E5FF305FA3F5C3449A48F84FB3F578F4D5C93D1C3BFE34209439310F33FB4F703712695F33F429F6765D786EE3FA630FA00DD78E13F0E1CD1D20626F53F15EF5CAD0BFFDD3FBE749E2CA155F43F4E42E4763B84F3BF52E2C28E42A20040CB52DEFDFEF7DD3F6DC189E4040FD63F9690A209F50D01C0176FA7F92500E13F3312569686F6F73FE1B5645B9A08F73F68B940BED6DFD5BF774543B29904E93F8C5071E717D2FCBF87EAE9587805DF3F3E5F255A35FB00C0EB1E9D445C8DF03F26880E5581E4F1BF1EDAEE8B1395C63F5597EC54CBE7FE3FCB1D68D117C5DB3F028D4E65EE0AF53FA9D41DC81B32EC3F388858A7C6EFF4BF5D4F37189EF200405C3C2A1EB3F4EB3F1F51799DFD7AE73F200222E3695600C08F59645FCBD0FB3F922D6E87B841E13FA82465D9D857F93FC54344CD29B5ECBFE42FC4A6566EFD3F8ADF73EC8B21F03FDDFC57A8A6C0FF3F7A4E32B5EB17F43F50B8DB3AE7DFF23F0021D09AF7AFEFBF7B4D115FB3ABDB3F17D9AEF8C60DFDBF3B5CEE53A744EB3F934EACF053BBE83F4FFFC56089DAE93FAE5448F1642E004050EFB10B74C2E13FE47ABF4D3696F83FEFF12A6B31CAF13FC4C2A09F4CF1EDBF7EA54E16498BD03FEDD7A7477BAEFA3F954A67333AB7F53FE990AD3963F4EB3FE7396DE06876FB3F6834B0B08EA5C7BFCBE9A3EB902B0040B4E8549787123ABFFBAD51C78BFADD3FAC2AF01CD574FBBFE27DF363BFA5DE3F75CBCEEDAC49FC3F8B558C4B9AB5F53F7C39A6090D11F23F8422006FAA4003403AD7FE2036B0D23FAC6BBF92B7F7C83F70FB415EC59F04C090EA06D30DF2F43FD98C1AAC6D1AF13F9004E8791060F43F6551B788EA66F43FD46568201184EF3FB9A000C7A6B502C00AA9419B6EECE93FB02C757D2D3CFDBF5C34A0DC8EE00040463188E38029F23F7F8E19093A17F53F4E5BA9098CCFD7BFD55CBE0022DDD83F7696412CA3B2F4BF72081D9EF480EC3F331114256A3FFA3FA6540E9CC933EF3FD9107894E305E03FED1DFF5299F1D23FA778E0833689FABF"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%0) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

