"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x30xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<20x30xf16>, tensor<20x30xf16>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<20x30xf16>
    %5 = "stablehlo.power"(%3#0, %3#1) : (tensor<20x30xf16>, tensor<20x30xf16>) -> tensor<20x30xf16>
    "stablehlo.custom_call"(%5, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x30xf16>, tensor<20x30xf16>) -> ()
    "func.return"(%5) : (tensor<20x30xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<20x30xf16>, tensor<20x30xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x82B76CC01C3C10C27C444F3CF9452B40B0400ABD2FC412B56EB9D7B9653B914720C6A1B6ED40B0C42044CEC2A3380D39F54010BD2841903F923E7AB6D9B62ABD103D99B40FB9DA3DFF4127C252B8C04317C481C0E7AEFA384242C3C36EC3A440F3400647183E96C30CC0BDC4E7BD49B7A8C4052C793EB13BECC5E041D542B440433CC7C46342604403478F3C4C2BEA3B0542CFC51240563D20ACDA4663C13D436C435F4315BB0FC36EC24FC03CBC963E593DB1C433407D396D47E6C3513FC1411DC6BDC568C6EB3667C0AA4370C422C2EB45EEC1583CA4C455BD62C0074203C296317840F53A17C34B402CBD74C3D5BC45BD9E39E7451CB8E1405834CF38F941C23CE44575B41A3CDA350340A1C3163DA63A76B155401F44EB3D9E407D406DB8273489C595381A3805C26CBAEDC411BE61BDC0C0A2BA0BB4F03BB6C1BCC057B5D6C00DC53AC022BFD7BDB2C0B5348D4022C4B542CF1AF244DEBB0B308EBD214338C5CC42ACBA2AB01CC2684302BF124749BDF2BB743DBD444FBB8E435FC4EFB7D4C0D5BFDA45043D9AC4B6C2E242AEC6093CF232B5438740273EE5BB13C4C0C18AC6CBBCE2B797C0CFBC6B45A74206C74BBC7B41C4C247B88044BFBA553C47B468B8F5304C3F58B21231DFC418C4FDC4BB3D0D467CC0D2435D381FBD42C2BF4254BAACC4AC2CEF4043BE314296BEA643904074C209B35CBF1B43AF43BAC2C7B91D355147BDC31B46083968C6A2BE42B300C785444EC1BB3EF144E8C05130EEC434B9E445E5BD9D3281BFD7BBDEC273BB35C501B844462ABECABB6D4459C4583D60BC613CCDBD66C2843ED3422FC4AA44633D37B16AC060BCE6BF173EA434A74364C3BABAB13FAABBE03E293145368742DAC044320CBDB6460C428AC3D2BE004493C00F419B3B87427441623A4D3D31411BC46DC5BCC2C5BA5244413FCB43043CC03EEEC12D3ED23CE4C04034D4C423BF8745D5BF6438AB3EB53D32C476C3773B8EC6ECC0534575C27AC2524135BA4FC5A53DDCC4D63DA0418DBF3C4111C1613DC746123A3FBF48419430C5C19DAF553489C46BC00CC462415ABAC5BB49C0E3C6E2BDA2C1514793337CBDAB48503D52398FC01AC3BC40EA404E3B433CDE3E04BB373D6AC4D233A535BEB903C08E438DBF3F44544579BE023D85BCC6392C442EC020C621B36A3C46C0E3B16F3B91BAEEC09840DC3C5E3548BD9945C13DEB3C82463017E243E04071BFF1BAA03A4B41CBBD5C3D20AFF2C215245A407D4023402ABD30391AC56D3FEA3D6B43DABFE43F014455406F4004C321C4713A3AB23CB87441C6417A42D6B981408C3B0934E2C5FA45E9334F33B8C49EB886B23643B23D843864416E43B840454229BD7CC6FEBD632DB4C01B413FC4ECC593422B3D953AE1BEF7AC0F4326C10FBD65C001BD9FB9FF3F3FC2BC40AF3A2EBD103B0E440F40BB440B46B7C3573C344061345A45D6C3AD42D8B72BAFB1AC5843524443C07B44A940263FC53F76C3A02CAB434B3CFC480FC41F3DC8407D433FC80DB55ABE584030443BC0F5B1F4BCE94354C2A1449EBB59C6EA4044BA50B9E33473C376BA94C42B468939BCC48E41CEBAC7C24B3CCE3FA2AB1BBC31B9D939183A8A3D32BBC0C5833AC4C29646593A52B86BBD9DAF323476B81CC498C143B91BC4A1C59333D9C19832D6C10AC3694341BECAC005C3C34323BD"> : tensor<20x30xf16>}> : () -> tensor<20x30xf16>
    %2 = "stablehlo.constant"() <{value = dense<"0x3EBF6F411A34AAC021436E3D14B9CE43563BC5C26DC37E30FC40AA4332BEE8B9B0C4B2385F37AFBC834060C034BE06BD0E43C84846C322C61EC32F3F684225C2D3405ABA69BEDBC43AB831C43DC5F7458D39E734B03E57BF78358038F9C4094151C2124278C1FEC47CC439C4733B83C30A3B49405D409237453FD1449F4881417AB54ABC36C24DC5204677C4093D8FA63340B83C1EC5403DEA408E3F8E3DAB3E553B05B1B739B34467C0953E89B45740CDB4EB363B358645D34223BB98BF51452D40E8425BBD16BCACBEEEBE133F5A3F12C0D5BCDAAE4544B0412EBE5240833C06C165BEB63C7F4067C184346CC227C1E13E83C38BBC6144874310C44442893D62C5F442F8BB2C386FBCB440E9BDB2408CBD743A2E404EBFD3BC44BE66C2EF3B14C12BC0BEC53740423D444197BAE1BAAEBDE43D1AB4F7C283BFC9BFB6BDC5BF4641A3C3F1C2CC41C6C4FAC1F4431FC48D459A40A03CAC46C341743F0923603FE141E5C3EE4013C3FDC4A2C5B744894476383338E444962A0CC39C4023C228C4353D2BC2EB3DB637E84188BE52432B39F1BD7244FF43F9C129AA723A7B36F243154075C589C7B642AEB55CC1C73C38B9A63D85BE95C4E0BA69434643B3BC9C44BDB64B3A0CBA0A3C81C1C542ED3B93C202380DC550349AC280C577C5F2C6893C93BA6EC5A4C4E5C1D4C012405BB557341E3F19B69537C93DA7C511C16CBAFD458D407E44D63F7443684003C445BB46C12440D7C277C4A63DC934B4C3C645403F4C2698B33CBFC5C00233294200C2F3C10331A43ACE383037E3BADA3E683E7F44463CBD3BE7BC73446C4303BD3140D1BC384675423BC4EB41313B3D3ABE3E3E451B46E6C0763E56384DC48A400CC5A645B7424BC0863F5932ABBC3C412AC052BA5A400ABD9A417DBD5740A23672C64FC43D391EC4613EC1C5494840C6713E85C4844514428B3C274030C48C429240DD44F9C387C05146443D5A375243B8C1F13F823FEA3507C2D93E1BC39A3DA93E7FB8B244EF3E53B5264046414B36CE41FE445344CCC290C533C0DABC743F683C89394844D4C1A0319E41DC3B264154B1BDBEE3373DB96CBE68C05C3A913E6F45CE448131B43DF9410E356E42EF3EC4438AC327B32340C3C14B4743C50A3ED537BD42113CA441944564B956B6AF3EE3C32F458544E036A43824C63FC7C1C0603C0EBE104815C010C2E8BDB2C0DFC401C0A93C883DF0C614C2FB351C4074C843BA423CE239093C5BC0F0B0BA3B52BC1AB964B61FC07B404C472FBBD9B3B6358DC2BFC622C3BFC082C050BFF543B432AFC48C3C0ABF74C10F41D3C57AC12FBF4D34803E83BE92443DC2BDC5C942A0BD1940B9AAF13D7ABC133EC0BBE93EB643FE400EC0E23E7B3D3C3AF7C4A642D042F5C311444044AFC4273E223BF4239DC1254055BD1CC426B8F63EC0C22A43FB3DB6BF5BC405BF2842FCC19F37C0BDE8C17D4050352C471F45122E1740DD2AAA4060BE2CB7133E69C099421CC3323C8A40E3C2C52FBBB47ABFA13DCAADA6467940B3C01A39053A28B9EE3CE4C09C43A12CB73440C5AAC69338CBC13D417B3BD72B2BBE30B883BB29394F416EBBD740A8C2FEBC02C671C31E4470410239C7BF7FC293BDEAC04D3CF0B9D2415FC2C0C0C8BD833F29C30040A6C77C40B33F62BF0F355F3019C314C59CC0D2BCAD40033D0C4051C00BC0"> : tensor<20x30xf16>}> : () -> tensor<20x30xf16>
    "func.return"(%1, %2) : (tensor<20x30xf16>, tensor<20x30xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x30xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00FE00FE073C00FE935A6D3C2535644C5E4000FE00FE00FE00FE00FE853C2F3300FE00FE0F3E00FE1D4E00FEA940203F254E00FE16282725793100FE00FE00FE0F3F00FE00FE0C317B3800FE00FE4E6A00FE00FE00FEC740E83D00FE00FE2948522BCC5D103500FE00FE00FE00FE00FE00FE7219B741DA3B00FE9C59007C4249D43B00FEF5268F0E007C7738F426003C0E4900FEBF26D73D00FEBD5000FE4448A742853A00FE00FE00FE00FE00FEE741553B00FE193DFC2F4F6300FE16354D5C00FE00FE00FEB64000FE3E2E00FE00FEDE2600FEEE3B00FE00FE00FE6B4900FE03556D34C93A00FE113000FE00FE00FE00FE8C43433000FE2C4F3E5A7F328B444A36705F00FE0D3C1942244500FE083F2C3D00FE0745D22CFD385134D22C00FEAE4F00FE224ED43300FE00FE00FE00FE00FE00FE00FE00FE0F3C00FE00FE00FE00FE00FE00FE00FE00FE00FEE51F532800FE0C4C95153A7900FE6C2500FE354900FE182000FE00FE00FE231100FEF36E00FE00FE8D44553C00FE594D00FE00FE00FE00FED34A763C00FE00FEC35500FEE63B9A14DF5A942DD63B00FE00FE00FE00FE00FE00FE00FE00FE8821344400FE00FE303200FE00FE1D5C00FE493B00FE00FE6133143900FEF95800FE00FE00FECA3C600700FEAF21044F00FE00FEF04300FE00FE007C792C00FEFC4800FEC13D574400FE00FE00FE55123B2800FE00FEC62C746F00FE9F62C23500FE00FE00FE00FEE31D00FE2C40733E00FE9F0000FE00FE413900FE5E5100FE00FE53A600FE00FE00FE054200FE00FE644A00FE5B4300FE5E3C00FE00FE1D46E03200FE03315E4600FE00FE00FE00FE8D3DF22F6F6400FE00FEC04100FE3B2E122412573B6200FE245000FED63D663400FE00FE5A3500FEF934F23A50326848493B3631342400FE00FE00FE00FE007C34267848DC3B7B4C00FE8D3EE43D00FEAE2200FE00FE7E1400FEC925D73FB63C00FE00FEF93A00FE00FEA21E00FE00FEDF4300FE00FE0A4500FE0E3B454800FED73D00FE6344AA6B1C4100FE2B30494900FE00FE7B3600FE00FE00FE084C00FE00FE00FE00FE00FE00FE3E29FB4D00FEE450AC44823000FE00FE8E4A503DF939773C144800FE8A3B00FE3E531B1000FE00FEAB3F00FE5944005700FEE03A00FEA438551B00FE00FE00FE3C3C00FE00FEC33C00FE00FEB7626139DA4E00FE7D2472314A396F481109D504452C00FE00FE5D45783700FEF63C00FE00FE933F3D40AF36083900FEE24000FEB555A239CD3900FEEA2EA605112CD73000FE00FEC23600FE00FE4142F430332900FE8720B13CEF4900FE914CDD48CD1400FE00FE00FE453120401F3C5C445E335D434A3500FE00FE00FEB85800FE394300FE00FE8852C940594000FE00FE931900FE00FE00FE00FE00FE5B3600FE1E39DA3900FE1E390E481734B514732900FE453AB33D71463A1F00FEF83D00FE00FE00FE264B543C00FEDD2D7A39D440673300FE8E711844B23CF70D00FE703B4732694600FE00FE00FE2C31FC4000FE00FE00FE702800FE783C00FE00FE201900FE00FEBB2900FE00FE00FE2C36A73D00FE864B00FE00FE533AF33600FE00FE00FED336BF3A403800FE00FEA23E00FEE633153800FE00FE00FE2E2D00FE394C00FE00FE00FE00FE133900FE405C00FE00FE9A3200FE00FE00FEDC2A00FE"> : tensor<20x30xf16>}> : () -> tensor<20x30xf16>
    "func.return"(%0) : (tensor<20x30xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

