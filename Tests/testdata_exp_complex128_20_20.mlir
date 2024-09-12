"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f64>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f64>>
    %4 = "stablehlo.exponential"(%2) : (tensor<20x20xcomplex<f64>>) -> tensor<20x20xcomplex<f64>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xcomplex<f64>>, tensor<20x20xcomplex<f64>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFB278A8F85E1F0BF620CAFAA52A5F4BF3F191CA64F6FF4BF0707E80B1BC0F3BF5097838EDB971040042A816A21DB17401C7E3E81DE561640E04242CD6C76F63FB83DCDC1FC35BF3F5EA14D428D9E10C072210CB99CBC10C0E11A54413977D83FF501C0C10D14F43F989EDB124A33F73F444D81CCE9730C4093BDAC0B8AE70B405A77785AD4B1084022084F8415220840CED4C3E7850D05C0167EA8759216E3BFBA28EA9E91DDFCBFB271D406C0A30D406A85A04B44DD05403630C0906CE815404AF37184A0A5ED3F769EE1CCAA031440C596081EC3ECD8BFC2B4F5E974F205402E2D1ECE1D560440EEA9CD831A931CC0D391CBADC72BFF3FD2792659CDA0DBBFD816583790F4FE3F41C0BE1BDD1D13C0F4F5A8B753BCFCBF621CE018F6E006400721B8450F07F8BF43D2488B812511C06C422CE1F3330C408038AEFD9576D03F74284E76F4311340926E47B7840CF4BF469A9D44E4FC0440965567874DE9FE3FB4B588A2F8B110C05B905788FF5F01406C84A73AD271F43FD1CC5861B66B04C041EC184CC03FF0BF2A547D5F8EDAE0BFAF1DDC2D3804014098AAE4A817F60240DE0F662FE006FA3FE860A64824F60FC0EE0DE6CA050C10C0D762E0AEE77DF9BFB1127323FB66D23F08343E4C2D78F0BF78B8A6106A6EF6BFDA60E3CBFD80FA3F35F6FFD45653D63F864A294D361D09404952146637C6FFBF3C8A99B951B211C05CCE8E851B7AF4BF5A57D7CC7116F43F8AB8B65E4082FEBF3D67577D8A41D8BF24018999DD75983F9B5D4A56688208C0D016624082B9BA3F246A44D6104D0FC0743BE0B2E824D5BF00954430F6A6F1BFA1DD3BE48086F6BF859A77BE8910D1BFD6686C447908094095DC7014BCF1F0BF3A69DDA27DC3E43FA6ECCF8C295DD1BF8CFCB928041606406A25D29B5EE805405DB60ABB724DF1BF00A7DAFABCDC0F40B09E3E12C736CD3F4A415D90748CE9BFA546BF4F0D32F53FC238E7E11300F2BF667E8627CF0A05C046948454E722114062567B7A34AC11401875AA94B9F90EC0B6FCE80B6F5CE7BF0316C5A009601A40DA6D2BEA541403C0332CBDC35EEFE43F3B691C9646CB08C08C13FB7EEC11F33FFF74937ABE27F3BF0914498E4F73F83FBB6DF5D12B58E3BF8093FD71CE85E8BF18F1EB37DD0515C0DC912E80C334F03FCE0E3BC193D6833FCB920B0AA8220C4034231DCD8871FBBFCA592B30EDF3EE3FF4B526A81F771E4087B995CE84420B40466CAA24F6140B40C2AC91D37C6017C05AB67C0CF565F0BF2F35A0F870F106C088A8C6B65872F6BF8BFC057BF822BE3F072CF0F1470BF9BF8018EC176AF70FC02C61618AE92AF53F7EF9D8977869EEBF22852523773C0F40E6800204C66C0540205042F6002AF33F2CF7287FA07EFC3F8A48A479A35A10405C74CF96423BA7BF0EDA1136566DFBBF54DEBC4DE3E112C0E8BB1BD44951184052CBA05B294EF0BFA62B4A1BDDCBE1BF9A186D29CCA1E0BF1C6093478F5016C02EB198DC8877EB3FB918B3C365CC06C0E775C9032E9BF2BF56E5B70F79470D40F6F629BA714403C00C3FE30BB02B07401E52B3ACD3A201C03CF92A03D89CE4BF01F38AAE427FB53F582E9CFCE3C808404BA928956591F33FAC32FE882E43DF3F5796A08FC8EE0EC03C9647219D2C18401835BCCEB9DCF2BF6AAB3478DD880040A64F45445A8F0AC06639D937B02102C074A7B26D587BFDBF233F58C93F69F1BF0AC1F5B31E49FABF9A6718F66CADF43FAC69AD1B3EAED0BF3E626B313132F4BF7AA32D465B9E9FBF1D2FBCE42BC0ED3F663E3D963B0AF93F12CE4809157601402E18F01A653FFABF24ECC91722C109C0EC920EB9ECF40740C0DC6C05CC5FE4BF6CDD4CE12B4202C0F09365BC9FFF15C08E49828A9B9CDDBFDD44A08307A504C0BA5AB5E87A5D04400A27EEBE987906401EA53EA23377F5BF6258884BDFAEFBBF8E6D96CF6749F2BFAB50EFCA23F3F73F8E9E8E9EBDBEEC3F6EE51F27943E04C0A3C93B00E8BE64BFD1104E043713EE3F26C74628CEF7104092F3E9305C2FC7BFFDF39CD824F20440CD7BB61C446CF43F874F57DE093DFC3F9B64AD6276F707C09A27FF1FCF8A0AC0E1C946CEA0DDDEBFEE184F5E475D1F407AEB8908F0C7DE3FAD56A8D5E4E90B40D426E0D5D6EBF1BF5EE3E98E0E8005403970F75A1878CCBFC0AD176B3E680D40A63936E1332DEF3F84FF57E247CB124037EEFBE51A9DD13F13D14F1663D6F43F3213936A4E9E00C00498233EA24F0BC080A64C18423806C0F66D94C80BA0E7BF3FDD072AE88103C022C4F585C76D09400BD205094F11D3BF8039E30915170D409B93D22D5E5512C0868DB8E6B29E04C0274710EFB0DBF23FDA0C015ADC480540F4C74DDF05B0DABF4AA1A4F0377200408A6C87CB0EBD00403C7D51795204EA3FAD31AF050C0602401229E0F4823A04C0FDAC43B04FD6E13F6ED16435B9FCBFBF0E40EC5582D9004024D5F3FA075011C0899DE94CBE2FF4BF6F55EEF39F57DF3F2E14571BB05517C090DD4D9408AAFD3F812F905EE2690540520363589A170140B9D015470AA3F43FA63C016A690802403F463C3B246117C07B5DF601193D14C0DCF0FEEEDF6110C0139A5D096FE3E7BF4E25430AB5B714409CC3FFA1689CECBFDD59CCE1FC1906C0B24CDC3F2433F33FE2A30C5E9A92F33FD5E02AA91FAFFC3FED267966E4B0E73F01969D3E4F49E43F24BAB6CFBCC20C40EC706C6DAE0516C064725B6CBC3AE9BFC4AEB7B3983EF03FBFC8A65A26CEFB3FC36B7C9CAF0C15C0C36D1544189A0DC00572BDD4223CC7BFA128CD1EDC84FF3FEA0974F4C5FBF6BF0DDEF809B1C205405FA91B31429EEE3FE22D26F6F61714C06643394F45DF07C0FA8F033B0BEBF63F311DA91DA092F2BFC7C318960DDDF43F4CD93237D22F1640DA8A5D977E4CE93F4A96EC14B7C0F7BFB6F1FD681ED7F1BF8085BD2ACAD9EBBFF1694839BC48C53F14F381703BEFE2BFFE2D820D355FDD3FA124FF887CA0054044E032F2917D10C08E5FFC2EF0D3E13F941F65295C940BC042693397034D0340AEFDFBBE112E01C00F37C694EC9ADCBF775E4CCECFD0FA3F40AFEED2804CE3BF160262D9E8E3E3BF8AC46FF87884C23F01116B74B842EC3F3F24C02558B9F13F81CC2ED7EDE80D40102331668500F3BF9BF9607B30200140C5625DA42FEDFC3F46DA0C99466006C000C37F0AB6C814409EEB6AB27F3D03406A0F27B62CF915C01E7E5AF17F1212C06AA2D92A2A1708C0125696A8390B0E406AFEB399E2390A40E987BC96068705C08BBB5930E958F7BFFFF739FA236DE7BFDCC10FA639BAED3F3AF2728CC56100C02784BF74BF5514407357E73CC874E93F2E44D9CFB48801C0989609853A30F8BF0EE7FE53F00011C0FEF587E675AC10C0F84E09DBEC38FB3F146DF48ACC421AC0D50C4BC8D87A05C052A6A290905FFA3F6936D6EB4A730940692C29215CCF03C089E5E55C9A84E3BF946BC99FBF0902C088C6301A0180DD3F4644C1E39451F3BF08D7A1461C8808C01AEA20FDC36CEBBF714D56801298F3BFC717F8184D7CFF3F530FD9F06C1CFA3F9DDC79FB9E1B0D40AAE944B2B1ED02C040864795D8F1FC3F10ED5174F4691A402C55F1CB2761ED3F9DA6D683D013F5BF09F150F6362307401C8E63BD09E5E6BF4DA1381CC6FBDE3F7EEFD3A2A2F5F7BF0F1B035AFB170040C401D7AE37F3F23F13E4324B684007C074BA1F14644C16C0122E8D8AC9B6CDBF78EE327AFE31EABF4B474716D60CFBBF13DD8B6E549388BFFC06D7B4384D10C041E88B840C7E0A40F422BA2D75F9FEBFC98D572CDD0D09C05714FBCAF21B15C095985582374FF9BF348D343C7445DB3FACEE949BF127FE3F1ADFDCFD79A603405CF171E99BFCEDBF40E4653B82BA03C0F21F54AAACCAD33FFFC78F4101ED1E4040C77792A318FFBF5C0A3D36435A14C0E9D6BE2337A5E5BF17AB11ED35B3104069D658CBE3E5C2BFAEE98CC8E34E683F9C7D800D13ECE7BF3E0173D51173EDBF1CDB3B720CC7ECBFED30EC75F20D05C0E071CFCE2A0F0AC0B985C4D39E99F2BF62A2158CAE6C074056DB2BEFD4E182BFE8A07FD13AE30B407ED139B6D6E3D83FE52543DD59050BC0C04B22954A1BEB3FB6F271AA97230540888F9D56048404C0D9DE85497E3D1DC0B67EC4C702F9ED3FD6AC6CE71FD9E9BFD4B01B60EFD5044017056F07E45BF4BFBC9E1D947380CABF200BE1DF4EDC0CC0A186198474A3E33FA59F6F0CE8A0D6BF9E34AABBE71A00C0C051BA472172034068AE6E5A38DDF5BFB06BD490E83017C0540B1158B522FFBFFECBFE50517BF4BF32B30EEB7192FDBFAF5EA2A8815A09408B338B7107FD0440093AD6B78EA302C0C56B9B018BF717C0B8A0131B689510408E578BD5E9EB08C03E2E5E7097FA06C0204DD85687531E40146E67F45D86FE3F06268DB30A1EDB3FBAB9787E886209C0DEF05F8DFE49FFBF0A74779D924E09C08475B470E12CFF3F7A332C8DE1D303C0A2FD7F4B45AB00C06F4E6C13B09FF63FA8EA284708E5E7BFE8E41A84C97E03C000F136CBF714FB3F9A09AA0F0FD2DEBF9E786BD161A1C4BFB8C6ACE7D7F81140E713152478021840DAC0F4CBE1AD10C05E8DA125430F02407C26D06654460DC06A011F2D47200AC0EBE93F5DE69DFEBFA2C293631CF0044088F8EA8A327BF5BF70319ADAB06712C03C1B4B83BE5E1340511D7EE80B561540FC3ED06D322B06402053669A37910640961D2C0634DE1AC0F8B8D8802B6AF43FB2EFD1FC2F3CD33FA0FE75C9D52E09400AD8DDEF89711BC0B28546283EC5F9BFB891410206C58FBFA3DAB864BBAFF63FF4C0F4173AB019C095DB5C0492F11640788FDF931226044084BF647B02C81040051B3B738D4D15C0A698274AF566FCBFE206D6C720840EC0AA5C84AE310801C01B07AE860F251240E5117D46725C04C0DC6F54D32A5A14C067461F5393C615402E49DC995D14FA3FAF69CB62F0BDDC3F9438BEDB7EE710C0B580CCA8067702C01888741911DFE5BFC4CA4622900A0A40634D7C4A29B2E4BFBC575B6B5AB80AC00076E1A54AC9D43FC8A5649E07B60F40D5DAE8B0D61EF13F7931E55A0E210940BC880C7090110D40E0FC8059F85CFE3F581EDA6E774F00C08C9CA94F067E8DBFE3DAC847034CAE3FDFEC8C3D167711C0F68E5C088740E3BFD067F3C22B4908C0443D330B971DEDBF61D5D726CACE0F40E417DD3574ABF23F6A2339826594F63F246977889C020EC0FE89EFEA990715C0AD337073A14020C028BBDB5D4DEAF63FC445BAC12751DE3F82130D1AC905DE3FB2F80BDB9352FC3F2ED789F6E00AEA3F5676BC57C966034070860946110BDB3F1C20BBD023CE17C0A388A34E41CC09C010B6C24E1981D33F4E97F8BBFAE81140138D1FE65072E83FD697FBEB8157FC3F92992C73B1C1D8BFC07533C4BEDE0B40D0053C478E57FCBFC771328C66B3EE3F48506565D8F90AC07071DE2ADA020740236A81A11044EFBFD8E91297AA0C9A3F6610DB6EEB191D4026C5C776574BF73FD130F850465305C0DCDEAF33BB25F53FFBBFF7D914CC20C093B443C35ED1D33FCD6BDACC522BF23F25DD865C768BFE3F16BD1DE25B731C401B2343AEAA04FEBF7FE9938894B508C049CE1CB0A093F0BFD0533FC0DC660040B0975805F5CDD7BF2C90B00F9F360B40C8C9BFAD29D2F1BF9F73F734E2F1D2BF1C6A4A0C595A0D40BB4041A589D2E3BF3E58D1C8816A0EC05E2DAF890B91E6BFEB831870BF6903C0FBC16A003870F13F080EE2A121DB0340A071B0B8464AE93F8894A25EC8506C3F1554DABB3B640BC0322472489DF405407C6F1811C73CF3BF798F8C2AF7F806C00C6C2B018531154004603294E6040AC0525EA90DB024D03F82D732DF462105C088EABBCC149806C0DB005D44C08FF6BFE6737EC84C9BF4BFCE07AD572F780CC0C59C3DCFF5DF00402D148F026EA608C09C22B75D7FCA0EC07D4BDCDF31FAF53F08BC88C1D4C412401D70B40518AB693F8857E2B18A870EC08D45FDBA7D1100C036D7E19B134C0A40C453FC05A4111E40840156F179DF01C0C0F98C0446AB96BF5139D57C76F0F13F4E44DF558ABCE83F3FECC20938E6FC3FD0480448FA47CFBFFEB26FDF796BA6BF48FF282AB99FF83FCE0F6AB1D4EBE7BF481A70E49BFDF4BF0A0A8F3846830740DA66214781C2114048339E572682F33F24CD2C52B244F93F76ED2C7AA304F6BF2AD2B81294E6D23FB2AFE74FF600F2BF4A57EBA013F1E4BF1184C8E57D5102C0583FE0DAADFAFD3F9D215A30EDCE00C09B3588C3695AFC3FC0BDF48C6810C03F7F11302C41DB0640473865C2D41100C0F43DCCFBFA3BF03F69E819643EB70F40B18749BE6A7B08C0CA28250D16E000C0DCE68561CDC00A40175EFF64B4F3EABF3B566F53A9501BC02ECF406030C7F6BF1256E5FA258102C0DA773B4C8138D0BF4AA578DB0DC102C095F670B5130FE53F0162A1CD5CDEF1BF21A2E81AE7C8044010A8DC761B3D054058E365637D71FBBFAC90CA0A41F4F6BFDE60CA5A5044FBBF3C5F05A03F0913C0520D2B6AE9DA04C004D326D46DFFE93F9627ADC6692F60BF92FDE2258094E5BFEC1ED74D571BE53F169944ACF0B60140C0393B00E0ED0840EB3BE4E16C050140425D15DA11D307402AE4D5F1CBD506C0BCDB99E55F24E93F509B8512E08B0A40186E3A428DBBC1BF7662BEF9F85C01404667A13D5F80F6BFC06DC115AEECEBBF27AAB6701E5D00C083C88228848E074058B35127B879FE3F40146CD0E430F33F87E222618EE6F53F5E5B990CA06D0E40AA0CF4DC373710402B717131742F1DC0FED0143D4175FEBFFA287A982B3EE9BFF2345AC7196912C0903B2DC4654BFBBF0E8C88A7564AF13FFD5C65212026E23F639650D5C387FC3FF4CF4A0D0A2708C08ABA3395F69C0D4070999F6E71E60A4062D54C2B225A0740E65C79CF86FAFB3FC974E9164B7BF7BF20A89867DEA3F4BF421E8F2A0ECACF3F1C909D5A834FF8BF66AA496590ED01C07C0DF2976199E5BF8C76331D1AF314C07F312703666D08408070B1AF8969FE3FF80480D612E41940DA4347C9B638F7BFB4519CDACD63B0BFAE5228D830C806408F831055CD7B0A406EC8C5A185F5024046FE5FAED3A1D6BF67E36C43569EE33F843D82AD22F7114056B8229AD3E7F8BFB491CC16F218FD3FF6B51CA30BF0F8BF2E3721641FA116C08B7BE7C6F23F0440538E62A85F63F2BF043C34718DFC0C40D1EC7F73CCA6FDBF5FEDCF6FFF05004010F77D14F549AABF8E60D603AF9A15C0EC79FF6B7B3B18C0D20C5C7AF57B0BC00D0A474D02DD0740420D319468010EC000274B097A4FCFBF11CF66B7C442144028846C126B3FFB3FAD69C1A84965FB3FCCCBDF4720EA06C0828D3E66E90608C0D82615E1AAC405C0420512CEA0BDF13FDE152CEA8754DF3FF87060A263C70DC0AAADF46156E4C53F0CD5390B305AE43FBCED3889FABAFF3F3627F05CB4569CBF9A71706B264FE2BF7BA0CA36E9B0F7BFE57BFDCAFF7C04C036EC2784887310C00E381D8377BB08C0BE16BE632FC7FCBF44B583E48E13F6BF43E1B77ABC74FD3F35E538AEBA220240E05300B70DDEC23FF4635B85F7FCD2BF5A363F6A200F0CC05A2B5152F8D41140ECADFDCCFAB70340D2715C6B81F10EC041ADBD37D29721C0AAF19B4B9B29CBBF3CD13BF6D4370DC0324A6565979A04C0821206EABE40EABFF88345D77EBDEE3F3F5F64B3125F0640EA7E5A32D390074074C7BEC98541E6BF1F2CFDA1FEBCE03FE76FD694C45B0A4022080AFF1C62FA3F2AD4544927C11A400A3F8BEEE12BE33FE6C186A38FCDE8BFD6B33731E04A084035E2AE2DA3D517408C6DBFC7D05611C004964C13B089E3BF14B3F8CDBD271340A895C09CC473E53F0D63A0DFED4D0640B8D46C80EC36FEBF42B5A24991C6F9BF2BC7300C7487EA3F164ACB5F552EFD3F060164723E2FC4BFFCA585934D83F43F693D7C33AB8C04C0E322B96CF2C4FCBFE183FAD67243E3BF6358728A9D02FABFFE6877B9E10DE13FCE07A84AA15EEDBFDA359C2B96C20A404C22A43E1E6ADA3F66247271DDC4F53F52402E331E86F53F3A104BBC0B9D12C06C9A5CAB46B5D23FEE121D9467381340895966DD849B0CC0C292E9D0287E15C02CB9BD50140BDE3F60916E3253ECDF3F524873185E3805404A7A39E1BEEFECBFE873686650E60EC0F60417F9D506104032E0096AE09F13C0483B78308D0CDCBF125D1CB6689DFA3F741ABD50BCD989BFE6CD4533EACDF63F06F892DD91F1DC3F7EB43BD972AC03C06277F3513F3DFD3FC28D6804255F05C095D7282927031640284B6DB73D1EE8BFCB0476BCB15513402E14523CD1A6E2BF70A125573A6011C0DE82E4111256F2BF625AD12DDD0B02C0E4E62650BCD70DC02A82D1E444A3D2BFED2CE147E1BF04C03E248C26394BEB3F359C7CB5A3821140EC425F96F0F21C40B03BB050FBC80C40A66D3C7D0CA0F5BFFCBDF5EDB0B4EF3F6EF436FF2825F1BF5DB7C5363B8712C008DC9188F952F9BF90591C22318DEDBF922CE0E07D3C02C05FE439E38ED9F1BF8CC925D5A4A0F1BF4C1E43B7EA22E93F52A7E020EB14DDBF046CDDA8951BF23FE45D90876714E7BF809B057AA6B4CCBF30F50BD8CE60D83F9E56191F5B1D00403A1C50F9ABD505C0FCC04C9B8B8AAE3FEEB19565326507409C8BDC626867ED3F245D2F805478E4BF8AA78D245CBCEA3FC4E95B010062CDBFFB4D33015712CC3F60190A427CA5F23FF25E792B6CD503C02C84B83A21DB064025C8B8C5650F16C0CC7E77F408311340E324512A6061F0BF1CAD60B87B351A40C2F17DCA0B81F8BFC3A6EA49C48DFEBFE4AD4C6D9EF0EA3FB780CD08DB77F13F354BD786F681F03FBB0D10B851D5EABFF2F62BD4C42C01C0E6B786EAF030EC3F4C10E7C484A2E3BFE2A6C5073085014082BD68D81CF8F3BFD227138E50C3CCBF7C830989B7EBFFBF76DD206EBCF911C000B12E643BA10640E3C43E8EEF991940626918688A4D03C09AB96FF3E3990440441C3DAB0F930BC00019EA27CE10FF3F"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%1) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f64>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x28FDD14659ABB83F9386E00E9B69D5BF76A2D14DAC8FB73FEC72B13333D8D0BF91FC190F53104E40C2EFDE6423DF33C0AFC8032A4B1F46407DA270867C6A7040A02B4CBF7E20E3BFF7E5C547A8ACEE3F39454F03BCF28C3FB6BAD7660C47773FD3B00D019F0BDB3F5FB186BEFADA0B40A1AC2742117B40C0D077C7E4C5CC27C076903B569BBC35C0183EBB4283D70540AF124B23687BAE3F1EEA44E4A8B2A4BF4F43E393C1D0C1BFA27AAAAC8E81B6BF73DD5B65EE4A254074B5A708A93226C02CF1F744D733E73F13D3C810A15A03C02336936C4BFBE3BF13179600CDCFD03F8D250377FC912040C22DF7D3FE4323C0FD5A681C427D1940944C7521F37B07C02C3EBD59548FDD3F84B0AC705AA01B4081F42C89E967C4BF7E6C2DB534A0A73F9D85B7D6238DB7BF066F76F951F7C93FB959C345A06C4040EDE1690F61482140BAD164C446F5424039C5D1956BD25CC084F533BB817B13C08089DD2EF0C929409BAC99993DD481BFC5D6287420008A3F55CD38F635DF07C06B10D2DDE0E5FFBFF4A67D720C0AD43F35E105CEE24DC7BFA063A4D0630F18C00F20A0C8AE651740FF14C70D95BF0AC07E13C2248EAB0E406E40929B189F3ABFAE0A5F09198892BF040523358CFCE53F54EF28F67347F2BF986D39F81C9295BF46C51D911963CF3F6603640ABAADF6BF08699A91D10A6B3FF4C68CAD48FAA3BFEA6F2E741CD8C03FAA770AC9DB13B63FF1B843A997EBD03FB864489C67AAC13F6CC736604324ACBF9340C0B94A56F0BFB2C208688768B4BF9D021205B279E9BF3074BCA204C1E83FE459F5DB8FBAD43F3F1A00F86587E4BF21AAB4141836CE3F94C1F59FF380B0BFD3D015EA81622640BF2CD79711ED33C05EA6B4CE767EFD3F20AB8A08A168E0BF363DC704A0162DC02A1475B6CDCF18409740DAA822EFCCBF6010F316B92DD0BF54287562860EEC3FDDC0D85FB3CBECBFAD7F872B44F2F93F8E3666EB0B260BC009AFC6976AA59EBFD78F9D67DAC7B0BFE1BA6C60FCE44EC0E97BF228C8AA4B404F42BF708A5DDD3F83348D511FDAC23F3DDDEAF15EB5B23F903A7C38CAB1AC3F03B32308C013913F90116AF5EB71A53F27BE374FE15E8A3F3D5572961650D33F50ED1544FE30D93F746E8B30AC3FD8BF198BB20CE5A0663FDB9027596221723FADD4FCC7EA0FEEBF52D9C4E17DB0D7BFE88924B93624BA3F07B8972E3CF6C23F6C9FC7DFE59E9EC0A47E9F319EAD80C01429E38BCEB93A40C987D67B96182940CD54C329511CD6BFB7CB20AF78D4B8BF40948D4B3141CF3F857A7ED946929D3F120B8DE4EC92C1BFDA015865052DC43FD9CABC2170760140C935F69F567008C0B7EE09F30A3246C0DA91CCA5AB2F364047312A2B341CE6BF35BF7E0123EB0940A94BD7AF7ECB4D405E0586E43DA505C0A6DF23820036583F728A99349C0DC73FA0D0E369A69D6C408EFCBB0B4E3F77C02E315E4642DADF3FFB0906421E3AD2BF5212EFE1A93A643FB34314A12B6B673FE27B60287180973FC09321251131ABBF51B0708F89DF3CC012D32E0C2B013AC0FBA7354EC37125C0ED66F4DFFC2E2DC0C43280CA8CBEE03FBAE0BC5FDA8CA63FD38AE9C084341E40920ECDD73CD43440C75FACB4D084F3BF3CEB4CA3F34AF13F278939187B1E644057CC145B1D5778C0EA2A40A7EE181FC0E7A290E1166EF63F5F28A79F15819CBF44FF2F124791B9BF254F8FD6B4D498BFDEEEC22E3380D5BF753CB86584250C40F27C6EACE306EEBF37A91CD0B61AD23F3808A45CEDE481BF69AEE7C44A168E3F87DB6F0813450440C80C0C8D0BC3E3BFEA72680457B221C0940D15451E40A4BF9C8CAE50D7FC773FA2A28D85E31BD6BF41884251BEA4D9BF65F7F7DA44F86D3F196F8C73C9E55DBF5428E8227F0BB0BF6F7898AA72C3A53FDA0347E99B2B0E407E77B9BC592A30C0C4A5C39A08D4B23F927220B286A4C4BFB0B47980DA4306400341199B74F50B404AB1CA6B9D61B43FF4A7A12F376D2ABFB10E35D1BF8FF2BF8BBA6422B34002C0D30297A1EB1FE7BF5EC89095F0AFDA3F04A742C9FA1EE6BF7C19AC6FD9210C40448ED61A8233A9BFA3CDAC0D47F2813F5729BD0EDB4B803FD34E3BAE1FC1E33F61F1BEED7455F8BF7A96C8C430A2E1BFD1EE062FB7C3D2BF1B75D3DC2D51C23FA1167B70470CE6BF15FC770DA517DABFC8B9CBA9F3D0A2BFB829BC47183105C02436E486855AD63F611EECA57850F43F59598A901DE3BEBFEBF6A668BF3FA13F2BD3436E7C8DA73F4EF6408B8C6EA5BF973DF7793955B6BF214E253F38766ABF7A57ACCD53E8E4BF45131943388ED6BF0E99252141B081BF29F63A54DF6176BF2CAAE5E0980C07C0D470C04D140FF83F90A21D28A0A9D3BFE70F026730A8E23F9D826EBC52471640BFC85FBE858B17404819577AE1211FC076648B9D3BE615C0F725C75574B8FB3FD60CF2B3C3DACBBF1527AFB81CA408C05CCE57B2B2781E40A0DCBD38B9FCCF3FBA2EB05FDB0CC13F10CB9C0F01CE4ABF85E8E5E56506673F2288EA9A542B1FC09ED636D2588B28402E1380F0655802C09B140CBB738806403BCACDD12B24503F9EC5908A474C663F13A76CB76606893FF2DBD282962687BF57617A4FFFCC5B40E0BE6482454E61C0218B1803276C973F41966C562E1FAE3F6A29784A90F0E7BF3478287018850A402674DCAE3907FB3F094B4606E1DEF33F4A5591B07BF339408B2F2086278D3940423E1D6061AFCE3F345755CEA8B7D83F7FE8D76292C5074047C927DBE9621340923ACA5619E5983FCB2FF306F44672BF42751C9A86BAEE3FDBD0C2D0296C1CC0BEF7F5B71A7E214012ACBDCC37D12840E858396050A07ABF03CB330634EC50BFC45A9CBDD8B6FA3F43A8ADC76EBB0EC0507CC69E4BD50540D2D1579A9DCB03C0339BDFA3074FC83FE23EA80A6D9201C0246A2F77DA0DCB3F7E6FB55B7D0BD0BFA1828230545DEF3F28384D2EE913E5BF5D447DB8FEECF6BF5E5F41B4A87CE53F83C37AD8EE298C3FE0813844678B813FE2270821CD4E98BFBF3AD49C90B5953FB16C6AAB45F5BA3F827E3C9BA4D7A9BFE16CFCA3E79A11404471DD6AA33F08C03BD96200F601E13FF2159ACA78D2B33FCF3C2AA8D548F13F392B65B37E4F014093D7B2EE16702F40CE610A98787F43C0211241CA6DF7FFBF952F7291FD882040362F0F76860A9D3F884E2E688FA5ABBF86531AE512321F40E8395D3782791F4092C7A34F8F2786BF532D950FE13857BFAEE4B175942D45C0E2CCC271FE4C17C020D9D37CE9EE7E3FE9D8203EDD40B1BF534C6DE6E76CD23F5264A9BF5DA7D83FE8C7C024CEF8A73FEB29A3E07EC7BEBFE379CB54E89FF4BF8CDCA2C6DBD4FCBFA1054E1EF222B9BF8311D5671846C93F6D3522ECE48260BFB1E4B16D276D8F3FD1D90E2A6FB654BF65A24525745944BF72C0DBB5DCC614C00705598B1369CABF3B4297327DA3B13F22E4129512A7A8BF6FAE5BEC2C0DB83F0B966DF284E3A73F269C2ACB7A14D3BFDBE4C648CDFA96BF328E5EE3C16EC23F95307FEE3A8DD9BFA6C1F4A57DFBDBBF6EB0C6E649911C40A901F6F287283BC0E5257328AAA03AC094638E25392D1740F5B9C5C75EC0FE3F3D55CF73C918E43FD3CBFFE4716503C08E3131DDDB382B40F7734E6164A827C07F24152E2B6FBE3FE8AEC46CE3E4F9BF93239CE34C8A06405A2077ED09B31B40DB3F4C6DC140A53FF95F84FE1C37A23F91EECFEC9655E13F6FA1C71B8686E2BF01FA06C6119AC73F18613F61702062BFC87E4E566C2491BF1A739E97C88867BF1BAE29F4FE77C2BF0831ADB8FE3857BF8147BA52A0900DBF47638A57F2E974BFE87C0C0FFD43DEBF8762B13B994DF73F67B23587E99E1B405BC183995ECB22C0FF17302432B5B43F892D3264D5769A3FD3B90D7BE3EE89C0E7A0D56ADF94A0C0E5221095C1B4733F1D07A73765A36FBF0D44C2C84E155040C5CCF6CEB32223C0D3E137896989E73F45022184F0D1E5BF3AB5F8C52BBACF3FFDAD35E93CF6D3BF560EB2DD9A4BB2BF1BEB0B73E406813FC4E95B03F58ED3BF67775A1EA6F6B03FFF2F4DAE4BD8EDBF18E52CC65F68D5BF05B6290190F3F6BF185EDB2F5A14D63F375DC637916200C07511C21D71DEF13F80AB04C03E64A43FE1CB8762CBDBB0BFDAADFA95A336FC3F398707F70581FDBF501CD754ACCD0F408E5406FD54DA29C0BB0FB379E23DE7BFAC1F78259D60D73FB67CEFB655BAFB3FA202D4F9B677E4BF64FD2C8970E9B9BF55AD4B059E4FB63F4B3A17516DDECC3F2A147C313F75BE3FE0E7198F5CF6A43F52FA86A57784C1BF9CA06BCE6327C4BF87CE91AAECCD71BF0D3F7DB46AF922C0F9CBCC7B360024C073887C5FD2F955BFE596CEA4AE4661BF451AD14E27E6A5BF60477674B12B88BFB4A6709E584584C07FFDB04D6FED9C40597C6446026EF8BF365B5AB7FBA5A83FF9BE01CC101BC2BF633FBE8ED53B693FD8DECFB3431F16C05B015F67CE4711C025F82C7BF0E7933F4C33949CAE79BF3F92BC577E531BD7BF9F3A62B6F8A5D3BFAB2DEAB549431340114AEEF7452204C0B50393C88BB5C7BF0C8290017495EABFE6AC8B7B041A6AC0315EDBCE25A57540883E07459F9C20C004B05FD160EC124001E3F97551458ABF2CCD0605BF67A2BFFCF9E2A0DACA08404CCDF83A78AF2AC0BC84AAB42B58553FF782679AE96284BFAD0DB91DA32668C0592D906227C35240FB95E26828792E407F89D8F19F3B1CC070395F38175F0B404D48D3D284F7F03FEDE7390CF1813340233517DBDA6E29C03997E253D790C93FC45877843E6269BFCF15C6D5D05A1040C466609EED4AE2BFB267197FFC726FC04E4D426B3E986640445CB5AD5C1A43402D2A0A846A234B406B1A5B0F5EF6C0BF47A91FB8EE09BB3F51D1F9AE125A95BFD9E7D65534FBBDBF901D38395A7A9D3F6E4F1B4C80AFB23F20CD35AA465D2BC0DE4D194B2ADE6C4056CD9965356FE7BFCA54D169E029F63FAECB7EB17DBDB33F461626E7BA13B0BFA231D88437B034400F85D8C5323E2FC0E9D73A34E331A13FD4368F55A028873F13B1DAC45B48394064EB7452FF184740DCD6C0150E6334C061C53E3630DA25C04881F7ACAD1208C069EAE4501DD017C025A3DF3EBE7CEF3FDA02DD62A9D8AD3F0A566B678E70853F28ACB2D317707DBFD0F2F2BA76329E3FAA23D367B36AA3BFD9BC54DE2AF334405F73C9C18E8148405C17523A9EE50AC038620C143CC90240F06070059FF256BFA6B40531908C74BF7EB4855D68D00D40D97585D34191FE3FC0B22716FD42D4BF3A37FE3D0212F93F67E9FE9B9E3AFBBFDBD0631C7FB5F73F7B6644A8D514F73FA8F4675249D1DF3FF7049D31146CA33FFCED3167F06F883F50C0E07E48C74F4098F0954DA8714E4053822B3929C71540138552D106BE01C032E957105DF719C0B8B775D6C8ED3FC030B04012495404C0489998BC0613E33FABC2615A6FDC2340DF572128B16C2DC02000FA10BAF3E13FBF49C676F97AEB3F62036E61717E0EC0A470750EAA6FFFBF6731DF247E15FFBFD79FCDD49EA809C05649BF676663E23FCCBBA6723AC6F33FAD4DFD922339124089E332FF96E7134055FAC5446294C3BF7B5D574CE49880BFAA585F2AA7F3C4BFCDD47FC87D26D43F17D580469251E5BF9D422FCB25B1C6BF246EFF880419D43FEFE9D1846084B8BFC318392738EE3F40CD5C7F85DEC436C0DBBB609FA368913F01E16E0D38A38DBF5F592574BDEBA43FB738E3776F0CB43F036ADE327CD620403EF7FE7F19012140897F6EF4DDD6EEBF4FE078F198E4D13F6D43D53A5B6916404CC29B1612062DC0DE1E1E075407A03F91B704AD2E26A8BFE4906305A12DA33F68DAFDE167C5833F6CC5F699AE55B1BF6375C1FEEEC596BFF3623B879971B13F489BCC496401CEBF82DB924B79E98DBFD170F70D3808993FD9730C2037DDA1BF11900ECD828A9E3F0839DBAC9864B4BF307B9A3DE1960FC00A8F9355BA11E9BF6D47919F970CE43F0E4E9D5975FEC0BFF90E6DD8E5ED93BF929BF47681B291C08B03E7019DA596C0017A72968534DB3FE804BC796330EC3F01B25CEF262BE0BFBABA756F31DA0040B3847646B709E93FA156FB33A58DA1BFAAB3AD34DB560B409010E836FF5709C047A49E171EE2D0BF8A13808AE9BAAB3FC6117FDC4E303D40CC5387058CE55340B81C08D7B907EE3F1CC2759C060A13C0786AB6CD6387E23F1E910C55F065F3BF9705000D7DE8D5BFA247ADA29207D9BF53FABC213C590AC03B5C6B4CF17816C078DB5534DA5817401B09390C8090E73FF14E52DAF4871DC04C2ABD21E8882FC0CCF320971E04FEBFC98F457D732D00C0D7A7D2E7369F98BF55C1BCCE289AA4BF9E59DCABAADD3240A7FA6171932435C05275FA4EA8CC243FCEAF806DF58951BF4F1FF7341F86B83FCEA1846D9B6899BFB31BF5874A6DB33FB51405D03309AE3FDD311D9532EED1BF521F90E6B6AAC53F2511D3E2CE5F00C013D640C135262CC0C9B7B47F1338A0BF66D483AB1A38CEBF7963519B84377EBFE1ACB20C5EE571BFD16AD097E8060240187C539E553C72BFBCB2CF0F58C4D93FFFEDB5C218FBD33F5E11B60CF74D22C022EDE8E8F8CFCD3F257A1E0EF39020C0C9ACAA0A37DEF53F85DA7A2210D8A43FD9FCEB632EDBA43F4E7129E248593B40E4D819AE78810EC0EA06464370F3F63F7938BC737E4921C063296E7FE071C8BF2E28B2961EC9D7BF9BE08615A4E918C0C40B634105F431401A38BE9CEF4DE53F35F7539B95010A408FC294FC10733BC048FF045683BD41C0414D09A1FE082DBF2F0D90B878FF44BFCAA60580307BA9BF8DF289AEA3E7DC3F0B1717E2DAE2B53FB3AFC9434682C43F4148D9728DC8D7BF581DE9F10394FB3FADA15D7B5C30A5BFD9E240AB59929ABF874B49295D263CC0C9D7539A217C19403A1C617AB1F1E23F2EE01C8489DD16C096158C73A113D13F8FA8610F6452B13F902250AA3366C1BF1F0F1FB4DBF3C5BFABA846C05955D03FB795E262AD32DC3F51A1F98FDB751BC04C2772B56A0B3440696C2754524853407CA88B7B141584C00A1A6D179CBAECBF84A90C2A0E63D13FE34311FEE4A233C08CC7228C1D1B33405C68DE125F60E23F94B792AA26DBD93FD5A43688AA45F43F3E73D50BDD4E56C01151AF93203CB33F35191A589CA618C03E403BB08D7067BF9513D9C52F65603F0CB2EAFACAF8D1BF0F3D2369A0CAC2BFE0D6B207CECDB0BF9ED53E30AA37C23F41F0B6848951E33FB31B3C0A6B78E73F5574B4A1985662BFA30A11CC4234463F8793D4D8C43130C0E4A1E079F397264099005EE4DD50D13FE1184FC04B83E7BF861E111FF1C3E8BF9B9DF699EEBD154039B7BF1FB3EAACBFBF1AC3B2001780BF47826F4A6A099E3FA5AAD2692F29AE3FE1A4735927D3F5BF7905A57948A5EC3FF3A4B1D97F8BEE3F5F160450998DE63F33B461B71C0D1D40378020AEDEBBC9BF776EA91A91FFA93F410E15C006FCE1BF98A39426AB4FA6BF4AA8040B4652B03FF18F777C090485BFB391D085FDA8A6BF69545319F331B1BF9E5D8F3B670ACF3FE85444863817234094DEF4E12CADF63FC12203751036E6BFC963D062E203D13F5F70CF9790D250C0B40006AE0D074B40739CB39A085191BFFB101C57AD2889BF8720C01A5994E6BF151E32E9BB4CD93F6609D24A7193AA3F4B1C15793E81ACBF62B13D4A74AC03C040C30313BA4BEC3F3052C8300F362D4041D96E7A5A6128C078363EF0D0ADFABF848CC73ECF7AD0BFFD0B129F8D1E13402556A5B1B2690040B8CA1E76C3CEF43F022E159C1762F4BFC35ED9F5ECBE33405A88B44AF7921AC061E4960954FD853F16A534433CC67EBF41D32102DE895740D83E78C3C3A9524039B1BD54734C14C0443A721989DF2EC02E186151B744C13FF347E6F665D8C23F8B26B29E67791840778C6A323222EFBF4C876E0D893A08C03693E4523A42FFBFD6269A31EF78C13F60CEFD414802B8BFFF5C3C5D00B2C53FD834C56D6198B93F9B7616ABC808D9BF811259BBD8A7B4BF483F98AB1A2ED43FB9A7060884A4F73FF98CDC94EAFCCCBFBDC4AD62D4A80E406609FAEEA6BDBF3FF1D716AB2257F5BF86F286184697913F8CC615C7AAA0963FE9B7CCA0C577F63F36B1F34A7D7AE83FBB9142C6C98B2140585E94A0384E26C0D9FCBB0266EA8BBF9BC18A47646190BFE9D3CF8321737B3F7626BA26BABA69BF4D049C8A9B1B154039FA614C6C0DB1BF21F0D5F552ED0D40D4EDC9102414FD3FBA9A9A76C73996BF78983C67FA2BB53F1CA8954CE82AA93FE73EC42D13E7A8BFF40CF339F827AD3F4505261115E6DDBF052BC6BE7ABCC9BFDCB30F2DAEAAE03FA4183F1787BEC9BFEF60DA412583CFBF7347C1B3B686973FB4AE2B64E9347CBFB08CB5CD1B2CA93F321391EB11D3AC3F7DF977CD45084740F3F8FA88343E504037DA07B7DCC61F405E0516E0F0D341C07168F92A9AA1F43F318839D0E1EA02C011623F70D7851EBF164E4D7567EF83BF5C938428048BD0BFC06B9774B54BD3BF88F078D97CF6C23F096D54974EB5D2BFEAC95392F788FF3F2F2EEC8937CFEEBFCC73369273A10240BF0A3F3F9D6100C0C951240B14BDE73F60E7FAEF5503D33FADD524C68B781BC0AA61051EF80608C0CAF7B45BA195F0BF2C930557BD46CD3FE363C2804D160040694D4F9587F0F7BF3CF65793B4F601409AA1AF795BCAE0BF4C51346B066EDF3F39A51B05014FF23F81140EB29897B4BF166D904E8A18983FC9191FFF8887363F6E1C0BC4EA6D70BF74A4BDC8FF29D63FC943E1B8F571B83FA74032188565B2BF59D3B04AFE19CABFE2D559615B1DF13FE20451DCD57900400E0EFE18A703FE3F71B1CED47EB100C031D63E3B7F09B33FA1240C060113B73F8609AF237C1CD4BF3FBBFCD75537DC3F1F39E722E6E8D13FC058D433E55FB0BF75F5DED2A6309EBFDC41303AB6FEC03F31FB5233D3CE3040AAE3F327CCA5FF3F601155999758B3BF515A21036D9BA83FF4737DC079A287BF6FF131A022659E3F"> : tensor<20x20xcomplex<f64>>}> : () -> tensor<20x20xcomplex<f64>>
    "func.return"(%0) : (tensor<20x20xcomplex<f64>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

