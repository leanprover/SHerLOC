"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf64>
    %4 = "stablehlo.rsqrt"(%2) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    "func.return"(%4) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0C72A718AE9BCF3F4EFB76A51B2CEDBFB28D592DE5D7EEBF462AE874D0A9F4BFD00AF8CE08F8DB3F66BEAFA00823F3BF6758DCD5C316F6BFCC4D216DD28C02C01202C478457BEBBF9AE562F07E8FA9BFF8B293418528E2BFFC9BA91B35CB0D408CF372C3205EF4BF2066C66C14300BC0949C9100E99114C07A25AFCEB425FE3F54C8E9312282FD3FB44D046CCDA2F0BFEE39F49972DED93F5935765BB3BB01C0A4F2257FCDF30BC07589CF7E3054FCBFDAB0141686D81040387DBEA62B4801405C1B8E8FAC230CC0A0562E5D593C1DC0C24F23D5993305C019293EE5F9EA10408F0E861E5985F53F839668780A59F7BFB0DEB102EFD8F13F38D80BDC767CF5BF384E2C1388A80EC0F3DC6ED428F006C0FB20E32351B3F03FCEEACFB863E3D3BFE2EEF375263505407A66BBCC298204406072E912250B0AC08854C82B5B2801C0C040A5540E2F0240E56DD842074ED43FA13937CC23E0F33FCA5CC53C7C110540D06D3DE1FA50F6BF236F0E08A85B16C0B43D7A12C4F80F40A70904A3641001406E48CC9429F6EDBF8C35DCDAB856DD3F2D00D6CFD117064067BCADAFD7A3DB3FD1E1FB0B59F804407A28FD380A5F0340584285E1C2DC0140B450D42D9967E3BF991511A25EB9D13F4AAC84CB8D9BF23FE5C802F58E5B07407C823104F05EF9BF349034A34E5BC93F466E377E0F150B40E4C59427D9C5B1BFD99C25E032CED53FF682EBC4806CDB3F5CE09FFCBF68D2BFCA7869FCCAA3DDBFC8E8326D2D831F40096DAFAEBFA70840DADA62A92284FF3F221D24F23C8B00406A0863B12B2401C0C0C8F74B345F0E40DB146D91A1DE00C0E6C3BAD2E53CAC3F9E093882D4A815C01367C7FA9619F73F46671B8D9EC71240AC36C5EEB78C13403693556665D014400EB642D71BBBFEBFD2BFE7A2C36100C0BF30408EB6E60940E148C74093BBF83F96FF3C55BB99CE3F2C6706C62EC70A404D6C23B7828FF13FF7B50DB264170640A81457C183040C403735E40186670540D66C6683AE24EA3F9C0C472A9F140DC07ADB1CC6DC2CE63F0E7C7706CBEC19C0D2C6089D59D909C00D5C71AD9003FA3F921120BC6B25E7BFC503CCB64C2E224059315A36CB16FBBF0004176F5FE80C408CAB6E404292E83F632E88ED53C5F13F96BD9C765E21F73F8D417629F2CE034091CFDBAA82B1E2BF287265310B2D0AC0AAF0C3FB8592ED3F7A9F2586A2A3F4BF6CEA05D28E3F00407E6698B4BD9BF0BF1EF224CABD5F01C0682D2AA7000C1CC06AE9B3D9756AF23F75809FCEDC8C0AC024A06CF5DF3109C07E8A138DC4A013409F8D52284DF0E53FB6B29AA169430E40785EE5D450190E40C92B66A97FE703C0AC5FA7D026BDF83FA467EE70F9CF1CC0E7D426426315034016783684EFFB0140A03814F0C1260C40E6916C24D255D33F54156495EB60DD3FDC6C80C7BCE1ED3FD8631B7D164605C00A37B01C1485D73F970BA8A42DF206C0B3DFD88C4DA8F73FAE384173EFCED7BF3B3CC27A9F65144043610EA76AD2FD3F0DC392409D60D9BF854D52BDBA72DA3F50FC6CFF30AB14C0C2F14DDF71EAF83FD10423DF0AF9F5BFCA0BEE31BDF610C00E94CC5EFFFAE53FBEF7F79AC5C300C0DE2CAD6B6E80F2BF44573067E575EABFF3306A2B6528F9BF4C8CFF6B421805C06B96BF8854E7C53F93A2FBE3FA66E33F24763CAAC54EDCBFDE8A563BCBA11040DAE666CC41280640F9934D341C4806C0337C98AB8A7800C00E884934231EE4BF2A8BE5321A690EC026B4E513BA1B0EC01ECBF4F0B1F70B40BE1BDA463A8ADCBFFB286B962505E93F54058910EBB906C01B11FF08F07600C00E134814B512F5BF8786B692D16E04409C3CDFB39AA618C040530B188A93C93F4A912E2D97BAC7BFD7B6769ECFBB164008804065B7D712401F1BD177C01DD23F9E11612452DC1C403CB26604CB0504C0D6039574A17002C0C02EF05CCD5CF43FBCF2C3E21DBB15C0470FE1208FF008C0A6DEEF03D98E12C0E458AD86CC36F43F2FC052E1E74EF0BF1337178F77E4EF3FF2F024564A39F73F3EBE9569814D094090C3F574ECA81A40AB9541FA01B90F40AD903011E66DC1BF445944EF1571F73F969442B29A33FFBF8479FDEBF8FFCF3F884B22AD7A67F5BF5B91C44D6694DBBF3215539F6FABCCBF02C3F13852C60D409A585EA1A658FD3F507EB82BFB7802C05A8D3686F800CABF9A87CEDBD310FDBF630189AE902803C02C813DFA943E19C052C165F816ED12C0FAA3FECBD79B01C0B59544DE14DCF9BF987C8CF7B7ABD1BF0A13BE7F70DEF8BFA7F9DB7E7A410740F56D4BCB89EF1E40708CED577D390440E39434554077FCBFB0F2167F385CE03FCCAB666906DD0DC09802CA88884A19401A95BCD53D83E13F54DB2ACB46C209C03AD986F389530A4025C660B2CF5ED5BF6096700B9AC0F73F34ED3846FB490BC0B64617924AB0044018716D704DB607C032C54AADFC7DFE3F6812B3B37EB60940F85B54BE3DB6993F3159EBAB3D471540FAD9D228A1E7F53FAFB3CC1CE21BB6BF3ECD593447CB06C0FC31A2D1907B0340106CE72B958200403411C1E313FCF3BF809492ADD227E63F7076053F376513C04EFD73B5B8231440F24C58BFC3FE02C08CCE534EC66407C00C5227695ADB0FC04E25BA132FA8D2BFE4B9ACB9B8721040B0579D8ECBA5C03FC0F3049E44330440F225F66327A00AC00B8CCA0CC5280F401EB25D89058002C05C2E31283D57F6BFABD780E85713EEBF0291EC7BAF4597BF28C8BEC3C97AEE3F6BEB7678363F03C063BE21E8E02ADABF7048ECAAA91DFE3F663E5F313726E53FB312442295B3F33FB67B2E358841E03F67D0DD0CBD76E7BF585AA329A608F33F6EF3D28AC58C0A408208649C86101140F2472B89FDF211C08FFB2D032041F7BF78B45580263F01408AF77466483FC1BF9E9252C4231E0A4060DD0F1934A4B6BFF61A73A08FF30440C82EFE0D8EFE1340A844594A7ECADFBF26CF6C5F27DFF83F303A959075EA02407A4A93F95E7CF63F645D046F0EFB00C09C40D19DC0CE01C0FBC66406D50FBB3FAAE0C3C89389E8BF50B4F255A780F3BF5CA5FC455ED8F63F3D41DCB0D00A0940F4489ECE69D8DABF0D48C1B74A38054046ACCCA1DB4410C0106F388302D2FABFAB567AF4D127E6BF970DAAB1193CC53FF87E56C8B27802C0B4DF4E09491D104095A18FC8019102402E994F703A940E407777D7252C1E03C01D92E89FD925C6BF3D92FF2C0DA1F6BF3475385932EEEB3F2447CE25C7E1D8BF1CC781C3F93805C0E3BDBC8A7DCD07403653B46EC870FFBF28CAACF65FFD03C0DA19D9E8289C07401B62DE047DA9D33F6C5E5041CDE405C014D474E1FF7302C0EFB0C11D2BA01340E665218201B00F40FE30EE37C81C104064E8767CEB410C4019E54FCC3BD8FCBF2D70DC879532DBBF6164AC836096F9BFFEEE11F08DEAC13F5EAF196F849510C0FC629647EBB5EDBF45C1706232C001C04E95A4EFE3E2D63FD64E5651C7281240B336769B9636F3BF5AB6E93C504DE5BF11BDAA99DE6FF03F153CCB6B89DEFEBFECD0BB488F5F05C0057BA0A9E6D605C0B1076A225762F2BFF150AB1162F1B9BFF28F1C845F18FD3FA713B177D7A80D401CE52DEA343005408233121A1A2AE83FBAC7F98112A1E53FE09646B7C7DA1C40AC44BA5A4ACAD2BFA7B9883B7FBB07409A10B4FA1F7617405C58A75B4609F8BF8E3F321D372008403936051B6345F23F7B1B6331C8EC00405AD0D0A6B1F6DC3F523D378C5992FD3F92AC0F01C9D3F73FFE5010A1544F04C098F2B186B743ED3F1443719C87FF1240046A398E8FC816406E50C6E1B28BF43F2EA2341AD0DC0CC03DCC64660EBDEDBF4C8A67EB67BCF63FF97B7689BF4005C0970545CA95DBF83FA91208BBF783F73F50F7E61630F1E83FABFDEDD8B5011240FC441744AE6E15C035DE8460BD181140729391252AA1134045485A79C44213404C57D545664B1AC00F6A8D31AF240340E9E39FB9F0ACEDBFC562D94FF7020540DCAA7296EDBDFC3FC85252DDFEABFE3F6BA3E90813EB0E400D4A5DEAF53CD2BF5941E550D048F0BF209BD182D31D0840F68784EC23B11340CA221708B0D10F401A4BBD1B5F0E19C014877679C63D04C0FC408DC79909EBBFAB8C0776B1A5D53F6CBEB4E4CF7FD9BF29608C346B9EF8BFCE16BE4C9FC40B408A5AB4880A82B3BFABEB19F3CC3DF6BF3E2DB39998920E402C4FE5C311F4104068DA8941360406403816B6C84E84EB3F66C63FEE45A0CF3F464AF70D29A6F43F5EF4030A7B331640A13DFD793313D6BF342C54BE072C13C0BAE3027D9B281EC088513A09413A00C05885EBF44C4F09402741873FB3EFF83F005EFBB23F00F0BF9AFCDC0059AF02404FE67B3AB5451240227687C77EF306C002D0922B33BEF6BFB617DC6E785102C0E44AB60B951EFC3F5DF111BB6498D4BFDC1DF841D11106400C54A4166F00EF3FADCF97B063D700C0E88A3BF32E1418C0F3C9E4A7123BF0BF0BC06C9CD728C3BF8394CA86FCA7C63F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%1) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xCB7CB20D50190040000000000000F8FF000000000000F8FF000000000000F8FF5668D87B0334F83F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF4C250F1AF294E03F000000000000F8FF000000000000F8FF000000000000F8FF3FD75260F54FE73FBACC19CC3990E73F000000000000F8FF67B004A2972AF93F000000000000F8FF000000000000F8FF000000000000F8FF797B9AFFB22FDF3FFE2DD517A4C5E53F000000000000F8FF000000000000F8FF000000000000F8FF586B9C6DAC1EDF3F937D4F6B7F97EB3F000000000000F8FF2F649DE4704CEE3F000000000000F8FF000000000000F8FF000000000000F8FF2D00F3245D52EF3F000000000000F8FFBCCC16D466A7E33FEA19948972FCE33F000000000000F8FF000000000000F8FF8325A6FAA839E53F668812F6F167FC3FF4C843410DB6EC3F0F54807302B8E33F000000000000F8FF000000000000F8FF3EC4EF49CF01E03F46819C821CE9E53F000000000000F8FFAD67E132A2A1F73F21185D6C8C41E33F706F8427C458F83F57F87A83D0C3E33FDC164BDB7C90E43F328C120A566AE53F000000000000F8FFCEBD64745F67FE3F0C881CF556ACED3F7597D51C3FBAE23F000000000000F8FFE793A74863F901409045ADCE6164E13F000000000000F8FFF0B175544169FB3F60FDD4774871F83F000000000000F8FF000000000000F8FF814E45D142CDD63F3849C82F613AE23F8BF3FF19EACCE63F1BD69C8A9A40E63F000000000000F8FF33E2A3FA596CE03F000000000000F8FF72B90E3F53081140000000000000F8FF604F740ECDA1EA3F33C07F4A7289DD3FDF7862510EF3DC3F0A8DAE336F0EDC3F000000000000F8FF000000000000F8FFBE8197ADC2C8E13F3DD96ABCEABCE93F94D94CC19D5C0040504909E3997DE13F501EE88B868BEE3F25C06FFABB41E33FC4E3110C6E19E13FF3AE22B13890E33F8B495E53A2B3F13F000000000000F8FFE6FFA0586738F33F000000000000F8FF000000000000F8FF7169E0F69C18E93F000000000000F8FF23A549FE193AD53F000000000000F8FF5C04C12383D5E03F17FB260B5842F23F2A539DF8225DEE3FA4727031529DEA3F43C8273A1356E43F000000000000F8FF000000000000F8FF7EC1FDDACEA4F03F000000000000F8FFD57A48613174E63F000000000000F8FF000000000000F8FF000000000000F8FFF1704D01CAD3ED3F000000000000F8FF000000000000F8FFA0D6D47841E4DC3F15E47671DC52F33F9C1ED4BCE273E03F9BB20C23607FE03F000000000000F8FFC67B00CB18BCE93F000000000000F8FF568D61E505B8E43F74100953BE57E53FA996558F040FE13FCE8E6A3C0A1CFD3F2D295FD7879DF73F49052AA4B08EF03F000000000000F8FFE837C8C4AB64FA3F000000000000F8FF8A203050FE50EA3F000000000000F8FF21F5C8367F57DC3F22CFC9B86C70E73F000000000000F8FF2D97966FA7E3F83F000000000000F8FFA854C309AAA4E93F000000000000F8FF000000000000F8FF60FB9C3F284EF33F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF603A1918D15603408B3E6237478CF43F000000000000F8FFBBE0BDC8D862DF3FE5191298663AE33F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFB3B0E4D8581DE13F000000000000F8FF091915973B18F23F000000000000F8FF000000000000F8FF000000000000F8FFDB629446E605E43F000000000000F8FF16A2D60E96E50140000000000000F8FFD35F23F382D8DA3FB0DEB4C3D17CDD3F5F836C9CAD12FE3FBCFA77188AD3D73F000000000000F8FF000000000000F8FF0E6EC60AA25DEC3F000000000000F8FF000000000000F8FF000000000000F8FF5C9324A43F78EC3F000000000000F8FF84B58490E606F03F4A32E6AA998FEA3F8F11E68D49FEE13F881B965F4ECAD83FDF8D4D40DD11E03F000000000000F8FF43F2ECABEA6FEA3F000000000000F8FFC6EC00C501000040000000000000F8FF000000000000F8FF000000000000F8FFE1BB7A5C4E96E03FDC809760DBA0E73F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF7CFFE766BCC4E23F3C8A27876B03D73FC0DF450F3B20E43F000000000000F8FFCA90B6607D60F63F000000000000F8FF28C7F4DFC473D93F688E01D9CCA0F53F000000000000F8FF5B722074DAA3E13F000000000000F8FFC3FEB7DE8443EA3F000000000000F8FF8C76B52B1EE6E33F000000000000F8FF7E7C889A1D2EE73F0CE27BAE67D9E13F09B2CF343D3E1940AB1283B2A6BFDB3F9AF51A4B5359EB3F000000000000F8FF000000000000F8FF6736894D6981E43F1692FFFD6E46E63F000000000000F8FFC819E6CF963AF33F000000000000F8FF9D819D64B885DC3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF3B90C524A28FDF3F34D8F0B2D32E0640C47EA7405423E43F000000000000F8FF28691540E436E03F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF760A32D7E964F03F000000000000F8FF000000000000F8FF605174251253E73F1E508A1356AEF33F553E027D72D6EC3F4170024DD472F63F000000000000F8FF0CB338E7D156ED3F165127D2CC90E13F757582625CFCDE3F000000000000F8FF000000000000F8FFB0E389DF54CBE53F000000000000F8FF6393BEBAD9B5E13F000000000000F8FF13EC616512C6E33F7150338A2EA0DC3F000000000000F8FF2876B7757BAAE93F7F21AC357BCFE43FC8C68F3547FEEA3F000000000000F8FF000000000000F8FF2C5BB6A1FE9A0840000000000000F8FF000000000000F8FF47337542B6C7EA3FD1963C372F16E23F000000000000F8FFFCA17C52F2A5E33F000000000000F8FF000000000000F8FF000000000000F8FFC816672D2FA40340000000000000F8FFE09B4CEDDEE2DF3F5198B82C6101E53FABA7449D165EE03F000000000000F8FF000000000000F8FF000000000000F8FFEEC210974120F13F000000000000F8FF000000000000F8FFC2FB021B378DE23F000000000000F8FF000000000000F8FFCF44B3838EA0E23F71B45E8ED8DDFC3F000000000000F8FF000000000000F8FFC4BB3468B2E4DC3F6F9CD56C2514E03F9AAB20635EE3DF3FCFC38AB4CF06E13F000000000000F8FF000000000000F8FF000000000000F8FF7148861A16620540000000000000F8FF000000000000F8FF000000000000F8FFD4668B928DC1FA3F080FFCCF8A09DE3F000000000000F8FF000000000000F8FFB6A789FD5E92EF3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF4235E1A8E6BAE73F68643A958A9EE03F172CEDA4B1A9E33FC83E67FA8769F23FDD0AB8982076F33F74BDEBEE2CD4D73F000000000000F8FF893674263E94E23F2B7E4EAF136DDA3F000000000000F8FFD2A785624D6DE23F66165D7DFDF1ED3F8A2022651D00E63F9BBC1DA1AEC8F73F580AFBF6C289E73F3D5C782CF038EA3F000000000000F8FF9662DED028BBF03FEF4133B3DB5DDD3F2711B4F2FED0DA3FFCDB64F82F3DEC3F000000000000F8FF000000000000F8FF9B9C3A0829D8EA3F000000000000F8FF87585B0953ACE93F060B565A4B65EA3F7C81AE86771FF23FFE5EBB070F2ADE3F000000000000F8FFFBE731F9E9F4DE3FD71B3DB4F6E3DC3F3B7CA6936C2ADD3F000000000000F8FFFC3A6720BDAFE43F000000000000F8FF55EF2764D1BEE33F0F0E1CB71FE0E73FF1DA9A67B41CE73F5D16D5A00947E03F000000000000F8FF000000000000F8FF7BE4F204376EE23FD25D5E393CD8DC3FAECA769EA00BE03F000000000000F8FF000000000000F8FF000000000000F8FF2C22D497DA82FB3F000000000000F8FF000000000000F8FF53B794DD0E2DE13F000000000000F8FF000000000000F8FF7D4BE274865EE03FF2A219C75216DF3F5D26229F1D4AE33F30FA47C41541F13F02951CBD24180040CFDBF115122BEC3F6635DCF0722ADB3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF5E394B31A6FDE13F53937623F6A1E93F000000000000F8FF6A7BDE524CF0E43F9BBD1332BAF1DD3F000000000000F8FF000000000000F8FF000000000000F8FF2A63ACDA6623E83F000000000000F8FFB96310942A44E33F378FE5276D41F03F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF6D0DF3C6E5030340"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%0) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

