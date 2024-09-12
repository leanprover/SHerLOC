"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf64>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf64>
    %4 = "chlo.asin"(%2) : (tensor<20x20xf64>) -> tensor<20x20xf64>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf64>, tensor<20x20xf64>) -> ()
    "func.return"(%4) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xBFBBFC63C61C1440887C0394AB50D5BF3A9C062896D300C070B68FFA59B9DBBF6E898B355DCA09C00AB48E9B727711C00837FA5F0F1D1DC088E5F2DDFC1D05C0A0A0D599636BF6BF02DF997312E412C09E9484B1295F0740B25D1101C4C712C0A0906003F109E03FAD7132BF6B8CFBBF5E35E537BF4BFC3F80E2DD847C501140398D16A82EECF43F3BF747208E61F6BF9BE703D588790440646A59AE872B0440584A88FEC79EC93F9611F8E77A86ED3F20AB6735BB16EC3F278D042D1D7F04C0C8E289C0C2F3C03FEE7DAD86BBB8F2BF4F88ECA60A11F0BFA6386D157437D63F1C4F86F2C00501C0459B0BBC729AF7BF9B80CBF74EFCE6BF30EAFCC9EE11DA3F5C613CE6635FFCBF98901792A7550340B992AD917FDC08402901649A5991124066F13F221461D93FDCEDED66EDB3174013F2AA516C01F63F7190248D3664FEBFDC77D2CF3CE70A4072CAA0977275FB3F096156FC467D0E402A0B930A90B4CABF8C4FEFD9002ADEBFCC20974D3339C0BF08FA9E1F15540CC07AB10D131ADAAE3FFB5F6B8C3DB9F4BFF369EB70571B15C09C03C8613D93F43FA525539CD21A19405861E5D2E6A709C0E8D7C80B110D0C402A7DF2D024000A401CAC471F515DFD3FDAF03E7D53B7EFBF5852516CA7A710C092839953C844FF3F3109C4A1CAF0FC3F7DC0C5139E5FF4BF34EB0BD6A8A0FABFD859A70DD6FCEF3F66430A434F7709C003E6C4012ECC02C0EFE0BD1D000D01C0551324C6C86DD23FEEAADB8203E5E43F55048FA9F8F207C0E42C96B1298E0540CC5F30130AA610C027ADEE363BDEE93F5FAC53B94054E63F2A949E4404A513404012774A20AE1840EAA83A9F15EDEC3F2AC658704EC3F2BF2513BF4D21D7F63F380C02D6E55D06C010FD38871696F83F5160985CE2810FC080527E0F7180DD3F089A3B27FB8C0CC0946E1B214EE202405EF161BA07FE07C0D7B1F0338D4402403194EB93B166EEBFFFB6B58727BAEEBFC551BDD6D749234082F0DD0554CF06403A47CA7B99831C407A7158FC2EEC0BC0DB23F85E2E1AFCBFFCC45225D1DDF03FFD025E107C05A3BFB84F7B692E8B0EC0ACE15B2A1F36F5BF738CD490958614C0B0FA42F191150240D80C99C638D90DC07615E157A01E07C02EBD9E42B3D514C0C7DAD60E964002C050D48660BFE615407C42220F78220C400C81FF32D764F43F6723B1497C7F0140C010DF20D9D6E73FDE3881F6C1A904C006DC42E72C5FDABFC067F9084522E6BFFA6063B59FE7F7BFB1A22E396719FB3F5BB6303B0D2AF53F9A7E8F79E8F6F43F1EFDCCBEE2FE1040000AD00EEF8CED3FEE1A8F7C85C11A40DECB8599359DF23F8F2C7FD09684F03F822C5EEFC42507C0AE4F88A703D10CC034DB98B29B1BF3BF554C3E8220000DC098987407B217FFBFAF7D50689353F43FCAF800808129EDBF590D23D23D4A0C40F6578D3D3611E63F7EE594012F2805408B029AD3D0990E408856FCA9CF2AF03F8F519E02AA10EBBF7911CFDE73CEFB3FDCC009B872A909C03ACD455EFF7506400C73A72CACF3BFBF1C39FD99D98F0A40BB0DD89A286403C022B9CBC4946A0AC0E06D0AA23669FB3FD02484CC4B96FC3FFE79AB8892F4FF3F18128D8449C00BC028E84BD10DF5FFBF5E2A51378F7A13408C631C466F00EF3FCC9A8F9CCA4D97BF5BBF3142D6B203400C6D84B17EFEF03F08F3C1B58BA909C0D2174F485035EF3FA061078D48E5FBBF2E1013D5E61AFDBFDAA2B9C2E25107C02ABC57F2AB7308402A1F1A2831E1F9BFBBAB553B0120F63F2AC0E7CD546107C0886B73639D7FF63F3A7B52740929E03F645E648CDBD6D83FB8C45F068318FEBF4876483C8427E83F63F7CAC8769ED03F732C20326599BB3F0C834E56C6FE893FC31293B58B8A09C00E3878DCDDDB064040828F093785DA3F6A49F397A78DF13F1F87E05C46D9F63FC5B86973C239064010A6265015DFFFBFAEBE824DC150EDBF204ED62B8B370C402E2699176109F9BF9489098A2CA3F6BF9C69510E9CA5CABF7FA26593FB8DF9BF6E7CE1D132D312C0882E480F6F410F405C22D364B06901C0BE2EACD985B418C0F5BE509C739906C02E1C8E5FF1D2FDBF77AA61D1FC29E7BF0E6B6C136DC006406A18AD29B379074042EB07EAA98DF03F2C80A9215F2408C0506A3DD85F3EEA3F09371ABBAB9AF03F5ED2C061B9421240737A16557BBF0AC0A96D6A760E98FFBFB43E2A17036001C020B49016E1A7F9BFAF6EFA80D477F53F53134D2B33A203C000C9FEF52A3DF5BF9350A2AFE8EFC1BF2E62DAB7621BD3BFE6CAEF693DC4F93F5B619D91B27F1640B0BFAA4D5C4CFEBF1AB0C096922B0540F336E020887301C0002C1F30DD831040B9D74EED0BCC01404CD5E155E0B3F83F63CA11C0B4B1FE3F5A8BCE2AA418F2BF78057BFCC27E0B40CE0BF164E7401A40AA79F1A005EB00C09E1C8F7986F803C09069006ABD4501C09D0BAD311741E63F2E1429C0E3D70DC06228B08C351FF8BFBAB73C958959D13FB1CBDD0FCEAB0640A4431ABA42DAF6BF8B3079C93DE6144038F97758122F08408E9A0B07047821403633A8D2C431D4BFA4316313858DF9BF503D97F7DA1FFA3FF10F820BA456F93FECD9D29E5990024005A1DD6469F4D9BF36AD612C1B8B01C00AAFDD1BA2E91740E19E7C38B6CABFBFE80779BB607F0F40AE381CB63017DA3F0C14BA3935EEE7BF0E9ED22DD3FBEC3F17B7E695F50C0FC04077C693C35020C0BB8FE59E266CE9BFE8FEE2A7C403EEBF1249994851FDCB3F2EF0C2CF009211408BD6CB11303AFABF9FE86F5D691515401C74F09848000BC0096E5DDD3201E73FA2569FD5816BD0BF7A38BDB0783606C0A276A696DD26F13FB05E512DC32B7B3FF3E2DECC26A1FFBF8450EFF93D6464BFBB92EEFA2D8CF4BF446F7FED97EFF43FD48EDBD8B1D90340F6036F08B98EF33FF7ABD8989ABEF03FC08D6854D41007C09018477F5B33DD3F2059AD235AD61AC0B4CB2F3E5FD0ECBF68B66BE941B6014090CB6F979C78F1BF90A616B99F7401C042A1E63B52D1E8BFB0DDC488F7E00840C2BFD09B6D7408C05902256712B9034012BBE9EE37AEF83FAF583E4DFECB15C0D326A83D58252340F6AEBAB82E00F9BFAEFB149B4699DC3F70FC496D42790D4052DC97DDAB9D0D401DEC900CE18711C057E8B3B10375084062135DB27687C0BF27016049D8900FC05D21CFF55AE0F2BF05DE5B25F660CABF5224E845760CE6BFD456C2EB3429BFBFF8D1CBC5FF5FE23FE6FB968912050B40F7A5940F5F7A0340D6D22AE022BBF8BF5A5E41183AB9214072909DCEC54603C024C25DFB83DE1640579B2C33A8F212C0A4EE220393BE06C01923D885541101C02CCADEBBEDA0D1BF79770881AA3E03C0B9AAB1D7393F0E40CAA780320DE006C075E25FF3DB7E1240D62EA96797C70C404D5F7480201DFEBF775FB0AF9578C73FEE9BF07896E706408FF0FDAD1C2A1240E94BA5A9B49F0840E016FB37E95217C0625CA3E63EC3F03F1E7C9940A6310E40740925531FB10A4028BE8518A6BBE4BF76000CD11FA409404A04CC620F6CFA3F44EA206DF96CFC3F622955F3C98010C032A4EBAA9035DABFA8083B16FE6507C046DF1D507D0312C099B4FE28CCEAE83F5E7CFDB2B913D63F5874D564912506C09F8882205532F1BF3D5B0E2C078707C0BBFEA811CB07C9BFE252DDB2261A0FC0C67BF5DDCCDFFC3F2B9666041348F2BF14F239198D42EBBF3E9C7E1355D6F93F6B352FD98F51BE3F99E7A063CF8CE33F260A121E9F8F0240DFCDAE1B893610C07C92A61F2259BBBFE86F63F74F870540441E07CFFEC20CC082DF7E79E23DE5BF32265E88DE450740A163358AD248E0BF47FFA3E6B11204406A49EBA8E4A71D4087CFCEB10D8A81BF76C79C19809712404768DDE99F9AE8BF9A9669033DB2FF3FEE55333F9EA8FD3F5EEFBF118EFB05404EDD72DC7A7E17409909230B8B8706C0E1064B7A961C16406C009A5DCEDD0A40084D5CF0BA9F08C02A95388E3BDBF23F6C2C06E49C500640D02895FE3153D33FAB33B9FAF9B2DD3FA36C4E5ECB8B0140AFFAD66B010804C0028CBC0F458EC53F44754B93C5C90E40A32EA19EFA1406C026113FBC33190EC014CE278C941502C0017A83AB2A7903C032A745C1E2B901C045E57ECD551A04C0F227AEE58546E93FBCD9D153372600C0DCA869974D05EB3F3C934699F993D53FA2D4C972A5BB0DC0AB030F891747EB3FF7BA0403F042F6BF4C2F057605CEE8BF56C5E4FFE094D9BF38DB52012741DC3F725CFC5E84870640215DBA9683D30240009B55AF4DD806403A63F339105BCDBF9E3D42D19DD9F93F04DD6B44C54102C03E676759B6FDD93F12119565AB5CE8BF00947E9BC2861040CEFA53F309C2EB3F786DC0B66A62F13F806AC2A49F931140CA13A692ED9409C019A04A5DDCDA1140BA0408488758104043BB78B6FDFEE43FAEAFAE769FE7D73F24652C07FEF40E40EC64BB8B6E840D40617F04CCE24F10C01C18774D7C63F7BFA7162124B811DE3F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%1) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000000000000F8FF4C962935F1BAD5BF000000000000F8FF00A876BD78ACDCBF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF25F96323CECCE03F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF015DD86E62CBC93F6527D60B8CCCF23FD8094FF89F23F13F000000000000F8FF87BE526F8C00C13F000000000000F8FF000000000000F8FF06E77D685BB0D63F000000000000F8FF000000000000F8FF2CD18FF3B2A4E9BFF8F3B150D1D9DA3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF38DB402EAC18DA3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFA5C77B6C28E7CABF1D035E29EF68DFBF32C0CEC06644C0BF000000000000F8FFCC67D6ABE3DEAE3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF9AC2DEF3FBFFF6BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF4340BBCC23B0F83F000000000000F8FF000000000000F8FF000000000000F8FFFF00F25C8AB1D23F2586F7A772C4E63F000000000000F8FF000000000000F8FF000000000000F8FFD51F417CE01FEE3FB3083103AFB6E83F000000000000F8FF000000000000F8FFE424BD27F60FF23F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF02E9144D46A9DE3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFAE89807CB90DF4BF30353761D79AF4BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF796E70009B06A3BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF0B8878B35FE5EA3F000000000000F8FF835D7F477E2EDBBFCF96D5E63071E8BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFC3FE669FEFD4F23F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF211056C9F157F2BF000000000000F8FFC84EA4B59A59E83F000000000000F8FF000000000000F8FF000000000000F8FF386A1784E720F0BF000000000000F8FF000000000000F8FF000000000000F8FF4DABF6AD8704C0BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFDEBCDB372C20F53FD934D1784E4E97BF000000000000F8FF000000000000F8FF000000000000F8FF281ADE02F090F53F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF0F31DD43C9F0E03FBDDFD1176782D93F000000000000F8FF6D04974C515FEB3F5710C3DEC9CFD03FAED9DD7027A7BB3F4BFE0416F4FE893F000000000000F8FF000000000000F8FF9043A21C4458DB3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF9F7AE5103E88F2BF000000000000F8FF000000000000F8FF000000000000F8FFE861377FDED7CABF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF3E9C242FA1E6E9BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFE1AF600683C5EE3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFC38CE27F12FFC1BFC404C94B2067D3BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF16BE3880F99BE83F000000000000F8FF000000000000F8FF6A878813D291D13F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF91BA23DFA48BD4BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF66CB22E581B9DABF000000000000F8FF000000000000F8FFED1F6E2EC7DFBFBF000000000000F8FF0A37C62993DFDA3FD2C243AC7908EBBF631B8CDB4521F23F000000000000F8FF000000000000F8FF64B7FB341D61EDBF11B208B99977F3BF0A389652AF37CC3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF874A8D23BBABE93F5A091D5D0B9BD0BF000000000000F8FF000000000000F8FF73EC963CD02B7B3F000000000000F8FF5DC63B5B3F6464BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFF2BA5FB18752DE3F000000000000F8FF8C8D6D76ADEEF1BF000000000000F8FF000000000000F8FF000000000000F8FF7EBAEBCF2067ECBF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF68750D99D6A5DD3F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF52DE202B5093C0BF000000000000F8FF000000000000F8FF9923CC02B391CABFC2AC80460C53E8BF52BB66C1093DBFBF1FF56FD9A192E33F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFE2354B6709DCD1BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFD36D1CBFC69AC73F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF5C978E28FD8DE6BF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFCE595ABCDA00DBBF000000000000F8FF000000000000F8FFD59BE0849B8FEC3F279FCCBA468AD63F000000000000F8FF000000000000F8FF000000000000F8FF0877E2F15931C9BF000000000000F8FF000000000000F8FF000000000000F8FFEAF749AA1650F0BF000000000000F8FFE5DBC14FD263BE3F3F845C592808E53F000000000000F8FF000000000000F8FF6A1918CE8466BBBF000000000000F8FF000000000000F8FFBEE7F5E5853AE7BF000000000000F8FF31191952A915E1BF000000000000F8FF000000000000F8FFFE76F3BE1B8A81BF000000000000F8FFF9B453F70911ECBF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FFC14BC1ACA2A1D33F10A351A945E2DE3F000000000000F8FF000000000000F8FF0A9044FEB0A8C53F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF8BEDA4E57523ED3F000000000000F8FF26A68E714716F03F6E3E27986002D63F000000000000F8FF1103885A6D54F03F000000000000F8FFF8795527E761ECBF46121E272451DABFDD6136D97A43DD3F000000000000F8FF000000000000F8FF000000000000F8FF728FF9618D9ECDBF000000000000F8FF000000000000F8FF06476905AFC3DA3F634F20FFD1B0EBBF000000000000F8FF52F79088E8CCF03F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF859A4AEFCEE6E63FBE8BDE51AB7FD83F000000000000F8FF000000000000F8FF000000000000F8FF000000000000F8FF32302F9F694DDF3F"> : tensor<20x20xf64>}> : () -> tensor<20x20xf64>
    "func.return"(%0) : (tensor<20x20xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

