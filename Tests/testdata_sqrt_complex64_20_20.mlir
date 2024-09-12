"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xcomplex<f32>>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xcomplex<f32>>
    %4 = "stablehlo.sqrt"(%2) : (tensor<20x20xcomplex<f32>>) -> tensor<20x20xcomplex<f32>>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_almost_eq", has_side_effect = true}> : (tensor<20x20xcomplex<f32>>, tensor<20x20xcomplex<f32>>) -> ()
    "func.return"(%4) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x31453F40AC26F03FA315A63F6F92CABEC719E9C0010D6D3FC92514BF9E1AF2BE9BD172BF2ABCC7C09BE510C07C11A04046B854C00D2500C029EC24BE955B2240273550BFD0745FC0FD9EB0C0E29B6540C84792BFC14121BF0FE79C3F4CB4DD3F77C4B4BF787E934033EA5FC0C1F4923FF42D97C02F337D3E3EB0E8BF711B13409AF9A540B90BC4BE2485D63F884013C0B448863D3B2746BDD22E3740094F8F3FB1318A3BFC81F1BF389F3DC0C8A72F3F254B49C0D5FD97C0639098C09222013CC213D33FB609F5BF218ED8BE94E41FBF2C39AEBE7D0249BF6DF09B40A5B45640C7BE2A400A04CABF918600BE5C5FCABFB034E63D879E723FB890233FC6A2383E0F79C5BD5A8A794011480C408A95A8BE8451A8C0104395C0E9D6A53F3D8B1C40DDDEC43FE35F72BFC3004340398224C0EC42E13E263FFFBEE3838AC0C5B71D40E5E157BF10ABED3FD14EA73FE8F94FC0D0A5D2BF02FAAE40D3D7F8BF71D3C43FB9310E4098AA30BE11C4E5BEFDE8B63E78F7BE3F577AFD3E933F08BF6AC501BFD6A271C0D4041BBD88E45FBFC5F6F8BE4130CFBF613C3140F8AF35406360074035EAD4BFFDB4ADC01FA9E13FDFB8173F6C5D81C07E542540569D95BF5B22FEBF0D92B3C02C190BBFE15E56BF58DCED3FE0EF9FBF92CF9EC0210106406EEC714049EA973F7A3BDB3F63CAFF3E5CE3B24055230640AAD1C1BFE34B0EC0CC3527406F9F9D401CD61E406ADC06C1D95823BF70D53FC0967B713FC0FDBEC0060C1AC0D17983C0D8420E407CB1A8BEEBE99FBF07DF9440122384C00E49EF3F326FAEC0C62DDBBF165CB83F7C93A940C0D0A83F610CE1BFDA13AB3F1EB20940B11085C0C47A76402CF121C0EDAC533FDA4C63C0B4C24440D6DD0D405F29C44092AD084185E938BE8DBE9140D641C1C088644E40C758A73F2DBA7BC00E6CB7BFF4A5C9BDCD1D873F0136B540DAD1AD40732BDEBF2E47343F14AA0EC115943D4000F442BF65080640830980BD72992E40CEAE82405DCEFE402E28B4C058FDB33F7B0FAC3BFBA31C406D114A408A941EC05B4D723F4B3302BE272F72C055CC39C02161CE3D9F1E80405B2AC0BE93F2A23F686FDC3F91ACC7C0AB6E48BF1EEB603F6031BC401D138A402536AABF488A7FBE05205FBFD05F8440987FD140AB065D400EA50340D6EB93BFE15A37C0C1F29B4020A4F13E6E4201BFCE81AB3FCEC8624003D9C8C0D75816401FF8304065BB883F4B559940BE338DBF62A4BB3D166120BFA81665BF139F364003F795C085863BC021E281BF2C4D1340CCDB2E40CE7EE0BE912CA53F820295C071732DC0A57E8B3DEFC505406643E0C0F6B914406F0808409F2A5FBF3C5EC53F53C5903FBBE83F4065D1AF3F6B0997C0226F72C0A3C39F40007B8BC02E94F1BFA659E1BE62F41BBC2E970CC0E434B63FE2FC44404FC01EC01759AD402A1AA3C00EEE98401B558040335A6EC087C2573F3D3F27BFA94F25C0CB1692C0DBF083C0AD346D40F67CF53F86F9E73FD58A9EBDD038BE3F04B2E6BD3568C73F93678AC08E9983C05A082FBF58081C3DC52765C0F77DD840974E46C0BD58FBC0A26F19C012C2A7BFF6F313C0FDF190BEC40F523B35F5F03FB97C03C0D8B844C0590E6440CED8A63FDB9AACBF9886713F3BB30DC12CAC563FAFFD9B40DE57A0401445413F979475C097EE4C40612B4540EF0D3340D93027403F17FB3FF0D46D403B915C401F8822C058C280BF0BA358C0E956893F414302BE4412FDC0569499BF9446FE3FE49B16C00E3D82C0265FD8C094658A3E845C14403D7B32BF4B7647C0D15CE140D83D3340F4880440C8FC92C0520D8940BEB827403CE3D8BFDB13764006FAAE3FD810D040CF25383E75CA03C1E23896C0C243EE3F070A41C075D3AEC028093DBF4B281FBE889667BFCE9DB240EAA3153F76B7B8BF9CD0DE40FC3960BFAECBDABF8437293FFB652F40F1DC8040981F91C04DC20C40E04BBE3FFE2B9EC0AE18D0406563973F3852C03FDF1EB83D4497B1C078BBA6BFBF0AF7BF241D0340B6719ABF5925BF3CF8F87C4037ED933F873CD93F3E0CC5BF6410CC3ED1D500BFC051BDC0A40F27BFF6156E40627E8A407D8ABA3DC8B807C0FE2B3840764EF0BF50AEA6BF02443C402C338D3EC7BC8140A03D90C00165DDBFB02F21C0AC0018407D64EEBFA7A16940B9855DC0FC2DBBBFFC2E4E4032CAC0BFA0C91340173459BF07DCAAC0142D1740C34A683F8FD81FC026CC8640FA418C3F40773240E3398C3F5F01063FC1F3F5BF0AAF1FBF35F1E7BE5089DABF13736B403CEFA7BDF7D918402107DA3F63AF19C06DC982403D3D86BFB88194C0BE82D8BEEBBB2CC01066333F08921AC0E49DDABF6C1311C002B685BE548BE03FAA75FFC0035974BFA62B3940803313C0D6262BC0D7BC5540A47C3740961354C008D8874086A5B8BF1D11363FBB92A73DA7E311C036402E404C6FB73F3C4E0F3F5AA5A9BF3D1C3A4161C008BE06A262C0A0D4FBC0B5D4BCBFC7E317C0990E8CBFFE3C11BFBAF4B23FFE2D5C405D048F40920CB0C06725BB3E00C40DC0ECDFEBBF78BECA40AD17764054EBA9400FD1CDC0814DCE40FAEA3C4061BF19BFF64A5A4088C9353F2F30ECBEA30BE93FA842E03FF46B1ABFEFB594C0C64294BF5DF0ADBF21F62F4097253EC0FA56683FC70A554040BCF53E138EA7BFE63E99C089FB6B40BD8CA8C0E6D8343E1DECB33E60E837BFBEA34040C01041C04B08A33FCD53E8BE12CE8B3FBB53F53FB71382C0DD166D40C99AF7400F3D2240DF936C3FB2C62CBF1FB8C1C0797CEE3FA3E7E63E1295743E1D322A40546015BF51C59B40439B68C0BFF99F405B0EE03FFE1217402A6A8DC0B79F10C00962393EC1EE0E40955E3EC0B355493F923260BFBE4FDABF910B4A4092666F406ABA75400D4FD3BF3C5EBC3FCF964BC0CE2682C08D46F940529FE03FD99337408F2206C10DA776403D5D4CC04BC5A73FACDB17C00DC2C6408E730BC014E9DCBF5A619BBDCC47D23FD07FDBBECD0F28BECFBC663FF27C7C4005B38840882E52BF37849DC03673AFBFCE3B823DD96C45C0F9E7123FB34A20BF951AC240AF6DCB3F30C26FBE22DE58C0DF5E5B3FFE417740E4E2E4BF7F060540AAB69E3FFF2D863EAB2953C00B7DC7C043569FBEB77EC5405DF1FA3F97100ABF206E8CBD1C66544061D64E404A7B64405C88E7BF61A20FC07F04C4C06A01C9BF77091EC0D238CDBF7DCDED3F2CEFD63FA6EF6F4089275A401D002340DFE29EBFD681BA3FD97015C04E05133F8D3EAD405B814EBFA06D2CBF7F8A8140FCBD58C0965E96C0EBAD08BF56C5EDBFED4BBE3F84E7B33F75B7BB3F973587C0A3C5ABC04FEFFFBE841E04C0B78BFB3F56AE1CC00CE4AE3F89852640B0ACFCBCD8F6FB3EC1230FC0992B1B40862B7B40D16932409C5385C0A4BB3A409E261340473F1EC0E58C19C0C4020DC0B93774BE07DCC8BFE82491C0A625A3C04E32613F404C6DC0CF967EC00A1806BFB93684C0E8B458BF20614EC0175631BF5D2D10BF08762A4093EA73406074AC3F05D945C02256C0BEBA0E4B3F26FF59C04C6F303FDCC8F93F6D723E406AFD79C0FC0453400E0B4440C49CB94051A90CC07BA7443F700A99BFAF87F53FA9762EBFE72920C004B345C067C80B3FC2F50A41A1CB4B40FCA8273EDB0DFFBE55AD1B3FCBB6683F786A5CBFDFBC35C07919C93F6A7728BF508E8C40E62085401ECFACC05E342B3F60CD5B3D53A53A406DC151C00E1EE6BF8092224057218ABF381E37C0CA3B21BFA846BBC0764315C04EB94D3FA247ADBFC2273D4026E6C33F8AA96FC0E52EA9BE5BBAA6C0720A9D40C8D9C73F64E00FC02FEC813D53B23B3C8D06C1C0854D2D408D1B5B40A3CFABBF8D60C5BF8463EE4031685BC0815D10403F7C29407DB8D03F2BDDF83FE2F60EC0403F233F8C2BA33ECB6B8A3CACA1F03F15B481BF63708CBFD8D01240A1B8803FB9838E3D49DF1E40AF7B03C005DE19C0DB3381C055468B3E1D8089C09C434DC0D3009B3F50806B3F84277A40F06F43C02AF21EBF94531A3F9D7EA0BF2A62C43F475955C0446F1D40107F85C0BC6302BF9633943E573ACA3F25EBF4BF993BB44024213840EB9A8DC0BF81B5C028A98BBE0C305FC02D1715BFCE0182C08B8E85BF02D849C0627E22C0AB1B9DBFC546EF3FE32D3ABEA27C6EC0F42795BE9FD7BC40B452CA40805B374028EE55C0B0EF4E3F533E6640BD3755C02F715240FCEA1CC0226803408663BE405EC2B2BE7FD290C0CF2547407A2BC1BEAB04CB3F0A4E7A4080A8CB3FB49D48BE1D48B23E335504C0ADAE5A40D0CF1440760402C0F43E2B3E2A771540077D5040207934BFC1D28CBFD8453BBD65AB94BE614E10402D94B64044B9AFBF458D4D3F95E1CCBFC71272BFF10F423FD2D63D3F153423BE15DAC940A5C78C3F8FC889400050654006AA4D402A3E87BE7E3A1C40790D6E3FEDB77ABFF50305C02503B43FE2B083BF6E44294071E307C0F23B553FA35C883FCECC1DC01289D63F43530EC00F187FC0"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%1) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xcomplex<f32>>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xEF0FE73FEC08053F9F73933F54D92FBEFC4E2F3EAC142D4074AA943E0E7350BFDFA3D13F89E7F3BFAE94A23F3B0BFC3FF509073F2BEEF2BF5AA88B3F27CE943FF8AE963F5DD1BDBF35AE3A3F366F1D40A7C3913EC29A8DBFA49AA53F845C2B3F4619A73F0FF7E13F7A1C9B3E618AF23F16E7683DC7270B4086F13E3F6D3AC53FE7DA1140E90BACBD567AC03F3ED943BF0DD58A3E4CB1B6BDA982DC3F915FA63E25ED783FDD5E78BF98C24A3E3BC7DD3F0B93903F1B9106C0188EEC3A4EBE0B40E2F3B83F499529BF3E67D03E106944BF4BF7013F13F845BF23A6144074E1383F165ED93F78EBEDBE80C35A3FA1D16CBF40F73A3FEE19263FDB9D4E3FFEC3E43D6186B03FBCF1B43F1E09BE3F351AE3BD64D9703FC5A61EC0BD73B63FE4A55B3FE183A53F4F70BBBECD08F03F66732FBF3E463E3F29B5ABBE6E34123F57140A40B0F7453F58AB993F797FC63FAB1C86BFBB79B63FC47AF53F895B043F8A58BE3F97EFBE3F32DE6CBD79D47F3E1C08373F466D9E3FC0CB4C3EA407A33E67C64BBFFA8E1F3CAEB7F8BF4D8E803E2FE377BF812F643F01D7C63F8899E43F5A9A173F7656B53F383AF5BF7E46AC3F4575613EF8461D3FC48D0640ACDB403F43ABA8BF3F9AEA3D08C917C0295E463FC27B993F2603B23FE562E4BFFC3DE53FB214873F66AFA33FF26F2B3FB1C3DF3FA6A8CC3FCFD5C33F475DFDBEA1E5463F4637D73F6F3B12405F080B3FC6E4E03DBAF039C015D3893EE944E03F4E6DF73E73621FC08BCB073F481806403D99313F148266BFE138154072B062BF7F0BFA3FB696B2BFC32F033FACE1B33FB0721440DA8F913EFEF3F23EBE43B43F7299EC3FEAF98FBFA29F0340DC7B1DBF8A6EBF3FB5FB97BF5A23ED3F8C26193FCFA038408683BD3F6F5EBD3F9B06C53FE19F223F117322401757D33FF57598BF1E5A283DD35099BF2F4CEC3FFF51C43FBF0117401F52BCBE1A920C400BE801C07015DE3FE1B960BEB541B93F24EEB0BCCC29FA3F5CBB853FD0783E40EA2272BF04C9973F1A19113BE5BAE53FC12C613F9749973E4501CD3F5F22AD3FA30CB3BFE936F23C1520DA3F3933004095DDBFBDFA3FA73F2FB4283FB729203EC42E20C0A673EC3F5DC0CB3FC3790640D903A2BE6FBB123FF2A342BFD7FC1B40E6E8AB3F966DF73F9934083FA5B67B3F387ABABF2C730D401AAADA3D6E482E3F2AEC7B3FDA6214400D41ADBF3E49DD3F22BB4C3F5B49DD3FFF62B13F4780323D068E863F591DF73E62536DBF87B10240D0DF92BF7497953ECB45DEBFA95CDB3F4B104C3F2A0B2E3F31F4723FD3CB1A3FED6C0FC0FF00853F86BD803FAEC5DD3E33AE2B40D758BE3FD41196BE8037A83F9E51DC3E9F1EE33FC42CC63E12ED503F688714C04E441A406E7667BFC8F1223EBF05B1BF79DB853F477086BF6799C63F40EC7D3FA9C8A83F177603404FDD783F94501D40CB5C0B4045EB5ABFBF1E7A3FC02DABBEA1AE933F093DFDBF8FC8573F1BB50C40F633C13FD6AF193FB100573F757E623F3EEC593FBF3F6A3FC008683F343111C0EEA0BC3C0BC3533F12AFB63F2AB017400041D13F7DBF19C0A57FD13E96FECCBF654EBE3D09FBC2BF9C93783F5927783FD901683FD810D9BF0482F53F34FAAD3E1A65C53ED39D9C3F531C103E65AC3E40CDE21B40F1A8833FDC80C33F57C9A0BF9F43FA3F3CB0493FCEF8E83F80B7373F12B5E03FDF79873F8499FB3FD15F25BFDBD48F3FB1CAC0BFCAD1843F6F127BBD25D3593EBE7E34C06EB8CB3F43423DBF84F2B03FCC841CC0240F923F8104823F96018F3F1F88B2BFDE0B2D402095043FB454F13F03EC9BBFAA090A407F861B3F6A518F3FC2C6DB3FF316004092EBCF3F9C4E0340BB7800C0F1E2D73EA2440D404986A23F34B009C04535B83DAD2F5DBFA342C53FEACDE73F8673843F488232BF433629409A9D29BE284F7E3E8657AA3FA2B5F93FF91B843FEEA8003F7E090C40705FE93FFA81ADBFD9DF2340997E6C3EF2F79C3F2224163D74998C3E80CA17C09E402A3F3926C53F77012E3CBB9B8C3FD49000409346933E41C2B43FE1880BBFC7A6383FD99DB2BEC528093EACE71BC0D7F20A406F297F3F27AA863F3C0181BF486CE33F4F4007BFCF7C7A3FA968C03F318BBC3F8027B03F97F2CC3E9C450AC069CE2F3F9D56DD3F1B55873F1DF9DC3F260CC53E0D2EF3BF15A3EB3F3873D1BE60A9C53F9FA78CBE6EC9FF3E534D1740DBE2AA3F01766FBF6F7104406E8D873E71AFD93F40E8A43E5D898F3F7A545BBF57E08A3EF9C655BFF0B38A3F0148D93FA17E893FDD4B8E3FC71EC33FFBA249BFFC6D024018BD83BEAACCC83DED030AC0559C583E7705D43F8B60053FA1CDD1BF7953B13DCA08C1BF50970E40A751E5BFA3EA823F980BB53F4289483F097DDABFE0D2FB3FA5873A3F8D45823F9A7905407069933E68179E3F59208B3FD83886BF4EF7D93F7271D73EAEE77F3F7CB529BF7A475A402B62A0BCA52BCC3F20E11DC021E94F3F8705BBBFBDA7863E5A0F8ABFF2A1CC3F91B9893F5FD91940557892BF163D923F752B78BF4060C53F317B034072F311403505953F6907943F9E633240130ADD3FA41032BEDCA8ED3FE0D0433E3179573F42708A3F6AD9AB3FDD0966BE8882883EBC040BC000936C3FFF68BE3F3648853EC621DF3F6A23EA3F0457063E010DAD3F9CB3E2BFFDF00F400FE295BFAAA7083FF086A83EB7848B3F51BCB03F80D8B73EA104E33FC88D1A3FFD91673F382AE53F184F91BF539B1E40E0D2C73FF20CCF3FF840923EA5A4D23F936EEBBF74F9AF3F87F4273E275E9A3FD81F8D3F5F1ABC3F4AFFD33F4E6D903FB6C70D407FFEC33FCC53453F108B053FE59E0AC05BD98C3FBDE4813F9B7E673EC5A5DE3F9ABC383F614397BFC16900406AA16E3F9E260040600FD3BE75E9C93F2F1081BFD6A5C43F87412240F1BFCC3F3A87653F993E263F02E93D408515B83E3F50E93F2A2ABB3FB6ED0740794C0C3F958BC9BFF5B8623F4B6F6D3FB140FC3DEF8E2ABF2255C93FDB85A03FCBE10440B6754ABE27AD9C3E6B560FC0149DA03F36569DBF69F3573FCC04BEBECEF21E40C2D1A33E05F5A03F6876ACBF1197C63F555E9F3F33C7303FC6A3C03F1F518F3FBBADEF3DA95CAF3F3F9C11C0BF41DB3F5E97E63F15E1B43F866743BEC430A33FD898A63FF8600040B6CE633FF4903B3F180AC4BFE121A13EA9AC1FC0007AF93E9296D2BFD208BD3F9389113F825E064056D04F3F48F0D13F1ABFC1BE9FB2B93F4C044EBF900BDE3F95BCC73FE6EAB23E02B776BFC0330A40FBBD48BF97CDFB3D26F50AC0B1BE023F2B4DBA3FD5CAA73F00330F3FAF1C923FAA7A16C0DEB1663F8C9C92BF8E7DCC3FD22544BF31CCBB3F46FF623FBB24F63EE306033F348C3A3FD5F0D43FDBC6054083B52A3F6AA12D3FBBA8094086A0D73F69E03BBFB6B1273FF743D7BF353A523FAD9774BF6CE9883F108718C0F004C43F6DF49ABF012D063E0BD8FFBF6520543E7EC302C01D64443E6A29E7BFD7FE843FEF0EA43F3DA5FD3F300EAE3E0E62DA3D8677E1BF5E7ABB3F26D694BF1955963FA2AD543F262CFE3F8FC97BBFF5BDFC3FFE91463FCCC91C401EABE5BED7EB853F304612BFF7F8B33FEB2978BEA8B25B3FCE5DE6BFA59F0940303E01408D7CE43F5FD93B3D0E46C23EE2234D3F3A0E853F7D0AD4BE4395E63E3644DF3F5A0CB03F9263CC3F14F015405F8693BF3985513F0748063DE899F43FD1875BBF0B854F3F548DC83F0BA47E3F7318B8BF5082CF3FF109E7BFADCE843EF246C63F9253793FE137C23F0DB9D53F08898FBF2327C83FAE3FD5BFCB840F40A43DB23E9249AD3CD1EFBF3F4981DE3F3515DEBF85B5F03FC106693FE9AD173F5190A6BFAC0733403CDE1CBF60BDD83F9B2F483F04C9B83F18632C3FDB4C583EA235C13FE191103FA61C753C0E5FB53F8712B7BE0BA7593FCBAEAC3FD26F803FA4070E3D2E2DD83F65B41BBF6F1D893F283AF1BF82A5C13F6CC6B5BF2232AA3EE825E93FE412C93F6A3E9F3F8CFD343EC8D1E0BF229F7F3F6FBB20BFCE78CE3F634384BFBB97F43FEAB88BBF5B46483E09703D3FAB66B63F26DF2BBF297C1C40C39C163FC29B963F97421AC0648BA23F4FC1AFBF2FDBA93FEBF0C3BF479F883F011BBDBF83F4BF3EC186D1BF9136AF3FDC0288BD0A6B9A3DD746F7BF5F9F2C40E405963F2FFFF33F537460BFF0E7BF3F2192993F4C47523F7E190040D95D1C3FE322D73FE02C1C40768292BDBBF6313F8B3C0F40F1A14A3F5E3E803FD50F01408BFBC93E5D5EA33E3BAF0B3F4F3A7B3F1BD6DE3F3F93D23F82101EBFC65A8F3FD474853FEF5BE83FC0D546BE0981B23C9F4986BF6FE97E3F14EC903FADF51940241892BEC9C3913F65E933BFC3D7BA3E20F2843FD3B45D3FA672BCBD1D582140095F5F3E7E710E40990F4E3F3AA6E53FD3C296BDF675CB3F20C3953E26D54F3FD9D7A3BFFD9BA03FF7E7D1BEEA5FDE3FA86F1CBFF3CB853F3D74023F72FF013F083DD33F2C778A3F3CD0EBBF"> : tensor<20x20xcomplex<f32>>}> : () -> tensor<20x20xcomplex<f32>>
    "func.return"(%0) : (tensor<20x20xcomplex<f32>>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

