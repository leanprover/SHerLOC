"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x30xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4 = "func.call"() <{callee = @inputs}> : () -> tensor<20x30xbf16>
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<20x30xbf16>
    %6 = "func.call"(%4) <{callee = @integer_pow}> : (tensor<20x30xbf16>) -> tensor<20x30xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x30xbf16>, tensor<20x30xbf16>) -> ()
    "func.return"(%6) : (tensor<20x30xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x30xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x5B3EF5BFEE4082BF914021BF143FD73E0AC159C0F6BFBD3F6D3F2BC08F3E1F40EB3F23C0A8BE333C89402EC0E3C0413E2BC065BFD64009BF52BE4240C5C03340FABE853F5740CB3FE2C01A40B13FC43FF33F07BFD6BE1C409A3F0640424034409AC01D3FBDC05CC0823FAEC032C097BFF43F3EC086C00CC0B7BF07401A401C4016C0ADBF03BF4BBDBFBFD2BF0141624024BF9ABFD9C004C0C2BEEEBF384055406A4084408F3FA5C054BE9AC08BC000405BC0743EA3BF36C01E3F60406E4044409AC0E9BD1BC06E405540CBBF47BFC0BF253E3540703E8BC02740D93FA140294088C0EDBF76C093C078C08A40B1BFA9409F402B3F2640E240123F96403D40274067408A3FC840C8C0174080BF2BC05FC072BDDB3F15C09A40B63FC840063FB73FFF3F62C00940CFC0C5BE38BF663F0E408740D1BFAEBF35C029C008C0CFBF2940AEC0843E4B3FABBFA04082C0F5BFF4BFB0BF04C036C0D13F2D40B6C088C084C07640AD40B9C0C03FABC043C084C082C02A405BBF9240164085C0EDBEB23F69C0F53E884046C05DC06AC0453EB6BF02400040D03FED3DA6408C403B40563E04C02ABD5BC086C02BBF5A4058BFB7BF2840C4C0B3BFE83F84BF1440A53F96BFD2404340E8BF23408F4089BF2AC0233FA93FC93FA640AEBF39408F40A04092405E3FBA3D483FE3BF92C0C2C0F13F203E0CC0003FA1BF854024C0B3BE99BE05C01C408EC0B33F45C09A3F3440513F63BE90C02C40254037BFAC3FEEC0B7BFB9BF87C02D407C3EEA3F7A3F81BF9A4053400BC0C3BF713F1B40F93F6ABF1D4004400641B640A5BF04401DBF76C08DC0DEBF674066404D4040C0B940C0BF36BF33C098C041C07C40E4C031C096BF77C0CDBE9E409B4004C0D23FE33E923F01405FC0233FD3BFC840B4BFA64033C033BE38402041D0BF853F8DC0463F41C034C03540673F893F49C0C23D1B3F3340A4400CBFCC40143FBB3F73BF5AC07C4061C0A83F1D411A3FEA3E72C016C0FCBF48BF2040C73F0F40493FE03FABBF95C0CB3F26C0184036C051C0B5BF36C08A3FDCBE093FDFBEF5BDC63F9A4071BF65C0D740E0BF00C04F3F0CBF0140FB3FE5BE66BE63BF11BF724035408A40234046BF9EC061C0C2BF5BC02AC0BF3F94BF42C0F440C6BFB54008C0C340234030C0BF3FB4BFAC3FEC40DD3FD3BF11403DBF97BF2FC08BBF2DC02D401440EF3F1040883F99402A403FBF2EBFA6C065BF05C08D3F744048406B3F29C043BD563F22401B4050C0373E54BF93BE26BFFEBF70404840A3BE253FE4405140A6404540154058BDADC0B3BFC0C0B7BFE240F13EC7C0343FAABD4A4044C03F3FC73F6D3FFEC0A5BF93C0EABF5BBF4DC019C006C0A93FC240494087BE004088407940B7403C3F84BFDD3F14C0A53E8FBFF93F84C0E8BFDCBFB040F33F234002BFA7409C408F3F46C089BF8B3FBAC0BC40F0BF8EC0714010407A401940423FAD40D4BE3BC03DC070C07E40B14099404E4061C03F3F87C060402F404CC031C0B6BFAB3F90BC0340A6408DC0C9BFC4405ABE37C04D402EBF0BC08EBE41C0B3BF03405EC0A74001C06A40C63F10C060C0BE406D3E52C04B4016C0A4409B40BD3F30BE2FC0A43F28C058BF3F3E3340A7C09D403D3D5C3F7FC0F5C089402AC07040393F913F22C07040FBBF22BFBDC009C020C029BE4AC042BF453C4140C9C01F40"> : tensor<20x30xbf16>}> : () -> tensor<20x30xbf16>
    "func.return"(%3) : (tensor<20x30xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x30xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x093B56413F45883FD243213EE43D003DAD4504435A4199403B3F4B42C83B1942364129423D3C7432A9435B421E45A73A4B42243FFA44A93DE73AA942B4447442693D953F0043CB401C4506426A40B04050419E3DFA3C0D4206409941A9427A420644123E99440C43883F5B447042F83F54419B429943B74186409E4106420D42F24156408C3DCB36A040E74084451C432C3E064004459041A93C3F418842F54233439043C83F3144F23A0644B24380410943543B29408242153E16433F43B042064430390A423F43F542CB40BC3EA240313A8042463BB2433A42044121444242A2433B415A43DF436143AD436A40424419444B3E35421C45DA3DF24399423A422943AD3FBE44BE44F841803F4B4213434D370941EA4106448240BE44993D86407C411C43A941DA44B43C883E273FC3419E43E4405B4080424242A241DA4042425B44903BCB3E4B401C44884356415441654090418242E44056428244A24390435A4356448C44A2404B44AD42904388434842093FDA43F24195433B3D70403043563DA243B7420F433343B43A824088418041DF403B393544B7439342FA3A90414836094399434B3E0743013F86403D42B04474402C41903FE4413140F23FE744AD422C412942C843A93F4842293E4240C34035445B408C42C8431C44DA43123F8E38BE3E1E41DA43A94449411C3AB741803D214095432C42743C033C95410D42C3437440B44206407A42E43E1E3BCD4350423142863E50403F4586408C409E435642703B3341693F843F0644ED42B241AD40493F0A426541333F124290419945824431409041123E5A43BC43124129432743D242A2428C44A240823E7442FD43A742704321456A42F23F5D43D23C15440A449041E7401E3DDA3F84411343293EED40BE447A4035447442743A88421C46DF40953FBC43B73EA7427A428042293FA93FC342A9380A3E74422C44B73DD044E43D9340503F0743704319433D401246063E333D4D43F2417041BE3E1C42BC40C841C33E16414B40EA43CB403542FD418242E44280408242AD3F0C3DA93D133D5639B7400644493F2443004516418041DA3EB73D84416C41243D273B1E3FD23D4D438042AD432942B73E15441943A94009434842A040E43FA9425445B7408044A241AD4429426542A0407A4050403A450F41ED40D241993EF83F5F42B23F56425642E4414241CD41A23F03444842A03E5B3E3544243F9541BC3F5443BE42363F4242AD36FA3E24420A42DF42863AF23EDF3B353E78414643BE42293C313E2145E4423544B442EA41013756447440A24486401C45493DBC447A3E4838C642B042A03EBC403B3F78453140DF433341093FD242034299414240A944C3429E3B8041A24365438644953E903F0F41E441313CC83F654190432C410C41654450412942883D3A440D44C83FB742A93FB23F8E4495444641C3434943CD4169430342A93E5644F23C93429942464378436A440344D7421943A03E9E4316435F42D0426A4282404B40CD338C413544BC43C340B044073B8642D2425B3EB241C33BA74274408C4112433A4484413343B740CD4116439B443B3BE742CB42F2412C440A449940653A5F422C403D42013FA03A74423A44124499360C3F7C435645A943484246438C3ED23F244246436C41243E9944A9411C42423AC642A93EB432A742C3441942"> : tensor<20x30xbf16>}> : () -> tensor<20x30xbf16>
    "func.return"(%2) : (tensor<20x30xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<20x30xbf16>) -> tensor<20x30xbf16>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<20x30xbf16>):
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<20x30xbf16>, tensor<20x30xbf16>) -> tensor<20x30xbf16>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<20x30xbf16>, tensor<20x30xbf16>) -> tensor<20x30xbf16>
    "func.return"(%1) : (tensor<20x30xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

