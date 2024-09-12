"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %4 = "chlo.cosh"(%2) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%4) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xC941062E93BDC8BDC3C5C0BF9FB66940D23B0E41A33C1F4301408D427EA92134B9B40B40653B7BBF64C3A73B3CC3B8351840253EA4BFD61D1B25BDC00546C4C0DDC549382FC5C5BCA62CA9C190C48B3EB1B94B374631D5C0F040C34160B6D94078C0D2C2662C3743D1B0EDBB913199B89EBE4B402BC2F9431C3CB0C6C7BDCB350A3D0E3E56422634E9B8FF3DDB3ED73E233D643DBA410A3B09C245432BC056423B44F03917C061B65B4144C39B3B2A3DACBD27417FB767C43ABB3042FCC0F244A0BE04C6A23DACC03BC080B813C0D439FCB682C040BDBB3F71B575BD20B968C263BDAB3CCBBC25BBC7B43A3D0EB92A3D42402AB37F3801B07838EF2F47310DC2A839B43E3BC153B9D1C2DCC036432FB9B840D2B8A4446244ABC15640E03F04BF074037C0D53DC6BFCDC0583E11C0ACC6ECB4ADC5CFBCBC17C8B956C13C406B414EC14AC5F1C594C06C4172C227C2034074C0823A3DBD0DBFD9B810C090B76F4372BC42462F3E2143CE40913C429F21C4603D98BD85BDC64304C512C4DBB5A9C25FB0A2BB47C2DAB2C7431138643B4046834657BED63C03C1153052C000C2B34445BB184056B6A034EF393CC4973B07417B400B3A7CC4D8C7DC3B8EBB2A1E4B449EBC7D3F02378C3ECF44E441E9B6B03732ABDF3F77C07A3FBABEB9C2563FD9C0A93B20432B42A3BB323C40B6A5395147C741F530D2BE31AC45C3CBB9A7C4B8441EC00A348B449DBD61C4A2C07F3C13BD52C3E93921C336C5183F3CB9E5402144B0C0903C8BACAEC4F73DB4C33CC958BC41C186BF393EA441ADA825C126421740E34039C13B470C4464BA0033C346D2B643B04BAAACC6F4AD2BBC533B783C7DBDA13F394467BBD1BC4ABAF5C13F33FEC0294043434144C4B3D73BEABA2F39BDBE5A4176B9A9BD7141104374C3FA43A23E2CB97B3111410AC8BC3B513ABC3971C7EFBB8BBE523A0343C441F145D32DB53DB34328360D3E67458EC06FA5E7C0B439173E5EC69F37D7C00D45D52B053C3DC58841D93C4B4167C12BB362BF70404FB35642172DE7BEF53414B1DEBBA7C7F2C52FC360C4DABC11AD544247473244B642ACC5A7B8A63EE641F03D5AC07BBF81C26747EEC2D3BC20C3A63CD442CD402E38"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%1) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x8648053C47407B40F8581643593C9744113E4D46013F674C8A439F4A013C223C2D3CAF43D53DA542094DF93DA84C423CDF43DD40E742003C003C64456E5A77458059963C9355323F033C4048FD5154410E3D6C3C0E3CA545F2457948523CB045B944934B023C9D4C0C3C213E103CAE3C6C4156447949BC4E4E3E455E7A40443C9E3FC340F349233CC73CB340BB41B641C73F1C406548A73D1F49BD4C1544F3494C50273DDC43523C4F47BB4CF33DD33F5F409D46723C1A51BF3D8749164664546E41675A554038453544A63CCC431C3D633CD144F83F0D433C3C2B40D93C2A4A1B400C3F3B3FB43D2E3CEE3FD33CD33F43441A3CA63C083CA43C083C0E3C2A490B3D8841DF46EB3C904BB9459A4CDF3C5745C03C7A52015145486D444D43F341A0432D44884020438F451741C5432C5E313C8F58413F003C173D3D4737448A4721473156F259FC448E47494A6E499143B044663DF33F0042C23CC143753C254DBC3E145CE8406B4C9245E73E003CC44F18404C403A40194EB654534F453CFE4A0A3CF73DC749183C1C4E873CD53D0C5C425D16414C3F2B46083C65440949DE52C53DDF43513C2B3C273D5150F03D3746C044323D8A51FB64173EEC3D003C9350F93EA842643C5541A953C448613C783C023C4B43B744A3428F41374B6B42B045FA3D694C7949F73D693E4F3C0A3DE16182480C3CAF41023CBD4C183D8D520153F643213CE0515140FC501E45CE3EAD3FDD4C243D6B4CBA551042E33CD245C44F4245E63E033CBC52AB40E34D4C74993EF446B642F3403648013C96466B49DC43CD45D8466561284F583D193CC05E5E3C093C013C2C5E043C603ECC3DC43E3240E2424450D73D443F4D3DED481A3C1C461144B94C66501E3C143E973DDF3C93414C47F83C5C40A147464C324DBF4E7141DE3C0F3C56464C66053E503D133DA962223E5441513D2B4C7B48F259043C6840E04D4D3CC240EF56ED44003CD8450F3DCD408D5C763CAB45E154023C323EE355FA47513F16477B471A3C7E42A7441B3CF349033CCB41323C0D3C183E1D64F8598A4CF750523F033CED49A76126502C4B8A58B23C7641C948A4407644A542784A6862004C483F694C053F9B4B8F458F3C"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%0) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

