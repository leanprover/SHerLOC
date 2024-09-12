"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<20x20xf16>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<20x20xf16>
    %4 = "stablehlo.rsqrt"(%2) : (tensor<20x20xf16>) -> tensor<20x20xf16>
    "stablehlo.custom_call"(%4, %3) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<20x20xf16>, tensor<20x20xf16>) -> ()
    "func.return"(%4) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x96C133B872441840554729C539C068BB8CBD5E488044AC40C1419EBBCEBA8EBE5FB503C36ABC7232203D794538B9EC40404016BE7143A7C0DC3C903B37C28439CAC5C345CD437FBFC33E92442EC46E41473D844250BC8A40D2BC84301840D42C8B3FF6C3D1398A4057A6FF4335C474C7ACB4EDBCA640973E9FC05141BB35044087449FB94E333E41424413B84FBC6237C442BB3CA740073B443CAF42E044243A5B40DFBC06AAB640D9C5B3BDD73F60BF41453137924011C2CCBED44283BEE842AC3D3AB935AF0A432EBD0734004668478DB97FC597B986B65FC48EB81AB4B8C579BC344807A74EC694C2D1C5263ED234F13EB5C580C60CC19E3F31C44D3FBC41D83CF639D4452FBDD1C3E23CA9C0D839AE3B27BF62C7F0BEBE3184C3FA3B5E32B4B39EC0A2C2D5C58CC1EB3C31BE1F4391C0AF417F41B5BC87C0B4B52CC3C2C21FC1383B2C41E1377E44A9C2BE427AC18CC415400C3E2D44FFB66B3CCD421D3E15BADF349144A5C3DDC146C6E8A039C0C3C09A4081BC89BD1BC1DD4679C06440894147420542F84415C381B35DBDCD3E09BB20B676C3743F28C0094087B863B706BE68447CC2DE4296B62DC2C3BCCFC03BC3FFC12143C5C17EC22CC62D4070342E4166C12DC1B93C7DBF33B6E9424543B9C487BB424099449FBCFAC085C0D145F9C01CB804BF71C215A984458B3DB6B42AC518BCBFC123C0CD3F9845A5C2A4BCCBB4724192C708394D39744175C123C455C264BD2F4235B9B04197C0B0344F37D9438E428C3A3234584554B868BF50C1D0309ABFF7C069C514C48DBE5FBC4BB869C708439DB9F7C32932753E4A40DDBF7D4393C474BE4336A03E67C4634330449FC4C63C293E91C4EDB52CC1153C1BC6113DEE3FCDBB72C34A45684664A50941B6318AC410C433B946C115B0F6C47646973F7B4281C185AC0CC0DD3CA5B61736E6ADFB3C433BD3416EC109C41BC0D242F5BA7144F5C55BC203C200444036F63BBC40F53E57C3EF3B70C3D33E99C361382740A2BE10B83C4028C29BC4273DB53E5FC4333FAD410B3DF7B190BB134177416B446BBFC7404FC103C4BE4258C3EEB83642A0C4BE3619C44EB6A5BFC9C40BC416B9AB2E8C3A24BD5441624553C45640"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%1) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x20xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00FE00FE98379839E93500FE00FE00FE00FE6A358B373C39B73800FE00FE00FE00FE00FE00FE7540113BD73600FE1A397D3900FE263800FE423B1D3C00FED13C00FEAA360D3800FE283A7B3700FEDB38F73A6F3800FE4F3900FE534198394843D43900FEB13C4F3900FE013800FE00FE00FE00FE3F393C3A00FEE838AF3EA539863700FE3040F138C23700FE00FEE43D5A385B3B3F39453CC03B60384037913C6C3900FE00FE363900FE00FEB63900FEFA36F73D4B3900FE00FE543800FE4E38B83A00FE00FE443800FEFA3F8836E13500FE00FE00FE00FE00FE00FE00FE00FE00FE853500FE00FE00FE00FE733A4A3F133A00FE00FE00FECC3900FEEC39B938453BA23CA13600FE00FE3E3B00FEAE3C153C00FE00FE00FEB94000FE023C7C4000FE00FE00FE00FE00FE383B00FE3D3800FEBF38D33800FE00FE00FE00FE00FE00FE363CF938B33D8D3700FE5C3800FE00FE9939823AD53700FE9D3B5738793A00FE403F7D3700FE00FE00FE00FE00FE00FE463900FE00FE00FE1C3600FE6639CF3884389C382E3700FE00FE00FE233A00FE00FE00FEDC3900FEA13900FE00FE00FE9F3700FE523800FE00FE00FE00FE00FE00FE3D3800FE00FE00FE8939983FF93800FE00FE5C3B00FE00FE4E38323800FE00FE7C39763700FE00FE00FEA23600FE00FE00FE00FE00FED136CC3A00FE00FE00FE00FE00FEBB39C43600FE00FE00FED93800FE0B3DEA3CD83800FE00FE00FE00FE8C3800FEBE3800FE633FEB3D0A386B386C3CCF3FEB3600FE00FE00FE294100FE00FE00FE00FE00FE00FE00FE00FE443800FE00FE8F404C3A763900FE233800FE00FE653E373A00FE2A38D13700FE523B723A00FE00FE00FEEC3B00FE1C3BAE3900FE00FEF636523600FE0B39BC4000FE00FE00FE00FE00FE00FE4C36CE39723800FE00FE00FE423B00FE7C3E00FE2B3B333CB03800FE00FE00FE553800FE983700FE00FE00FE0038663E033C3439113A00FE053C00FE1F3A00FE683D8E3900FE00FE7F3900FE00FE0D3B2E3A00FEF639C038203B00FE00FE0639D7389D3700FE2D3900FE00FE5C3800FE00FE8A3800FE2A3E00FE00FE00FE00FE00FE00FE32426C3C00FEE738E53600FE6F39"> : tensor<20x20xf16>}> : () -> tensor<20x20xf16>
    "func.return"(%0) : (tensor<20x20xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

