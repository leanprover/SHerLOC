"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x40xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x40xf32>, tensor<3x5x2xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x40xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<3x5x40xf32>, tensor<2x1xi64>, tensor<3x5x2xf32>) -> tensor<3x5x40xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x40xf32>, tensor<3x5x40xf32>) -> ()
    "func.return"(%6) : (tensor<3x5x40xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x40xf32>, tensor<3x5x2xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x7B7BBB3F2C270DC0D12F8FBF1FF6A53F0AFF95403D48093F7B5E943F0AAE75BC98E41EC03E23D8BF650E4C4079AF87C083EFBF3FBB8851C0FBCB163F39A0C4BFE25AFBBF0334F63FFCBAF03F6785873FD28E0641E57E334027BB24BF5A1C9A3FAF8034C0BFF79F3F26A7703F2448E9BE3A17CA409DBAB4BEA61801C0A851CD40EDA88040485708C07E599AC0F9184C4029DAA33FC1FD89404AD33F3F25EA1C400127FE3FD5401A3FEB83C13EDDA9493FE6E422C0889114C0CD538040B9FD31407764BDC0F577A7BFA54702C09206F03F53AC973E78A94E406609EE40D9A37C40DB36B43F4DCBEDBFCBDD5D3E645041C05FFC91BF1E47B440B1D84FC0B469FFBF9F1C82BF6B0B05C069DE7C3FEE205940B8FB5E40FE8206BFFF8FA0C00B236DC0935B1A3FDA818F3F4B69953F2941E93F85409E3F37157C40B45CD33F661C5B4029B9C1C064E884C05CD477BFD152BABF555A15C0C72B29400C5431C0EE79E6BF6C4E03C0AC45CFC07B0A2FBECC43EB3FE39301C08761873F27113BC051FB6B40689C0540C0FFD2BF3560F23F1D5FB940FD6DD5C09B41AABC78D5F83F4C2CCCBFBD253140D59246BC7591B2BF2B0A4940163D0541A4FB37C0582DAE3F332B11BEDB189DBE6DBB21C07B1DA0406FB0D84005FEA3BEB730893FADCBFA3D58BE313EB88BE53F5E111FC039F513BFFBA944400F26BC3FC8073E3F16AD8EBFEB64A6BFB296833F614C00C042A227BFA42FFDBF59B2B03FFAA42BBEF0E7F53F82570C40A5883940D5B708BFFF1F213EDDF33E3F11360640302725409DECBB4008CA0640798BEDBF9256C6BFE52544C0A3A3DEBF95947F3EB73787C0393691BF737D98BF88DA0B3FD105B4404D8898BFD3477D3FCB66CEBFACFAE33F46B4A2402BADF9BF3EEE6AC014C196C0B2D5ED3F856FF13F5F78D640E98E85BF23AF0DC02251A43FF8C327C0446B94C0F11537BF32242540F320D2C02517BF3F8444F93E4E3D57C0C8892B4049F2B840821DBA3F85DFFEBE3DB3E2BF7E34AD3F36AAB1404A82D53DC71F63C00F8A4D40AF010540FD3595C0B9E9D43F6A3421408C4090407F76C03EC3725E4049258F40581F46408E4528C0B2A291C0D4F98F3F5CBC903FBF7088C0314FE8403B8A7F40B1B0C4BD4497DEBFEADC8840693E843FB5BDA23E9DFC184043353DBF6D680DBFEA7E893E4A80ADC02065C2BEE205E93F86C614C0D9A2BAC0296B52BF7D6EAC3F8BFD94C093116940BB61B3BF6F17C6BF831CA640F461BB3E485B85BF04C89D404AABC83F1C7B34C06F9F5E3E9639AF404E850240F0051440B322E43F204EE63F11CC77407D3A0C4062A1A4BFB3E524407019C0C0A8254F40F628BA3FB0FB3A3FF9FE263FAB50164073945EBF76BF2840B3316640E0C25A40169282C0D17FCCBF02A87BC00B2109C1394EBF40D9949FC0DE515C4000E864BBE2313D401A0F5A3F1EFBD7BFE048A33F1E7411C00F645CBF23224EC0EE7B14C020E088C09EE3BB3FE1464940B2BE0D3F0957CDC0721DEF3F606DF5C0ED47213E470F8F3F17F1BCC0116C68BDE7CB06C0F631834037B50F40DADBE63F3017FBBF4DDF42BD336B80BF93DF04C03B11B4BF1448B3C0CDEF02C021F2013FA52956C08DD40CC0F2F7D2BF9C37CC3F57CD12BF85902AC034AE124078328240A08ED03E9CF7004086DA25C09A3F963FAFE4C53FE69C963EEF6B284098E40DC02CCDD4BFA885D83ED5243AC046F904C0262467BF846E26C0EAB48A3E091A1E4028AF9B40F9851F406585D13E7E00843FB143CF3F3A6315C0A9215D40D3FD2D408DAACB3F08DB3EC02C0AA9BF1F5B11C07955E33FF5EB4F4015C233C0305FB43FAEEF7DC05E12AB40FC5902415D54D23FBD8E9C4087ECFABFDFE1094095FF8E404403D6BF466A8D40B4C119BF57E6CDC0CFB5164086D44340BC51D4C028E42C3F9656903CE40098C0EF20FAC0952D5AC020BE8DC04DFAF73FA37786BE84D3B6C009CDDA3F799364C0603E93C052201B40D59AD23E7057A6C03E6807BFC694ABBFAE673040FDB8F4BF8BA712C14DC6E1BE423DDEC094BBABBEAB8F85BF690E1CC0382A0EC11E55EFBFC553A6BF004ADB3F807AAE3FB7E4853FDDD4BDBF2F4D1AC033A3083FC519653FDD07B3BF1FA84740DE0D104000F8E2BF347D853E575770C0324C91404B16FABFB69E903E0380BA3F0B96C3402E1C85405FB832C0F0403C3F154D6EBD0A9794C05D0038C0D74BCE3FBE7458C0850083BFFD575C40CB1D87BEECA893C093A990BEE644844097EB8EBFD47FB7BE282B9AC0276B0940FF9284C0C079124028995040E74A4940433CA23E98E5B2C07AC31540454348C0AD8B5FBFF55CE33F724EE93F1A22054004E324BF19078D3C95019140D7F83DC05333213FADC55140D47F1440993B46C077011BC079C8B9BF1F8843403B72B4401582E23E64298BC06C4A37401AFAA1C00456D2BFC325A04078DFA1BF06959E3F0E510E3FEB92CC3EA600DE3F1A28A14070C1ED3F5E6494BFA624B8407A0E86C066F5A13F7A6397C08271DA3F6C1D2ABF2998693F3B415BBDC700E9BFB5DAB3BC0E4F55BFE43D2640E5649BC0DA579C40D4F5843F1B6EC1BFA1EC933F71F82FC01FB578C015D1B0C09CCF15C06D0848BF72E03DC0180C53C049F6A43F5A968C40101490408CC4AF3FDF6179C028BCDE40D4B42BC0C42B48BF29941240FB181A40DA48EBBF4A65EABF6250B240D1E63040606B993F23881A40633C5AC0193387C01335FD3FCB3967BF0EF107406EBB8E4093202DC08376F3BB646620C035539640C8DC2EBE71FD3DBF4B00FD3E705F7FC0A840D6BFB4DDC4BF2736A83F9178CFC06B1BCBBF13CC2B40C725DAC0FE14C3406D042D40BB4FAB40FA98BE3F8FA6763F2F3881BFA01F683F798417C0DFE74D40667B6140F39A1BBE5C20A740FDDE86BF006B11BFEBD89DC083A98AC09EB0C6C0CDC481BE9FF90BC0ECCD98BFAA73673E818148C063A7933F740B2E40AB5574BF0F946E3D9227A93F7D698A4029C9E9BFE91B993F474803C097C300C0174F213F9C286940B90898C0AF92633F3E102FC0596B63404F1E2ABFB936E43F7D51823FDEBC2EC00EF629BE33FBC4C084A52F400BC83940DA19FABF301D3C4086A62B402A18093EA48F2EBF6F6FC33F425822C019B5323F5F078B3F42716DC09E3C13406287623FD0BD23C03A62AA3F06F4813FBB1031C022662CBF89AC8CBFF63B1D407D1B80C0F51C58BF51A461C0E8D38D3FCE894CC083F16B404EE5C43F418C16C08CCBCFBFE869B240AF1CA2BF7597CB3F62E06DBF2DB23140951C2BC00363B23F01DE16C0C3AF95C084240AC0B4B8E3BF34B54BC0907AEF3FE63E6CC046AEB13F8229BBBFF45382BFFA91CEBFCDE5983F7CE794BFF0F27DBF"> : tensor<3x5x40xf32>}> : () -> tensor<3x5x40xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[[4.45158768, 6.71890974], [-2.41928864, 1.79930842], [-0.170640275, 1.95324767], [3.1499424, 1.76437771], [1.23784339, -2.13207674]], [[0.252204955, 0.888183951], [-1.92308128, -3.47330046], [1.58392239, -1.89726233], [0.402788848, -0.33233875], [-3.74422836, 4.80488586]], [[3.38082385, -4.21664524], [6.54100561, 1.83753514], [0.712574541, 0.109676763], [-2.31464648, -6.50513029], [3.36480594, -4.880548]]]> : tensor<3x5x2xf32>}> : () -> tensor<3x5x2xf32>
    "func.return"(%1, %2) : (tensor<3x5x40xf32>, tensor<3x5x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x40xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x7B7BBB3F2C270DC0D12F8FBF1FF6A53F0AFF95403D48093F7B5E943F0AAE75BC98E41EC03E23D8BF650E4C4079AF87C083EFBF3FBB8851C0FBCB163F39A0C4BFE25AFBBF0334F63FFCBAF03F6785873FD28E0641E57E334027BB24BF5A1C9A3FAF8034C0BFF79F3F26A7703F2448E9BE3A17CA409DBAB4BEA61801C0A851CD40EDA88040485708C07E599AC0F9184C4029DAA33FC1FD89404AD33F3F25EA1C400127FE3FA0D51AC0EB83C13EDDA9493FE6E422C0889114C0CD538040B9FD31407764BDC0F577A7BFA54702C09206F03F53AC973E78A94E406609EE40D9A37C40DB36B43F4DCBEDBFCBDD5D3E645041C05FFC91BF1E47B440B1D84FC0B469FFBF9F1C82BF6B0B05C069DE7C3FEE205940B8FB5E40FE8206BFFF8FA0C00B236DC0935B1A3FDA818F3F4B69953F2941E93F85409E3F37157C40B45CD33F661C5B4029B9C1C064E884C05CD477BFD152BABF555A15C0C72B29400C5431C0EE79E6BF6C4E03C0AC45CFC07B0A2FBECC43EB3FE39301C08761873F27113BC051FB6B40689C0540C0FFD2BF3560F23F1D5FB940FD6DD5C09B41AABC78D5F83F4C2CCCBFBD253140D59246BC7591B2BF2B0A4940163D0541A4FB37C0582DAE3F332B11BEDB189DBE6DBB21C07B1DA0406FB0D84005FEA3BEB730893FADCBFA3D58BE313EB88BE53F5E111FC039F513BFFBA944400F26BC3FC8073E3F16AD8EBFEB64A6BFB296833F614C00C042A227BFA42FFDBF59B2B03FFAA42BBEF0E7F53F82570C40A5883940D5B708BFFF1F213EDDF33E3F11360640302725409DECBB4008CA0640798BEDBF9256C6BFE52544C0A3A3DEBF95947F3EB73787C0393691BF737D98BF88DA0B3FD105B4404D8898BFD3477D3FCB66CEBFACFAE33F46B4A2402BADF9BF3EEE6AC014C196C0B2D5ED3F856FF13F5F78D640E98E85BF23AF0DC02251A43FF8C327C0446B94C0F11537BF32242540F320D2C02517BF3F8444F93E4E3D57C0C8892B4049F2B840821DBA3F85DFFEBE3DB3E2BF7E34AD3F36AAB1404A82D53DC71F63C00F8A4D40AF010540FD3595C0B9E9D43F6A3421408C4090407F76C03EC3725E4049258F40581F46408E4528C0B2A291C0D4F98F3F5CBC903FBF7088C0314FE8400221813EB1B0C4BD4497DEBFEADC8840693E843FB5BDA23E9DFC184043353DBF6D680DBFEA7E893E4A80ADC02065C2BEE205E93F86C614C0D9A2BAC0296B52BF7D6EAC3F8BFD94C093116940BB61B3BF6F17C6BF831CA640F461BB3E485B85BF04C89D404AABC83F1C7B34C06F9F5E3E9639AF404E850240F0051440B322E43F204EE63F11CC77407D3A0C4062A1A4BFB3E524407019C0C0A8254F40F628BA3F8E4A5EC0F9FE263FAB50164073945EBF76BF2840B3316640E0C25A40169282C0D17FCCBF02A87BC00B2109C1394EBF40D9949FC0DE515C4000E864BBE2313D401A0F5A3F1EFBD7BFE048A33F1E7411C00F645CBF23224EC0EE7B14C020E088C09EE3BB3FE1464940B2BE0D3F0957CDC0721DEF3F606DF5C0ED47213E470F8F3F17F1BCC0116C68BDE7CB06C0F631834037B50F40DADBE63F3017FBBF4DDF42BD7ED9F2BF93DF04C03B11B4BF1448B3C0CDEF02C021F2013FA52956C08DD40CC0F2F7D2BF9C37CC3F57CD12BF85902AC034AE124078328240A08ED03E9CF7004086DA25C09A3F963FAFE4C53FE69C963EEF6B284098E40DC02CCDD4BFA885D83ED5243AC046F904C0262467BF846E26C0EAB48A3E091A1E4028AF9B40F9851F406585D13E7E00843FB143CF3F3A6315C0A9215D40D3FD2D408DAACB3F08DB3EC02C0AA9BF1F5B11C07955E33FF5EB4F4015C233C0305FB43FAEEF7DC05E12AB40FC5902415D54D23FBD8E9C4087ECFABFDFE1094095FF8E404403D6BF466A8D40B4C119BF57E6CDC0CFB5164086D44340BC51D4C028E42C3F9656903CE40098C0EF20FAC0952D5AC020BE8DC04DFAF73FA37786BE84D3B6C009CDDA3F799364C0603E93C052201B40D59AD23E7057A6C03E6807BFC694ABBFAE673040FDB8F4BF8BA712C14DC6E1BE423DDEC094BBABBEAB8F85BF690E1CC0382A0EC11E55EFBFC553A6BF004ADB3F807AAE3FB7E4853FDDD4BDBF2F4D1AC033A3083FC519653FDD07B3BF1FA84740DE0D104000F8E2BF347D853E575770C0324C91404B16FABFB69E903E0380BA3F0B96C3402E1C85405FB832C0F0403C3F154D6EBD0A9794C05D0038C0D74BCE3FBE7458C0850083BFFD575C40CB1D87BEECA893C093A990BEC2EE86C097EB8EBFD47FB7BE282B9AC0276B0940FF9284C0C079124028995040E74A4940433CA23E98E5B2C07AC31540454348C0AD8B5FBFF55CE33F724EE93F1A22054004E324BF19078D3C95019140D7F83DC05333213FADC55140D47F1440993B46C077011BC079C8B9BF1F8843403B72B4401582E23E64298BC06C4A37401AFAA1C00456D2BFC325A04078DFA1BF06959E3F0E510E3FEB92CC3EA600DE3F5A34EB3F70C1ED3F5E6494BFA624B8407A0E86C066F5A13F7A6397C08271DA3F6C1D2ABF2998693F3B415BBDC700E9BFB5DAB3BC0E4F55BFE43D2640E5649BC0DA579C40D4F5843F1B6EC1BFA1EC933F71F82FC01FB578C015D1B0C09CCF15C06D0848BF72E03DC0180C53C049F6A43F5A968C40101490408CC4AF3FDF6179C028BCDE40D4B42BC0C42B48BF29941240FB181A40DA48EBBF4A65EABF6250B240369EE03D606B993F23881A40633C5AC0193387C01335FD3FCB3967BF0EF107406EBB8E4093202DC08376F3BB646620C035539640C8DC2EBE71FD3DBF4B00FD3E705F7FC0A840D6BFB4DDC4BF2736A83F9178CFC06B1BCBBF13CC2B40C725DAC0FE14C3406D042D40BB4FAB40FA98BE3F8FA6763F2F3881BFA01F683F798417C0DFE74D40667B6140F39A1BBE5C20A740FDDE86BF006B11BFEBD89DC083A98AC0072AD0C0CDC481BE9FF90BC0ECCD98BFAA73673E818148C063A7933F740B2E40AB5574BF0F946E3D9227A93F7D698A4029C9E9BFE91B993F474803C097C300C0174F213F9C286940B90898C0AF92633F3E102FC0596B63404F1E2ABFB936E43F7D51823FDEBC2EC00EF629BE33FBC4C084A52F400BC83940DA19FABF301D3C4086A62B402A18093EA48F2EBF6F6FC33F425822C019B5323F5F078B3F42716DC0732D9CC06287623FD0BD23C03A62AA3F06F4813FBB1031C022662CBF89AC8CBFF63B1D407D1B80C0F51C58BF51A461C0E8D38D3FCE894CC083F16B404EE5C43F418C16C08CCBCFBFE869B240AF1CA2BF7597CB3F62E06DBF2DB23140951C2BC00363B23F01DE16C0C3AF95C084240AC0B4B8E3BF34B54BC0907AEF3FE63E6CC046AEB13F8229BBBFF45382BFFA91CEBFCDE5983F7CE794BFF0F27DBF"> : tensor<3x5x40xf32>}> : () -> tensor<3x5x40xf32>
    "func.return"(%0) : (tensor<3x5x40xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

