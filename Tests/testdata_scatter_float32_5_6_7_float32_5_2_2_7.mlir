"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      "stablehlo.return"(%arg1) : (tensor<f32>) -> ()
    }) : (tensor<5x6x7xf32>, tensor<2x2x1xi64>, tensor<5x2x2x7xf32>) -> tensor<5x6x7xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf32>, tensor<5x6x7xf32>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xD6669540242805C001C3D53FF4240C4065F77540F0B662C0F93F92BF697308C18FC469C05740F1BECBC51640433F6F3FAD387FC059AB18407097A0C0FD5003C032C0013E72889E3F61CCE940680AF23EE91C7BBFB2101CC0DD6DC43FDB4009BF0B444B40BEA068C0D5DF02C0DB8605411155D83F13EA434017A2B2BFB3FA95BD7F6110C0EBBBD43E8392DA4090F93540BDFE39C0E296BDC0C57472C0018C6DC0E238CE3E84B62840CB0015C0093217C0D71240BFB4A724409413AF3FD0B7984026D23F402C6760BF428CBA3F1C0C544083A5C63F00DE61C04698674091FD283F56435E40075BA33E6657663F79279FC084AB8EBFFAE2643F9D6B8DBEBB8EB13F312423C0FB415F3F18E954BFF425AA40585596C004154240B80F51400BFE8B3E5AB9BFC01E59A9C0E5C66140088B0EBF59AA6AC0A8F000C0E96F4BC07E733740F18F4FC0A77FE2BFF9D1AC3F196394405E018DBE353A36C0D9A710C0326692C0407EE53F9FA99C3E0B262CBF8D6A65BF12DFA73F20830D406DA5DE3F913BA03D47F908BF1E57E3C0FC9269C06697D7BF4DCF24407BBCD13D803844BF702EF7C00D3A70BEA6592B3F592A5CBE74C013BD1583CCBF4254973EF52EFC400DBEAE3FD42CC3BE786787C0EEFC0340DD8A4D4053E5EEC09374A03F41DC2BC0626E42C0CCAB2CC099A6F2BFDACA434092B5763F932069C0ED4B43BFB7F69FC09296E63EA4BE1FC0E035003F9F548240EF111C40217122C0C92421BF74790CBF8AD90C40622A7A406ECC89C0AA3F9B3F60421F3F23C717C0D180EBBFA39A2CBD181E5FBF98CBAFBFF7178440EA2BCA3F903C173E77EFAE3F75E9CE3C3724BCC0120C1340433B99BF1955944001D85140AC041840206A3E3D71C48D40F7AA0BBF56D890C0764944C06F3414C0755697C0AF1E12BF0949233FC264CEBE9A2D7B3FF355BAC0DC7F373B3FDDCA3F147DE83F6368F9BF27CA0A40AD5CBF3F061410BF168DCB4006A4064092AB1FC0F2C838C062DD6840BE420D3F8521B4BFFE3C993F0BB6404002310EC037372EC054DF80BE6B5BD1BDD357184061E30E40F23B55BFF7822BBFE520043FFAFF92BFE8386340AD5A4F3E3590B5C0B1019C3D2556113E97CADEBF5FC4223EF96E27C0B2A8DCBF4CDF27BEE46E9CC01B6614C0856F8E40CF1AED3E343A35C0622AFDBF"> : tensor<5x6x7xf32>}> : () -> tensor<5x6x7xf32>
    %2 = "stablehlo.constant"() <{value = dense<"0x50648BBE40ED14C0B2CAABC0688A083F794D0D3F72A8DBBF466B49BD7A0128403AE6E73F444026C0DC51D1C09712BBC00F312D40F9DF7DC04EF84D40A99323C07DC1BABFC0114E3E8EA4163FF259634030F641C04387D4BFF327E03F12F02740A216D0C06BF338C0C570053F0502AD3D14D0C0BFD232D6BF67D506407EA660BD692E10406E8379C02D7D18BFBF9E4A3ED008BF40B4343D3E5F680B3F977F543F6FE9504036FC91C0915668C0D4D0B6C0B864A73F46BFE2C02A38C4BF137C1040164ECBBFEA512C4069A44CBFD7AD74BE9DBC6F3F26660040C86A85C0D0B919C0A8DFF7BFF21425C04589EC3FDA4872C04360973FD09870C0F3A986C012263A40A9ED07BE17B46DC04DB30DC0D74F2A3E23EC18C049993A3E549882BE394662BE4FCAA5C005D3C43DED0D073FBEFA5E406B46B2BF085825408E3CEBBE6B2475C0CA1EA53F992880BFE99004BE8E2D7240B7CFA73F0887A1BFE8FB94C085B7FC3F9B8BB93D2EF222C09FD02E40D54687BF254913C054BF843FAE9F2CBC684EB940BD11EA3F6D6B194000C6BABF49EBCC3F4C2EC43F1D2BE2BE275517C02C11203F0A5E9240C33D664031667D3F1912C640E7F09EC04BF77C40660D77C018860240CB8BA6BFEA86FE4083D98B40D50078BF5B3FD3BFCBEC953F805537C06829BCBE9319CC3FC2B8533F2E7905C094E134C09035ACC052A3CFBF7ADF3BC092E6ACC057DB6EC076E636BFC027A43E29BA8040758243405D33C940109A1A40365A4E4008318E3F766AB5BFCCEDC03D905EA5BC"> : tensor<5x2x2x7xf32>}> : () -> tensor<5x2x2x7xf32>
    "func.return"(%1, %2) : (tensor<5x6x7xf32>, tensor<5x2x2x7xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x50648BBE40ED14C0B2CAABC0688A083F794D0D3F72A8DBBF466B49BD7A0128403AE6E73F444026C0DC51D1C09712BBC00F312D40F9DF7DC04EF84D40A99323C07DC1BABFC0114E3E8EA4163FF259634030F641C04387D4BFF327E03F12F02740A216D0C06BF338C0C570053F0502AD3D1155D83F13EA434017A2B2BFB3FA95BD7F6110C0EBBBD43E8392DA4090F93540BDFE39C0E296BDC0C57472C0018C6DC0E238CE3E84B6284014D0C0BFD232D6BF67D506407EA660BD692E10406E8379C02D7D18BFBF9E4A3ED008BF40B4343D3E5F680B3F977F543F6FE9504036FC91C0915668C0D4D0B6C0B864A73F46BFE2C02A38C4BF137C1040164ECBBFEA512C4069A44CBFD7AD74BE9DBC6F3F26660040C86A85C0D0B919C0B80F51400BFE8B3E5AB9BFC01E59A9C0E5C66140088B0EBF59AA6AC0A8F000C0E96F4BC07E733740F18F4FC0A77FE2BFF9D1AC3F19639440A8DFF7BFF21425C04589EC3FDA4872C04360973FD09870C0F3A986C012263A40A9ED07BE17B46DC04DB30DC0D74F2A3E23EC18C049993A3E549882BE394662BE4FCAA5C005D3C43DED0D073FBEFA5E406B46B2BF085825408E3CEBBE6B2475C0CA1EA53F992880BFE99004BE8E2D7240D42CC3BE786787C0EEFC0340DD8A4D4053E5EEC09374A03F41DC2BC0626E42C0CCAB2CC099A6F2BFDACA434092B5763F932069C0ED4B43BFB7CFA73F0887A1BFE8FB94C085B7FC3F9B8BB93D2EF222C09FD02E40D54687BF254913C054BF843FAE9F2CBC684EB940BD11EA3F6D6B194000C6BABF49EBCC3F4C2EC43F1D2BE2BE275517C02C11203F0A5E9240C33D664031667D3F1912C640E7F09EC04BF77C40660D77C01886024001D85140AC041840206A3E3D71C48D40F7AA0BBF56D890C0764944C06F3414C0755697C0AF1E12BF0949233FC264CEBE9A2D7B3FF355BAC0CB8BA6BFEA86FE4083D98B40D50078BF5B3FD3BFCBEC953F805537C06829BCBE9319CC3FC2B8533F2E7905C094E134C09035ACC052A3CFBF7ADF3BC092E6ACC057DB6EC076E636BFC027A43E29BA8040758243405D33C940109A1A40365A4E4008318E3F766AB5BFCCEDC03D905EA5BC3590B5C0B1019C3D2556113E97CADEBF5FC4223EF96E27C0B2A8DCBF4CDF27BEE46E9CC01B6614C0856F8E40CF1AED3E343A35C0622AFDBF"> : tensor<5x6x7xf32>}> : () -> tensor<5x6x7xf32>
    "func.return"(%0) : (tensor<5x6x7xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

