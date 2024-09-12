"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x5x6xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x1x6xf32>, tensor<4x5x6xf32>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<4x5x6xf32>
    %5 = "stablehlo.broadcast_in_dim"(%3#0) <{broadcast_dimensions = array<i64: 0, 1, 2>}> : (tensor<4x1x6xf32>) -> tensor<4x5x6xf32>
    %6 = "stablehlo.power"(%5, %3#1) : (tensor<4x5x6xf32>, tensor<4x5x6xf32>) -> tensor<4x5x6xf32>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x5x6xf32>, tensor<4x5x6xf32>) -> ()
    "func.return"(%6) : (tensor<4x5x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x1x6xf32>, tensor<4x5x6xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[[-8.88444901, 2.3156395, -1.93535292, 0.505103946, 1.9416374, 2.08314252]], [[-0.842454433, -0.398483545, -1.76412654, -1.48839188, -0.527319252, 3.69354391]], [[-1.57986867, -4.6202879, -0.552776694, 0.795943558, 3.30360889, -0.916407347]], [[-0.322809756, 2.65766501, 0.705473124, -1.01220083, 2.89223814, -0.472224563]]]> : tensor<4x1x6xf32>}> : () -> tensor<4x1x6xf32>
    %2 = "stablehlo.constant"() <{value = dense<"0x58D382BF46D557C075C708BF0778C93F4C970F3E082AB740B852B0C0A7678ABFC8F0F7BF2A3D79C04C97A63E4E8501BFDB3303C08BAAA2C099D88C40A2AB76403C908E3F9EC47F40F81A934057289F3FA1EEC43F891785C0293C6FBFDBA1B03FCF47C0C063BAC4BFA7C8E8BF21BF0EBFA6C1ADBE9BC49ABF692006BF10F559C0E004C03FF86A5B3F68BB0540BC405FBF3DDE69C01B0BBD40E6543C400B3889C0B5FEDABDEB890F40A84186C0F794C2C04E4AF9BF04C51EC09FB7FF3F6754AD400A4D363F99EF3FBF0AA87340615AACC063024BC06B82513FFAF2AFBF5A9348BFA9D9C94007A68740F4E36940F159A6409E8331C0D90F72BF2FF840C0626B123F51711840D31BA63FB37F3D40D883063FEAD71BC0EE5782C025F2883EFE75D1BFF5777A40BB5EC93F7BA1A33D92FE52C09020C4C0DD5A8A3E3D46E53F3F5550BF596311BFFE2834407F6390BE024A8840B0CECF40B29AE53F8D345840514D0D4090173AC078FBF2C02EE19CBF75C04740B0062E406E79BC3F3DD8B2404D6E653FB4EC3CBF5A1CBCBF8F03E83F5FAC08C0DE71A84034E60840C71DAB3FA29F943F9FB025C0495910C164519CC06473C23F03778FC0E421FFBF6F6B48C0AC2F48403ED8A13F156CF73FB1C4EFBF0D6CC0BF2A7386C06B40253F5D71533F7EB2973F"> : tensor<4x5x6xf32>}> : () -> tensor<4x5x6xf32>
    "func.return"(%1, %2) : (tensor<4x1x6xf32>, tensor<4x5x6xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x5x6xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0000C0FF434B713D0000C0FF15BEAE3E367B8C3FD87485420000C0FF5284CE3E0000C0FF93B2644130D89E3F3C99303F0000C0FF4F72653C0000C0FF3A43933D61020640AB3F96410000C0FF73CF35400000C0FFA9038941F9B3093FD93130400000C0FFA0DC8C3E0000C0FF5654BB3F11624C3F51D0D23E0000C0FF0000C0FF0000C0FF0000C0FF0000C0FF63D6A33E0000C0FF0000C0FF0000C0FF0000C0FF0000C0FFA2E195410000C0FF0000C0FF0000C0FF0000C0FF0000C0FF7E1494440000C0FF0000C0FF0000C0FF0000C0FF0000C0FF6A743A400000C0FF0000C0FF0000C0FF0000C0FF0000C0FF50BB5E440000C0FF0000C0FF0000C0FF1AAC603F18CF89410000C0FF0000C0FF0000C0FF0000C0FFB32522403235B03F0000C0FF0000C0FF0000C0FF0000C0FFD4D00740D5D92C3A0000C0FF0000C0FF0000C0FF0000C0FFC0A7063F7CC2363F0000C0FF0000C0FF0000C0FF0000C0FF48AB1A3F0EB5FD3C0000C0FF0000C0FFF90BA941FC45C63E0000C0FF2721BD430000C0FF0000C0FFAD78733E7204083F0000C0FFA8EC85430000C0FF0000C0FF331A4740AFEC1D400000C0FF40F5B63B0000C0FF0000C0FFDBF0113E56D73E400000C0FF6A1D75400000C0FF0000C0FF11966B3EFF998A400000C0FF0DDD19400000C0FF"> : tensor<4x5x6xf32>}> : () -> tensor<4x5x6xf32>
    "func.return"(%0) : (tensor<4x5x6xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

