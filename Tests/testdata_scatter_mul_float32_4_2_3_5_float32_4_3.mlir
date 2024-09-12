"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xf32>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f32>, %arg1: tensor<f32>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f32>, tensor<f32>) -> tensor<f32>
      "stablehlo.return"(%7) : (tensor<f32>) -> ()
    }) : (tensor<4x2x3x5xf32>, tensor<2xi64>, tensor<4x3xf32>) -> tensor<4x2x3x5xf32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3x5xf32>, tensor<4x2x3x5xf32>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xf32>, tensor<4x3xf32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x84FBC8C021A4C4C00431C13FF5DED4404CBF98BE7E376C40EBF77AC02EA8E53F7EA8803F0D56763FDB55C93F85C23C4053355FC00E9D4E405147084023E56940065F364066BB6ABEDF2B20C068E03040BBFE13BF07213A40616D454056C504C0F3A3EA40E17EC4BFFA60D0BE3A61D4BF5B5BB03F664BD140D18754C02F0F20C0B23B5A3F827F27C0A55E2740C5A5A4C0125B0DBFE7A590BF191B013F53E3BBC065599140CC7A9C3F5808B23F36DFF8BFC2EEC3BF3536783F7BAE704063D95A3F93496B3E16FA5C3DA305B43F3078A7C021EDFEBF97CE79BFA7868340FEF79AC0C6BD4740C8BD69BFCB8F9EC00C0931405FACABC0AA1A9F40E7632AC0D285664053200C3FD0B1864049150A40C4511CBFDA2F9940ADDCA9BF95E77DC03C6CA4C07D1C79C0D71D14BDB98D803FA79B03C09686ACBF6A3607C0721A42C0F824AB4052E5A5408B338FBFE772C93F22597740D949BF3FE05492BF252AF040941DD93FFAF5ADC055DFA93E217B384033959CBEAA5A183F72EE12C07F1C8F3F2EB5DABE8A1C4EC09A8765405095963FD14266BFB6D2D03F50E4343F49A042C05DD47BC0FA3984C0B319273E83B50BC0C3396DC06B8A9240F3FAB1C020DE074080234340E5659D3F9693FABF506B9F4097F1A7C01C8F1EC0DA90B0C04D6BA63FE022ED40"> : tensor<4x2x3x5xf32>}> : () -> tensor<4x2x3x5xf32>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.343577415, 3.2909205, 2.0481863], [1.4194442, 1.15045309, 5.463560e-01], [3.94921708, 1.17333698, -0.110294797], [-1.61713457, 1.74694538, 0.211081758]]> : tensor<4x3xf32>}> : () -> tensor<4x3xf32>
    "func.return"(%1, %2) : (tensor<4x2x3x5xf32>, tensor<4x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x84FBC8C021A4C4C00431C13FF5DED4400AECD13D7E376C40EBF77AC02EA8E53F7EA8803F0DAB4A40DB55C93F85C23C4053355FC00E9D4E40DD8F8B4023E56940065F364066BB6ABEDF2B20C068E03040BBFE13BF07213A40616D454056C504C0F3A3EA40E17EC4BFFA60D0BE3A61D4BF5B5BB03F664BD140D18754C02F0F20C0B23B5A3F827F27C06C926D40C5A5A4C0125B0DBFE7A590BF191B013F0428D8C065599140CC7A9C3F5808B23F36DFF8BF141956BF3536783F7BAE704063D95A3F93496B3E16FA5C3DA305B43F3078A7C021EDFEBF97CE79BFA7868340FEF79AC0C6BD4740C8BD69BFCB8F9EC00C0931405FACABC0AA1A9F40E7632AC0D2856640E6580A40D0B1864049150A40C4511CBFDA2F99402E4EC7BF95E77DC03C6CA4C07D1C79C0D71D14BD57DCE2BDA79B03C09686ACBF6A3607C0721A42C0F824AB4052E5A5408B338FBFE772C93F22597740D949BF3FE05492BF252AF040941DD93FFAF5ADC055DFA93E217B384033959CBEAA5A183F72EE12C0256EE7BF2EB5DABE8A1C4EC09A8765405095963F6F20C9BFB6D2D03F50E4343F49A042C05DD47BC0E7485FBFB319273E83B50BC0C3396DC06B8A9240F3FAB1C020DE074080234340E5659D3F9693FABF506B9F4097F1A7C01C8F1EC0DA90B0C04D6BA63FE022ED40"> : tensor<4x2x3x5xf32>}> : () -> tensor<4x2x3x5xf32>
    "func.return"(%0) : (tensor<4x2x3x5xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

