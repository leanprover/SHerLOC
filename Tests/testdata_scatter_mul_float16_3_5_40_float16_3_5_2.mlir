"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5x40xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<1> : tensor<2x1xi64>}> : () -> tensor<2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<3x5x40xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [2], scatter_dims_to_operand_dims = [2], index_vector_dim = 1>}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<3x5x40xf16>, tensor<2x1xi64>, tensor<3x5x2xf16>) -> tensor<3x5x40xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x5x40xf16>, tensor<3x5x40xf16>) -> ()
    "func.return"(%6) : (tensor<3x5x40xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<3x5x40xf16>, tensor<3x5x2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x9AC3AAB639C2AABE2BC1C9C625B82ABE04BB4EBA0CC3513DC13E18405241B4BD0CBB67C0E6B8C1C1663D9540954149BC0A3A16C0DFC1BBBDD53E9A3EB5409E3C294069BFE141CA41843C70C2953A84423DC3F3402DC5E6C2483E2A42D5406EC1863754C026C00146D7463D3E0A440E3FCAC35BC1F0419F4508BAEFA60240FEC2E5C2BDB910BB82A173C308C1DCC3A23D9F42E5C135C1433E7638483CC841D2C0183C57C08D3B9AB5BF331A3F124058C14DC189BD1EC12E435C44843D39B9F8C20ABE693C4ABFDBBE8138974681B60A4159341B429339E143F8B07CB766B9D63D2544B7C53D41C4444F2EADB00C48E6C3333930BC87C25B2DE7C12EC4FD44D2452541064590401743C13866A1C23AE8BC8D3DD8C4FDB6D3C2AC43BD3A4944FCB58CBA9ABD40C133B6F9C25731E7389CC2ADC1AF42F4BC70C18E41C743F1430FBB99BF4E44A6C6013C7EBA4DBDBDB2353E14BFD5C897C462C1AFBA6740AA4311C1E3400D4373C59CA7BB3C093CA7C2B93F83C40D42C6410E4603C42FC368B3C04172BAFC426BC5974057B9F835CF3A44B44F286638A2C1A6461D409F3F73440D3A7BC30942B02AF54137C129390AC454C556C3C7C47CBFCEBAE242D43FF2361246253EC2C05230F440C6C10A3A78C54A45A945F5AB4FC13DBEC83E49BB06B751442B3CDEC10EBF2C401FC0A0C4184123BD34234B3BE2442BBFC9C263C5283C55C20A40A4BA3C423CBE54434CC3A2BC96C2E8BEA93E90C088C1503F753F9FC44CBE62C25F398638E34443C1F340A0BCADBE0DC4FABFDE2EE043B5BC6DBCF3B90238244117C474BCADA98FC5F63C5F34B4420541D7BD0A3A453CF93876BF5F42BE41FD37933AB4C4B6C436417A3EAEC38CA3FE22E23D072F0BC284BD48C4C3C3753D57B6C7BEAFBE2ABD2EAF1842F340B43594C16CBFAA3A4CC2BCC3B2C02C3911B831C3F23E1FB928458CC09E4128C612371DB97FBB32C5A6BF803C013EAA3810BDDAB9B9BC01C66DC5463A05B1212B7D407E4157426EC6D62F0FC23AC0F03CFAC1B43A60C1683D95B0183F65419FC0AF36833CE74043BB91BDCEC0A1416A40ECC2C721A1BE72412B4003408943ADBA3F450A41FCC24AB59A4321AD9F4314B5FFC4473F803AE9BED9BDB9C09644BBB9EB42E43C9CB21EC31FBB123C7ABB59B9DA4288C37AC560427C37B53C07BC9A412CB23A45E7BCD6C1E2C03630583E73C61040D63EE2C321BB66BDEEC6713B5845C33F1DC4374507308341F6B142BB164181BB21C15340C531B43B34BC6EBDD4C18C42F9C1C6C17743CEBD33C3E7C24F3CB23DFA456B470AC0DAB2DBAA1DC284C3FA3F35C3B93CBA450FBD19C68FBA64BC96408F42FCC06E37D93FDFC1F23E573BCCC0C83275383AC560A6033F9F3426449728CAC26D3CB13ECEBFAEB8AE34A33D3EBB66B628AF40BD4FBD003EF1C52BAE50C114C1844706AD2644B33F4740CDC1BFC49C3E0FB5EB336FBD23378E3552409BBFA4BD66BD1440424134C30A4141B891404BBB19C2E13A893CEC41E041A9C617B234B8E7BDB7421CBD00C47E42F7C379C503386843A73B9343433534355CB8B1C27243943E66C0D1C2CCBBFE3D05410C3BEF37352E77BD9BC17CC2003600442B462645EDBFA7BF5DB237C3C2C20DC32EBF8E454DC01F4339C75B45E8488CC3C542AE3A383C28C3A446A94672C1C83F11C1"> : tensor<3x5x40xf16>}> : () -> tensor<3x5x40xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[9.965820e-01, 6.218750e+00], [4.812500e+00, 4.285160e+00], [-1.906250e+00, -2.439450e+00], [2.953130e+00, -1.923830e+00], [-1.714840e+00, -3.303220e-01]], [[-5.179690e+00, 6.628900e+00], [1.573240e+00, -3.460940e+00], [-2.537110e+00, -3.958980e+00], [-3.126950e+00, 4.027340e+00], [2.062500e+00, 2.138670e+00]], [[-1.911130e+00, 1.304690e+00], [-1.436770e-01, 2.777340e+00], [2.025150e-01, -1.497070e+00], [-5.332030e-01, -2.575680e-01], [6.772460e-01, 2.705080e+00]]]> : tensor<3x5x2xf16>}> : () -> tensor<3x5x2xf16>
    "func.return"(%1, %2) : (tensor<3x5x40xf16>, tensor<3x5x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5x40xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x9AC329C139C2AABE2BC1C9C625B82ABE04BB4EBA0CC3513DC13E18405241B4BD0CBB67C0E6B8C1C1663D9540954149BC0A3A16C0DFC1BBBDD53E9A3EB5409E3C294069BFE141CA41843C70C2953A84423DC361522DC5E6C2483E2A42D5406EC1863754C026C00146D7463D3E0A440E3FCAC35BC1F0419F4508BAEFA60240FEC2E5C2BDB910BB82A173C308C1DCC3A23D9F42E5C135C1433E7638483CC841D2C0183C0CC98D3B9AB5BF331A3F124058C14DC189BD1EC12E435C44843D39B9F8C20ABE693C4ABFDBBE8138974681B60A4159341B429339E143F8B07CB766B9D63D2544B7C53D41C4444F2EADB00C48E6C33339F34587C25B2DE7C12EC4FD44D2452541064590401743C13866A1C23AE8BC8D3DD8C4FDB6D3C2AC43BD3A4944FCB58CBA9ABD40C133B6F9C25731E7389CC2ADC1AF42F4BC70C18E41C743F1430FBB99BFE140A6C6013C7EBA4DBDBDB2353E14BFD5C897C462C1AFBA6740AA4311C1E3400D4373C59CA7BB3C093CA7C2B93F83C40D42C6410E4603C42FC368B3C04172BAFC426BC5974057B9F835CF3A44B44F28B8CCA2C1A6461D409F3F73440D3A7BC30942B02AF54137C129390AC454C556C3C7C47CBFCEBAE242D43FF2361246253EC2C05230F440C6C10A3A78C54A45A945F5AB4FC13DBEC83E49BB06B751442B3CFD4B0EBF2C401FC0A0C4184123BD34234B3BE2442BBFC9C263C5283C55C20A40A4BA3C423CBE54434CC3A2BC96C2E8BEA93E90C088C1503F753F9FC44CBE62C25F398638E34443C1F340A0BCADBE0DC402CDDE2EE043B5BC6DBCF3B90238244117C474BCADA98FC5F63C5F34B4420541D7BD0A3A453CF93876BF5F42BE41FD37933AB4C4B6C436417A3EAEC38CA3FE22E23D072F0BC284BD48C4C3C3753D57B6554DAFBE2ABD2EAF1842F340B43594C16CBFAA3A4CC2BCC3B2C02C3911B831C3F23E1FB928458CC09E4128C612371DB97FBB32C5A6BF803C013EAA3810BDDAB9B9BC01C66DC5463A05B1212B7D407E41FE4A6EC6D62F0FC23AC0F03CFAC1B43A60C1683D95B0183F65419FC0AF36833CE74043BB91BDCEC0A1416A40ECC2C721A1BE72412B4003408943ADBA3F450A41FCC24AB59A4321AD9F4314B5FFC4473F0DC0E9BED9BDB9C09644BBB9EB42E43C9CB21EC31FBB123C7ABB59B9DA4288C37AC560427C37B53C07BC9A412CB23A45E7BCD6C1E2C03630583E73C61040D63EE2C321BB66BDEEC6713B5845C33F1DC429C007308341F6B142BB164181BB21C15340C531B43B34BC6EBDD4C18C42F9C1C6C17743CEBD33C3E7C24F3CB23DFA456B470AC0DAB2DBAA1DC284C3FA3F35C3B93CBA450FBD19C68FBA64BC96408F420C3A6E37D93FDFC1F23E573BCCC0C83275383AC560A6033F9F3426449728CAC26D3CB13ECEBFAEB8AE34A33D3EBB66B628AF40BD4FBD003EF1C52BAE50C114C1844706AD2644B33F4740CDC1BFC49C3E8FA9EB336FBD23378E3552409BBFA4BD66BD1440424134C30A4141B891404BBB19C2E13A893CEC41E041A9C617B234B8E7BDB7421CBD00C47E42F7C379C503386843A73B9343433534355CB8B1C27243064266C0D1C2CCBBFE3D05410C3BEF37352E77BD9BC17CC2003600442B462645EDBFA7BF5DB237C3C2C20DC32EBF8E454DC01F4339C75B45E8488CC3C542AE3A383C28C3A446A94672C1C83F11C1"> : tensor<3x5x40xf16>}> : () -> tensor<3x5x40xf16>
    "func.return"(%0) : (tensor<3x5x40xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

