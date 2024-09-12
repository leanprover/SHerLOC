"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2xi64>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<2x7xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0BC8EEC1C938943F90C7ACC3A845F039533C23BEF13AC7BAE1C2873B96BE8041CF289A428C471B4618435AC15941213934453EC453C009C40F439EC05FB1A84520C5953F3DC2F5C3CBC264C0ABAFE93C41C37DBBCDC37E40813C2241204005C361ACAAB972BECF3F5846E3C22BC5C5464239B9C3F3395EC2FD4076413D3A58C06A4481452433B03303B82E40D53EE8C089B9A2343F4322B56DBFCF4141392D2FEA418F44AEB75B40FDB4B0C2E1387240D2BD2343143F51313FBDAFC4D6434C3D51C3444020446F39E14449C441BF003AD6407FC12C3FB6C614B476BFCC374EBEBAC43BBB64C6E0C049BB34C05D412D391D4090BCD142DDBD7A473F3E41BE09B0C5C3FF317AC01CBD893E37BDF33BD042FBB93C4020BCEE4409BFB1C17DC0F34587C3D0C52FC1C0451CBDDBBCC0C0443FF4B800AC17B1243F11BF763545A5263985C1254181C1963F84BE75239BC4DEBE3142393469C1F038E7C6814093C39ABEBFBB2F3093BC08C3DBC0832ADE3C13C167BFC0BC38BF0A439DBAAC3865453E400F27454498BDCFC426BD74AF393D27322335613C1EC131C2E7407BC55E3936C251C26AB9"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-7.573240e-01, -6.106570e-02, -5.308590e+00, -4.632810e+00, -2.304690e+00, 3.986330e+00, -4.210940e+00], [-3.023440e+00, -8.007810e-01, -1.676760e+00, -2.931640e+00, 6.664060e+00, -9.655760e-02, 2.437740e-01]]> : tensor<2x7xf16>}> : () -> tensor<2x7xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<2x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0BC8EEC1C938943F90C7ACC3A845C0A3143CD8C688C34EC260388AC296BE8041CF289A428C471B4618435AC15941213934453EC453C009C40F439EC05FB1A84520C5953F3DC2F5C3CBC264C0ABAFE93C41C37DBBCDC37E40813C2241204005C361ACAAB972BECF3F5846E3C22BC5C5464239B9C3F3395EC2FD4076413D3A58C06A4481452433B03303B82E40D53EE8C089B9A2343F4322B56DBFCF4141392D2FEA418F44AEB75B40FDB4B0C2E1387240D2BD2343143F51313FBDAFC4D6434C3D51C3444020446F39E14449C441BF003AD640C6C5F03B32C860C2CC44403654BDBAC43BBB64C6E0C049BB34C05D412D391D4090BCD142DDBD7A473F3E41BE09B0C5C3FF317AC01CBD893E37BDF33BD042FBB93C4020BCEE4409BFB1C17DC0F34587C3D0C52FC1C0451CBDDBBCC0C0443FF4B800AC17B1243F11BF763545A5263985C1254181C1963F84BE75239BC4DEBE3142393469C1F038E7C6814093C39ABEBFBB2F3093BC08C3DBC0832ADE3C13C167BFC0BC38BF0A439DBAAC3865453E400F27454498BDCFC426BD74AF393D27322335613C1EC131C2E7407BC55E3936C251C26AB9"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

