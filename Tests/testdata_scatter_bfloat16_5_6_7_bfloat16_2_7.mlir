"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      "stablehlo.return"(%arg1) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2xi64>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xADC01240A8BFC4C0213E66C0024046408440043FEA40F43E6ABEFEBF8FBF7A4038C023C01FC139BFB2C09CC075C0883FC3BD36C033403C4015BF5B3FC9BF2C401C4081C08F406B408CC0F13FB13F6440BABF4AC0843FC63F94BFA13F36BF323FB8402740EE4087BF5A40BABFAFC061409640913FA240964050C0FE3DAC3F7BC07DBD9440E5BFC6BF984040C0DABF9AC03EC07D4029C0CEBF9FBFAA3F3B409CBE19C10840F33FFD3FA53E45C094404B3D88BF45C06AC05F4065C05140D3C0CC40064012C08D3FAD3FA63F0E4039BE0840B04010C06B402F40853FB4BED4BFF13F87BF4DBF2140594058C080C0803E4B40E840B6BFAB3E1DC0C83E9B3F8040FCBF86C00AC0C9BFC5BE174082C006406840C03FA0BE67C08C408FBF0740D8403AC0F73E2C401FC097C091402A405CC0003FC4BF88C0D9BE87BFA8BFCE4003C0C040263FFABF163E7440DB3F0A409E3F1BC0963E7F4081BC84403CBF303EC43F6D3F93C08A40C1BF6D3F40C0854062406B404EBED7BF814033403EC0604054C092C0B8C0523EBD40C5C0E8BF25C0C73E85401140C1BE85C00F40A53FAB40143F7FC0533D9ABF"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[3.078130e+00, -1.304690e+00, 3.453130e+00, 1.289060e+00, -3.296880e+00, -4.125000e+00, 4.843750e+00], [-3.718750e+00, 7.937500e+00, 3.234380e+00, 1.738280e-01, 2.265630e+00, 1.767580e-01, 2.015630e+00]]> : tensor<2x7xbf16>}> : () -> tensor<2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xADC01240A8BFC4C0213E66C002404540A7BF5D40A53F53C084C09B408FBF7A4038C023C01FC139BFB2C09CC075C0883FC3BD36C033403C4015BF5B3FC9BF2C401C4081C08F406B408CC0F13FB13F6440BABF4AC0843FC63F94BFA13F36BF323FB8402740EE4087BF5A40BABFAFC061409640913FA240964050C0FE3DAC3F7BC07DBD9440E5BFC6BF984040C0DABF9AC03EC07D4029C0CEBF9FBFAA3F3B409CBE19C10840F33FFD3FA53E45C094404B3D88BF45C06AC05F4065C05140D3C0CC40064012C08D3FAD3FA63F0E4039BE0840B0406EC0FE404F40323E1140353E014087BF4DBF2140594058C080C0803E4B40E840B6BFAB3E1DC0C83E9B3F8040FCBF86C00AC0C9BFC5BE174082C006406840C03FA0BE67C08C408FBF0740D8403AC0F73E2C401FC097C091402A405CC0003FC4BF88C0D9BE87BFA8BFCE4003C0C040263FFABF163E7440DB3F0A409E3F1BC0963E7F4081BC84403CBF303EC43F6D3F93C08A40C1BF6D3F40C0854062406B404EBED7BF814033403EC0604054C092C0B8C0523EBD40C5C0E8BF25C0C73E85401140C1BE85C00F40A53FAB40143F7FC0533D9ABF"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

