"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x2xi64>, tensor<5x2x2xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x084049C0734039BF7E4007C06B3E114018C04C40414089C0A0C0D9C09C40F5BEA53F584038BF56409A40D740D03F2FC08F40B9BFC93FFDBF3C406F3FCEC09C403240043E134095BF733E66C0073F3CC00B40243F18C082C019BF05C1E83F433E0E4078C0BC402F40F2BEAFC08D3FC5BFF2BF2E407AC0D0BEEEBF7C3FA3BF0BC0FFBE593E5C3FC1409B40D1BF5940EA3F86C092BF87C0C83F7B40143F8F3F394006BF54BE3F401D3F40C0A6BFAAC0E2C0833F2340123FEBBED04055C05C40F63F3C403240AE3E44C068C03B4098BE633FF93F5EC03BC053C008C095404C409F4038404F3C9CC07140FA3F95C00DC090C09040B93F1440C2BFBDC081C0DCBFC6BF0AC0C73F09C08940B33F13C0ACBF3D40CCBF4E3E044013BF6D40233F31400E402BC04BBFDA3FC5BEA83FF63ED7BF03C06A401A4094BF8140713E4D3FC23F113F75C019C02DC08ABF9D3FFABD27403640EE3FABBFE7BE7AC0E83E5340C83E714055C02BC0D840943F0C4023BF36BDE03F3D400FC00F4087BDE13FE1BE90BF6FC05EC099C049402640FD3ED4BFDD3EDD3F8340CD3F28C01840C3C07DC013BDBE3FB5BF6140"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[1.367190e+00, 2.109380e+00], [-2.968750e-01, -2.515630e+00]], [[3.812500e+00, -2.328130e+00], [-1.867190e+00, -1.015630e+00]], [[2.921880e+00, -2.281250e+00], [1.210940e+00, -2.484380e+00]], [[5.437500e+00, -4.843750e+00], [1.308590e-01, -7.539060e-01]], [[-6.367190e-01, -4.062500e+00], [-1.906250e+00, 3.078130e+00]]]> : tensor<5x2x2xbf16>}> : () -> tensor<5x2x2xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x084049C0734039BF7E4007C06B3E114018C021C0414089C0A0C0D9C09C40F5BEA53F074038BF56409A40D740D03F2FC08F40B9BFC93FFDBF98BE6F3FCEC09C403240043E134095BF733E66C0073F3CC00B40243F18C082C019BF05C1E83F433E0E4078C0BC4082BFF2BEAFC08D3FC5BFF2BF2E407AC015C0EEBF7C3FA3BF0BC0FFBE593E5C3FC1409B40D1BFEFBFEA3F86C092BF87C0C83F7B40143F8F3F394006BF54BE3F401D3F40C0A6BFAAC0E2C0833F2340123FEBBED04055C05C40F63F3C403240AE3E44C068C012C098BE633FF93F5EC03BC053C008C095404C409F409B3F4F3C9CC07140FA3F95C00DC090C09040B93F1440C2BFBDC081C0DCBFC6BF0AC0C73F09C08940B33F13C0ACBF41BFCCBF4E3E044013BF6D40233F31409BC02BC04BBFDA3FC5BEA83FF63ED7BF03C06A401A4094BF8140713E4D3FC23F113F75C019C02DC08ABF9D3FFABD27403640EE3FABBFE7BE7AC0E83E5340C83E714055C02BC0D840943F0C4023BF36BDE03F3D4082C00F4087BDE13FE1BE90BF6FC05EC099C049402640F4BFD4BFDD3EDD3F8340CD3F28C01840C3C07DC013BDBE3FB5BF6140"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

