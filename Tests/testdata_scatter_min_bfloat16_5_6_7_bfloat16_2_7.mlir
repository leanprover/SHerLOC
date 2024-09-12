"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2xi64>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xE34009BF2FBE82C0123F48C0AE3F83C088C04AC086C0343FCE40E24050C00AC0613F733F22BF14BF9F3FB1BF1CC098BEBD3E7FBF883EABBFB6405FBE3DC0E2BF85C03141C9BF8EBF3F408CC006C04E406BBEB53F41C06CBF8BBF7C3F853D234024BF6DBFD83F7A406ABFC340BB3F1DC01ABEB34082BF80406ABF71BE264037C03ABE38C0D7C0C43FF2C0E3BE2DC0DD3F6340EDC0944087C08B40E43F8F409FBF83BF703F3D4054C0113F45BE87BF8CC033C066C019C00A41AF3FDEBEE53F53BE25C0B3C08540DF3F123F9DBF6BC0DD4057BF58C08FBF54C0F43FBC3F0B40803F51C0203FD73F97C0E63F2A401A4095C0BC404040BA3FA93F9640834072C0F53F47400EC0214067C0D7BE4CC0C53ED63FC93E5DC00FC0193F84C04D4065BF6E3D4AC052C02240EAC0F0BED1BF4840F33F04C0014041C05FBD39C036BFD23E39C07240094065BF8ABF3EBF1940D33F88C047C09140103F82C07F40A83D34BF6B40E63D05404C4087BF9A40AB40B3BE1740B0BF074031C0643F00BEA1404E3E243E3FC086BED4BF8AC0923F84BF0EC0A7C0353F09C087BFF93F2340AFBFCEC082BD6EBFD93F"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-1.960940e+00, -1.281250e+00, 3.140630e+00, 2.859380e+00, -1.000000e+00, 4.656250e+00, -3.593750e+00], [-4.875000e+00, 8.945310e-01, 2.171880e+00, -9.296870e-01, 3.250000e+00, 7.226560e-02, 3.066410e-01]]> : tensor<2x7xbf16>}> : () -> tensor<2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xE34009BF2FBE82C0123F48C0AE3F83C088C04AC086C080BF954066C050C00AC0613F733F22BF14BF9F3FB1BF1CC098BEBD3E7FBF883EABBFB6405FBE3DC0E2BF85C03141C9BF8EBF3F408CC006C04E406BBEB53F41C06CBF8BBF7C3F853D234024BF6DBFD83F7A406ABFC340BB3F1DC01ABEB34082BF80406ABF71BE264037C03ABE38C0D7C0C43FF2C0E3BE2DC0DD3F6340EDC0944087C08B40E43F8F409FBF83BF703F3D4054C0113F45BE87BF8CC033C066C019C00A41AF3FDEBEE53F53BE25C0B3C08540DF3F123F9DBF6BC0DD4057BF9CC08FBF54C06EBFBC3F943D9D3E51C0203FD73F97C0E63F2A401A4095C0BC404040BA3FA93F9640834072C0F53F47400EC0214067C0D7BE4CC0C53ED63FC93E5DC00FC0193F84C04D4065BF6E3D4AC052C02240EAC0F0BED1BF4840F33F04C0014041C05FBD39C036BFD23E39C07240094065BF8ABF3EBF1940D33F88C047C09140103F82C07F40A83D34BF6B40E63D05404C4087BF9A40AB40B3BE1740B0BF074031C0643F00BEA1404E3E243E3FC086BED4BF8AC0923F84BF0EC0A7C0353F09C087BFF93F2340AFBFCEC082BD6EBFD93F"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

