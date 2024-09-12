"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2xi64>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xAB3E404047408F3FBABE7E3F19C04A3FFDBFBC405940B4BF4F3F79408B40FFBFF43F363E31C08AC0DCC0BE3F26C0A84013C1BBBFB0C0C740684006C0723EB3BF3E3FC83F454089C0123FB6BF09C0AFC05340E83F4C40C73E903EB83ECEC067406C401040CE409F3F1640A1BF21401A3F2BBF31C0ACC07040D1BEABBFAA40F63F6A40CF3F063F52C00440204004C013C077C057C08EC0A4408BC0F33EA3C01140BFBFA0BE5BC02B409CC04CBF9540AF3EC63F464022BF5CBF0140DEBF5740A5C03D3F433F963FBAC0883E1F4038C0104087BE43C0B13E34C0553F98BFB5BE76C06D3E233F28BF9040443F683F57C0E2BF844001C0D83F59C0C34078C00CBFBF3EBC3F1C40B5C06040F33F4DBF42C06DC098BFA53E00BF903FA140103FDA3FD53F4D3E5DBE2840573F4D3FDFBE4CC0774053BF06BF23C049C08FBD01400AC052400C4019BF3DC0A03F173E9240D7BF6240ACBDD83F4C40E2BE983F8F3F43C00740D5BF0540B040BDBF6D4083C04BC0BFBF09C054406940223F1ABEC83FA5C03EBF01C06E4028C0A3C048C042BF73C06D40044015411EC09F40BD3E6DBF81C06D3F43C01CBE"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.062500e+00, 2.500000e+00, -1.953130e-01, -7.617180e-01, 1.781250e+00, 4.593750e+00, 9.062500e-01], [1.343750e+00, -2.843750e+00, -9.648430e-01, -2.216800e-01, -1.890630e+00, -2.140630e+00, -4.406250e+00]]> : tensor<2x7xbf16>}> : () -> tensor<2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xAB3E404047408F3FBABE7E3F19C0D0BF9EC093BF25C020C06E4062408B40FFBFF43F363E31C08AC0DCC0BE3F26C0A84013C1BBBFB0C0C740684006C0723EB3BF3E3FC83F454089C0123FB6BF09C0AFC05340E83F4C40C73E903EB83ECEC067406C401040CE409F3F1640A1BF21401A3F2BBF31C0ACC07040D1BEABBFAA40F63F6A40CF3F063F52C00440204004C013C077C057C08EC0A4408BC0F33EA3C01140BFBFA0BE5BC02B409CC04CBF9540AF3EC63F464022BF5CBF0140DEBF5740A5C03D3F433F963FBAC0883E1F4038C0104087BE83C07CBF2E403DBE1040423F87416D3E233F28BF9040443F683F57C0E2BF844001C0D83F59C0C34078C00CBFBF3EBC3F1C40B5C06040F33F4DBF42C06DC098BFA53E00BF903FA140103FDA3FD53F4D3E5DBE2840573F4D3FDFBE4CC0774053BF06BF23C049C08FBD01400AC052400C4019BF3DC0A03F173E9240D7BF6240ACBDD83F4C40E2BE983F8F3F43C00740D5BF0540B040BDBF6D4083C04BC0BFBF09C054406940223F1ABEC83FA5C03EBF01C06E4028C0A3C048C042BF73C06D40044015411EC09F40BD3E6DBF81C06D3F43C01CBE"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

