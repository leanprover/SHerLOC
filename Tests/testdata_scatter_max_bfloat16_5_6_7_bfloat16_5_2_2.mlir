"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x2xi64>, tensor<5x2x2xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xF240EC40FFBF1FC07FC07F3F16BFEB3F50C08240EC3F92BF533FBC3F8BBFDD3FC53F2E4048BED5BF19C0BEBE92C03AC0D23F6D405540673FFCC0AE3F4040EC3D97408EBF903FDABD72BFE24084BF3FBE0A4000409B3F21C04CC0AA3E1540A7C08B4032C0A74097BDE8BFBCC04740EB3F803F96C04240F73D8740B83E184025C0413FD13EA13FC53F683E64C06DBFAE3EAFBF33C0F43F3A40B7BF9C3FB63F88BF36BF293FE53E433E0DBF2EC0AD3D21BF30C085C042BF0E407F406140414031C080C0CBC00B3F5DC0DAC0A9C0C93FE83F65C09ABFE73F243EBC4090C0E23F11C03BC0E1BFF9BD7AC0DC40E33E184071C039C06CBD28C0934061BF74BFEC3CD4BF1B403AC0A8BF19C0F53E9D3FD93EB9BF6FBFC2BFFCBED2C05FC088C08C3FABBE80BEB9C0D63E083F09BED240BA3F43BE8EC0E63E3E40FBBF09408E4036C0B13F7740013E8E40973FE43F7A3F30C0943F2B401EBFDF3F303FDFBF69C0B3BF89BF1EC014BE4E3F354036C0F7BD0C3FFD3FDBBD5A40A5404E40823F38403640AB3F83C017C0D53FA340173F0DBCFD3FE5BF164088C076C0B93F8D3F64BF7940AAC04FBE7840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-7.562500e+00, 3.140630e+00], [-1.125000e+00, 3.703130e+00]], [[3.531250e+00, 1.367190e+00], [-1.765630e+00, -1.484380e+00]], [[-6.562500e-01, 1.070310e+00], [5.812500e+00, -2.015630e+00]], [[-3.421880e+00, -1.343750e+00], [-5.703130e-01, -1.226560e+00]], [[2.312500e+00, 2.265630e+00], [-1.671880e+00, -3.078130e+00]]]> : tensor<5x2x2xbf16>}> : () -> tensor<5x2x2xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xF240EC40FFBF1FC07FC07F3F16BFEB3F50C08240EC3F92BF533FBC3F8BBFDD3FC53F494048BED5BF19C0BEBE92C03AC0D23F6D405540673F90BFAE3F4040EC3D97408EBF903FDABD72BFE24084BF3FBE0A4000409B3F62404CC0AA3E1540A7C08B4032C0A74097BDE8BFBCC04740EB3F803F96C04240AF3F8740B83E184025C0413FD13EA13FC53F683E64C06DBFAE3EAFBF33C0F43F3A40B7BF9C3FB63F88BF36BF293FE53E433E0DBF28BFAD3D21BF30C085C042BF0E407F406140414031C080C0CBC00B3F5DC0DAC0893FC93FE83F65C09ABFE73F243EBC4090C0E23F11C0BA40E1BFF9BD7AC0DC40E33E184071C039C06CBD28C0934061BF74BFEC3CD4BF1B403AC0A8BF19C0F53E9D3FD93E9DBF6FBFC2BFFCBED2C05FC088C08C3FABBE80BEB9C0D63E083F09BED240BA3F43BE8EC0E63E3E40FBBF09408E4036C0B13F7740013E8E40973FE43F7A3F30C0943F2B401440DF3F303FDFBF69C0B3BF89BF1EC014BE4E3F354036C0F7BD0C3FFD3FDBBD5A40A5404E40823F38403640AB3F83C017C0D53FA340173F0DBCFD3FE5BF164088C076C0B93F8D3F64BF7940AAC04FBE7840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

