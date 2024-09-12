"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      "stablehlo.return"(%arg1) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x2xi64>, tensor<5x2x2xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x76BE0940F8C0843EC4BF87C092C02AC036BF094041C0144013C0D53F3D4002C0C3BF2DC0AF409AC077405FBFCDBF653FD8BFAAC0AA3FC13F4540863E544085404CBFE7BF0940A140C13E6DC0DF3E4F3FEABF78C066C00CC09DBFAB3F89C0A53E1640393F69BF38C08D3FB2BE27C01B402E4004C06FC0AFC06C4099407140F33EAA3E9D3F92C0F13F9BBE37C0613F76BF5EC033406CC0C63EF4BF40BD14BFABC03EBFE2BFA03E21C06DC0DB3F6E3F1C40AFBFA63F1040FAC00E40A03BD13E3C401040753F54C0144028C038C004C12FBE9FC01640FABD89C02740A6BF143F783B5E3E7E3EA5405040C43F35BF8CBF2440A23F18C114BF12C0FDC05AC083C0C73F06C07240FC3F783F50408BC07340084040403E3F9B408B3F973F82C071BF91C01A408F404F40913FC53FA73FAEC0FFBF8E40D9BF28C02C406DC0BC3F004082BF19C01340BABFDF3F733F0FBFB0BF48404C3F9E408AC0CEBF1EBFC03E7AC0A0BD12BFB9BFE5403640D6BF453F86BF6BC0DBBE9340A1C0D53DC43E0AC0824056C08440DE3F16C053C0813FBFC0564063C001C11DBFBBBF14C0BFBF3CBF433FE840C4BE2840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-3.093750e+00, -3.979490e-02], [2.312500e+00, -4.296880e-01]], [[-3.710940e-01, -2.968750e+00], [2.875000e+00, -1.640630e+00]], [[-1.023440e+00, 4.875000e+00], [2.171880e+00, -5.000000e+00]], [[1.992190e-01, 2.080080e-01], [-3.218750e+00, -3.312500e+00]], [[2.617190e-01, 2.050780e-01], [1.523440e+00, 2.546880e+00]]]> : tensor<5x2x2xbf16>}> : () -> tensor<5x2x2xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x76BE46C0F8C0843EC4BF87C092C02AC036BFDCBE41C0144013C0D53F3D4002C0C3BF23BDAF409AC077405FBFCDBF653FD8BFAAC0AA3FC13F1440863E544085404CBFE7BF0940A140C13E6DC0DF3E4F3FEABF78C066C0BEBE9DBFAB3F89C0A53E1640393F69BFD2BF8D3FB2BE27C01B402E4004C06FC03EC06C4099407140F33EAA3E9D3F92C0F13F9BBE37C0384076BF5EC033406CC0C63EF4BF40BD14BFABC03EBFE2BFA03E21C06DC083BF6E3F1C40AFBFA63F1040FAC00E40A0C0D13E3C401040753F54C0144028C09C4004C12FBE9FC01640FABD89C02740A6BF143F783B0B407E3EA5405040C43F35BF8CBF2440A23F18C114BF12C0FDC05AC083C04C3E06C07240FC3F783F50408BC0734054C040403E3F9B408B3F973F82C071BF553E1A408F404F40913FC53FA73FAEC0FFBF8E40D9BF4EC02C406DC0BC3F004082BF19C01340BABFDF3F733F0FBFB0BF48404C3F863E8AC0CEBF1EBFC03E7AC0A0BD12BF2340E5403640D6BF453F86BF6BC0DBBE523EA1C0D53DC43E0AC0824056C08440DE3F16C053C0C33FBFC0564063C001C11DBFBBBF14C0BFBF3CBF433FE840C4BE2840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

