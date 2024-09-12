"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<2x7xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2xi64>, tensor<2x7xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<2x7xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x2F2DD8C24D3D0A3F823FB5C106C07D45F1C389BC2FBDBBBB75BD4347EEC40442DB3B60B5D542BC404AC41E442DBC703DDE3063B764C008C49BC1D0C0373F7EB6BAC101C4F3C37F44883D16BC2E429443094051C45D4300460245D540D5B9913554BD2EC463BB7C4562C1ECC3F44095BB8D45424446C48F3E1941CCBEE9BC5535C540F5C1CBC6D3C180C301C1C941723F3640194658C060B86141DDC14F3C313894421330914258C8803D53C5C642513D624777C10F4171406B489D41EB3E49C1B9412131A3C4E4405ABF2042943D9DB8F93F293BECBCCEB99CBC6A382CC0EBC3FD357F422044923B5242ABC152BB36B6CAC359AE6EBDE6C430B9B1C51E2C6FBC13BF173D513DA5C1FB415CB4DAB3DEB772BD13432D34703F8B389AB1D6C2084528B93C402FBD67408BBC30C03244F13E51C301C0B842DA3CF9C624BE2344C840C34119C53440A641042658C1FB422B3B86C39F4213C1D144513B2DBDF3C061C249BD8734B5C22A43823A32BA1344E6C3A7431342E8C0CEBCFCC302C47DBEB345D8326DB2D9C26DC104C4DF41C345C4441FC4DDB88AB7193FF63FDF388532D1B6E9BDC5C0"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[3.884770e+00, 3.884770e+00, 1.958010e+00, 1.750980e+00, -8.234380e+00, -2.351560e+00, 4.421390e-01], [-8.447270e-01, 6.459960e-01, -4.589840e+00, -1.073240e+00, 3.373050e+00, 1.296880e+00, -4.042970e+00]]> : tensor<2x7xf16>}> : () -> tensor<2x7xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<2x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x2F2DD8C24D3D0A3F823FB5C106C0554DB6CB71C08AC0F5476B426C42EEC40442DB3B60B5D542BC404AC41E442DBC703DDE3063B764C008C49BC1D0C0373F7EB6BAC101C4F3C37F44883D16BC2E429443094051C45D4300460245D540D5B9913554BD2EC463BB7C4562C1ECC3F44095BB8D45424446C48F3E1941CCBEE9BC5535C540F5C1CBC6D3C180C301C1C941723F3640194658C060B86141DDC14F3C313894421330914258C8803D53C5C642513D624777C10F4171406B489D41EB3E49C1B9412131A3C4E4405ABF2042943D9DB8F93F0CBA5CBAA942F23C723F69C1004CFD357F422044923B5242ABC152BB36B6CAC359AE6EBDE6C430B9B1C51E2C6FBC13BF173D513DA5C1FB415CB4DAB3DEB772BD13432D34703F8B389AB1D6C2084528B93C402FBD67408BBC30C03244F13E51C301C0B842DA3CF9C624BE2344C840C34119C53440A641042658C1FB422B3B86C39F4213C1D144513B2DBDF3C061C249BD8734B5C22A43823A32BA1344E6C3A7431342E8C0CEBCFCC302C47DBEB345D8326DB2D9C26DC104C4DF41C345C4441FC4DDB88AB7193FF63FDF388532D1B6E9BDC5C0"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

