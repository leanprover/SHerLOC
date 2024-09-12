"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<5x6x7xf16>, tensor<2x2x2xi64>, tensor<5x2x2xf16>) -> tensor<5x6x7xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xf16>, tensor<5x6x7xf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xf16>, tensor<5x2x2xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x4A3F12C4A6B051BC3F3B12C403C1F63E00C4BB368EBC8342CC3B0E4627C1753D88BE2445B8B729C51CC33B3E4FBF254131C0F5C3D9C63A3A08C171C56A45D9C12CC55140BDB1D93A4B432CC396C6C2B92D40C643264131C7A14464340644E53F194558B816C37D4496C6DFC018C405C034AA934409BFE9421E34243CE7C7213D74C5FB42E943443F11C45637D2B434422BC188A9EE38E9C0E130A3409B39D4C1E8B87C3C293A8FC1C533E8444CB9BE3BA933C7BB7D45EDC2AE461F3FF5B504C05331A14507318B3DFABCAB350E4353B9824505BF4C3D4A392E3DA1BC36C0F1410EBC7CC2F537A23A65C4633F42C15B4075BA35B950BCD243ABB9A947C0C378442DC0C9BDF34455BE2BC467434ABD6445DDC00AB9B2432942CEBC17B81DC412BDEFC114BF79BCE138673029B5D73CBEC2EDBE744142B7C83D5D3A56419FC1B2C141C53933A0BFDD3A9C384E41EC42D5C27FBE3D42E4364AC5BEC5033CFB3EE7C2EC40C5B138C1422E2BC1D8C0424170415E2FABC0E7C3FEB8283BF0453BB59D3F88BD43398C4408BE803EC5C388BC81B8A7C24FC05BC138416C3EAD40DEBE64454244E143"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[2.298830e+00, -2.277340e+00], [-2.005860e+00, -3.138670e+00]], [[-1.824220e+00, 3.845700e+00], [-3.527830e-01, -1.957030e+00]], [[2.488280e+00, 2.406250e+00], [3.945310e-01, 1.807620e+00]], [[1.765140e-01, 4.096680e-01], [-7.421880e-01, -1.053710e+00]], [[1.026370e+00, -1.147460e+00], [-2.074220e+00, 1.888670e+00]]]> : tensor<5x2x2xf16>}> : () -> tensor<5x2x2xf16>
    "func.return"(%1, %2) : (tensor<5x6x7xf16>, tensor<5x2x2xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x4A3F16BFA6B051BC3F3B12C403C1F63E00C470C18EBC8342CC3B0E4627C1753D88BEBA41B8B729C51CC33B3E4FBF254131C0F5C3D9C63A3A86C471C56A45D9C12CC55140BDB1D93A4B432CC396C6C2B92D40C643264182C8A14464340644E53F194558B816C3104196C6DFC018C405C034AA934409BF4D471E34243CE7C7213D74C5FB42E943443F11C456373CB934422BC188A9EE38E9C0E130A3409B39D4C1E8B87C3C293A8FC1C53365474CB9BE3BA933C7BB7D45EDC2AE462D43F5B504C05331A14507318B3DFABC85410E4353B9824505BF4C3D4A392E3DA1BC36C0F141F4B87CC2F537A23A65C4633F42C15B4075BA35B950BCD243ABB9A947C0C3A5442DC0C9BDF34455BE2BC467434ABD5644DDC00AB9B2432942CEBC17B81DC4DDBAEFC114BF79BCE138673029B5D73CBEC2EDBE7441C8BCC83D5D3A56419FC1B2C141C53933A0BFDD3A9C384E41EC42D5C27FBE2544E4364AC5BEC5033CFB3EE7C2EC40D53E38C1422E2BC1D8C0424170415E2FF6C2E7C3FEB8283BF0453BB59D3F88BD43398C4408BE30B7C5C388BC81B8A7C24FC05BC138416C3EAD40DEBE64454244E143"> : tensor<5x6x7xf16>}> : () -> tensor<5x6x7xf16>
    "func.return"(%0) : (tensor<5x6x7xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

