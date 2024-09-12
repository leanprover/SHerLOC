"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<4x2x3x5xf16>, tensor<2xi64>, tensor<4x3xf16>) -> tensor<4x2x3x5xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3x5xf16>, tensor<4x2x3x5xf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xf16>, tensor<4x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x8AC410AC4FC12A3EE334ACC3943DC5BFE8360136694181B153C55ABC49B903452244AF281241EC3982C208C14040BFC643C0213DF2B8BEBD03C174C5E2BC093BB2C1A93D2335163D49B475C5354864C1D8B2A6C282BD03BBA9BC633BCA3E3FBC46BE1F3C64BC63BBA8303640863CA0BDD7408D4013C306B6F93E5840614145427E3E19B8F23C384361BF65B36843A3446E4451377FC4254386BD1EC3FDB9F3C2F8C5B0BC3DBAD53BA7BAE62EAD41B342AFC5244215403C41C3403D3ECB3E17431DBE04C2A3B938C31A3E694048B942C02E330C41844036C1CABF62BA59C294437FC32540743CCCC48F34DCC00A3CB235"> : tensor<4x2x3x5xf16>}> : () -> tensor<4x2x3x5xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[-2.597660e+00, -3.593750e+00, -1.796880e+00], [-6.937500e+00, -1.564450e+00, -4.363280e+00], [2.152340e+00, -4.138180e-01, 6.778720e-03], [6.054690e-01, 2.955080e+00, -5.078130e+00]]> : tensor<4x3xf16>}> : () -> tensor<4x3xf16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xf16>, tensor<4x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x8AC410AC4FC12A3E96C0ACC3943DC5BFE83670C2694181B153C55ABCEAC003452244AF281241EC3982C208C14040BFC643C0213DF2B8BEBD03C174C5E2BC093BB2C1A93D9EC6163D49B475C5354842C4D8B2A6C282BD03BB87C5633BCA3E3FBC46BE1F3C64BC63BBA8303640863CA0BDD7408D4013C306B6F93E5840614145428D4319B8F23C384361BF29B96843A3446E4451377DC4254386BD1EC3FDB9F3C2F8C5B0BC3DBAD53BA7BAE62EAD41B342AFC5244215403C41C3403D3E9C4017431DBE04C2A3B93CB91A3E694048B942C0DBC40C41844036C1CABF62BA59C294437FC32540743CCCC48F34DCC00A3CB235"> : tensor<4x2x3x5xf16>}> : () -> tensor<4x2x3x5xf16>
    "func.return"(%0) : (tensor<4x2x3x5xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

