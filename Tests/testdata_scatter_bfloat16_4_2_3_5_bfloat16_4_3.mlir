"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      "stablehlo.return"(%arg1) : (tensor<bf16>) -> ()
    }) : (tensor<4x2x3x5xbf16>, tensor<2xi64>, tensor<4x3xbf16>) -> tensor<4x2x3x5xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<4x2x3x5xbf16>, tensor<4x2x3x5xbf16>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xBFBF3140BFBF3F40743D7340B33F54BFD33F263FC83F493F17C08B40BDBF983F49C0903F26C01B4060C0C640513FA1BF54BF544038C0E7BE0B4184C0AB3E82408B4069409EBF0F40544005BF29BE804002C08F3F95C03EC0EFBF64BF8FC06C3F84BFB5BF8B4094BC58C095C040C012408CC08040034032BF3640B9C0354091C09440E83EF03F0F3F0840134033C09840DA3F404029408340DE3FA840D63F0FBF0D3FA140653DC8BF22C073BE88408E4027BD5ABE1FC0E9BF45C04FC0623EBFC09FC0A03F38BF85C075BF39C0194009C0D84092C0EC3F0CC0F83F0DC024BF983F4CBF833FA13FC53E20C0013F61408740"> : tensor<4x2x3x5xbf16>}> : () -> tensor<4x2x3x5xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[1.765630e+00, -1.250000e+00, -1.453130e+00], [-3.531250e+00, -1.179690e+00, 2.562500e+00], [1.632810e+00, -1.359380e+00, 2.750000e+00], [-3.093750e+00, 1.289060e+00, -3.359380e+00]]> : tensor<4x3xbf16>}> : () -> tensor<4x3xbf16>
    "func.return"(%1, %2) : (tensor<4x2x3x5xbf16>, tensor<4x3xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xBFBF3140BFBF3F40E23F7340B33F54BFD33FA0BFC83F493F17C08B40BABF983F49C0903F26C01B4060C0C640513FA1BF54BF544038C0E7BE0B4184C0AB3E82408B40694062C00F40544005BF29BE97BF02C08F3F95C03EC0244064BF8FC06C3F84BFB5BF8B4094BC58C095C040C012408CC08040034032BF3640B9C0354091C0D13FE83EF03F0F3F0840AEBF33C09840DA3F404030408340DE3FA840D63F0FBF0D3FA140653DC8BF22C073BE88408E4027BD5ABE1FC0E9BF45C04FC046C0BFC09FC0A03F38BFA53F75BF39C0194009C057C092C0EC3F0CC0F83F0DC024BF983F4CBF833FA13FC53E20C0013F61408740"> : tensor<4x2x3x5xbf16>}> : () -> tensor<4x2x3x5xbf16>
    "func.return"(%0) : (tensor<4x2x3x5xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

