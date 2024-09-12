"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2x2xi64>, tensor<5x2x2xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFBFA73F97C0DDBFFBBFE3BD29C04140C2BF9B3ED740FDBFD0C0A440783F38BF263E523F8140C4BE964007C0B4BD2A407340EEBE163FA3BFE5BE114056C013BF26BF843FA33F173F084121BFA13F5C3F8EBE65409AC009C0FEBF03C082BFA5C0B4407A3E02406FC0143F35C05A3F9140C0BF8740B0BFD13F1F4030C036BD47407A401F402740CFBF4CBD3D3FF8409EC0ACBFB44080C088C007C0864092403CC0C2BFA4BF26C00D3F48C057C068BF59C0C6BF33BF0A408940D4BC02C0793E56402940E2BF19C0B5BF45C092BFDB40B1BF0FC0B33F19BF513ED23FACC0ECBF03C0854051BF7A4026BF594009C0204029404D4028C0943F76C08EC0993FBC4044401C408DBF90C03640943F08C1C2BF44BF1241A1C0853FE4BF97BF5D406BC037C06AC0253F93C00E406EC0FF3F1D40B440343F81C09CC0C1BFB4409B40144066BF77C08FC0A33FBE3EDA3F82C0DD3F8E3F7540F93E2FC085BFC6BF00C051C038400841C3BE45BF94C098409CBFBBC03F3FA23F77403540A54088BF16C0A73F6540913EFC3E8A404EC024408D3B22409EC0913F41C020C0C6BEF93F673FA9BF9840873F3840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[[-4.160160e-01, -2.546880e+00], [2.234380e+00, -2.937500e+00]], [[-3.703130e+00, 5.820310e-01], [-2.593750e+00, 1.085940e+00]], [[-6.562500e+00, -1.757810e+00], [3.750000e+00, 5.976560e-01]], [[2.502440e-02, -1.171880e+00], [2.578130e+00, 1.664060e+00]], [[-7.382810e-01, -1.554690e+00], [-2.937500e+00, -4.406250e+00]]]> : tensor<5x2x2xbf16>}> : () -> tensor<5x2x2xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<5x2x2xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFBF0BBF97C0DDBFFBBFE3BD29C04140C2BF64BFD740FDBFD0C0A440783F38BF263E06C08140C4BE964007C0B4BD2A407340EEBE163FA3BF80BF114056C013BF26BF843FA33F173F084121BFA13F5C3F8EBE65409AC0FE40FEBF03C082BFA5C0B4407A3E024082C0143F35C05A3F9140C0BF8740B0BF733F1F4030C036BD47407A401F402740CFBF4CBD3D3FA1C19EC0ACBFB44080C088C007C0864092403CC0C2BFA4BF26C00D3F48C0B04168BF59C0C6BF33BF0A408940D4BC9BBF793E56402940E2BF19C0B5BF45C00040DB40B1BF0FC0B33F19BF513ED23FACC0ECBF03C0794151BF7A4026BF594009C0204029404D4028C0943F76C08EC0993FBC409D3D1C408DBF90C03640943F08C1C2BFA3BF1241A1C0853FE4BF97BF5D406BC056406AC0253F93C00E406EC0FF3F1D40B440343F81C049C1C1BFB4409B40144066BF77C08FC0A33FBE3EDA3F82C0DD3F8E3F7540B8BE2FC085BFC6BF00C051C038400841D73F45BF94C098409CBFBBC03F3FA23FC0C03540A54088BF16C0A73F6540913EFC3E8A404EC0F1C08D3B22409EC0913F41C020C0C6BEF93F673FA9BF9840873F3840"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

