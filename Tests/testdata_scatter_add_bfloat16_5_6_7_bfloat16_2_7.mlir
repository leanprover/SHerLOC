"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xbf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<bf16>, %arg1: tensor<bf16>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<bf16>, tensor<bf16>) -> tensor<bf16>
      "stablehlo.return"(%7) : (tensor<bf16>) -> ()
    }) : (tensor<5x6x7xbf16>, tensor<2x2xi64>, tensor<2x7xbf16>) -> tensor<5x6x7xbf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<5x6x7xbf16>, tensor<5x6x7xbf16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xbf16>, tensor<2x7xbf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x89C08CBFA73EAA3FB340B13FC9BF86BF7FC083BF1BBDA1BF6BC077C0A93F98C0B6BFE0BFE33E05C0B43FEC3EB7BF70408F409A40A33E6ABFBC3F4A3F4EBF90BE84BF264098BF253F7F3E56C0063F3ABF4AC0043EDF4094C08E4023BF5140163E563F064021C04DC059C051BF4D40683EAB3FF1BF3F40A7C08E40D23FC1BEECBF4D40FFBFE2BFDCBE7C4081C09A40D94085BE6FBFBAC05EC006BF813F70C0313F3CC0B83F22BF7D400C3F7C3FE0BFA14055C059BFB6C082C034C078BF44C086C00DBF9D40AB40DD3F1E3C7DBF9B3F70407E3F1A40963F60BF3E3F55C0C23FD8BFB6BF5AC05840C8BF4CC0084081C067404DBF753F5CBED9BF5040F23F2CC084C04BC0AD4084402DC04F3FE13F8340D83F14C0CA40E83F3B40533FF03F4740A03F8AC0193DC44039C07D401840B0401AC02540F63F2BC0A1406E3FE6BF8A3F34C0FDBE523E88C0E43FA2C006BFABC0C140A8401D3F08C122C036403840C9BF9E4098C00BBF8FC061C06AC099C0E9BC9640F93E0D40274146C069C0C03FF0404D408BC02E3F86C0F5BD7F4057BF91C03640D03E60405DC0CD3FCCBFEEBE28BE0E40BBBF0C40"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    %2 = "stablehlo.constant"() <{value = dense<[[3.734380e+00, -2.988280e-01, 5.656250e+00, -1.085940e+00, 4.250000e+00, 1.054690e+00, -6.250000e-01], [-1.257810e+00, -6.523440e-01, 2.359380e+00, 1.070310e+00, 5.195310e-01, -5.390630e-01, -5.000000e-01]]> : tensor<2x7xbf16>}> : () -> tensor<2x7xbf16>
    "func.return"(%1, %2) : (tensor<5x6x7xbf16>, tensor<2x7xbf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xbf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x89C08CBFA73EAA3FB340B13FC9BF2C4089C0944090BF404028C090C0A93F98C0B6BFE0BFE33E05C0B43FEC3EB7BF70408F409A40A33E6ABFBC3F4A3F4EBF90BE84BF264098BF253F7F3E56C0063F3ABF4AC0043EDF4094C08E4023BF5140163E563F064021C04DC059C051BF4D40683EAB3FF1BF3F40A7C08E40D23FC1BEECBF4D40FFBFE2BFDCBE7C4081C09A40D94085BE6FBFBAC05EC006BF813F70C0313F3CC0B83F22BF7D400C3F7C3FE0BFA14055C059BFB6C082C034C078BF44C086C00DBF9D40AB40DD3F1E3C7DBF9B3F70407E3F933F053FBE3FE83F34C07A3F0CC0B6BF5AC05840C8BF4CC0084081C067404DBF753F5CBED9BF5040F23F2CC084C04BC0AD4084402DC04F3FE13F8340D83F14C0CA40E83F3B40533FF03F4740A03F8AC0193DC44039C07D401840B0401AC02540F63F2BC0A1406E3FE6BF8A3F34C0FDBE523E88C0E43FA2C006BFABC0C140A8401D3F08C122C036403840C9BF9E4098C00BBF8FC061C06AC099C0E9BC9640F93E0D40274146C069C0C03FF0404D408BC02E3F86C0F5BD7F4057BF91C03640D03E60405DC0CD3FCCBFEEBE28BE0E40BBBF0C40"> : tensor<5x6x7xbf16>}> : () -> tensor<5x6x7xbf16>
    "func.return"(%0) : (tensor<5x6x7xbf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

