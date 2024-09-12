"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xf16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xf16>, tensor<1x3xf16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xf16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f16>, %arg1: tensor<f16>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<f16>, tensor<f16>) -> tensor<f16>
      "stablehlo.return"(%7) : (tensor<f16>) -> ()
    }) : (tensor<1x50x3xf16>, tensor<1xi64>, tensor<1x3xf16>) -> tensor<1x50x3xf16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x50x3xf16>, tensor<1x50x3xf16>) -> ()
    "func.return"(%6) : (tensor<1x50x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xf16>, tensor<1x3xf16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xB9BCF43D404773B83BC039BDC1C4563A803464C4BFBCF93FC84166C2293CF3473047C4BCFABF3A3C823D3C359DC5E13E17422B3EEB4271C3ACC08547F6AC31425B3A4BC06B3BB9AB98C456C0CD4127B89340CF4377487DC34D3D8EAFFE3E7B3AAE42C74099BA663DD33F9243CAC2F942673D7FC246BF993D25355C279BBE044240BE452FC8BFB63E25C448C028421BBA8DC3D241463FA5C5204545BA8C40DBC0804068459EBD1EC3BB3C4E3C323FADC2DF46AA34A33E8AC167BDB1448DC68A359C42F3C09CBE39B6B2430C2D71411BC5744466C495C359411CBF88B933C518B942C0ED3486C36EC27BB6563F593779382F394242244205C0A6C0FF42383EEBBC2BC0EB3BBFBE833D23C60F34C5B4FFB814C59BBD4AC2224001C224462B381D44343AA5BD25C651C3073FD03E"> : tensor<1x50x3xf16>}> : () -> tensor<1x50x3xf16>
    %2 = "stablehlo.constant"() <{value = dense<[[2.130860e+00, -2.222660e+00, 2.972660e+00]]> : tensor<1x3xf16>}> : () -> tensor<1x3xf16>
    "func.return"(%1, %2) : (tensor<1x50x3xf16>, tensor<1x3xf16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xf16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xB9BCF43D404773B83BC039BDC1C4563A803464C4BFBCF93FC84166C2293CF3473047C4BCFABF3A3C823D3C359DC5E13E17422B3EEB4271C3ACC08547F6AC31425B3A4BC06B3BB9AB98C456C0CD4127B89340CF4377487DC34D3D8EAFFE3E7B3AAE42C74099BA663DD33F9243CAC2F942673D7FC246BF993D25355C279BBE044240BE452FC8BFB63E25C448C028421BBA8DC3D241463FA5C5204545BA8C40DBC0804068459EBD1EC3BB3C4E3C323FADC2DF46AA34A33E8AC167BDB1448DC68A350B478045E9C439B6B2430C2D71411BC5744466C495C359411CBF88B933C518B942C0ED3486C36EC27BB6563F593779382F394242244205C0A6C0FF42383EEBBC2BC0EB3BBFBE833D23C60F34C5B4FFB814C59BBD4AC2224001C224462B381D44343AA5BD25C651C3073FD03E"> : tensor<1x50x3xf16>}> : () -> tensor<1x50x3xf16>
    "func.return"(%0) : (tensor<1x50x3xf16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

