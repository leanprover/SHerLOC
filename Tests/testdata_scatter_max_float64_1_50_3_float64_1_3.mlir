"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xf64>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xf64>, tensor<1x3xf64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xf64>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<f64>, %arg1: tensor<f64>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<f64>, tensor<f64>) -> tensor<f64>
      "stablehlo.return"(%7) : (tensor<f64>) -> ()
    }) : (tensor<1x50x3xf64>, tensor<1xi64>, tensor<1x3xf64>) -> tensor<1x50x3xf64>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<1x50x3xf64>, tensor<1x50x3xf64>) -> ()
    "func.return"(%6) : (tensor<1x50x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xf64>, tensor<1x3xf64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xC6B91C245B38F93F8D64C4954F45E13F6BE7EBB7C7D0FCBF46FD190C13BF18C0E7B6CCD7B0460AC0FC056017DA8ACDBFA01742436B6AF4BFD6A802A1E1251340D3244EF8DA4E13C0B6F38D2F1584EDBF7D57CF6F8ABC0A402ADE6BF3DE17D8BF09ADEBF44F3AF53FD8A3D216E0E2E2BFDE9E87A67862014038277E600D091B40557C818AF02DFFBF10E416827746D43FE442C91D324C06C0BA7B0849095A06C0FC3AED2FAEC215C0AD421E9F4F3510400C673BD99B9DF8BFB833FD3DCC14E6BF66AB3E28FB571840E7CE55211495F1BF522509D21DBADABFB970B7063835EA3FBD3C21D912421440FF60520C872FE23FCDCC71EF2AFF0CC0EC9F46827B6ABE3F9ACB0E55683F0740D830504AB7DA00409EC8F70D7C98D33F3A67416BF01E0440641B096DF449DBBF2EE6A8740037FB3F8ED120E171320AC0716994FBCF480CC0A07359C4AF9B2140ED4ED6AF94A6F1BF6ADAC440F4E5F3BFFE4C18565CF80EC0A6DA9DF538F4E4BF5F47D9BD4D66F53F44FD4FE846DE12404EF3B37DA0A3D63FFC9670AB875706C0F650A66B4DB2E9BFD5ECD91A03E911C050021884973FA13F8D6D4AF9784307C09983EAB4E4C002C01465D543DD530D40242640B60A780F40902A2F60CE30E2BF16C6CF623552B03FAC7C1CC2B55E0940B84A851E7C49CE3F1292561CA76D1240D82BAA8A25C20CC0157D48F99B1B13C04C71EA56F0DAF5BF5602E8474A540340FC1C033DBABCFA3FF3858DB865EAF23F66F2D682D71A0A40CA659DAD5981ED3FFBA5C6E99774F03FF73B1F6A834BB6BFFC895DD834E2FA3F7A78E43BE0FA18C054C88B1070A2F43FB40517EA330110C0A1D61F94D8DBE4BF184132F08CC4EF3F7453199411E5F2BF24C5B6A0709B14401650BCDB1399FA3F3872AB103316EDBFE5B90BAE5FD617C00689B296E17BE93F314DDEDD9C5A0CC071FDF0AB8EBDF7BF121219003F4DF73F468CF1F7917813408463AEABE346F43F8414F7FF7FB10540E7E2260BBDA1104058CD53822E22EEBF46AD6042461A16C018E53466245708C01EEF9A5016390BC0491AF9ABF5C7FB3F12F48DB2CA65074047D23733B61BE03F938BA9B8EFBD1C40209DC3B6ED57CC3F223EBF2F33ACF23F53BE7A09BDF512C0044E27759FADE93F68A7E2E9474CD3BF565A6AA879C90F40F8C9F20E7938E1BF2A08C961A0E013402C0F5FF9F2220A402D08E983D61C004062BC0CEB9A71B2BFEB5633C2289902C02FFC9EB45FAC144040C49F55DF7DF43FC58AE7B3497CFE3F66750F747772F03FA6A3490C2AB6054045D254A2B0D819C0F4D43F2013570DC090D7319358370EC08B04CB28F9ABF63F505FE5BB7BB5D7BF6E133EFA58E0ECBFEA7C6BB380A3C8BF94BB270FA33AC0BF8C81E5D39F1302C0CAFB05B2EEBC0E4077866D8991FB06C09B6389D379EC1640AE90C6AA5896F6BF4355459C12B001C0827264241291EABFC0EE0D978038E2BF561274E371B9DBBF268AEAE74FCDFCBFC8116EE9E7D9ED3F3D22FD45C147CE3F983634C745F5D7BF6C00E9F5C8C5FA3F5817778CA41B14C0791C1C1D93E9E03FC26F0C1D3A08F2BF85BC0C2E669F16C0424852531A83F03F9F456770132E12C0EE2D1892517F0140EAEC8AEC54BEBD3FCF6F37E223E7F1BF7771AC819F4908C0542C2E99206901C051A3847A16F4E03F2D5F3CD864A80340"> : tensor<1x50x3xf64>}> : () -> tensor<1x50x3xf64>
    %2 = "stablehlo.constant"() <{value = dense<[[-0.78079662675684358, 0.2106770315053868, -5.7162267541805845]]> : tensor<1x3xf64>}> : () -> tensor<1x3xf64>
    "func.return"(%1, %2) : (tensor<1x50x3xf64>, tensor<1x3xf64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xf64>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xC6B91C245B38F93F8D64C4954F45E13F6BE7EBB7C7D0FCBF46FD190C13BF18C0E7B6CCD7B0460AC0FC056017DA8ACDBFA01742436B6AF4BFD6A802A1E1251340D3244EF8DA4E13C0B6F38D2F1584EDBF7D57CF6F8ABC0A402ADE6BF3DE17D8BF09ADEBF44F3AF53FD8A3D216E0E2E2BFDE9E87A67862014038277E600D091B40557C818AF02DFFBF10E416827746D43FE442C91D324C06C0BA7B0849095A06C0FC3AED2FAEC215C0AD421E9F4F3510400C673BD99B9DF8BFB833FD3DCC14E6BF66AB3E28FB571840E7CE55211495F1BF522509D21DBADABFB970B7063835EA3FBD3C21D912421440FF60520C872FE23FCDCC71EF2AFF0CC0EC9F46827B6ABE3F9ACB0E55683F0740D830504AB7DA00409EC8F70D7C98D33F3A67416BF01E0440641B096DF449DBBF2EE6A8740037FB3F8ED120E171320AC0716994FBCF480CC0A07359C4AF9B2140ED4ED6AF94A6F1BF6ADAC440F4E5F3BFFE4C18565CF80EC0A6DA9DF538F4E4BF5F47D9BD4D66F53F44FD4FE846DE12404EF3B37DA0A3D63FFC9670AB875706C0F650A66B4DB2E9BFD5ECD91A03E911C050021884973FA13F8D6D4AF9784307C09983EAB4E4C002C01465D543DD530D40242640B60A780F40902A2F60CE30E2BF16C6CF623552B03FAC7C1CC2B55E0940B84A851E7C49CE3F1292561CA76D1240D82BAA8A25C20CC0157D48F99B1B13C04C71EA56F0DAF5BF5602E8474A540340FC1C033DBABCFA3FF3858DB865EAF23F66F2D682D71A0A40CA659DAD5981ED3FFBA5C6E99774F03FF73B1F6A834BB6BFFC895DD834E2FA3F7A78E43BE0FA18C054C88B1070A2F43FB40517EA330110C0A1D61F94D8DBE4BF184132F08CC4EF3F7453199411E5F2BF24C5B6A0709B14401650BCDB1399FA3F3872AB103316EDBFE5B90BAE5FD617C00689B296E17BE93F314DDEDD9C5A0CC071FDF0AB8EBDF7BF121219003F4DF73F468CF1F7917813408463AEABE346F43F8414F7FF7FB10540E7E2260BBDA1104058CD53822E22EEBF46AD6042461A16C018E53466245708C01EEF9A5016390BC0491AF9ABF5C7FB3F12F48DB2CA65074047D23733B61BE03F938BA9B8EFBD1C40209DC3B6ED57CC3F223EBF2F33ACF23F53BE7A09BDF512C0044E27759FADE93F68A7E2E9474CD3BF565A6AA879C90F40F8C9F20E7938E1BF2A08C961A0E013402C0F5FF9F2220A402D08E983D61C004062BC0CEB9A71B2BFEB5633C2289902C02FFC9EB45FAC144040C49F55DF7DF43FC58AE7B3497CFE3F66750F747772F03FA6A3490C2AB6054045D254A2B0D819C0F4D43F2013570DC090D7319358370EC08B04CB28F9ABF63F505FE5BB7BB5D7BF6E133EFA58E0ECBFEA7C6BB380A3C8BF94BB270FA33AC0BF8C81E5D39F1302C0CAFB05B2EEBC0E4077866D8991FB06C09B6389D379EC1640AE90C6AA5896F6BF4355459C12B001C0827264241291EABFC0EE0D978038E2BF561274E371B9DBBF268AEAE74FCDFCBFC8116EE9E7D9ED3F3D22FD45C147CE3F983634C745F5D7BF6C00E9F5C8C5FA3F5817778CA41B14C0791C1C1D93E9E03FC26F0C1D3A08F2BF85BC0C2E669F16C0424852531A83F03F9F456770132E12C0EE2D1892517F0140EAEC8AEC54BEBD3FCF6F37E223E7F1BF7771AC819F4908C0542C2E99206901C051A3847A16F4E03F2D5F3CD864A80340"> : tensor<1x50x3xf64>}> : () -> tensor<1x50x3xf64>
    "func.return"(%0) : (tensor<1x50x3xf64>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

