"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> (tensor<5x7x4xui32>, tensor<5x7x4xui32>), res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}, {jax.result_info = "[1]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2:2 = "func.call"() <{callee = @expected}> : () -> (tensor<5x7x4xui32>, tensor<5x7x4xui32>)
    %3 = "stablehlo.iota"() <{iota_dimension = 0 : i64}> : () -> tensor<5x7x4xui64>
    %4 = "stablehlo.iota"() <{iota_dimension = 1 : i64}> : () -> tensor<5x7x4xui64>
    %5 = "stablehlo.iota"() <{iota_dimension = 2 : i64}> : () -> tensor<5x7x4xui64>
    %6 = "stablehlo.constant"() <{value = dense<28> : tensor<ui64>}> : () -> tensor<ui64>
    %7 = "stablehlo.broadcast_in_dim"(%6) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x7x4xui64>
    %8 = "stablehlo.multiply"(%7, %3) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %9 = "stablehlo.constant"() <{value = dense<4> : tensor<ui64>}> : () -> tensor<ui64>
    %10 = "stablehlo.broadcast_in_dim"(%9) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x7x4xui64>
    %11 = "stablehlo.multiply"(%10, %4) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %12 = "stablehlo.constant"() <{value = dense<1> : tensor<ui64>}> : () -> tensor<ui64>
    %13 = "stablehlo.broadcast_in_dim"(%12) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x7x4xui64>
    %14 = "stablehlo.multiply"(%13, %5) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %15 = "stablehlo.add"(%8, %11) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %16 = "stablehlo.add"(%15, %14) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %17 = "stablehlo.constant"() <{value = dense<32> : tensor<ui64>}> : () -> tensor<ui64>
    %18 = "stablehlo.broadcast_in_dim"(%17) <{broadcast_dimensions = array<i64>}> : (tensor<ui64>) -> tensor<5x7x4xui64>
    %19 = "stablehlo.shift_right_logical"(%16, %18) : (tensor<5x7x4xui64>, tensor<5x7x4xui64>) -> tensor<5x7x4xui64>
    %20 = "stablehlo.convert"(%16) : (tensor<5x7x4xui64>) -> tensor<5x7x4xui32>
    %21 = "stablehlo.convert"(%19) : (tensor<5x7x4xui64>) -> tensor<5x7x4xui32>
    "stablehlo.custom_call"(%21, %2#0) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
    "stablehlo.custom_call"(%20, %2#1) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
    "func.return"(%21, %20) : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x7x4xui32>, tensor<5x7x4xui32>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<0> : tensor<5x7x4xui32>}> : () -> tensor<5x7x4xui32>
    %1 = "stablehlo.constant"() <{value = dense<"0x000000000100000002000000030000000400000005000000060000000700000008000000090000000A0000000B0000000C0000000D0000000E0000000F000000100000001100000012000000130000001400000015000000160000001700000018000000190000001A0000001B0000001C0000001D0000001E0000001F000000200000002100000022000000230000002400000025000000260000002700000028000000290000002A0000002B0000002C0000002D0000002E0000002F000000300000003100000032000000330000003400000035000000360000003700000038000000390000003A0000003B0000003C0000003D0000003E0000003F000000400000004100000042000000430000004400000045000000460000004700000048000000490000004A0000004B0000004C0000004D0000004E0000004F000000500000005100000052000000530000005400000055000000560000005700000058000000590000005A0000005B0000005C0000005D0000005E0000005F000000600000006100000062000000630000006400000065000000660000006700000068000000690000006A0000006B0000006C0000006D0000006E0000006F000000700000007100000072000000730000007400000075000000760000007700000078000000790000007A0000007B0000007C0000007D0000007E0000007F000000800000008100000082000000830000008400000085000000860000008700000088000000890000008A0000008B000000"> : tensor<5x7x4xui32>}> : () -> tensor<5x7x4xui32>
    "func.return"(%0, %1) : (tensor<5x7x4xui32>, tensor<5x7x4xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

