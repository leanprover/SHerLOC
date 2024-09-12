"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x1x2xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<7x5x3xf32>, tensor<3xi64>)
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<3x1x2xf32>
    %5 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %6 = "stablehlo.reshape"(%5) : (tensor<1xi64>) -> tensor<i64>
    %7 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %8 = "stablehlo.reshape"(%7) : (tensor<1xi64>) -> tensor<i64>
    %9 = "stablehlo.slice"(%3#1) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %10 = "stablehlo.reshape"(%9) : (tensor<1xi64>) -> tensor<i64>
    %11 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %12 = "stablehlo.compare"(%6, %11) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %13 = "stablehlo.constant"() <{value = dense<7> : tensor<i64>}> : () -> tensor<i64>
    %14 = "stablehlo.add"(%6, %13) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %15 = "stablehlo.select"(%12, %14, %6) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %16 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %17 = "stablehlo.compare"(%8, %16) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %18 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %19 = "stablehlo.add"(%8, %18) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %20 = "stablehlo.select"(%17, %19, %8) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %21 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %22 = "stablehlo.compare"(%10, %21) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %23 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %24 = "stablehlo.add"(%10, %23) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %25 = "stablehlo.select"(%22, %24, %10) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %26 = "stablehlo.dynamic_slice"(%3#0, %15, %20, %25) <{slice_sizes = array<i64: 3, 1, 2>}> : (tensor<7x5x3xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<3x1x2xf32>
    "stablehlo.custom_call"(%26, %4) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<3x1x2xf32>, tensor<3x1x2xf32>) -> ()
    "func.return"(%26) : (tensor<3x1x2xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<7x5x3xf32>, tensor<3xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xC4F2F1BFBAADEFC068289D3FBCEB153FBFDFE2BFE6D3C23FD9961C4004A2F03F1FC4FC3FDD00A7BFEC78E63F30C60D40CEC096C04786E63F7FCE0440D94D713ED0D601C1A9B4713FBA6A2840B56BDD3F49322BC054861EC031FD62C01C78AEC0BD6BDC40EC2EBFC016D10240AE7EC03F2BC91FC074B68A3EC54564400D008540ABDE43C074493FC088EB74C0B6E872BF0DCDC6C0EF1E4E4003B5563F3E3A773FC5CE004166FF943FFF57DB3F7650633F93283ABF1159EFBFD6DE533DCC7527C0B4FF80BF9FDAA9BF8972CCBEA8962A40AA441EC03848AA3FB0B50040732D8ABF55164E3FFEA6E0BF5DA0034094E036BF9499074051E793BDAA3034C028AB444083B89B40C9C011BFDD3FB7BEE8AE8740421A46C030D7BD3FC1F331400435C9405AB555C02F652840056B59BFC169B53FDEEFD73E9BECE23DDBDE144072B20240029BBEC08CBF1B409E4850BF479D07BFADDB02C08570E63F4795824033F99540C2355BBFB63978400019FEC0ADF9C3BF618BCD404EF9274038F0EB3F0DF6E23F2AE915C0EA0C21C02D931D40ADB90BC01C0BF0BF6F55EDBF5579B5BDD20EC33E920D01C0"> : tensor<7x5x3xf32>}> : () -> tensor<7x5x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<[4, 0, 1]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "func.return"(%1, %2) : (tensor<7x5x3xf32>, tensor<3xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x1x2xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[[-0.0722185448, -2.81547022]], [[0.421751916, 0.110802852]], [[-1.531057, 6.42326403]]]> : tensor<3x1x2xf32>}> : () -> tensor<3x1x2xf32>
    "func.return"(%0) : (tensor<3x1x2xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

