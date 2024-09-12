"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<7x5x3xf32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4:3 = "func.call"() <{callee = @inputs}> : () -> (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi64>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<7x5x3xf32>
    %6 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 1>, start_indices = array<i64: 0>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %7 = "stablehlo.reshape"(%6) : (tensor<1xi64>) -> tensor<i64>
    %8 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 2>, start_indices = array<i64: 1>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %9 = "stablehlo.reshape"(%8) : (tensor<1xi64>) -> tensor<i64>
    %10 = "stablehlo.slice"(%4#2) <{limit_indices = array<i64: 3>, start_indices = array<i64: 2>, strides = array<i64: 1>}> : (tensor<3xi64>) -> tensor<1xi64>
    %11 = "stablehlo.reshape"(%10) : (tensor<1xi64>) -> tensor<i64>
    %12 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %13 = "stablehlo.compare"(%7, %12) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %14 = "stablehlo.constant"() <{value = dense<7> : tensor<i64>}> : () -> tensor<i64>
    %15 = "stablehlo.add"(%7, %14) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %16 = "stablehlo.select"(%13, %15, %7) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %17 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %18 = "stablehlo.compare"(%9, %17) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %19 = "stablehlo.constant"() <{value = dense<5> : tensor<i64>}> : () -> tensor<i64>
    %20 = "stablehlo.add"(%9, %19) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %21 = "stablehlo.select"(%18, %20, %9) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %22 = "stablehlo.constant"() <{value = dense<0> : tensor<i64>}> : () -> tensor<i64>
    %23 = "stablehlo.compare"(%11, %22) <{compare_type = #stablehlo<comparison_type SIGNED>, comparison_direction = #stablehlo<comparison_direction LT>}> : (tensor<i64>, tensor<i64>) -> tensor<i1>
    %24 = "stablehlo.constant"() <{value = dense<3> : tensor<i64>}> : () -> tensor<i64>
    %25 = "stablehlo.add"(%11, %24) : (tensor<i64>, tensor<i64>) -> tensor<i64>
    %26 = "stablehlo.select"(%23, %25, %11) : (tensor<i1>, tensor<i64>, tensor<i64>) -> tensor<i64>
    %27 = "stablehlo.dynamic_update_slice"(%4#0, %4#1, %16, %21, %26) : (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<i64>, tensor<i64>, tensor<i64>) -> tensor<7x5x3xf32>
    "stablehlo.custom_call"(%27, %5) <{call_target_name = "check.expect_close", has_side_effect = true}> : (tensor<7x5x3xf32>, tensor<7x5x3xf32>) -> ()
    "func.return"(%27) : (tensor<7x5x3xf32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi64>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xC063C5BFB1CE253E70918BBE0615D3C0EAA797BF0A9D7540CAABA3BFF77C0A4025F37DC0CA8A434016954740329700C033DAF83F6D75883F032A273F5EDF303F028CE3BF946050C022EF4EBF75CDA83FD8BA88408C2AE0BE155D49C0ED9A84C0FDC28CBF815215C094B34F3FC49A35C08285D83F45CC92BF710553BF44AD6840A96551BE1ACD4A3E1752A1BFC3E533407344FF3FB7ADA4BFE1609BBF709BD9BFC7A9953F04276E4001DCB7BFD2CACE4033FABE3FF990943F65E1503E4A91B5BF873C4740143EE8BFA360263E3D3A6BC074E452BF70C4A340EFE6393F26DCF7BE9EB608C098526D3FF8B92740F431B93F02B44BBFDDBF73BFE2DF46BF932C90BD98C5AF3F4A89893F344F49C0CF157CBEBFDB9A400452CE3F43737C40573AD5C0ADCC27406787913DAD95E7BFA277E5BFC6BF99BF25194AC038996E4041A3A640246439C0B9A15F3E803CB1BFE2F2F0BE0023ED3FC5ADA4401CA9054082B00B40E1BC19BF60DE4AC0936B3BBFCA525DBE943151BFE6044940940ABFBFE12116C07F4D0FBF06F3B23ED81002C0E99894401D03733FA43F39C0D316393EDD1A66C02791963F"> : tensor<7x5x3xf32>}> : () -> tensor<7x5x3xf32>
    %2 = "stablehlo.constant"() <{value = dense<> : tensor<2x0x1xf32>}> : () -> tensor<2x0x1xf32>
    %3 = "stablehlo.constant"() <{value = dense<[4, 1, 0]> : tensor<3xi64>}> : () -> tensor<3xi64>
    "func.return"(%1, %2, %3) : (tensor<7x5x3xf32>, tensor<2x0x1xf32>, tensor<3xi64>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<7x5x3xf32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xC063C5BFB1CE253E70918BBE0615D3C0EAA797BF0A9D7540CAABA3BFF77C0A4025F37DC0CA8A434016954740329700C033DAF83F6D75883F032A273F5EDF303F028CE3BF946050C022EF4EBF75CDA83FD8BA88408C2AE0BE155D49C0ED9A84C0FDC28CBF815215C094B34F3FC49A35C08285D83F45CC92BF710553BF44AD6840A96551BE1ACD4A3E1752A1BFC3E533407344FF3FB7ADA4BFE1609BBF709BD9BFC7A9953F04276E4001DCB7BFD2CACE4033FABE3FF990943F65E1503E4A91B5BF873C4740143EE8BFA360263E3D3A6BC074E452BF70C4A340EFE6393F26DCF7BE9EB608C098526D3FF8B92740F431B93F02B44BBFDDBF73BFE2DF46BF932C90BD98C5AF3F4A89893F344F49C0CF157CBEBFDB9A400452CE3F43737C40573AD5C0ADCC27406787913DAD95E7BFA277E5BFC6BF99BF25194AC038996E4041A3A640246439C0B9A15F3E803CB1BFE2F2F0BE0023ED3FC5ADA4401CA9054082B00B40E1BC19BF60DE4AC0936B3BBFCA525DBE943151BFE6044940940ABFBFE12116C07F4D0FBF06F3B23ED81002C0E99894401D03733FA43F39C0D316393EDD1A66C02791963F"> : tensor<7x5x3xf32>}> : () -> tensor<7x5x3xf32>
    "func.return"(%0) : (tensor<7x5x3xf32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

