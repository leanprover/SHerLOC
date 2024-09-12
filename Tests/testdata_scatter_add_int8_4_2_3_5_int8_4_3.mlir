"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x03FC01FF00FFFD0003FB00020002FCFF020002FD0500FCFC000101000000010004FF00000100040203FF0403FF0002FE0000FEFE010104FF02000100030A040500FD0102FFFF00FE0306FFFFFFFEFD0300040200FD000700FF01FF03FEFF00FD00FE0000FC000503040003040500FF000104FAFF0000FEFF"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-4, 4, 0], [4, -2, -1], [-5, -6, 1], [4, 3, 0]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi8>, tensor<4x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x03FC01FFFCFFFD0003FF00020002FCFF020002FD0500FCFC000101000000010004FF04000100040003FF0403FE0002FE0000FEFE010104FF02000100030A0405FBFD0102FFF900FE030600FFFFFEFD0300040200FD000700FF01FF03FEFF04FD00FE0003FC000503040003040500FF000104FAFF0000FEFF"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    "func.return"(%0) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

