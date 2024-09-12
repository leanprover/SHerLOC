"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2x2xi64>, tensor<5x2x2xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<5x2x2xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x00FEFF0003FFFFFEFC0102FD030000FFFE010402050003FC040304FD07FC010200FDFEFBFE000200FD0106FF01030600000000FCFD00FE01FF00FFFD01FCFE0103FA00010100FF02FEFD0601000003000201FE01FCFC050303FEFB0301FFFEFD02FCFC000100FB000100FF0001FB04FDFB000004000202FF000000FD02050100040000FC01000000010105FC01FDFDFD00FFFF00FEFDFD040102FF0000FF01FC02FFFE01FF0300FDFFFEFD02FB00010000FE02FD00FF03FB010300FF05FE0200FEFF000001060000FF03FF0407000004FDFF"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[[-4, -5], [2, -7]], [[-1, 3], [-1, 5]], [[-2, 0], [0, 2]], [[-1, 3], [2, 2]], [[1, -1], [3, 3]]]> : tensor<5x2x2xi8>}> : () -> tensor<5x2x2xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<5x2x2xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00FEFF0003FFFFFEFC0102FD030000FFFE010402050003FC040304FD07FC010200FDFEFBFE000200FD0106FF0103060000000005FD00FE01FF00FF0301FCFE0103FA00010100FF02FEFD0601000003000201FE01FCFE050303FEFB030102FEFD02FCFC000100FB000100FF0001FB04FD00000004000202FF000000FD02050100040000FC01000002010105FC01FDFD0300FFFF00FEFDFD040102020000FF01FC02FFFE01FF0300FDFF01FD02FB000100000302FD00FF03FB010300FF05FE0200FEFF000003060000FF03FF0407000004FDFF"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

