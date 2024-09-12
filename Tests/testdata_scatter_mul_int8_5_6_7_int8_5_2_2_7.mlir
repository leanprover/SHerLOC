"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2x1xi64>, tensor<5x2x2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01FDFD0303000004010100FF0200FE0003FE03FAFF02FF03010003FCFC00FD010000FFFFFF0203FF000000FF05FE02FCFD00FEFD03FFFE0100FEFDFF04F9000003FC01FAFF0903000502FFFDFEFE02FDFDFE0000FF04000000FD00FC00FE010003FE00050502FF06FF020102FEFEFF02FC00FEFB0100020201000001FEFE030204FF02FE0103FBFEFF00FE03FE0002FD0001040101FFFD0200050000010002FD00FFFD05FC040002FE0200FF0201FEFF00FDFEFEFFFCFBFA04000201FF040300040100FCFBFF00060004FCFF01FC02FFFE00"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<"0xFB0301000100FE01FD08FD0000FCFFFC0001FD000302020000FC00FF0100FD05FE0000020003FC020101040000000000FE02F900FE01FDFF00FDFDFEFDFEFE0000FE00040003FE00FEFC000100FD03FE02010004FD030001010200FAFDFE0400FEFA010101FBFC00FDFDFF0200FEFEFF0101FFFF03FE0100000301FEFDFA0200FFFEFE020100FD0504FD0000"> : tensor<5x2x2x7xi8>}> : () -> tensor<5x2x2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<5x2x2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFBF7FD0003000004FD0800000000020000FEF700FD04FE0000000004FC00FD010000FFFFFF0203FF00000000F1F6FC00000000F7F4FEFE010000000000000000EB00FEFA03F703000502FFFDFEFE02FDFDFE000000F40000000600000004000000FA0000F6F8000600FA03FCFCFE0008FC00FEFB0100020201000001FEFEF70600FF02FC00EE0F04FC0004EEFE00020F0000F4FDFFFE00FC00FB0000010002FD00FFFD05FC040002FE02000106FEFE0000F7FE040318F600FC00FC02FF00F70010FD0000FBFF00060004FCFF01FC02FFFE00"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

