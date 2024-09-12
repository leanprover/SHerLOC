"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFE00FF0000FE0201FF01000000FF00FD020001040200FDFF07010800FDFF0503FD0200FA01FDFF000003FEFAFE0101040201FDFF00FAFF0000FAFE03FDFCFC0102FE02FF050003FD01FF020003FE00FEFD010002FB0304020401FF0403FEFEFFFA040002FEFD010302FB00FF04FFFB0001FE0000FAFDFF010403040100000204FFFFFFFDFF01FE0200FAFF04000000FDFE00FFF9FD0200000200FEFD01000000FFFD01FF01FD00FF00FFFEFDFE0200FFFF00FD05FFFDFD000000FE01FE01000600FEFE04FF0600020402FDFE010200F90000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 0, 2, -1, -5, -1, 4], [1, 3, -3, 3, 4, 0, 0]]> : tensor<2x7xi8>}> : () -> tensor<2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFE00FF0000FE020100020000000400FD020001040200FDFF07010800FDFF0503FD0200FA01FDFF000003FEFAFE0101040201FDFF00FAFF0000FAFE03FDFCFC0102FE02FF050003FD01FF020003FE00FEFD010002FB0304020401FF0403FEFEFFFA040002FEFD0103020103FF0404000001FE0000FAFDFF010403040100000204FFFFFFFDFF01FE0200FAFF04000000FDFE00FFF9FD0200000200FEFD01000000FFFD01FF01FD00FF00FFFEFDFE0200FFFF00FD05FFFDFD000000FE01FE01000600FEFE04FF0600020402FDFE010200F90000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

