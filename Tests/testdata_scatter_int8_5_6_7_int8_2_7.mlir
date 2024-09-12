"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      "stablehlo.return"(%arg1) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFD0403FE00FB020002FC0100FD00FEFF0104FF0001FEFD00000100FF01020001020101040202FE02FB0100FEFD010106FB0201FF00000302FC0300FFFF00000101FEFFFC01FFFDFF00FC04FE0300030000FFFEFF00000204FE0301FD020102FEFF020201000701FE02030603FDFAFFFBFBFD00FE04000100010400FFF802FC03FEFFFEFB00FE050000FCFAFFFF00FF0002FE00FE020000FF05050000000300030002000101FDFD000005020001FB02FFFC0002FF040200FD00FCFF0000FFFD07FF000400FF0000020600030102FF030005FD"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 1, 2, -3, 1, 6, -2], [-4, 0, 1, 0, 3, 0, 1]]> : tensor<2x7xi8>}> : () -> tensor<2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFD0403FE00FB02010102FD0106FEFEFF0104FF0001FEFD00000100FF01020001020101040202FE02FB0100FEFD010106FB0201FF00000302FC0300FFFF00000101FEFFFC01FFFDFF00FC04FE0300030000FFFEFF00000204FE0301FD020102FEFF020201000701FE02FC000100030001FBFD00FE04000100010400FFF802FC03FEFFFEFB00FE050000FCFAFFFF00FF0002FE00FE020000FF05050000000300030002000101FDFD000005020001FB02FFFC0002FF040200FD00FCFF0000FFFD07FF000400FF0000020600030102FF030005FD"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

