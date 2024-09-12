"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFEFDFC0006FE03020300FF00FE01FE00FFFD000103FDFFFC000604FF00FE0204000403FE01010604FE00FC0300FF000300FEFC00FD02FD00010402FA02FB0200000100FFFF08FD0204020204FDFE040101FD00060008FC00000501FEFE030001FB03FDFD0304FD03000000010002000000000300FF0000FF03000408FFFF00FF0200F702030002040100FCFF0403060100FDFF0005FF01FFFE000406FF040100FCFDFDFB02FE02000200FD030200000000FC00FFFD000302FEFBFFFD030300FFFFFE0100FFFF0403FD0000FCFBFC00FE0503"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 4, -3, 4, 2, -3, 0], [1, 1, 1, -3, -3, 6, 2]]> : tensor<2x7xi8>}> : () -> tensor<2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFEFDFC0006FE030003FDFF00FD00FE00FFFD000103FDFFFC000604FF00FE0204000403FE01010604FE00FC0300FF000300FEFC00FD02FD00010402FA02FB0200000100FFFF08FD0204020204FDFE040101FD00060008FC00000501FEFE030001FB03FDFD0304FD0300000001FDFD000000000300FF0000FF03000408FFFF00FF0200F702030002040100FCFF0403060100FDFF0005FF01FFFE000406FF040100FCFDFDFB02FE02000200FD030200000000FC00FFFD000302FEFBFFFD030300FFFFFE0100FFFF0403FD0000FCFBFC00FE0503"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

