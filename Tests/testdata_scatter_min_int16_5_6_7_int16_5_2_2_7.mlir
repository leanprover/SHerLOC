"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi16>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i16>, %arg1: tensor<i16>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i16>, tensor<i16>) -> tensor<i16>
      "stablehlo.return"(%7) : (tensor<i16>) -> ()
    }) : (tensor<5x6x7xi16>, tensor<2x2x1xi64>, tensor<5x2x2x7xi16>) -> tensor<5x6x7xi16>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi16>, tensor<5x6x7xi16>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFCFFFEFF01000300FEFF0000FFFFFEFF04000200000001000300FCFF0800FDFF0300FFFF010002000300040001000000FFFF02000000FFFFFEFF03000300FEFF0000FEFFFDFFFBFFFEFF0100FEFFFDFF070000000200020000000000FAFF0300FBFF0300000001000400FEFF00000000FFFFFEFF0000FEFF02000100FFFF0500FEFFFCFFFEFF010002000000FDFF010001000300FFFFFAFFFFFF0000010005000000000000000000F9FFFEFFFDFF0600FFFF0300FCFF0000FBFF08000300FFFFFCFFFDFF00000400FBFF0000FBFF00000100FCFF03000700020000000300FDFF00000100010001000700FDFFFEFFFFFF00000100FFFF0000FFFF000000000500FDFF01000000020005000100FFFFFCFFFEFF0200FFFF01000100FFFFFEFFFBFF0500FCFF0300F9FF00000000FFFF06000500FEFFFEFF0100FBFFFFFF080000000000FCFF00000000FFFFFBFF02000000FEFF0300FDFFFCFF01000000FFFF0400FFFF04000000FDFFFEFFFFFFFEFF00000100FFFF0000FEFF0000010001000000F9FF040001000100FAFF02000000030001000000FFFF00000000FFFFFCFF0100FBFF0000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    %2 = "stablehlo.constant"() <{value = dense<"0xFEFFFEFFFEFFFCFF0200FDFF010001000400FFFF02000100FCFF0000000000000100FBFF0300FDFFFDFF010000000200040003000300FCFFFDFFFDFF030000000200FEFFFCFFFFFF000000000000000000000000FDFFFDFFFEFFFFFF0100FFFF00000A00FCFFFCFF02000000FEFF000000000300FDFF0000FEFF020000000400FFFFFAFFFDFF0200FEFF010001000000FFFF000001000300FFFF01000000FEFF0200FFFF00000000FDFF0300FFFF0400FFFFFFFF03000000FCFF05000000FEFF040002000400FEFF00000000FEFF00000000000000000200FDFF0200FDFF0400FDFFFAFF0000000005000200030002000000FCFF05000000FEFF0100FDFFFFFF02000100FDFF00000200FCFF0100F7FF0000050007000200"> : tensor<5x2x2x7xi16>}> : () -> tensor<5x2x2x7xi16>
    "func.return"(%1, %2) : (tensor<5x6x7xi16>, tensor<5x2x2x7xi16>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi16>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFCFFFEFFFEFFFCFFFEFFFDFFFFFFFEFF0400FFFF00000100FCFFFCFF0000FDFF0100FBFF0100FDFFFDFF010000000000FFFF02000000FCFFFEFF03000300FEFF0000FEFFFDFFFBFFFEFF0100FEFFFDFF07000000FDFFFDFF00000000FAFFFEFFFBFFFFFF000000000000FEFF00000000FDFFFDFFFEFFFEFF0100FFFFFFFF0500FCFFFCFFFEFF0000FEFF0000FDFF010001000300FFFFFAFFFFFF0000010005000000000000000000F9FFFEFFFDFF0000FEFF0200FCFF0000FBFFFAFFFDFFFFFFFCFFFDFF00000000FBFF0000FBFF0000FFFFFCFF0000FEFF0200FFFF0000FDFF00000100010001000700FDFFFEFFFFFF00000100FFFF0000FFFF0000FDFF0300FDFF0100FFFFFFFF03000000FCFFFCFFFEFFFEFFFFFF01000100FEFFFEFFFBFFFEFFFCFF0000F9FF00000000FDFF0200FDFFFEFFFEFF0100FBFFFFFF080000000000FCFF00000000FFFFFBFF02000000FDFFFAFFFDFFFCFF01000000FFFF0200FFFFFCFF0000FDFFFEFFFFFFFDFFFFFF0100FFFFFDFFFEFF0000FCFF0100F7FFF9FF040001000100FAFF02000000030001000000FFFF00000000FFFFFCFF0100FBFF0000"> : tensor<5x6x7xi16>}> : () -> tensor<5x6x7xi16>
    "func.return"(%0) : (tensor<5x6x7xi16>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

