"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000302030300000000020004040501000000010201050000010100030001010106030201020100010002000302020002000103000001020301010106000201020100000103010503020201000101000200020200000300000101060102030106020400030204010206010203020204040104010402020100060000030305010601040201030000000003000103010003050402000001010102020601000004010200000002000503000103010102000301010604000300000100010001000102000101000100000306000302020001010400"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<"0x0304000503060001050000000103010203020102030402040300020201010003000200000304010203020302020000020101030600010301010803000102010201000201000002020102000705030001010301010202010301010300030003020103010002000000010103030301030201000302020104000504010101010101050601060301030104030302"> : tensor<5x2x2x7xui8>}> : () -> tensor<5x2x2x7xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000300030300000000000000010301000000010201040000010000020001010106030201020100010002000100020002000003000001020201010100000201010100000103010503020201000101000200020200000300000101010101000101000000020102000205010001010201010104010402020100060000030305010201030101030000000002000101000000000001000001010102020601000004010200000002000503000003010101000001010101000100000100010001000101000101000100000306000302020001010400"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

