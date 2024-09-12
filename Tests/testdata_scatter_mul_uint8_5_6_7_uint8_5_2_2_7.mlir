"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x040201000103000402050102000202030107010301030300030300000104010201040400040300010205000000020601010204010200000305010206000804010203020205020008010205050204050302000505010201000201050101040003020004050100020000010101020001020501000000000101020208000700000203000201030301030003030200080203040000020502000001030301000003020004020303000001060000010100010000010104000102000005020302000402020201020003010302010301020101010100"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<"0x0101000104000201000104010101020104000300040402000204010401030300050404020002030101010303020201000104020301020601000703010B01000400010001010300010000050202030700000504010100020300010101010407000004010001000002040002010200010209090103040106000100000200010000050401000000050105010402"> : tensor<5x2x2x7xui8>}> : () -> tensor<5x2x2x7xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0402000004000004000504020002040304000300040C0600060C00000104010201040400040300010205000000001E0404040002060000030F03040C00000404040902041E020008010205050204050302000505000E030016010004000400030200000500000A00000307000000040205010000000001010202080007000000060000010303010C0000000800000200000000000A020000010603010000030200040203030000013600000304000600000000080001000000140200000014020A0204040003010302010301020101010100"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

