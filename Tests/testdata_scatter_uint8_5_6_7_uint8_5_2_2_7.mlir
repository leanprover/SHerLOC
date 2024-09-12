"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0], [1]], [[2], [3]]]> : tensor<2x2x1xi64>}> : () -> tensor<2x2x1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 3], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      "stablehlo.return"(%arg1) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2x1xi64>, tensor<5x2x2x7xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x040402020507070202070201040101020002030001020202020000030401000104070500050107050101060200040002000405000402030002000301040201000002010200030201070004000201020401010204010100000201010100020201020003000001020102000001040003030001000502050002030206000001030403020402020506020001000001020201000000000006020501050200000103000100040502010602000105000303030300000400030305000100000206000405010107040003000102000401000600040004"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<"0x0103000201030104000003010400000000010303030004020001030102040200000002020206020501050302000006000401010000010105020006030400020301010401060001010101000002000000050001040101020301010104000500000204010202000102010302010106010002030203000101020000010102000000020102040001000003030000"> : tensor<5x2x2x7xui8>}> : () -> tensor<5x2x2x7xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<5x2x2x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010300020103010400000301040000000001030303000402000103010401000104070500050107050101020402000000020202060205010503020000060004010100000101050201070004000201020401010204020006030400020301010401060001010101000002000000050001040001000502050002030206000001010102030101010400050000020401020200010201030201010601000200000103000100040502010602020302030001010200000101020000000201020400010000030300000003000102000401000600040004"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

