"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[[0, 1], [2, 3]], [[4, 0], [1, 2]]]> : tensor<2x2x2xi64>}> : () -> tensor<2x2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1, 2], scatter_dims_to_operand_dims = [1, 2], index_vector_dim = 2>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2x2xi64>, tensor<5x2x2xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<5x2x2xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x010104010300020504000602010001010204010303000203020103030300010002030203000405010300010205020200020302010205000200030404040201000100010202030100020000000705010100000304000200000101010100020003010000020601040000030203040700010207010000000105030000010202000001020200000201030000000205010101010000030203000300020402010701040003000000020901010003000404010501010102020000040100020603000004050102000101010004020303010100010000"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[[3, 3], [2, 2]], [[2, 3], [0, 2]], [[3, 1], [1, 2]], [[3, 4], [2, 0]], [[3, 4], [1, 3]]]> : tensor<5x2x2xui8>}> : () -> tensor<5x2x2xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<5x2x2xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x010304010300020504020602010001010204010303000203020103030300010002030203000405010300010205020200020302020205000200030404040201000100010202030100020000000705010100000304000300000101010100020003010000020601040000030203040700010207010000000105030000010202000301020200000201030000000205010104010000030203000300020402010701040003000000020901010303000404010501030102020000040104020603000004050102000101010004020303010100010000"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

