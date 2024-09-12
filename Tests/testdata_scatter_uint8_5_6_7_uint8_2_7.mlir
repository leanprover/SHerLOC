"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xui8>, tensor<2x7xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      "stablehlo.return"(%arg1) : (tensor<ui8>) -> ()
    }) : (tensor<5x6x7xui8>, tensor<2x2xi64>, tensor<2x7xui8>) -> tensor<5x6x7xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xui8>, tensor<5x6x7xui8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xui8>, tensor<2x7xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x030104000004020204030502000300000001000200010500050204040201000303020300010403020000040000030302000402000201040201030701000301000001020202000503000402020000030101010001030201030101000000010206000000000000000000020500020200010101030002010005010506030100050503000301000200040002050101040202000005010200020104020100020201020103020500000705010104000200020204040101030400050201000200060400010302000102010002010700020401000108"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 1, 2, 3, 0, 2, 0], [2, 0, 1, 3, 7, 1, 0]]> : tensor<2x7xui8>}> : () -> tensor<2x7xui8>
    "func.return"(%1, %2) : (tensor<5x6x7xui8>, tensor<2x7xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x030104000004020101020300020000000001000200010500050204040201000303020300010403020000040000030302000402000201040201030701000301000001020202000503000402020000030101010001030201030101000000010206000000000000000000020001030701000101030002010005010506030100050503000301000200040002050101040202000005010200020104020100020201020103020500000705010104000200020204040101030400050201000200060400010302000102010002010700020401000108"> : tensor<5x6x7xui8>}> : () -> tensor<5x6x7xui8>
    "func.return"(%0) : (tensor<5x6x7xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

