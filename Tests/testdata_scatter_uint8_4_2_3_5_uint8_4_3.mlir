"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      "stablehlo.return"(%arg1) : (tensor<ui8>) -> ()
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x050000020402020202050002010201030201030006000101040103020104020002060003030002000102000003010704020103020202010306000403030401030304010400000000010200000504030502020304020402030200000002030704010705030104000500030000000002020300010003000101"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 0, 1], [5, 4, 2], [0, 3, 1], [0, 3, 4]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui8>, tensor<4x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x050000020102020202000002010201030201030006000101040103020104020002060503030002040102000002010704020103020202010306000403030401030004010400030000010201000504030502020304020402030200000002030004010705030104000504030000000002020300010003000101"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    "func.return"(%0) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

