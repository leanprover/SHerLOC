"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x00030A000001020202050100020002020101060200000003000103020603000205040100000002000000000503020100000104020700000403010203000201020002000300050003010401020004020000040100000204010106060101020002000103010201010104070400020005030004020201020200"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 2, 0], [4, 2, 2], [1, 0, 0], [1, 5, 3]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui8>, tensor<4x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x00030A000101020202050100020002020101060200000003000103020603000205040400000002020000000503020100000104020700000403010203000201020102000300050003010401020004020000040100000204010106060101020102000103050201010104070400020005030004020201020200"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    "func.return"(%0) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

