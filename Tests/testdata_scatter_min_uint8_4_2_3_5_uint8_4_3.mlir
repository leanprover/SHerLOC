"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x040400000003010001020503000001020000010401010201050102010306060001040000000006000500030307020902020200030104010201020202020202010001000002010100010304050204040103040000000303060203020200000201000002050007000205010001080106000201010300040304"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 3], [4, 5, 3], [3, 4, 7], [3, 0, 2]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui8>, tensor<4x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x040400000003010001010503000001020000010401010201050102010306060001040000000006000500030303020902020200030104010201020202020202010001000002010100010304050204040103040000000303060203020200000201000002000007000202010001080106000201010300040304"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    "func.return"(%0) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

