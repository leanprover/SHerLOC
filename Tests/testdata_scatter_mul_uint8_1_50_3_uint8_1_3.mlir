"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui8>, tensor<1x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000000040104010201050102030201000300020201020101030502010002030200000300040201020101040000010006030001030801010000050000010002030303010102000404010304030202000102010200000200010600030001030601030404000103040500000101050101020000010000010002050203010303010204010209010403000000010103030603000302050300"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[6, 1, 0]]> : tensor<1x3xui8>}> : () -> tensor<1x3xui8>
    "func.return"(%1, %2) : (tensor<1x50x3xui8>, tensor<1x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000000040104010201050102030201000300020201020101030502010002030200000300040201020101040000010006030001030801010000050000010002030303010102000404010304030202000102010200000200010600030001030601120400000103040500000101050101020000010000010002050203010303010204010209010403000000010103030603000302050300"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    "func.return"(%0) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

