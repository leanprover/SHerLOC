"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui8>, tensor<1xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      "stablehlo.return"(%arg1) : (tensor<ui8>) -> ()
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    "func.return"(%6) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui8>, tensor<1xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0002020101000501040100040003030003010201030401040400010203010103000002020100010501030201020106030300020002030104000101010000050000030000010000010103040300020506030005000206000206020402000305070203000003040003020500000405010104000301000204020100010102"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    %2 = "stablehlo.constant"() <{value = dense<1> : tensor<1xui8>}> : () -> tensor<1xui8>
    "func.return"(%1, %2) : (tensor<1x125xui8>, tensor<1xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0102020101000501040100040003030003010201030401040400010203010103000002020100010501030201020106030300020002030104000101010000050000030000010000010103040300020506030005000206000206020402000305070203000003040003020500000405010104000301000204020100010102"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    "func.return"(%0) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

