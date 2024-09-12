"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui8>, tensor<1xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    "func.return"(%6) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui8>, tensor<1xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0102030406010100000105000103010003010200010408010001010200010301010002010103000304000201010001030401050000010002000101010304010003000202020200020203050303020202050102010700000200000000020101060100070000030102000200020402000301010100000300030004000401"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xui8>}> : () -> tensor<1xui8>
    "func.return"(%1, %2) : (tensor<1x125xui8>, tensor<1xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0102030406010100000105000103010003010200010408010001010200010301010002010103000304000201010001030401050000010002000101010304010003000202020200020203050303020202050102010700000200000000020101060100070000030102000200020402000301010100000300030004000401"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    "func.return"(%0) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

