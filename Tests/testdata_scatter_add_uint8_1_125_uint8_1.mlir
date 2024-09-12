"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xui8>, tensor<1xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<1x125xui8>, tensor<1xi64>, tensor<1xui8>) -> tensor<1x125xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xui8>, tensor<1x125xui8>) -> ()
    "func.return"(%6) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xui8>, tensor<1xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x0100000002020500030401040203020300010300010100050001000002020305020002040002040001040000050303030401030104020202030201000102000203000103010004050201080101000001090100020203000100020100000201000001010106000206030502010000010105040000030201010306000800"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    %2 = "stablehlo.constant"() <{value = dense<0> : tensor<1xui8>}> : () -> tensor<1xui8>
    "func.return"(%1, %2) : (tensor<1x125xui8>, tensor<1xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x0100000002020500030401040203020300010300010100050001000002020305020002040002040001040000050303030401030104020202030201000102000203000103010004050201080101000001090100020203000100020100000201000001010106000206030502010000010105040000030201010306000800"> : tensor<1x125xui8>}> : () -> tensor<1x125xui8>
    "func.return"(%0) : (tensor<1x125xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

