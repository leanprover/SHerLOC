"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x000201040103050002020401000200000200020402010200010000030002000300030403010101000100000000010302050101020201000000050304010603010000010101010203040201050204030202050000020303000003020100050504030002020202010101020202090002000001000201000402"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 0], [1, 2, 2], [0, 3, 2], [1, 1, 1]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui8>, tensor<4x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x000201040103050002030401000200000200020402010200010000030002000300030503010101020100000002010302050101020201000000050304010603010000010101040203040203050204030202050000020303000003020100050604030002030202010102020202090002000001000201000402"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    "func.return"(%0) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

