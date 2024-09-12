"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<4x2x3x5xui8>, tensor<2xi64>, tensor<4x3xui8>) -> tensor<4x2x3x5xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xui8>, tensor<4x2x3x5xui8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xui8>, tensor<4x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x020404030204000402020100000101000002020400010004010000020302010203000200000201000100000006000207020701010002010004010000030004020100000406010005040102030500000202000504020204010202020002000201040205050106030200010004050002000501010502030202"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, 0, 3], [1, 1, 0], [2, 3, 2], [5, 2, 2]]> : tensor<4x3xui8>}> : () -> tensor<4x3xui8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xui8>, tensor<4x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x020404030204000402000100000103000002020400010004010000020302010203000200000201000100000000000207020701010002010004010000030004020200000406030005040104030500000202000504020204010202020002000A010402050A0106030200010004050002000501010502030202"> : tensor<4x2x3x5xui8>}> : () -> tensor<4x2x3x5xui8>
    "func.return"(%0) : (tensor<4x2x3x5xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

