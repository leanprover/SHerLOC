"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<ui8>, tensor<ui8>) -> tensor<ui8>
      "stablehlo.return"(%7) : (tensor<ui8>) -> ()
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui8>, tensor<1x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x020301020101030303000504010302010002040101020201000300040301000402010103040401000101020200010000040003000004030500030002020200010303030700000202010105010100050004010201000101050706020004010404030101020000010003070002010002010103050103010201000500050001020600060704010101000005000400000102000407000104"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 3, 3]]> : tensor<1x3xui8>}> : () -> tensor<1x3xui8>
    "func.return"(%1, %2) : (tensor<1x50x3xui8>, tensor<1x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x020301020101030303000504010302010002040101020201000300040301000402010103040401000101020200010000040003000004030500030002020200010303030700000202010105010100050004010201000101050706020004010404030404020000010003070002010002010103050103010201000500050001020600060704010101000005000400000102000407000104"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    "func.return"(%0) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

