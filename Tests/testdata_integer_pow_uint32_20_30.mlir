"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<20x30xui32>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %4 = "func.call"() <{callee = @inputs}> : () -> tensor<20x30xui32>
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<20x30xui32>
    %6 = "func.call"(%4) <{callee = @integer_pow}> : (tensor<20x30xui32>) -> tensor<20x30xui32>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<20x30xui32>, tensor<20x30xui32>) -> ()
    "func.return"(%6) : (tensor<20x30xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x30xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %3 = "stablehlo.constant"() <{value = dense<"0x0200000001000000000000000200000002000000000000000200000003000000000000000400000003000000020000000300000001000000090000000400000001000000020000000300000003000000000000000000000002000000010000000000000000000000020000000000000003000000020000000300000003000000030000000300000003000000000000000100000002000000050000000200000001000000060000000200000002000000010000000200000001000000020000000300000002000000020000000200000003000000010000000100000001000000010000000200000002000000030000000000000002000000020000000500000002000000020000000100000001000000000000000100000003000000040000000300000001000000000000000500000002000000040000000000000002000000000000000000000000000000000000000100000003000000010000000200000000000000020000000400000000000000020000000100000002000000000000000300000001000000000000000500000000000000010000000100000000000000000000000000000002000000000000000300000002000000040000000100000002000000030000000100000002000000020000000100000005000000000000000900000000000000010000000000000002000000010000000700000003000000000000000400000001000000000000000000000001000000000000000000000000000000000000000300000001000000020000000200000000000000020000000000000001000000020000000100000000000000030000000200000001000000000000000100000003000000020000000100000003000000020000000000000001000000030000000100000000000000020000000600000002000000030000000200000005000000070000000000000003000000010000000300000000000000020000000200000002000000020000000400000000000000030000000700000006000000050000000000000002000000010000000500000001000000010000000100000001000000020000000000000001000000000000000100000000000000020000000000000002000000010000000000000001000000020000000500000000000000000000000300000004000000010000000000000000000000040000000500000000000000010000000100000004000000030000000100000000000000020000000400000002000000020000000000000001000000010000000100000004000000020000000200000005000000000000000000000002000000000000000000000002000000030000000000000002000000000000000100000002000000090000000000000005000000010000000100000002000000010000000600000002000000040000000200000003000000030000000700000002000000030000000600000001000000010000000000000001000000050000000300000001000000020000000000000003000000080000000200000003000000020000000100000004000000000000000100000008000000020000000100000002000000000000000000000000000000010000000300000003000000040000000200000003000000020000000300000001000000010000000100000005000000010000000200000000000000040000000300000001000000000000000100000002000000030000000200000005000000030000000300000001000000010000000500000000000000020000000000000004000000030000000000000000000000000000000000000005000000040000000200000005000000050000000200000000000000020000000200000000000000050000000000000003000000000000000000000005000000020000000000000004000000000000000000000000000000010000000000000000000000000000000000000001000000010000000000000000000000010000000200000004000000010000000100000001000000030000000000000000000000010000000000000001000000010000000100000000000000020000000100000004000000010000000100000004000000000000000400000001000000060000000100000002000000010000000000000000000000020000000300000001000000010000000500000000000000060000000200000005000000020000000000000004000000030000000000000004000000000000000500000002000000010000000100000004000000000000000300000006000000030000000200000002000000020000000200000001000000030000000400000000000000030000000400000004000000000000000400000001000000030000000100000002000000030000000A0000000000000004000000000000000400000000000000020000000200000006000000000000000600000003000000000000000100000000000000040000000000000002000000050000000100000001000000000000000400000000000000050000000000000003000000020000000300000001000000000000000100000005000000030000000100000001000000000000000100000001000000050000000400000002000000010000000100000000000000000000000000000000000000020000000000000000000000030000000100000002000000000000000000000000000000000000000200000000000000000000000100000001000000000000000200000002000000010000000100000003000000020000000400000000000000000000000400000005000000050000000500000002000000030000000600000002000000000000000500000002000000020000000300000000000000000000000100000001000000020000000000000004000000030000000000000000000000000000000000000004000000020000000200000002000000000000000100000000000000010000000000000004000000040000000300000003000000030000000200000003000000000000000200000000000000000000000300000002000000010000000100000001000000010000000200000000000000020000000300000001000000000000000100000005000000020000000400000002000000020000000300000001000000040000000200000004000000040000000000000002000000010000000000000000000000000000000400000001000000010000000200000002000000040000000400000000000000020000000100000004000000010000000100000002000000040000000200000002000000010000000100000004000000"> : tensor<20x30xui32>}> : () -> tensor<20x30xui32>
    "func.return"(%3) : (tensor<20x30xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<20x30xui32>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %2 = "stablehlo.constant"() <{value = dense<"0x1000000001000000000000001000000010000000000000001000000051000000000000000001000051000000100000005100000001000000A1190000000100000100000010000000510000005100000000000000000000001000000001000000000000000000000010000000000000005100000010000000510000005100000051000000510000005100000000000000010000001000000071020000100000000100000010050000100000001000000001000000100000000100000010000000510000001000000010000000100000005100000001000000010000000100000001000000100000001000000051000000000000001000000010000000710200001000000010000000010000000100000000000000010000005100000000010000510000000100000000000000710200001000000000010000000000001000000000000000000000000000000000000000010000005100000001000000100000000000000010000000000100000000000010000000010000001000000000000000510000000100000000000000710200000000000001000000010000000000000000000000000000001000000000000000510000001000000000010000010000001000000051000000010000001000000010000000010000007102000000000000A119000000000000010000000000000010000000010000006109000051000000000000000001000001000000000000000000000001000000000000000000000000000000000000005100000001000000100000001000000000000000100000000000000001000000100000000100000000000000510000001000000001000000000000000100000051000000100000000100000051000000100000000000000001000000510000000100000000000000100000001005000010000000510000001000000071020000610900000000000051000000010000005100000000000000100000001000000010000000100000000001000000000000510000006109000010050000710200000000000010000000010000007102000001000000010000000100000001000000100000000000000001000000000000000100000000000000100000000000000010000000010000000000000001000000100000007102000000000000000000005100000000010000010000000000000000000000000100007102000000000000010000000100000000010000510000000100000000000000100000000001000010000000100000000000000001000000010000000100000000010000100000001000000071020000000000000000000010000000000000000000000010000000510000000000000010000000000000000100000010000000A1190000000000007102000001000000010000001000000001000000100500001000000000010000100000005100000051000000610900001000000051000000100500000100000001000000000000000100000071020000510000000100000010000000000000005100000000100000100000005100000010000000010000000001000000000000010000000010000010000000010000001000000000000000000000000000000001000000510000005100000000010000100000005100000010000000510000000100000001000000010000007102000001000000100000000000000000010000510000000100000000000000010000001000000051000000100000007102000051000000510000000100000001000000710200000000000010000000000000000001000051000000000000000000000000000000000000007102000000010000100000007102000071020000100000000000000010000000100000000000000071020000000000005100000000000000000000007102000010000000000000000001000000000000000000000000000001000000000000000000000000000000000000000100000001000000000000000000000001000000100000000001000001000000010000000100000051000000000000000000000001000000000000000100000001000000010000000000000010000000010000000001000001000000010000000001000000000000000100000100000010050000010000001000000001000000000000000000000010000000510000000100000001000000710200000000000010050000100000007102000010000000000000000001000051000000000000000001000000000000710200001000000001000000010000000001000000000000510000001005000051000000100000001000000010000000100000000100000051000000000100000000000051000000000100000001000000000000000100000100000051000000010000001000000051000000102700000000000000010000000000000001000000000000100000001000000010050000000000001005000051000000000000000100000000000000000100000000000010000000710200000100000001000000000000000001000000000000710200000000000051000000100000005100000001000000000000000100000071020000510000000100000001000000000000000100000001000000710200000001000010000000010000000100000000000000000000000000000000000000100000000000000000000000510000000100000010000000000000000000000000000000000000001000000000000000000000000100000001000000000000001000000010000000010000000100000051000000100000000001000000000000000000000001000071020000710200007102000010000000510000001005000010000000000000007102000010000000100000005100000000000000000000000100000001000000100000000000000000010000510000000000000000000000000000000000000000010000100000001000000010000000000000000100000000000000010000000000000000010000000100005100000051000000510000001000000051000000000000001000000000000000000000005100000010000000010000000100000001000000010000001000000000000000100000005100000001000000000000000100000071020000100000000001000010000000100000005100000001000000000100001000000000010000000100000000000010000000010000000000000000000000000000000001000001000000010000001000000010000000000100000001000000000000100000000100000000010000010000000100000010000000000100001000000010000000010000000100000000010000"> : tensor<20x30xui32>}> : () -> tensor<20x30xui32>
    "func.return"(%2) : (tensor<20x30xui32>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = (tensor<20x30xui32>) -> tensor<20x30xui32>, sym_name = "integer_pow", sym_visibility = "private"}> ({
  ^bb0(%arg0: tensor<20x30xui32>):
    %0 = "stablehlo.multiply"(%arg0, %arg0) : (tensor<20x30xui32>, tensor<20x30xui32>) -> tensor<20x30xui32>
    %1 = "stablehlo.multiply"(%0, %0) : (tensor<20x30xui32>, tensor<20x30xui32>) -> tensor<20x30xui32>
    "func.return"(%1) : (tensor<20x30xui32>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

