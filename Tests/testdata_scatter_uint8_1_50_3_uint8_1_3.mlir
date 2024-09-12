"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xui8>, tensor<1x3xui8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xui8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<ui8>, %arg1: tensor<ui8>):
      "stablehlo.return"(%arg1) : (tensor<ui8>) -> ()
    }) : (tensor<1x50x3xui8>, tensor<1xi64>, tensor<1x3xui8>) -> tensor<1x50x3xui8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xui8>, tensor<1x50x3xui8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xui8>, tensor<1x3xui8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x060000000000020501070205040301040002010301020503000202050004030301010005030304010101000500000000050407040201000200020200030300020201010303040400060502040001010602030402030003000204000101020100050600020200000201030300000001020402020200000000000001050105000105030001030207030400010100000202030405000302"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    %2 = "stablehlo.constant"() <{value = dense<[[4, 0, 1]]> : tensor<1x3xui8>}> : () -> tensor<1x3xui8>
    "func.return"(%1, %2) : (tensor<1x50x3xui8>, tensor<1x3xui8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xui8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x060000000000020501070205040301040002010301020503000202050004030301010005030304010101000500000000050407040201000200020200030300020201010303040400060502040001010602030402030003000204000101020100040001020200000201030300000001020402020200000000000001050105000105030001030207030400010100000202030405000302"> : tensor<1x50x3xui8>}> : () -> tensor<1x50x3xui8>
    "func.return"(%0) : (tensor<1x50x3xui8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

