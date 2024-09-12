"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi8>, tensor<1x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFFFB0300F90003FBFF000002FD01000004010303FC0200FF0002010000FBFE0300FEFD000000040200FFFE03000200000000FFFF020204000402FC0203FFFF040000000403010300FE00FF00FE02FE0602010100FC04FFF80201FA0000010000FE02FF01FD0105020205000003FFFE000103FC030504000401FDFCFA000100030004000B0004FEFF0406000101FB0000FEFD01010100"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[1, -2, 5]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
    "func.return"(%1, %2) : (tensor<1x50x3xi8>, tensor<1x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFFFB0300F90003FBFF000002FD01000004010303FC0200FF0002010000FBFE0300FEFD000000040200FFFE03000200000000FFFF020204000402FC0203FFFF040000000403010300FE00FF00FE02FE0602010100FC04FFF80201FA0000010000FF000401FD0105020205000003FFFE000103FC030504000401FDFCFA000100030004000B0004FEFF0406000101FB0000FEFD01010100"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    "func.return"(%0) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

