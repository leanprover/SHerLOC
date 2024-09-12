"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x125xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<0> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x125xi8>, tensor<1xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x125xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<1x125xi8>, tensor<1xi64>, tensor<1xi8>) -> tensor<1x125xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x125xi8>, tensor<1x125xi8>) -> ()
    "func.return"(%6) : (tensor<1x125xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x125xi8>, tensor<1xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFE040000010007FAFDFD01FF00F9FCFEFE0102000202FD030002FEFDFF060000FDFFFFFE020203FE000005010100070002FE00FE000302020400FD0002030200FDFB0100FE04FC0400FCFE00000200000003FD030003FD0000FD03FF01FD02FCFF0301FBFD00FF00FF040003FE00FB00FC04010000FFFEFD0001010003"> : tensor<1x125xi8>}> : () -> tensor<1x125xi8>
    %2 = "stablehlo.constant"() <{value = dense<-4> : tensor<1xi8>}> : () -> tensor<1xi8>
    "func.return"(%1, %2) : (tensor<1x125xi8>, tensor<1xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x125xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFC040000010007FAFDFD01FF00F9FCFEFE0102000202FD030002FEFDFF060000FDFFFFFE020203FE000005010100070002FE00FE000302020400FD0002030200FDFB0100FE04FC0400FCFE00000200000003FD030003FD0000FD03FF01FD02FCFF0301FBFD00FF00FF040003FE00FB00FC04010000FFFEFD0001010003"> : tensor<1x125xi8>}> : () -> tensor<1x125xi8>
    "func.return"(%0) : (tensor<1x125xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

