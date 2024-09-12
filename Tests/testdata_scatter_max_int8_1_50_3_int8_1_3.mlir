"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.maximum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi8>, tensor<1x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFDFD00FF00F802FCFD0002040302010007FE050202FEFE00060100FEFBFA0001FB00FF000000FF020105FDFCFFFEFF020002040100050000000000FC08FE02FE020403020301FEF60001FCFD0006F901FF0100FB010100FAFC0002FD000001010500FD00FC0001FCFE02010700FE00000101000003FBFCFA0103FF00FD040307020300FF000002FFFE0002FF0004FF0203FF01FF0001"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-1, 4, 2]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
    "func.return"(%1, %2) : (tensor<1x50x3xi8>, tensor<1x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFDFD00FF00F802FCFD0002040302010007FE050202FEFE00060100FEFBFA0001FB00FF000000FF020105FDFCFFFEFF020002040100050000000000FC08FE02FE020403020301FEF60001FCFD0006F901FF0100FB010100FAFC0002FD0000010105040200FC0001FCFE02010700FE00000101000003FBFCFA0103FF00FD040307020300FF000002FFFE0002FF0004FF0203FF01FF0001"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    "func.return"(%0) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

