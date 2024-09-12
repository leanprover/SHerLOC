"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      "stablehlo.return"(%arg1) : (tensor<i8>) -> ()
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi8>, tensor<1x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFF0403020000FFFE03FD0300FE010300030707000101F603FA0000FE040106FE00FE0100030002FC04FDFD00060305020002FC0402000001FE01010304FD02FEFD01FEFB01FF02FF0100FF05FFFEFB00FF07020202FFFBFF05030000FF05010200000001FF000006FEFEFFFC03FFFEFD03030100FF07FF00FBFE0002FEFEFC0003FEFFFE0100040102FFFC00FEFF00FEFF03FEFD02FD"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 1]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
    "func.return"(%1, %2) : (tensor<1x50x3xi8>, tensor<1x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFF0403020000FFFE03FD0300FE010300030707000101F603FA0000FE040106FE00FE0100030002FC04FDFD00060305020002FC0402000001FE01010304FD02FEFD01FEFB01FF02FF0100FF05FFFEFB00FF07020202FFFBFF05030000FF05010200010101FF000006FEFEFFFC03FFFEFD03030100FF07FF00FBFE0002FEFEFC0003FEFFFE0100040102FFFC00FEFF00FEFF03FEFD02FD"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    "func.return"(%0) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

