"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<32> : tensor<1xi64>}> : () -> tensor<1xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<1x50x3xi8>, tensor<1x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<1x50x3xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1], scatter_dims_to_operand_dims = [1]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<1x50x3xi8>, tensor<1xi64>, tensor<1x3xi8>) -> tensor<1x50x3xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<1x50x3xi8>, tensor<1x50x3xi8>) -> ()
    "func.return"(%6) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<1x50x3xi8>, tensor<1x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01FFFE0101FFFE020100060203FE0003020300FEFE00FF02090406FF00FF0005000000020500FD000100FE0001FF0000010605FEFC01040402FE00FEFAFE01FEFC0007FCFC0001FF05010000FDFE01FDFB00FB00010201FC0001020200FFFDFFFE030501010001FFFCFBFB0402030106020000FF00FE01FAFF010003FCFC01FD07010204FF0102FE0203FE00010200FE010401030202"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[6, 2, -2]]> : tensor<1x3xi8>}> : () -> tensor<1x3xi8>
    "func.return"(%1, %2) : (tensor<1x50x3xi8>, tensor<1x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<1x50x3xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01FFFE0101FFFE020100060203FE0003020300FEFE00FF02090406FF00FF0005000000020500FD000100FE0001FF0000010605FEFC01040402FE00FEFAFE01FEFC0007FCFC0001FF05010000FDFE01FDFB00FB00010201FC0001020200FFFDFFF406F601010001FFFCFBFB0402030106020000FF00FE01FAFF010003FCFC01FD07010204FF0102FE0203FE00010200FE010401030202"> : tensor<1x50x3xi8>}> : () -> tensor<1x50x3xi8>
    "func.return"(%0) : (tensor<1x50x3xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

