"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.multiply"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x01F8000204010001FAFFFFFE00FF0101FF02FD00040006000000FDFF0002FD01020000000100FDFD03FF0201FFFB06FCFF0601FDFC00FD03010000FB0000FDFCFF040105FEFEFB0102FF01FF02FC00FE0003FF00FFFA0005FD00FE00FE05FCFC00000003FDFC0100FD00030202010005FD0000010200FF00"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[0, 1, 2], [-6, 0, 3], [0, -1, 0], [-3, 2, 1]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi8>, tensor<4x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x01F8000200010001FAFFFFFE00FF0201FF02FD00040006000000FDFF0002FD01020000000100FD0003FF0201FDFB06FCFF0601FDFC00FD03010000FB0000FDFC00040105FE02FB0102FF00FF02FC00FE0003FF00FFFA0005FD00FE00FE050CFC00000006FDFC0100FD00030202010005FD0000010200FF00"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    "func.return"(%0) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

