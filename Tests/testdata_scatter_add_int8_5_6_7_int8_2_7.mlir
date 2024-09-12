"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[[0, 1], [2, 3]]> : tensor<2x2xi64>}> : () -> tensor<2x2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<5x6x7xi8>, tensor<2x7xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<5x6x7xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [1], inserted_window_dims = [0, 1], scatter_dims_to_operand_dims = [0, 1], index_vector_dim = 1>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<5x6x7xi8>, tensor<2x2xi64>, tensor<2x7xi8>) -> tensor<5x6x7xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<5x6x7xi8>, tensor<5x6x7xi8>) -> ()
    "func.return"(%6) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<5x6x7xi8>, tensor<2x7xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0xFDFF0500030103000000FDFC050201030300FDFD00000200FF06000AFE00020003050404FBFBFC010103FF00010001FEFFFD0100FE00FE000501F802FEFF00FE0205FB02FC00010000FEFDFD0002F90301FE03FCFDFDFC01FAFF00FF00FE02FD01FD000005FE000102FFFA00FD00010302FEFDFEFE06000000FEFF0003FFFE0006FEFE010202FF020501FEFF02020002FF00FEFDFEFE01FD00FE0000FEFCFD030103FB00FF000202FEFCFF0009040101FD0100030204030201FEF70400FD00FFFEFFFF06010402FFFF030002020003FE0000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-2, 2, -3, 0, 1, 1, -2], [6, -1, 0, 1, -5, 0, -1]]> : tensor<2x7xi8>}> : () -> tensor<2x7xi8>
    "func.return"(%1, %2) : (tensor<5x6x7xi8>, tensor<2x7xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<5x6x7xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0xFDFF0500030103FE02FDFDFD060001030300FDFD00000200FF06000AFE00020003050404FBFBFC010103FF00010001FEFFFD0100FE00FE000501F802FEFF00FE0205FB02FC00010000FEFDFD0002F90301FE03FCFDFDFC01FAFF00FF00FE02FD01FD000005FE00010205F900FEFB010202FEFDFEFE06000000FEFF0003FFFE0006FEFE010202FF020501FEFF02020002FF00FEFDFEFE01FD00FE0000FEFCFD030103FB00FF000202FEFCFF0009040101FD0100030204030201FEF70400FD00FFFEFFFF06010402FFFF030002020003FE0000"> : tensor<5x6x7xi8>}> : () -> tensor<5x6x7xi8>
    "func.return"(%0) : (tensor<5x6x7xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

