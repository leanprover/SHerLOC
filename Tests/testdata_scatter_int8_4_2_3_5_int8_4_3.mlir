"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "stablehlo.constant"() <{value = dense<[0, 4]> : tensor<2xi64>}> : () -> tensor<2xi64>
    %4:2 = "func.call"() <{callee = @inputs}> : () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>)
    %5 = "func.call"() <{callee = @expected}> : () -> tensor<4x2x3x5xi8>
    %6 = "stablehlo.scatter"(%4#0, %3, %4#1) <{scatter_dimension_numbers = #stablehlo.scatter<update_window_dims = [0, 1], inserted_window_dims = [1, 3], scatter_dims_to_operand_dims = [1, 3]>, unique_indices = true}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      "stablehlo.return"(%arg1) : (tensor<i8>) -> ()
    }) : (tensor<4x2x3x5xi8>, tensor<2xi64>, tensor<4x3xi8>) -> tensor<4x2x3x5xi8>
    "stablehlo.custom_call"(%6, %5) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<4x2x3x5xi8>, tensor<4x2x3x5xi8>) -> ()
    "func.return"(%6) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> (tensor<4x2x3x5xi8>, tensor<4x3xi8>), res_attrs = [{mhlo.layout_mode = "default"}, {mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<"0x02FDFF01000004FD05000207FC04FF02FF0102FA0003000000FF0402FF03FD0504000400FD0300FE04FF000103FF02FFFAF8FF000603030200F90000F7000101FEFF00000004000102FCFFFC02FDFE00000101FBFD010103FFFEFF000404F9000200000400FD0303010100FF02020600FD00FF00F800FCFE"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-6, 1, 0], [1, 0, 3], [0, 2, 1], [4, 4, 3]]> : tensor<4x3xi8>}> : () -> tensor<4x3xi8>
    "func.return"(%1, %2) : (tensor<4x2x3x5xi8>, tensor<4x3xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x2x3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<"0x02FDFF01FA0004FD05010207FC040002FF0102FA0003000000FF0402FF03FD0504000100FD03000004FF000103FF02FFFAF8FF000603030200F90000F700010100FF00000002000102FC01FC02FDFE00000101FBFD010103FFFEFF00040404000200000400FD0303030100FF02020600FD00FF00F800FCFE"> : tensor<4x2x3x5xi8>}> : () -> tensor<4x2x3x5xi8>
    "func.return"(%0) : (tensor<4x2x3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

