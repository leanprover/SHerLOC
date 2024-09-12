"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<3x5xi8>, res_attrs = [{jax.result_info = "", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %2 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi8>
    %3 = "func.call"() <{callee = @expected}> : () -> tensor<3x5xi8>
    %4 = "stablehlo.constant"() <{value = dense<127> : tensor<i8>}> : () -> tensor<i8>
    %5 = "stablehlo.broadcast_in_dim"(%4) <{broadcast_dimensions = array<i64>}> : (tensor<i8>) -> tensor<i8>
    %6 = "stablehlo.reduce_window"(%2, %5) <{window_dimensions = array<i64: 2, 2>}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.minimum"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x6xi8>, tensor<i8>) -> tensor<3x5xi8>
    "stablehlo.custom_call"(%6, %3) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<3x5xi8>, tensor<3x5xi8>) -> ()
    "func.return"(%6) : (tensor<3x5xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[-3, -2, 2, 0, -3, 0], [0, 0, 0, 1, 3, -2], [-4, 0, -5, -2, 3, -3], [-7, -1, 0, -2, 1, 1]]> : tensor<4x6xi8>}> : () -> tensor<4x6xi8>
    "func.return"(%1) : (tensor<4x6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<3x5xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[[-3, -2, 0, -3, -3], [-4, -5, -5, -2, -3], [-7, -5, -5, -2, -3]]> : tensor<3x5xi8>}> : () -> tensor<3x5xi8>
    "func.return"(%0) : (tensor<3x5xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

