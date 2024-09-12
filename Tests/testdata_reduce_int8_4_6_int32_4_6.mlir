"builtin.module"() <{sym_name = "jit_main"}> ({
  "func.func"() <{function_type = () -> tensor<6xi8>, res_attrs = [{jax.result_info = "[0]", mhlo.layout_mode = "default"}], sym_name = "main", sym_visibility = "public"}> ({
    %3 = "func.call"() <{callee = @inputs}> : () -> tensor<4x6xi8>
    %4 = "func.call"() <{callee = @expected}> : () -> tensor<6xi8>
    %5 = "stablehlo.constant"() <{value = dense<3> : tensor<i8>}> : () -> tensor<i8>
    %6 = "stablehlo.reduce"(%3, %5) <{dimensions = array<i64: 0>}> ({
    ^bb0(%arg0: tensor<i8>, %arg1: tensor<i8>):
      %7 = "stablehlo.add"(%arg0, %arg1) : (tensor<i8>, tensor<i8>) -> tensor<i8>
      "stablehlo.return"(%7) : (tensor<i8>) -> ()
    }) : (tensor<4x6xi8>, tensor<i8>) -> tensor<6xi8>
    "stablehlo.custom_call"(%6, %4) <{call_target_name = "check.expect_eq", has_side_effect = true}> : (tensor<6xi8>, tensor<6xi8>) -> ()
    "func.return"(%6) : (tensor<6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<4x6xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "inputs", sym_visibility = "private"}> ({
    %1 = "stablehlo.constant"() <{value = dense<[[2, -1, 0, 0, 2, 1], [1, 0, -1, -4, 1, -4], [2, 1, 1, -3, 0, 6], [0, 0, -3, 3, 6, 0]]> : tensor<4x6xi8>}> : () -> tensor<4x6xi8>
    %2 = "stablehlo.constant"() <{value = dense<[[-3, 1, 2, 2, -1, -1], [1, -1, -1, 3, 0, -2], [-3, 0, 0, -2, -2, 1], [5, 1, 1, -3, -3, 1]]> : tensor<4x6xi32>}> : () -> tensor<4x6xi32>
    "func.return"(%1) : (tensor<4x6xi8>) -> ()
  }) : () -> ()
  "func.func"() <{function_type = () -> tensor<6xi8>, res_attrs = [{mhlo.layout_mode = "default"}], sym_name = "expected", sym_visibility = "private"}> ({
    %0 = "stablehlo.constant"() <{value = dense<[8, 3, 0, -1, 12, 6]> : tensor<6xi8>}> : () -> tensor<6xi8>
    "func.return"(%0) : (tensor<6xi8>) -> ()
  }) : () -> ()
}) {mhlo.num_partitions = 1 : i32, mhlo.num_replicas = 1 : i32} : () -> ()

